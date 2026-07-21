"""
ats_engine.py — HireFlow-Lite ATS Scoring Engine
==================================================

Standalone module implementing four ATS (Applicant Tracking System) resume-
scoring algorithms.  Every function is **synchronous**; no async anywhere.

Algorithms
----------
1. **BM25**            — Classic lexical scoring via rank-bm25.
2. **Neural (SBERT)**  — Semantic similarity via sentence-transformers.
3. **Hybrid**          — Weighted blend of BM25 (40 %) + SBERT (60 %).
4. **ColBERT**         — Late-interaction retrieval via ragatouille.

Public API
----------
- extract_jd_keywords(jd_text)          → list[str]
- compute_skill_match(resume, jd_kws)   → (matched, missing)
- score_bm25(resume_text, jd_text, resume_skills)
- score_neural(resume_text, jd_text, resume_skills, device)
- score_hybrid(resume_text, jd_text, resume_skills, device)
- score_colbert(resume_text, jd_text, resume_skills)
- run_ats(resume_text, jd_text, resume_skills, algo, device) → dict
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import warnings
from typing import Any

import httpx

# Force UTF-8 stdout/stderr on Windows to avoid UnicodeEncodeError
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


# ── NLTK bootstrap (runs on import — lightweight, silent) ───────────────
import nltk

nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

# ── Standard-lib / lightweight imports ──────────────────────────────────
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ── Module-level caches for lazily-loaded heavy models ──────────────────
_spacy_nlp: Any | None = None
_sbert_model: Any | None = None
_sbert_device: str | None = None

# ── NLTK stopword set (cheap to build once) ─────────────────────────────
_NLTK_STOPWORDS: set[str] = set(stopwords.words("english"))


# =====================================================================
#  Helper — lazy spaCy loader
# =====================================================================
def _get_spacy_nlp():
    """Return the cached spaCy ``en_core_web_sm`` pipeline, loading it once."""
    global _spacy_nlp
    if _spacy_nlp is None:
        print("  → Loading spaCy en_core_web_sm model...")
        import spacy

        try:
            _spacy_nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Model not installed — attempt a download first.
            print("  → en_core_web_sm not found; downloading…")
            from spacy.cli import download as spacy_download

            spacy_download("en_core_web_sm")
            _spacy_nlp = spacy.load("en_core_web_sm")
        print("  → spaCy model loaded.")
    return _spacy_nlp


# =====================================================================
#  Helper — lazy SBERT loader
# =====================================================================
def _get_sbert_model(device: str = "cpu"):
    """Return the cached SentenceTransformer model, loading it once.

    If the caller changes ``device`` between calls the model is reloaded.
    """
    global _sbert_model, _sbert_device
    if _sbert_model is None or _sbert_device != device:
        print(f"  → Loading SBERT model (all-MiniLM-L6-v2) on '{device}'...")
        from sentence_transformers import SentenceTransformer

        _sbert_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        _sbert_device = device
        print("  → SBERT model loaded.")
    return _sbert_model


# =====================================================================
#  1. JD Keyword Extraction
# =====================================================================
def extract_jd_keywords(jd_text: str) -> list[str]:
    """Extract deduplicated noun/proper-noun keywords from a job description.

    Uses spaCy ``en_core_web_sm`` (lazy-loaded) to tokenise the text,
    removes stopwords, and keeps only tokens tagged as **NOUN** or
    **PROPN**.

    Parameters
    ----------
    jd_text : str
        Raw job description text.

    Returns
    -------
    dict[str, list[str]]
        Dictionary containing 'required', 'preferred', and 'all' keyword lists.
    """
    if not jd_text or not jd_text.strip():
        return {"required": [], "preferred": [], "all": []}

    print("  → Extracting JD keywords with spaCy…")
    nlp = _get_spacy_nlp()
    
    # Split JD into required and preferred
    required_text = jd_text
    preferred_text = ""
    
    import re
    pref_match = re.search(r"(preferred|nice to have|bonus|desired) skills?:?(.*?)(?:\n\n|\Z)", jd_text, re.IGNORECASE | re.DOTALL)
    if pref_match:
        preferred_text = pref_match.group(2)
        required_text = required_text.replace(pref_match.group(0), "")
        
    req_match = re.search(r"(required|core|must have|essential) skills?:?(.*?)(?:\n\n|\Z)", required_text, re.IGNORECASE | re.DOTALL)
    if req_match:
        required_text = req_match.group(2)

    STOP_SKILLS = {
        # Generic JD structure words
        "developer", "candidate", "service", "api", "system", "application",
        "project", "experience", "engineer", "work", "team", "skills",
        "knowledge", "ability", "responsibilities", "opportunity", "company",
        "years", "role", "job", "software", "development", "environment",
        "requirements", "requirement",
        # Generic adjectives / nouns that leak from JD prose
        "title", "type", "time", "minimum", "year", "senior", "junior", "mid",
        "good", "strong", "deep", "solid", "excellent", "familiarity",
        "communication", "teamwork", "principle", "design", "process",
        "cross", "functional", "end", "complete", "full", "plus", "nice",
        "backend", "frontend", "fullstack", "stack", "level", "day", "week",
        "month", "basis", "way", "understanding", "understanding", "aspect",
        "use", "used", "using", "build", "building", "create", "working",
        "including", "related", "relevant", "key", "core", "bonus", "desired",
        "preferred", "required", "must", "nice", "basic", "advanced",
        "field", "area", "domain", "industry", "sector", "position",
        "hire", "hiring", "seek", "looking", "ideal", "following", "below",
        "range", "scale", "scope", "impact", "value", "high", "low",
        "new", "existing", "large", "small", "complex", "simple",
        "platform", "tool", "framework",  # too generic on their own
        # Single tokens that are fragments of multi-word tech terms.
        # 'boot' is blocked here because 'spring boot' is captured as a
        # multi-word noun chunk in the chunk phase below.
        "boot", "skill", "expertise",
        "proficiency", "proficient", "language",
    }
    
    tech_keywords = {
        "python", "java", "c++", "c", "c#", "javascript", "typescript", "react", "angular",
        "vue", "node.js", "express", "django", "flask", "fastapi", "spring", "spring boot",
        "hibernate", "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ci/cd",
        "linux", "git", "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
        "machine learning", "deep learning", "ai", "nlp", "computer vision",
        "pytorch", "tensorflow", "keras", "pandas", "numpy", "scikit-learn",
        "data science", "data engineering", "spark", "hadoop", "kafka",
        "devops", "agile", "scrum", "html", "css", "next.js", "nextjs",
        "graphql", "rest api", "restful api", "rest apis", "restful apis",
        "microservices", "bash", "shell", "powershell", "golang", "rust", "ruby",
        "llms", "genai", "prompt engineering", "rag", "vector databases",
        "junit", "mcp", "langchain", "langsmith", "openai", "huggingface",
        "spring boot", "hibernate", "jpa", "maven", "gradle", "junit", "jvm",
    }

    def _extract_from_text(text: str) -> list[str]:
        if not text.strip(): return []
        doc = nlp(text)
        seen = set()
        kws = []
        for token in doc:
            if token.is_stop or token.is_punct or token.is_space: continue
            lower = token.lemma_.lower()
            if lower in STOP_SKILLS: continue
            if token.pos_ in ("NOUN", "PROPN") or lower in tech_keywords:
                if lower not in seen and len(lower) > 1:
                    seen.add(lower)
                    kws.append(lower)
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.lower().strip()
            if chunk_text.startswith("a "): chunk_text = chunk_text[2:]
            if chunk_text.startswith("an "): chunk_text = chunk_text[3:]
            if chunk_text.startswith("the "): chunk_text = chunk_text[4:]
            if chunk_text in STOP_SKILLS: continue
            if chunk_text not in seen and len(chunk_text.split()) > 1:
                if not any(stop in chunk_text.split() for stop in STOP_SKILLS):
                    seen.add(chunk_text)
                    kws.append(chunk_text)
        return kws

    req_kws = _extract_from_text(required_text)
    pref_kws = _extract_from_text(preferred_text)
    all_kws = list(dict.fromkeys(req_kws + pref_kws))

    # If parsing failed to find explicitly labeled required skills, fallback to treating all as required
    if not req_kws and all_kws:
        req_kws = all_kws

    print(f"  → Extracted {len(req_kws)} Required, {len(pref_kws)} Preferred keywords.")
    return {"required": req_kws, "preferred": pref_kws, "all": all_kws}


# =====================================================================
#  2. Skill Matching
# =====================================================================
def compute_skill_match(
    resume_skills: list[str],
    jd_keywords: list[str],
    raw_text: str = "",
) -> tuple[list[str], list[str]]:
    """Compare resume skills against JD keywords (case-insensitive).

    Uses two strategies:
    1. Match JD keywords against the parsed skills list.
    2. Fallback: if not found in parsed skills, check if the keyword
       appears anywhere in the raw resume text.  This prevents
       disagreement between matched_skills and explanation text.
    """
    import re
    resume_lower: list[str] = [s.lower().strip() for s in resume_skills if s.strip()]
    jd_lower: list[str] = [kw.lower().strip() for kw in jd_keywords if kw.strip()]
    raw_lower = raw_text.lower() if raw_text else ""

    matched: list[str] = []
    missing: list[str] = []
    for kw in jd_lower:
        found = False
        # Strategy 1: check parsed skills list
        for skill in resume_lower:
            pattern = rf"\b{re.escape(kw)}\b"
            if re.search(pattern, skill) or re.search(rf"\b{re.escape(skill)}\b", kw):
                found = True
                break
        # Strategy 2: fallback — check raw resume text
        if not found and raw_lower and len(kw) > 1:
            try:
                if re.search(rf"\b{re.escape(kw)}\b", raw_lower):
                    found = True
            except re.error:
                pass
        if found:
            matched.append(kw)
        else:
            missing.append(kw)

    # Deduplicate while preserving order
    matched = list(dict.fromkeys(matched))
    missing = list(dict.fromkeys(missing))

    return matched, missing


# =====================================================================
#  MODE 1 — BM25 Scoring (corpus-relative)
# =====================================================================
def score_bm25(
    resume_text: str,
    jd_text: str,
    resume_skills: list[str],
) -> dict[str, Any]:
    """Score a resume against a JD using BM25Okapi over pseudo-documents.

    The resume is split into per-line pseudo-documents (each non-empty
    line ≥15 chars becomes one document in the corpus).  This gives
    BM25's IDF a real multi-document corpus to work with, rather than
    a single-document corpus where IDF is meaningless.

    The MAX score across all pseudo-documents represents how well the
    best-matching section of the resume aligns with the JD.

    Normalisation uses a fixed empirical ceiling (12.0) derived from
    typical BM25 score ranges for short text segments.

    Parameters
    ----------
    resume_text : str
        Plain-text resume.
    jd_text : str
        Plain-text job description.
    resume_skills : list[str]
        Pre-extracted resume skills for skill matching.

    Returns
    -------
    dict
        Keys: ``score``, ``algo_used``, ``matched_skills``,
        ``missing_skills``, ``explanation``.
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as exc:
        return _error_result(
            "bm25",
            resume_skills,
            jd_text,
            f"rank-bm25 is not installed: {exc}",
        )

    print("  → Tokenizing with BM25…")

    jd_tokens = _clean_tokens(jd_text)
    if not jd_tokens:
        return _error_result(
            "bm25",
            resume_skills,
            jd_text,
            "JD produced no tokens after cleaning.",
        )

    # Split resume into per-line pseudo-documents for a real BM25 corpus
    pseudo_docs = []
    for line in resume_text.split("\n"):
        line = line.strip()
        if len(line) >= 15:
            tokens = _clean_tokens(line)
            if tokens:
                pseudo_docs.append(tokens)

    if not pseudo_docs:
        # Fallback: if no lines are long enough, use the whole resume as one doc
        resume_tokens = _clean_tokens(resume_text)
        if not resume_tokens:
            return _error_result(
                "bm25",
                resume_skills,
                jd_text,
                "Resume produced no tokens after cleaning.",
            )
        pseudo_docs = [resume_tokens]

    print(f"  → Building BM25 index ({len(pseudo_docs)}-doc corpus)…")
    bm25 = BM25Okapi(pseudo_docs)
    # Do NOT manually override IDF values — with a real corpus, IDF is meaningful

    print("  → Scoring JD query against resume pseudo-docs…")
    raw_scores = bm25.get_scores(jd_tokens)
    # Take the MAX score: the best-matching section represents the resume
    raw_score: float = float(max(raw_scores)) if len(raw_scores) > 0 else 0.0

    # Normalise using a fixed empirical ceiling (increased to 25.0 to prevent easy 100%)
    BM25_CEILING = 25.0
    normalised = (raw_score / BM25_CEILING) * 100.0
    normalised = round(min(max(normalised, 0.0), 100.0), 2)

    # Top matching terms
    all_resume_tokens = set()
    for doc in pseudo_docs:
        all_resume_tokens.update(doc)
    top_terms = [t for t in jd_tokens if t in all_resume_tokens]
    top_terms = list(dict.fromkeys(top_terms))[:15]

    # Skill matching (with raw_text fallback)
    jd_kws = extract_jd_keywords(jd_text)
    req_kws = jd_kws["required"]
    pref_kws = jd_kws["preferred"]

    req_matched, req_missing = compute_skill_match(resume_skills, req_kws, raw_text=resume_text)
    pref_matched, pref_missing = compute_skill_match(resume_skills, pref_kws, raw_text=resume_text)

    # Skill score calculation:
    # If preferred keywords exist   → 40% req + 20% pref + 20% semantic (total 80)
    # If NO preferred keywords exist → 60% req + 0% pref + 20% semantic (total 80)
    # This prevents giving free 20/20 points when JD has no preferred section.
    if req_kws and pref_kws:
        req_score  = 40.0 * min(1.0, len(req_matched) / len(req_kws))
        pref_score = 20.0 * min(1.0, len(pref_matched) / len(pref_kws))
    elif req_kws:
        req_score  = 60.0 * min(1.0, len(req_matched) / len(req_kws))
        pref_score = 0.0
    else:
        req_score  = 60.0
        pref_score = 0.0
    semantic_score = normalised * 0.20
    
    partial_score = req_score + pref_score + semantic_score

    explanation = (
        f"BM25 semantic: {normalised:.1f}/100. "
        f"Required skills missing ({len(req_missing)}): -{len(req_missing)*7}pts. "
        f"Preferred skills missing ({len(pref_missing)}): -{len(pref_missing)*2}pts. "
    )

    print(f"  → BM25 Partial Score (out of 80): {partial_score:.1f}")
    return {
        "score": partial_score,  # This is out of 80 right now
        "semantic_score": normalised,
        "req_score": req_score,
        "pref_score": pref_score,
        "algo_used": "bm25",
        "matched_skills": req_matched + pref_matched,
        "missing_skills": req_missing + pref_missing,
        "explanation": explanation,
    }


# =====================================================================
#  MODE 2 — Neural (SBERT) Scoring
# =====================================================================
def score_neural(
    resume_text: str,
    jd_text: str,
    resume_skills: list[str],
    device: str = "cpu",
    hyre_text: str = "",
) -> dict[str, Any]:
    """Score a resume against a JD using SBERT cosine similarity.

    Uses ``all-MiniLM-L6-v2`` from sentence-transformers.  The model is
    lazy-loaded and cached at module level.

    Parameters
    ----------
    resume_text : str
        Plain-text resume.
    jd_text : str
        Plain-text job description.
    resume_skills : list[str]
        Pre-extracted resume skills for skill matching.
    device : str, optional
        Torch device for model loading **and** encoding (default ``'cpu'``).
    hyre_text : str, optional
        Hypothetical resume text to expand the job description representation.

    Returns
    -------
    dict
        Standard result dict.
    """
    try:
        from sentence_transformers import util as st_util
    except ImportError as exc:
        return _error_result(
            "neural",
            resume_skills,
            jd_text,
            f"sentence-transformers is not installed: {exc}",
        )

    print("  → Loading SBERT model…")
    model = _get_sbert_model(device)

    print("  → Encoding resume and JD with SBERT…")
    
    # Split resume into chunks (sentences/lines) rather than one massive vector
    import torch
    resume_sentences = [s.strip() for s in resume_text.split('\n') if len(s.strip()) > 15]
    if not resume_sentences:
        resume_sentences = [resume_text]
        
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        resume_embs = model.encode(resume_sentences, convert_to_tensor=True, device=device)
        # Apply HYRE expansion if present
        effective_jd = jd_text
        if hyre_text:
            effective_jd = f"{jd_text}\n\n[HYPOTHETICAL IDEAL RESUME REFERENCE]\n{hyre_text}"
        jd_emb = model.encode(effective_jd, convert_to_tensor=True, device=device)

    print("  → Computing cosine similarity…")
    # Compute similarity between JD and every resume chunk
    cosine_sims = st_util.cos_sim(jd_emb, resume_embs)[0]
    
    # Take the top K similarities to reward dense relevant sections
    k = min(5, len(cosine_sims))
    top_k_sims = torch.topk(cosine_sims, k=k).values
    cosine_sim = float(torch.mean(top_k_sims))
    
    # Calibrate score: raw cosine similarity rarely goes above 0.5 for different texts
    # We stretch the range [0.15, 0.5] -> [0, 100]
    normalised = max(0.0, (cosine_sim - 0.15) / 0.35) * 100.0
    normalised = round(min(normalised, 100.0), 2)

    jd_kws = extract_jd_keywords(jd_text)
    req_kws = jd_kws["required"]
    pref_kws = jd_kws["preferred"]

    req_matched, req_missing = compute_skill_match(resume_skills, req_kws, raw_text=resume_text)
    pref_matched, pref_missing = compute_skill_match(resume_skills, pref_kws, raw_text=resume_text)

    # Skill score: no free preferred points when JD has no preferred section
    if req_kws and pref_kws:
        req_score  = 40.0 * min(1.0, len(req_matched) / len(req_kws))
        pref_score = 20.0 * min(1.0, len(pref_matched) / len(pref_kws))
    elif req_kws:
        req_score  = 60.0 * min(1.0, len(req_matched) / len(req_kws))
        pref_score = 0.0
    else:
        req_score  = 60.0
        pref_score = 0.0
    semantic_score = normalised * 0.20
    
    partial_score = req_score + pref_score + semantic_score

    explanation = (
        f"SBERT semantic: {normalised:.1f}/100. "
        f"Required skills missing ({len(req_missing)}). "
        f"Preferred skills missing ({len(pref_missing)}). "
    )

    print(f"  → Neural Partial Score (out of 80): {partial_score:.1f}")
    return {
        "score": partial_score,
        "semantic_score": normalised,
        "req_score": req_score,
        "pref_score": pref_score,
        "algo_used": "neural",
        "matched_skills": req_matched + pref_matched,
        "missing_skills": req_missing + pref_missing,
        "explanation": explanation,
    }


# =====================================================================
#  MODE 3 — Hybrid (BM25 40 % + SBERT 60 %)
# =====================================================================
def score_hybrid(
    resume_text: str,
    jd_text: str,
    resume_skills: list[str],
    device: str = "cpu",
    hyre_text: str = "",
) -> dict[str, Any]:
    """Score a resume using a weighted hybrid of BM25 and SBERT.

    ``final = (bm25_norm × 0.4 + sbert_norm × 0.6) × 100``

    Parameters
    ----------
    resume_text, jd_text, resume_skills, device
        Same as the individual scorers.
    hyre_text
        Hypothetical resume text to expand the job description representation.

    Returns
    -------
    dict
        Standard result dict.
    """
    print("  → Running hybrid scoring (BM25 40 % + SBERT 60 %)…")

    # ── BM25 component ──────────────────────────────────────────────
    print("  → [Hybrid] Computing BM25 component…")
    bm25_result = score_bm25(resume_text, jd_text, resume_skills)
    bm25_norm = bm25_result["semantic_score"] / 100.0  # 0-1

    # ── SBERT component ─────────────────────────────────────────────
    print("  → [Hybrid] Computing SBERT component…")
    neural_result = score_neural(resume_text, jd_text, resume_skills, device=device, hyre_text=hyre_text)
    sbert_norm = neural_result["semantic_score"] / 100.0  # 0-1

    # ── Combine ─────────────────────────────────────────────────────
    semantic_score_norm = (bm25_norm * 0.4 + sbert_norm * 0.6) * 100.0
    
    # We use the neural_result's skill extraction logic since it's the same
    req_score = neural_result["req_score"]
    pref_score = neural_result["pref_score"]
    semantic_score = semantic_score_norm * 0.20
    
    partial_score = req_score + pref_score + semantic_score
    partial_score = round(min(max(partial_score, 0.0), 80.0), 2)

    explanation = (
        f"Hybrid semantic: {semantic_score_norm:.1f}/100. "
        f"Required skills missing: -{(40.0 - req_score):.0f}pts. "
        f"Preferred skills missing: -{(20.0 - pref_score):.0f}pts. "
    )

    print(f"  → Hybrid Partial Score (out of 80): {partial_score:.1f}")
    return {
        "score": partial_score,
        "semantic_score": semantic_score_norm,
        "req_score": req_score,
        "pref_score": pref_score,
        "algo_used": "hybrid_efficient",
        "matched_skills": neural_result["matched_skills"],
        "missing_skills": neural_result["missing_skills"],
        "explanation": explanation,
    }


# =====================================================================
#  MODE 4 — ColBERT Scoring
# =====================================================================
def score_colbert(
    resume_text: str,
    jd_text: str,
    resume_skills: list[str],
) -> dict[str, Any]:
    """Score a resume against a JD using ColBERT late-interaction retrieval.

    Uses ``ragatouille`` with the
    ``answerdotai/answerai-colbert-small-v1`` model.  If ragatouille is
    not installed the function returns a graceful error result with
    ``score=0``.

    Parameters
    ----------
    resume_text : str
        Plain-text resume.
    jd_text : str
        Plain-text job description.
    resume_skills : list[str]
        Pre-extracted resume skills for skill matching.

    Returns
    -------
    dict
        Standard result dict.
    """
    try:
        from ragatouille import RAGPretrainedModel
    except ImportError:
        return _error_result(
            "colbert",
            resume_skills,
            jd_text,
            (
                "ragatouille is not installed.  "
                "Install it with:  pip install ragatouille"
            ),
        )

    try:
        print("  → Loading ColBERT model (answerai-colbert-small-v1)…")
        rag = RAGPretrainedModel.from_pretrained(
            "answerdotai/answerai-colbert-small-v1"
        )

        # ragatouille needs a writable index directory
        index_dir = os.path.join(tempfile.gettempdir(), "hireflow_colbert_index")
        os.makedirs(index_dir, exist_ok=True)

        print("  → Indexing resume text with ColBERT…")
        rag.index(
            collection=[resume_text],
            index_name="resume_index",
            max_document_length=512,
            split_documents=True,
        )

        print("  → Querying index with JD…")
        results = rag.search(query=jd_text, k=1)

        if results and len(results) > 0:
            raw_score = float(results[0].get("score", 0.0))
        else:
            raw_score = 0.0

        normalised = (raw_score / 150.0) * 100.0
        normalised = round(min(max(normalised, 0.0), 100.0), 2)

        jd_kws = extract_jd_keywords(jd_text)
        req_kws = jd_kws["required"]
        pref_kws = jd_kws["preferred"]

        req_matched, req_missing = compute_skill_match(resume_skills, req_kws, raw_text=resume_text)
        pref_matched, pref_missing = compute_skill_match(resume_skills, pref_kws, raw_text=resume_text)

        # Skill score calculation:
        # If preferred keywords exist   → 40% req + 20% pref + 20% semantic (total 80)
        # If NO preferred keywords exist → 60% req + 0% pref + 20% semantic (total 80)
        if req_kws and pref_kws:
            req_score  = 40.0 * min(1.0, len(req_matched) / len(req_kws))
            pref_score = 20.0 * min(1.0, len(pref_matched) / len(pref_kws))
        elif req_kws:
            req_score  = 60.0 * min(1.0, len(req_matched) / len(req_kws))
            pref_score = 0.0
        else:
            req_score  = 60.0
            pref_score = 0.0
        semantic_score = normalised * 0.20
        
        partial_score = req_score + pref_score + semantic_score

        explanation = (
            f"ColBERT semantic: {normalised:.1f}/100. "
            f"Required skills missing ({len(req_missing)}): -{len(req_missing)*7}pts. "
            f"Preferred skills missing ({len(pref_missing)}): -{len(pref_missing)*2}pts. "
        )

        print(f"  → ColBERT Partial Score (out of 80): {partial_score:.1f}")
        return {
            "score": partial_score,
            "semantic_score": normalised,
            "req_score": req_score,
            "pref_score": pref_score,
            "algo_used": "colbert",
            "matched_skills": req_matched + pref_matched,
            "missing_skills": req_missing + pref_missing,
            "explanation": explanation,
        }

    except Exception as exc:
        return _error_result(
            "colbert",
            resume_skills,
            jd_text,
            f"ColBERT scoring failed: {exc}",
        )


# =====================================================================
#  Dispatcher
# =====================================================================
_ALGO_MAP: dict[str, str] = {
    "bm25": "score_bm25",
    "neural": "score_neural",
    "hybrid_efficient": "score_hybrid",
    "colbert": "score_colbert",
}


def run_ats(
    resume_text: str,
    jd_text: str,
    resume_skills: list[str],
    algo: str = "hybrid_efficient",
    device: str = "cpu",
    hyre_text: str = "",
) -> dict[str, Any]:
    """Dispatch to the requested ATS scoring algorithm.

    Parameters
    ----------
    resume_text : str
        Plain-text resume content.
    jd_text : str
        Plain-text job description.
    resume_skills : list[str]
        Skills pre-extracted from the resume.
    algo : str, optional
        Algorithm name — one of ``'bm25'``, ``'neural'``,
        ``'hybrid_efficient'``, ``'colbert'`` (default ``'hybrid_efficient'``).
    device : str, optional
        Torch device string (default ``'cpu'``).
    hyre_text : str, optional
        Hypothetical resume text to expand the job description representation.

    Returns
    -------
    dict
        Keys: ``score``, ``algo_used``, ``matched_skills``,
        ``missing_skills``, ``explanation``.

    Raises
    ------
    ValueError
        If ``algo`` is not a recognised algorithm name.
    """
    algo_key = algo.lower().strip()
    if algo_key not in _ALGO_MAP:
        supported = ", ".join(sorted(_ALGO_MAP.keys()))
        raise ValueError(
            f"Unknown algorithm '{algo}'. Supported: {supported}"
        )

    print(f"\n{'=' * 60}")
    print(f"  ATS Engine — algorithm: {algo_key}")
    print(f"{'=' * 60}")

    # Resolve function reference
    fn_name = _ALGO_MAP[algo_key]
    fn = globals()[fn_name]

    # Functions that accept a device parameter and optionally hyre_text
    if algo_key in ("neural", "hybrid_efficient"):
        result = fn(resume_text, jd_text, resume_skills, device=device, hyre_text=hyre_text)
    else:
        result = fn(resume_text, jd_text, resume_skills)

    print(f"\n  ✓ ATS scoring complete — {result['score']}/100 ({algo_key})")
    print(f"{'=' * 60}\n")
    return result


# =====================================================================
#  Internal helpers
# =====================================================================
def _clean_tokens(text: str) -> list[str]:
    """Tokenise text with NLTK, lowercase, remove stopwords & non-alpha."""
    tokens = word_tokenize(text.lower())
    return [
        t for t in tokens
        if t.isalpha() and t not in _NLTK_STOPWORDS and len(t) > 1
    ]


def _error_result(
    algo: str,
    resume_skills: list[str],
    jd_text: str,
    message: str,
) -> dict[str, Any]:
    """Build a standardised error result dict with score=0."""
    print(f"  ✗ Error ({algo}): {message}")

    # Still attempt skill matching so the caller gets partial info
    try:
        jd_keywords = extract_jd_keywords(jd_text)
        matched, missing = compute_skill_match(resume_skills, jd_keywords)
    except Exception:
        matched, missing = [], []

    return {
        "score": 0.0,
        "algo_used": algo,
        "matched_skills": matched,
        "missing_skills": missing,
        "explanation": f"Error: {message}",
    }


def _is_ollama_running(host: str) -> bool:
    """Helper to check if Ollama server is reachable."""
    try:
        with httpx.Client(timeout=1.0) as client:
            resp = client.get(host)
            return resp.status_code == 200
    except Exception:
        return False


def _get_ollama_model(ollama_host: str, preferred_model: str) -> str:
    """Fetch installed models from local Ollama and pick the best match."""
    try:
        url = f"{ollama_host}/api/tags"
        with httpx.Client(timeout=2.0) as client:
            resp = client.get(url)
            if resp.status_code == 200:
                models = [m.get("name") for m in resp.json().get("models", [])]
                # If preferred model matches exactly
                if preferred_model in models:
                    return preferred_model
                
                # Check priority list
                priority_list = ["lfm2.5-thinking", "qwen3.5:2b", "minicpm-v4.6:1b"]
                for p in priority_list:
                    p_lower = p.lower()
                    p_base = p_lower.split(":")[0]
                    for m in models:
                        m_lower = m.lower()
                        # Match exactly, by substring, or by base prefix (e.g. qwen, minicpm, lfm)
                        if p_lower == m_lower or p_lower in m_lower or p_base in m_lower:
                            return m
                
                # If preferred model base matches
                pref_base = preferred_model.split(":")[0]
                for m in models:
                    if pref_base in m or m.startswith(pref_base):
                        return m
                
                if models:
                    return models[0]
    except Exception:
        pass
    return preferred_model


def _query_llm(prompt: str, system_instruction: str = "") -> str:
    """Send a prompt to local Ollama, Gemini, or OpenAI API via HTTP."""
    gemini_key = os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"
    ollama_model = os.getenv("OLLAMA_MODEL", "lfm2.5-thinking")
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    # Auto-detect if Ollama is running and no cloud API keys are set
    if not use_ollama and not gemini_key and not openai_key:
        if _is_ollama_running(ollama_host):
            use_ollama = True

    # 1. Local Ollama Route
    if use_ollama:
        model_name = _get_ollama_model(ollama_host, ollama_model)
        print(f"  → [Local LLM] Querying model '{model_name}' via Ollama...")
        
        url = f"{ollama_host}/api/chat"
        headers = {"Content-Type": "application/json"}
        
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})
        
        data = {
            "model": model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.2
            }
        }
        
        try:
            # First-time local LLM load and execution on laptops might take time, set to 300s
            with httpx.Client(timeout=300.0) as client:
                resp = client.post(url, headers=headers, json=data)
            if resp.status_code != 200:
                raise RuntimeError(f"Ollama returned status code {resp.status_code}: {resp.text}")
            
            resp_json = resp.json()
            content = resp_json.get("message", {}).get("content", "")
            return content.strip()
        except Exception as e:
            # Fall back to cloud APIs if keys are available
            if gemini_key or openai_key:
                print(f"  → [Local LLM WARNING] Ollama call failed: {e}. Falling back to Cloud API.")
                use_ollama = False
            else:
                raise RuntimeError(f"Failed to query local Ollama model: {e}") from e

    # 2. Gemini API Route
    if not use_ollama and gemini_key:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}"
        headers = {"Content-Type": "application/json"}
        
        combined_prompt = prompt
        if system_instruction:
            combined_prompt = f"{system_instruction}\n\nUser Request:\n{prompt}"
            
        data = {
            "contents": [{
                "parts": [{"text": combined_prompt}]
            }]
        }
        
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(url, headers=headers, json=data)
            if resp.status_code != 200:
                raise RuntimeError(f"Gemini API returned status code {resp.status_code}: {resp.text}")
            
            resp_json = resp.json()
            candidates = resp_json.get("candidates", [])
            if candidates:
                text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                return text.strip()
            raise RuntimeError(f"Gemini API returned no candidates: {resp.text}")
        except Exception as e:
            raise RuntimeError(f"Failed to query Gemini API: {e}") from e

    # 3. OpenAI API Route
    elif not use_ollama and openai_key:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {openai_key}",
            "Content-Type": "application/json",
        }
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})
        
        data = {
            "model": "gpt-4o-mini",
            "messages": messages,
            "temperature": 0.2,
        }
        
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(url, headers=headers, json=data)
            if resp.status_code != 200:
                raise RuntimeError(f"OpenAI API returned status code {resp.status_code}: {resp.text}")
            
            resp_json = resp.json()
            choices = resp_json.get("choices", [])
            if choices:
                text = choices[0].get("message", {}).get("content", "")
                return text.strip()
            raise RuntimeError(f"OpenAI API returned no choices: {resp.text}")
        except Exception as e:
            raise RuntimeError(f"Failed to query OpenAI API: {e}") from e
            
    return ""


def generate_hypothetical_resume(jd_text: str) -> str:
    """Generate a hypothetical resume from a job description (HYRE)."""
    gemini_key = os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    # Auto-detect if Ollama is running and no cloud API keys are set
    if not use_ollama and not gemini_key and not openai_key:
        if _is_ollama_running(ollama_host):
            use_ollama = True

    if not gemini_key and not openai_key and not use_ollama:
        return ""
        
    system_instruction = (
        "You are an expert technical recruiter. Your task is to generate a brief, "
        "hypothetical ideal resume candidate description matching the given job description. "
        "Include candidate summary, key skills, and brief past experience outlines. "
        "Output ONLY the text of the hypothetical resume."
    )
    
    prompt = f"Job Description:\n{jd_text}\n\nGenerate the hypothetical ideal resume:"
    
    try:
        print("  → [HYRE] Querying LLM to generate hypothetical resume...")
        hypothetical = _query_llm(prompt, system_instruction)
        print("  → [HYRE] Hypothetical resume successfully generated.")
        return hypothetical
    except Exception as e:
        print(f"  → [HYRE WARNING] Failed to generate hypothetical resume: {e}. Falling back.")
        return ""


def re_rank_candidates_llm(candidates_list: list[dict], jd_text: str) -> list[dict]:
    """Perform Stage 2 listwise re-ranking using an LLM (Ollama, Gemini, or OpenAI)."""
    if not candidates_list:
        return []
        
    gemini_key = os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    # Auto-detect if Ollama is running and no cloud API keys are set
    if not use_ollama and not gemini_key and not openai_key:
        if _is_ollama_running(ollama_host):
            use_ollama = True

    if not gemini_key and not openai_key and not use_ollama:
        print("  → [Re-ranking WARNING] No API keys or local LLM found. Skipping Stage 2 LLM Re-ranking.")
        # Attach default empty explanations
        for c in candidates_list:
            if "llm_explanation" not in c:
                c["llm_explanation"] = {
                    "strengths": [],
                    "gaps": [],
                    "fit_justification": "Stage 1 scoring retained (LLM re-ranker bypassed)."
                }
        return candidates_list

    print(f"  → [Re-ranking] Running Stage 2 LLM listwise re-ranking on {len(candidates_list)} candidates...")
    
    # Construct a single compact prompt listing candidates and their profiles
    candidates_summary = []
    for idx, c in enumerate(candidates_list):
        name = c.get("name", f"Candidate_{idx+1}")
        skills = ", ".join(c.get("matched_skills", []))
        missing = ", ".join(c.get("missing_skills", []))
        stage1_score = c.get("score", 0.0)
        resume_snippet = c.get("resume_text", "")[:1200]
        
        candidates_summary.append(
            f"Candidate ID: {idx+1}\n"
            f"Name: {name}\n"
            f"Stage 1 Score: {stage1_score}\n"
            f"Matched Skills: {skills}\n"
            f"Missing Skills: {missing}\n"
            f"Resume Text Snippet: {resume_snippet}\n"
            f"----------------------------------------"
        )
        
    prompt = (
        f"Job Description:\n{jd_text}\n\n"
        f"Candidates to Rank:\n" + "\n".join(candidates_summary) + "\n\n"
        f"Your task is to evaluate these candidates based on their fit for the role. "
        f"You must ONLY cite strengths that are explicitly present in the candidate's 'Matched Skills' list. "
        f"Do NOT infer or hallucinate skills that are not explicitly provided. "
        f"For each candidate, output:\n"
        f"1. 2-3 specific strengths (grounded ONLY in their matched skills).\n"
        f"2. 1-2 gaps or missing qualifications.\n"
        f"3. A 1-sentence justification of their fit.\n\n"
        f"Respond ONLY with a valid JSON block of this format:\n"
        f"{{\n"
        f"  \"evaluations\": [\n"
        f"    {{\n"
        f"      \"candidate_id\": 1,\n"
        f"      \"strengths\": [\"Python\", \"FastAPI\"],\n"
        f"      \"gaps\": [\"Missing Kubernetes\"],\n"
        f"      \"justification\": \"Solid backend developer with required Python/FastAPI experience.\"\n"
        f"    }}\n"
        f"  ]\n"
        f"}}"
    )
    
    system_instruction = (
        "You are an expert recruitment system. Output ONLY a clean, valid JSON object matching "
        "the requested schema. Do not output any markdown wrapper or surrounding text. "
        "Never hallucinate skills not present in the candidate's profile."
    )
    
    try:
        response_text = _query_llm(prompt, system_instruction)
        
        # Strip markdown syntax if returned
        json_clean = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", response_text.strip())
        
        ranking_data = json.loads(json_clean)
        evaluations = ranking_data.get("evaluations", [])
        
        # Build lookup map
        rank_map = {int(r["candidate_id"]): r for r in evaluations if "candidate_id" in r}
        
        updated_candidates = []
        for idx, c in enumerate(candidates_list):
            cand_id = idx + 1
            if cand_id in rank_map:
                r_info = rank_map[cand_id]
                c["llm_explanation"] = {
                    "strengths": r_info.get("strengths", []),
                    "gaps": r_info.get("gaps", []),
                    "fit_justification": r_info.get("justification", "No justification provided.")
                }
            else:
                c["llm_explanation"] = {
                    "strengths": c.get("matched_skills", [])[:3],
                    "gaps": c.get("missing_skills", [])[:2],
                    "fit_justification": "Retained Stage 1 scoring due to parser fallback."
                }
            updated_candidates.append(c)
            
        # Re-sort by score descending
        updated_candidates.sort(key=lambda x: x["score"], reverse=True)
        print("  → [Re-ranking] Successfully re-ranked candidates and attached explanations.")
        return updated_candidates
        
    except Exception as e:
        print(f"  → [Re-ranking WARNING] LLM Re-ranking failed: {e}. Returning Stage 1 results.")
        # Attach default explanation block to avoid crashing UI
        for c in candidates_list:
            if "llm_explanation" not in c:
                c["llm_explanation"] = {
                    "strengths": [],
                    "gaps": [],
                    "fit_justification": "Stage 1 scoring retained (LLM re-ranker bypassed)."
                }
        return candidates_list
        print("  → [Re-ranking] Successfully re-ranked candidates and attached explanations.")
        return updated_candidates
        
    except Exception as e:
        print(f"  → [Re-ranking WARNING] LLM Re-ranking failed: {e}. Returning Stage 1 results.")
        # Attach default explanation block to avoid crashing UI
        for c in candidates_list:
            if "llm_explanation" not in c:
                c["llm_explanation"] = {
                    "strengths": [],
                    "gaps": [],
                    "fit_justification": "Stage 1 scoring retained (LLM re-ranker bypassed)."
                }
        return candidates_list
