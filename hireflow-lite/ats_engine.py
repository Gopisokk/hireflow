"""
ats_engine.py — HireFlow-Lite: Evidence Aggregation ATS Scoring Engine v2
=========================================================================

Root-cause fixes from v1:
  - Sigmoid midpoint was 0.65 → destroyed semantic for cosines in 0.18-0.36 range.
    Fixed: midpoint=0.30, k=15 now maps 0.28→44%, 0.36→71%, 0.50→95%.
  - Project evidence was SBERT-first — fails for short project texts.
    Fixed: keyword-first (required skills found in projects), SBERT secondary.
  - Skills in projects were weighted equal to skills just listed.
    Fixed: evidence multiplier — each project demonstrating a skill adds +50% weight.
  - Semantic was over-weighted (15%) and dominated ranking.
    Fixed: semantic reduced to 10% (supporting signal, not primary).

Component Weights (sum = 1.00):
  35%  Required Skill Match   (+ evidence multiplier from projects)
  25%  Project Evidence        (keyword-first, SBERT secondary)
  10%  Semantic Alignment      (calibrated cosine — supporting signal only)
  10%  Experience Relevance    (seniority + domain)
   8%  Keyword Coverage        (BM25-inspired JD term coverage)
   5%  Resume Quality          (structure, sections, completeness)
   4%  Education               (field + degree level)
   3%  Certifications          (cloud certs, open source, competitions)
   +0 to -15  Penalty Layer

Confidence Tiers:
  95–100  Exceptional Match   → Strongly Recommend
  85–94   Strong Match        → Recommend
  70–84   Good Match          → Recommend with Notes
  55–69   Potential Match     → Further Review
  40–54   Moderate Match      → Hold for Review
  25–39   Weak Match          → Not Recommended
  0–24    Poor Match          → Do Not Proceed
"""

from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN_WARNING"] = "1"

import re
import math
import warnings
from collections import Counter
from typing import Optional

import numpy as np

# ── SBERT Embedding Model (runs on CPU — 5ms per doc, leaves 100% VRAM for Ollama GPU) ──

DEVICE = "cpu"
_MODEL = None


def _get_model():
    from minilm import get_minilm_model
    return get_minilm_model(device="cpu")


def _encode(texts: list[str], model=None) -> np.ndarray:
    """Batch encode a list of texts. Returns (N, 384) normalized array."""
    m = model or _get_model()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return m.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            device=DEVICE,
            show_progress_bar=False,
            convert_to_numpy=True,
        )


# ── Confidence Tier System ────────────────────────────────────────────────────

TIERS = [
    (95, "Exceptional Match",  "Strongly Recommend",    "Interview immediately"),
    (85, "Strong Match",       "Recommend",             "Strong technical fit"),
    (70, "Good Match",         "Recommend with Notes",  "Most requirements met"),
    (55, "Potential Match",    "Further Review",        "Partial requirements met"),
    (40, "Moderate Match",     "Hold for Review",       "Some transferable skills"),
    (25, "Weak Match",         "Not Recommended",       "Significant gaps"),
    (0,  "Poor Match",         "Do Not Proceed",        "Does not meet requirements"),
]


def get_tier(score: float) -> dict:
    for threshold, label, recommendation, note in TIERS:
        if score >= threshold:
            return {
                "label":          label,
                "recommendation": recommendation,
                "note":           note,
                "threshold":      threshold,
            }
    return {"label": "Poor Match", "recommendation": "Do Not Proceed", "note": "Does not meet requirements", "threshold": 0}


# ── Skill Synonym Map ─────────────────────────────────────────────────────────

SKILL_SYNONYMS: dict[str, list[str]] = {
    "javascript":       ["javascript", "js", "ecmascript"],
    "typescript":       ["typescript", "ts"],
    "python":           ["python", "py", "django", "flask", "fastapi"],
    "java":             ["java", "spring boot", "spring", "maven", "gradle", "jvm"],
    "c++":              ["c++", "cpp", "c plus plus"],
    "c#":               ["c#", "csharp", ".net", "dotnet", "asp.net"],
    "rust":             ["rust", "tokio", "cargo", "rustlang", "async rust", "serde",
                         "actix", "axum", "wasm", "webassembly"],
    "async":            ["async", "asynchronous", "tokio", "async/await", "concurrency",
                         "concurrent", "threading", "parallelism"],
    "systems":          ["systems programming", "systems", "embedded", "os", "kernel",
                         "memory management", "memory safety", "zero-cost", "ffi",
                         "low-level", "performance-critical"],
    "go":               ["go", "golang", "goroutine"],
    "kotlin":           ["kotlin", "coroutines"],
    "swift":            ["swift", "swiftui", "xcode"],
    "ruby":             ["ruby", "rails", "ruby on rails"],
    "php":              ["php", "laravel", "symfony"],
    "react":            ["react", "reactjs", "react.js", "react native", "nextjs", "next.js"],
    "vue":              ["vue", "vuejs", "vue.js", "nuxt"],
    "angular":          ["angular", "angularjs"],
    "node.js":          ["node.js", "nodejs", "express.js", "fastify", "nestjs"],
    "machine learning": ["machine learning", "ml", "machinelearning", "sklearn",
                         "scikit-learn", "scikit learn"],
    "deep learning":    ["deep learning", "dl", "neural network", "pytorch",
                         "tensorflow", "keras"],
    "llm":              ["llm", "large language model", "gpt", "claude", "gemini",
                         "generative ai", "genai", "openai", "rag", "vector"],
    "nlp":              ["nlp", "natural language processing", "spacy", "nltk",
                         "huggingface", "bert", "transformers", "sentence transformer",
                         "sbert", "bm25", "embedding"],
    "computer vision":  ["computer vision", "cv", "opencv", "image processing",
                         "object detection", "yolo"],
    "sql":              ["sql", "mysql", "postgresql", "postgres", "sqlite",
                         "oracle", "tsql", "mssql", "database"],
    "nosql":            ["nosql", "mongodb", "cassandra", "dynamodb", "redis",
                         "couchdb", "firestore"],
    "aws":              ["aws", "amazon web services", "ec2", "s3", "lambda",
                         "rds", "cloudfront", "sagemaker"],
    "gcp":              ["gcp", "google cloud", "bigquery", "gke", "cloud run", "vertex ai"],
    "azure":            ["azure", "microsoft azure", "azure devops", "azure ml"],
    "docker":           ["docker", "container", "dockerfile", "podman", "containerization"],
    "kubernetes":       ["kubernetes", "k8s", "kubectl", "helm", "k3s", "openshift"],
    "ci/cd":            ["ci/cd", "cicd", "github actions", "jenkins", "gitlab ci",
                         "travis", "circleci", "pipeline"],
    "git":              ["git", "github", "gitlab", "bitbucket", "version control"],
    "linux":            ["linux", "unix", "bash", "shell", "ubuntu", "debian",
                         "centos", "rhel", "posix"],
    "rest api":         ["rest", "restful", "rest api", "http api", "openapi",
                         "swagger", "api design"],
    "graphql":          ["graphql", "graph ql", "apollo"],
    "microservices":    ["microservices", "micro services", "service mesh",
                         "kafka", "rabbitmq", "event driven"],
    "devops":           ["devops", "sre", "infrastructure", "terraform", "ansible",
                         "puppet", "chef", "iac"],
    "agile":            ["agile", "scrum", "kanban", "jira", "sprint"],
    "solana":           ["solana", "anchor", "spl token", "solana web3",
                         "solana rpc", "solana program"],
    "blockchain":       ["blockchain", "web3", "ethereum", "solidity",
                         "smart contract", "defi", "dapp", "nft", "x402"],
    "testing":          ["unit testing", "integration testing", "pytest", "jest",
                         "junit", "tdd", "test driven"],
    "data structures":  ["data structures", "algorithms", "dsa", "leetcode",
                         "competitive programming"],
    "open source":      ["open source", "opensource", "github contribution",
                         "open-source", "contributor", "maintainer"],
}

# Build reverse lookup: alias → canonical
_ALIAS_MAP: dict[str, str] = {}
for _canonical, _aliases in SKILL_SYNONYMS.items():
    for _alias in _aliases:
        _ALIAS_MAP[_alias.lower()] = _canonical


def _normalise(skill: str) -> str:
    return _ALIAS_MAP.get(skill.strip().lower(), skill.strip().lower())


def _skill_present(skill: str, text_lower: str, skill_set: set[str]) -> bool:
    """Check skill against resume text + skill set, with full synonym expansion."""
    key = skill.lower().strip()
    canonical = _normalise(key)

    # Direct check
    if key in skill_set or canonical in skill_set:
        return True
    if key in text_lower or canonical in text_lower:
        return True

    # Expand all aliases
    for alias in SKILL_SYNONYMS.get(canonical, [key]):
        if alias in text_lower:
            return True
    # Also check all aliases of the raw key (if it has its own list)
    for alias in SKILL_SYNONYMS.get(key, []):
        if alias in text_lower:
            return True
    return False


# ── Stop Words ────────────────────────────────────────────────────────────────

_STOP = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "is","are","was","be","has","have","we","you","your","our","their","will",
    "can","must","should","may","not","strong","good","excellent","great",
    "experience","knowledge","skills","ability","understanding","proficiency",
    "familiarity","working","minimum","preferred","required","years","year",
    "plus","bonus","join","team","work","company","candidate","role","position",
    "job","looking","seeking","ideal","responsibilities","including","using",
    "comfortable","demonstrated","proven","track","record",
}

_MULTI_SKILLS = sorted(
    [c for c in SKILL_SYNONYMS if " " in c] +
    [a for aliases in SKILL_SYNONYMS.values() for a in aliases if " " in a],
    key=len, reverse=True,
)


# ── JD Keyword Extraction ─────────────────────────────────────────────────────

def extract_jd_keywords(jd_text: str) -> dict:
    """
    Extract required and preferred skill keywords from a Job Description.
    Returns {"required": [...], "preferred": [...], "all": [...]}.
    """
    jd_lower = jd_text.lower()

    # Multi-word first (longest match)
    found_multi = [t for t in _MULTI_SKILLS if t in jd_lower]

    # Single token extraction
    words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9#+.\-]{1,30}\b", jd_text)
    single = [
        w.lower() for w in words
        if w.lower() not in _STOP and len(w) > 2
        and not any(w.lower() in m for m in found_multi)
    ]
    all_kws = list(dict.fromkeys(found_multi + single))[:60]

    # Detect req / preferred sections
    req_m  = re.search(
        r"(?:requirements?|must.have|required|mandatory|qualifications?)[:\n](.*?)"
        r"(?:preferred|nice.to.have|bonus|responsibilities|\Z)",
        jd_lower, re.DOTALL | re.IGNORECASE,
    )
    pref_m = re.search(
        r"(?:preferred|nice.to.have|bonus|plus)[:\n](.*?)(?:\Z)",
        jd_lower, re.DOTALL | re.IGNORECASE,
    )

    req_text  = req_m.group(1)  if req_m  else jd_lower
    pref_text = pref_m.group(1) if pref_m else ""

    required  = list(dict.fromkeys(k for k in all_kws if k in req_text ))[:30]
    preferred = list(dict.fromkeys(k for k in all_kws if k in pref_text))[:15]

    # Canonical dedup for required
    seen: set[str] = set()
    req_deduped = []
    for k in required:
        c = _normalise(k)
        if c not in seen:
            seen.add(c)
            req_deduped.append(k)

    return {"required": req_deduped, "preferred": preferred, "all": all_kws}


# ── Helper: project text list ─────────────────────────────────────────────────

def _build_project_texts(projects: list, raw_text: str) -> list[str]:
    """Extract a list of project text strings from structured or raw data."""
    texts = []
    for proj in projects:
        if isinstance(proj, dict):
            pt = f"{proj.get('name','')} {proj.get('description','')}"
            techs = proj.get("technologies") or proj.get("tech") or []
            if isinstance(techs, list):
                pt += " " + " ".join(techs)
        elif isinstance(proj, str):
            pt = proj
        else:
            continue
        if pt.strip():
            texts.append(pt.strip())

    if not texts:
        m = re.search(
            r"(?i)(?:projects?|portfolio)[:\n](.*?)(?:education|experience|skills|certif|\Z)",
            raw_text, re.DOTALL,
        )
        if m:
            texts = [s.strip() for s in m.group(1).split("\n") if len(s.strip()) > 20]
    return texts


# ═══════════════════════════════════════════════════════════════
# COMPONENT 1 — Required Skill Match (35%)
# WITH EVIDENCE MULTIPLIER: skills proven in projects count 1.5×
# ═══════════════════════════════════════════════════════════════

def _c1_required_skills(
    skills: list[str],
    raw_text: str,
    projects: list,
    jd_kw: dict,
) -> dict:
    """
    Score: 0–100.
    Evidence multiplier: if a required skill appears in N projects,
    it gets a weight of (1 + 0.5*N) instead of 1.
    This ensures 'Rust in 4 projects' >> 'Rust in skills list only'.
    """
    required  = jd_kw.get("required",  [])
    preferred = jd_kw.get("preferred", [])

    if not required:
        required = jd_kw.get("all", [])[:20]

    skill_lower = {s.lower() for s in skills}
    text_lower  = raw_text.lower()

    # Build project texts for evidence counting
    proj_texts = [pt.lower() for pt in _build_project_texts(projects, raw_text)]

    matched_req = []
    missing_req = []
    evidence_weights: dict[str, float] = {}

    for kw in required:
        if _skill_present(kw, text_lower, skill_lower):
            matched_req.append(kw)
            # Count how many projects demonstrate this skill
            proj_count = sum(
                1 for pt in proj_texts if _skill_present(kw, pt, set())
            )
            # Evidence multiplier: base 1.0 + 0.5 per project (uncapped)
            evidence_weights[kw] = 1.0 + (proj_count * 0.5)
        else:
            missing_req.append(kw)
            evidence_weights[kw] = 0.0

    total_required = max(1, len(required))

    # Simple ratio (base signal)
    req_ratio = len(matched_req) / total_required

    # Evidence-boosted ratio
    # If all required skills are in 2 projects each → max weight = len(req) * 2.0
    evidence_sum  = sum(evidence_weights.values())
    max_possible  = total_required * 2.0
    evidence_ratio = evidence_sum / max_possible

    # Combined score: 55% evidence-boosted + 45% simple ratio
    # This rewards proven depth over breadth of mention
    combined_ratio = (evidence_ratio * 0.55) + (req_ratio * 0.45)
    base = combined_ratio * 100.0

    # Preferred skills bonus (up to +15)
    matched_pref = [k for k in preferred if _skill_present(k, text_lower, skill_lower)]
    pref_ratio   = len(matched_pref) / max(1, len(preferred)) if preferred else 0.5
    pref_bonus   = pref_ratio * 15.0

    score = min(100.0, base + pref_bonus)

    # Hard cap for very poor match
    if req_ratio < 0.20:
        score = min(score, 30.0)

    return {
        "score":           round(score, 1),
        "matched":         matched_req + matched_pref,
        "missing":         missing_req,
        "req_ratio":       round(req_ratio, 3),
        "matched_count":   len(matched_req),
        "total_required":  total_required,
        "evidence_weights": evidence_weights,
    }


# ═══════════════════════════════════════════════════════════════
# COMPONENT 2 — Project Evidence (25%)
# KEYWORD-FIRST: required skills found in projects dominate.
# SBERT is secondary (helps for semantic proximity of descriptions).
# ═══════════════════════════════════════════════════════════════

def _c2_project_evidence(
    projects: list,
    raw_text: str,
    jd_kw: dict,
    jd_emb: np.ndarray,
    model,
) -> dict:
    """
    Score: 0–100.
    Primary signal: how many required JD skills does each project demonstrate?
    Secondary signal: SBERT similarity of project description to JD.

    A project using 3+ required skills = strong evidence (high score).
    Multiple such projects = stacking bonus.
    """
    required_skills = jd_kw.get("required",  [])
    all_jd_terms    = set(
        k.lower() for k in jd_kw.get("required", []) + jd_kw.get("preferred", [])
    )

    proj_texts = _build_project_texts(projects, raw_text)

    if not proj_texts:
        return {"score": 12.0, "detail": "no_projects_found"}

    # Encode project texts for SBERT (secondary signal)
    proj_embs = _encode(proj_texts, model)

    proj_scores: list[float] = []
    for i, ptext in enumerate(proj_texts):
        plow = ptext.lower()

        # ── PRIMARY: keyword evidence ──────────────────────────────────────
        # Count required JD skills demonstrated in this project
        req_hits  = sum(1 for r in required_skills if _skill_present(r, plow, set()))
        req_ratio = req_hits / max(1, len(required_skills))
        # Amplify: 33% of required skills → 100 score (generous)
        keyword_score = min(100.0, req_ratio * 300.0)

        # Count any JD terms (preferred + required)
        any_hits  = sum(1 for t in all_jd_terms if t in plow)
        any_ratio = any_hits / max(1, len(all_jd_terms))
        any_score = min(80.0, any_ratio * 400.0)

        # Take the better of the two keyword signals
        kw_score = max(keyword_score, any_score)

        # ── SECONDARY: SBERT semantic ──────────────────────────────────────
        sem_raw   = float(np.dot(proj_embs[i], jd_emb))
        sem_score = _sigmoid_calibrate(sem_raw)

        # 65% keyword evidence + 35% semantic
        combined = (kw_score * 0.65) + (sem_score * 0.35)
        proj_scores.append(combined)

    if not proj_scores:
        return {"score": 12.0, "detail": "empty_project_texts"}

    max_s = max(proj_scores)
    avg_s = sum(proj_scores) / len(proj_scores)

    # Count strong-evidence projects (score > 35)
    strong = sum(1 for s in proj_scores if s > 35)
    # Stacking bonus for multiple relevant projects
    stack_bonus = min(18.0, (strong - 1) * 6.0) if strong > 1 else 0.0

    final = (max_s * 0.60 + avg_s * 0.40) + stack_bonus
    return {
        "score":          round(min(100.0, final), 1),
        "project_scores": [round(s, 1) for s in proj_scores],
        "strong_count":   strong,
    }


# ═══════════════════════════════════════════════════════════════
# COMPONENT 3 — Semantic Alignment (10%)
# RECALIBRATED: midpoint=0.30, k=15
# Maps real-world resume↔JD cosines to a meaningful 0–100 range.
# ═══════════════════════════════════════════════════════════════

def _sigmoid_calibrate(raw_cosine: float) -> float:
    """
    Sigmoid calibration for MiniLM cosine similarity → 0–100.

    Observed real-world cosine ranges (MiniLM, resume vs. JD, same domain):
      0.18  → very weak match   → calibrated  ~16%
      0.28  → weak match        → calibrated  ~44%
      0.36  → decent match      → calibrated  ~71%
      0.50  → strong match      → calibrated  ~95%

    sigmoid: k=15, midpoint=0.30
      score = 1 / (1 + exp(-15*(cosine - 0.30))) * 100
    """
    k, mid = 15.0, 0.30
    return round(100.0 / (1.0 + math.exp(-k * (raw_cosine - mid))), 1)


# ═══════════════════════════════════════════════════════════════
# COMPONENT 4 — Experience Relevance (10%)
# ═══════════════════════════════════════════════════════════════

_SENIORITY: dict[str, list[str]] = {
    "entry": ["entry level", "0-1 year", "fresh", "fresher", "junior", "intern", "trainee"],
    "mid":   ["2 year", "3 year", "mid level", "intermediate", "associate", "2-3", "2-4"],
    "senior":["senior", "5 year", "4 year", "sr.", "3-5", "4-6", "5+", "lead developer"],
    "lead":  ["lead", "principal", "staff", "architect", "manager", "10+", "director"],
}

_DOMAINS: dict[str, list[str]] = {
    "systems":    ["kernel", "memory management", "os", "syscall", "embedded", "firmware",
                   "driver", "concurrency", "threading", "posix", "zero-copy", "unsafe"],
    "web":        ["frontend", "backend", "full stack", "html", "css", "browser", "spa",
                   "ssr", "seo", "web performance"],
    "data":       ["data pipeline", "etl", "bigquery", "spark", "hadoop", "warehouse",
                   "dbt", "airflow", "data engineering"],
    "ml":         ["model training", "dataset", "inference", "experiment", "jupyter",
                   "colab", "gpu training", "fine-tuning", "prompt", "embedding"],
    "devops":     ["deployment", "monitoring", "observability", "helm", "ci/cd",
                   "sre", "incident", "on-call", "uptime"],
    "mobile":     ["ios", "android", "flutter", "react native", "swift", "kotlin"],
    "blockchain": ["wallet", "smart contract", "dapp", "defi", "nft", "web3",
                   "solana", "ethereum", "anchor"],
    "security":   ["security", "vulnerability", "penetration", "cve", "owasp",
                   "encryption", "authentication"],
}


def _c4_experience(raw_text: str, jd_text: str, projects: list) -> dict:
    """Score: 0–100. Seniority alignment + domain expertise match."""
    text_l = raw_text.lower()
    jd_l   = jd_text.lower()

    # Detect JD seniority level
    jd_level = "mid"
    for lvl, patterns in _SENIORITY.items():
        if any(p in jd_l for p in patterns):
            jd_level = lvl
            break

    has_senior = any(p in text_l for p in _SENIORITY["senior"] + _SENIORITY["lead"])
    has_mid    = any(p in text_l for p in _SENIORITY["mid"])
    proj_count = len(projects) if projects else len(re.findall(r"(?i)\bproject\b", text_l))

    seniority_score = {
        "entry": 75 if not has_senior else 55,
        "mid":   80 if (has_mid or has_senior) else (65 if proj_count >= 2 else 45),
        "senior":85 if has_senior else (72 if proj_count >= 3 else 50),
        "lead":  80 if has_senior else 55,
    }.get(jd_level, 60)

    # Domain match
    domain_score = 0.0
    for domain, signals in _DOMAINS.items():
        jd_hits = sum(1 for s in signals if s in jd_l)
        if jd_hits >= 2:
            res_hits    = sum(1 for s in signals if s in text_l)
            domain_score = max(domain_score, min(100.0, res_hits / len(signals) * 150.0))

    if domain_score == 0:
        gen = ["api", "test", "debug", "deploy", "review", "database", "server",
               "code review", "performance", "optimize", "architecture"]
        domain_score = min(75.0, sum(7 for g in gen if g in text_l))

    return {
        "score":        round(seniority_score * 0.40 + domain_score * 0.60, 1),
        "level":        jd_level,
        "domain_score": round(domain_score, 1),
    }


# ═══════════════════════════════════════════════════════════════
# COMPONENT 5 — Keyword Coverage (8%)
# ═══════════════════════════════════════════════════════════════

def _c5_keyword_coverage(raw_text: str, jd_kw: dict) -> dict:
    """BM25-inspired term coverage. Score: 0–100."""
    text_l  = raw_text.lower()
    all_kws = jd_kw.get("all", [])[:50]
    if not all_kws:
        return {"score": 50.0, "hits": 0, "total": 0, "coverage": 0.0}

    hits     = [k for k in all_kws if k in text_l]
    coverage = len(hits) / len(all_kws)

    # Non-linear mapping (sqrt amplifies low-coverage)
    score = min(100.0, (coverage ** 0.55) * 100.0)
    return {
        "score":    round(score, 1),
        "hits":     len(hits),
        "total":    len(all_kws),
        "coverage": round(coverage, 3),
    }


# ═══════════════════════════════════════════════════════════════
# COMPONENT 6 — Resume Quality (5%)
# ═══════════════════════════════════════════════════════════════

def _c6_resume_quality(
    raw_text: str,
    skills: list,
    projects: list,
    email: str = "",
    phone: str = "",
    github_url: str = "",
) -> dict:
    """Score: 0–100. Completeness, format, ATS compliance."""
    score = 0
    text_l = raw_text.lower()

    # Contact info (20 pts)
    score += 8  if email                               else 0
    score += 6  if phone                               else 0
    score += 6  if (github_url or "github" in text_l)  else 0

    # Sections present (35 pts — 7 pts each)
    for pat in [r"\beducat", r"\b(?:experience|work history)\b", r"\b(?:skills?|technologies)\b",
                r"\bprojects?\b", r"\b(?:achievements?|awards?|publications?)\b"]:
        if re.search(pat, text_l):
            score += 7

    # Content quality (45 pts)
    n_skills = len(skills)
    score += (12 if n_skills >= 15 else 8 if n_skills >= 8 else 4 if n_skills >= 3 else 0)

    metrics = re.findall(
        r"\b\d+%|\b\d+x\b|\b\d+ms\b|\b\d+k\b|\$\d+|\b\d+\s*(?:users|requests|seconds|customers)\b",
        raw_text,
    )
    score += min(10, len(metrics) * 2)

    wc = len(raw_text.split())
    score += (12 if wc >= 500 else 8 if wc >= 350 else 4 if wc >= 200 else 0)

    n_proj = len(projects)
    score += (11 if n_proj >= 4 else 8 if n_proj >= 2 else 4 if n_proj >= 1 else 0)

    return {"score": min(100, score), "word_count": wc}


# ═══════════════════════════════════════════════════════════════
# COMPONENT 7 — Education (4%)
# ═══════════════════════════════════════════════════════════════

_CS_FIELDS = [
    "computer science", "computer engineering", "software engineering",
    "information technology", "electronics and communication",
    "electrical engineering", "cs", "cse", "it", "ece", "eee", "btech",
]
_STEM_FIELDS = [
    "mathematics", "physics", "statistics", "data science",
    "information systems", "bioinformatics", "mechanical",
]


def _c7_education(raw_text: str, education: str = "") -> dict:
    edu = (education + " " + raw_text).lower()
    level = 5 if any(d in edu for d in ["phd","ph.d","doctorate"]) else \
            10 if any(d in edu for d in ["master","m.tech","m.e.","msc","m.s."]) else \
            5  if any(d in edu for d in ["bachelor","b.tech","b.e.","bsc","b.sc","be "]) else 2
    field = 65 if any(f in edu for f in _CS_FIELDS) else \
            45 if any(f in edu for f in _STEM_FIELDS) else 20
    bonus = 0
    if re.search(r"\b(?:gpa|cgpa)[\s:]+[89]\.", edu) or re.search(r"\b[89]\.\d+\s*/\s*10\b", edu):
        bonus += 10
    if any(t in edu for t in ["distinction","first class","iit","nit","bits","gold medal"]):
        bonus += 5
    return {"score": min(100, field + level + bonus)}


# ═══════════════════════════════════════════════════════════════
# COMPONENT 8 — Certifications (3%)
# ═══════════════════════════════════════════════════════════════

_CERT_CHECKS = [
    (r"\baws\s+certified|\bawssa\b|\baws\s+saa\b|\bcloud\s+practitioner\b", 25),
    (r"\bgcp\s+certified|\bgoogle\s+cloud\s+certified", 25),
    (r"\bazure\s+certified|\baz-\d{3}\b", 25),
    (r"\bcka\b|\bckad\b|\bkubernetes\s+certified\b", 20),
    (r"\bopen.source\s+(?:contributor|maintainer)|\bcore\s+maintainer\b", 20),
    (r"\bhackathon\b.*(?:won|winner|prize|1st|first)|\bwon\b.*hackathon", 15),
    (r"\bhackathon\b", 8),
    (r"\bcompetitive\s+programming|\bleetcode\b.*(?:top|rank)|\bcodeforces\b.*rank", 10),
    (r"\bpublication|\bieee\b|\bacm\b|\bconference\s+paper\b|\bjournal\b", 15),
    (r"\bcertif", 8),
    (r"\bcoursera|\budemy|\bedx\b", 5),
]


def _c8_certifications(raw_text: str) -> dict:
    score = 30.0
    text_l = raw_text.lower()
    for pattern, pts in _CERT_CHECKS:
        if re.search(pattern, text_l, re.IGNORECASE):
            score += pts
    return {"score": min(100.0, score)}


# ═══════════════════════════════════════════════════════════════
# PENALTY LAYER (0 to -15)
# ═══════════════════════════════════════════════════════════════

def _penalty(skill_info: dict, projects: list, raw_text: str) -> float:
    p = 0.0
    req_ratio = skill_info.get("req_ratio", 1.0)
    if req_ratio < 0.20:
        p -= 15.0
    elif req_ratio < 0.30:
        p -= 8.0
    elif req_ratio < 0.45:
        p -= 3.0

    if not projects:
        has_proj = bool(re.search(r"(?i)\bproject\b", raw_text))
        if not has_proj:
            p -= 5.0

    if len(raw_text.split()) < 180:
        p -= 5.0

    words = re.findall(r"\b[a-z]{4,}\b", raw_text.lower())
    freq  = Counter(words)
    stuffed = [w for w, c in freq.most_common(30) if c > 10 and w not in _STOP]
    if len(stuffed) > 4:
        p -= 5.0

    return max(-15.0, p)


# ═══════════════════════════════════════════════════════════════
# STRENGTHS & WEAKNESSES (for Hiring Confidence display)
# ═══════════════════════════════════════════════════════════════

def _compute_strengths(
    matched_skills: list[str],
    evidence_weights: dict,
    proj_texts: list[str],
    top_n: int = 8,
) -> list[dict]:
    """
    Compute star rating for matched skills.
    Stars = min(5, base_1 + floor(proj_count * 0.8))
    Skills backed by multiple projects get 4-5 stars.
    """
    strengths = []
    for skill in matched_skills:
        proj_count = sum(1 for pt in proj_texts if _skill_present(skill, pt, set()))
        # Stars: 1 base + 1 per project (capped at 5)
        stars = min(5, 1 + proj_count)
        # Boost for high evidence weight
        ev = evidence_weights.get(skill, 1.0)
        if ev >= 2.0:
            stars = min(5, stars + 1)
        strengths.append({"skill": skill, "stars": stars, "projects": proj_count})

    return sorted(strengths, key=lambda x: x["stars"], reverse=True)[:top_n]


# ═══════════════════════════════════════════════════════════════
# WEIGHTS & LABELS
# ═══════════════════════════════════════════════════════════════

WEIGHTS = {
    "required_skill_match": 0.35,
    "project_evidence":     0.25,
    "semantic_match":       0.10,
    "experience_relevance": 0.10,
    "keyword_coverage":     0.08,
    "resume_quality":       0.05,
    "education":            0.04,
    "certifications":       0.03,
}

COMPONENT_LABELS = {
    "required_skill_match": "Required Skill Match",
    "project_evidence":     "Project Evidence",
    "semantic_match":       "Semantic Alignment",
    "experience_relevance": "Experience Relevance",
    "keyword_coverage":     "Keyword Coverage",
    "resume_quality":       "Resume Quality",
    "education":            "Education",
    "certifications":       "Certifications",
}


# ═══════════════════════════════════════════════════════════════
# MAIN SCORING FUNCTION
# ═══════════════════════════════════════════════════════════════

def score_candidate(
    raw_text:   str,
    skills:     list[str],
    projects:   list,
    education:  str = "",
    email:      str = "",
    phone:      str = "",
    github_url: str = "",
    jd_text:    str = "",
    model       = None,
    device:     str = "cpu",
) -> dict:
    """
    Evidence-aggregation ATS score for one candidate.

    Returns a comprehensive dict with final_score, all 8 component scores,
    confidence tier, strengths with star ratings, and weaknesses.
    """
    if not jd_text.strip():
        return {"final_score": 0.0, "error": "Empty job description"}

    m = model or _get_model()
    jd_kw = extract_jd_keywords(jd_text)

    # ── Encode resume + JD ───────────────────────────────────────────────────
    embs = _encode([raw_text, jd_text], m)
    resume_emb, jd_emb = embs[0], embs[1]
    raw_cosine = float(np.dot(resume_emb, jd_emb))

    # ── All 8 components ─────────────────────────────────────────────────────
    c1 = _c1_required_skills(skills, raw_text, projects, jd_kw)
    c2 = _c2_project_evidence(projects, raw_text, jd_kw, jd_emb, m)
    c3 = _sigmoid_calibrate(raw_cosine)
    c4 = _c4_experience(raw_text, jd_text, projects)
    c5 = _c5_keyword_coverage(raw_text, jd_kw)
    c6 = _c6_resume_quality(raw_text, skills, projects, email, phone, github_url)
    c7 = _c7_education(raw_text, education)
    c8 = _c8_certifications(raw_text)
    pen = _penalty(c1, projects, raw_text)

    raw_scores = {
        "required_skill_match": c1["score"],
        "project_evidence":     c2["score"],
        "semantic_match":       c3,
        "experience_relevance": c4["score"],
        "keyword_coverage":     c5["score"],
        "resume_quality":       c6["score"],
        "education":            c7["score"],
        "certifications":       c8["score"],
    }

    weighted_sum = sum(raw_scores[k] * w for k, w in WEIGHTS.items())
    final = round(max(0.0, min(100.0, weighted_sum + pen)), 1)

    # ── Confidence tier ───────────────────────────────────────────────────────
    tier_info = get_tier(final)

    # ── Strengths / weaknesses ────────────────────────────────────────────────
    proj_texts_lower = [pt.lower() for pt in _build_project_texts(projects, raw_text)]
    strengths  = _compute_strengths(
        c1["matched"], c1["evidence_weights"], proj_texts_lower
    )
    weaknesses = c1["missing"]

    # ── Explanation ───────────────────────────────────────────────────────────
    matched = c1["matched"]
    missing = c1["missing"]
    explanation = (
        f"{tier_info['label']} ({final}/100) — {tier_info['recommendation']}. "
        f"Skill match: {c1['matched_count']}/{c1['total_required']} required "
        f"({'evidence-boosted' if any(v>1 for v in c1['evidence_weights'].values()) else 'listed only'}). "
        f"Semantic: {c3}/100 (cosine: {raw_cosine:.3f}). "
        f"Projects with strong evidence: {c2.get('strong_count', 0)}. "
        + (f"Missing: {', '.join(missing[:5])}. " if missing else "No critical gaps. ")
        + (f"Penalty: {pen:.0f}." if pen < 0 else "")
    )

    # ── Component breakdown for UI ────────────────────────────────────────────
    components = {
        k: {
            "label":        COMPONENT_LABELS[k],
            "score":        round(raw_scores[k], 1),
            "weight":       round(WEIGHTS[k] * 100, 0),
            "contribution": round(raw_scores[k] * WEIGHTS[k], 2),
        }
        for k in WEIGHTS
    }

    return {
        "final_score":      final,
        "raw_cosine":       round(raw_cosine, 4),
        "tier":             tier_info["label"],
        "recommendation":   tier_info["recommendation"],
        "tier_note":        tier_info["note"],
        "components":       components,
        "penalty":          pen,
        "matched_skills":   matched,
        "missing_skills":   missing,
        "strengths":        strengths,
        "weaknesses":       weaknesses[:8],
        "jd_required":      jd_kw.get("required", []),
        "jd_preferred":     jd_kw.get("preferred", []),
        "explanation":      explanation,
    }
