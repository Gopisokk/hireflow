"""
run_ats.py -- ATS End-to-End Test Runner (GPU-Accelerated)
===========================================================
Accepts a resume PDF/DOCX path + a JD text and runs the complete
ATS pipeline: Parse -> Index -> Hybrid Search -> Score.

NO GitHub verification. ATS scoring only.

Usage:
    python run_ats.py --resume path/to/resume.pdf --jd "We are hiring..."
    python run_ats.py --resume path/to/resume.docx --jd-file jd.txt
    python run_ats.py --resume path/to/resume.pdf --jd-file jd.txt --algo hybrid_efficient
    python run_ats.py --resume path/to/resume.pdf --jd "..." --save results.json
"""

import os
import sys
import json
import time
import argparse
import tempfile
from pathlib import Path

# Force UTF-8 stdout on Windows
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# ── GPU Configuration ─────────────────────────────────────────────────────────
import torch

def _setup_gpu() -> str:
    """Auto-detect CUDA, unlock full GPU power, return device string."""
    if torch.cuda.is_available():
        device   = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
        # Unlock full GPU throughput
        torch.backends.cudnn.benchmark         = True   # auto-tune conv kernels
        torch.backends.cudnn.deterministic     = False  # allow non-deterministic for speed
        torch.backends.cuda.matmul.allow_tf32  = True   # TF32 for faster matmul on Ampere+
        torch.backends.cudnn.allow_tf32        = True
        # Pre-warm the CUDA context
        _ = torch.zeros(1, device=device)
        print(f"  [GPU] DETECTED: {gpu_name}  ({vram_gb} GB VRAM) -- CUDA ENABLED")
    else:
        device = "cpu"
        print("  [INFO] No CUDA GPU found - running on CPU")
    return device

DEVICE = _setup_gpu()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _banner(text: str, char: str = "=", width: int = 64):
    print()
    print(char * width)
    print(f"  {text}")
    print(char * width)

def _step(n: int, text: str):
    print(f"\n  +-- Step {n}: {text}")

def _ok(text: str):
    print(f"  |   [OK]   {text}")

def _warn(text: str):
    print(f"  |   [WARN] {text}")

def _info(text: str):
    print(f"  |   [INFO] {text}")


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1: Resume Parsing
# ─────────────────────────────────────────────────────────────────────────────

def layer1_parse(resume_path: str) -> dict:
    """Parse the PDF/DOCX resume using the existing resume_parser.py."""
    _step(1, "Resume Parsing  (resume_parser.py)")

    if not os.path.exists(resume_path):
        raise FileNotFoundError(f"Resume not found: {resume_path}")

    from resume_parser import parse_resume

    parsed = parse_resume(resume_path)

    _ok(f"Parsed  : {parsed.get('name', 'Unknown')}")
    _ok(f"Email   : {parsed.get('email', '-')}")
    _ok(f"GitHub  : {parsed.get('github_username', '-')}")
    skills = parsed.get("skills", [])
    _ok(f"Skills  ({len(skills)}): " + ", ".join(skills[:8]) + ("..." if len(skills) > 8 else ""))
    projects = parsed.get("projects", [])
    _ok(f"Projects({len(projects)}): " + ", ".join([p.get("name","") for p in projects[:4]]))

    return parsed


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2: Build In-Memory Database for this Single Run
# ─────────────────────────────────────────────────────────────────────────────

def _adapt_parsed_to_schema(parsed: dict) -> dict:
    """
    Map resume_parser.py output keys to the schema used by build_database.py.
    """
    skills_list = parsed.get("skills", [])
    primary     = [{"skill_span": s, "context": ""} for s in skills_list[:10]]
    secondary   = [{"skill_span": s, "context": ""} for s in skills_list[10:]]

    projects = []
    for p in parsed.get("projects", []):
        if isinstance(p, dict):
            projects.append({
                "project_name": p.get("name", ""),
                "description":  p.get("description", "")
            })

    work_exp = []
    for w in parsed.get("experience", []):
        if isinstance(w, dict):
            work_exp.append({
                "job_title":   w.get("title", ""),
                "company":     w.get("company", ""),
                "dates":       w.get("dates", ""),
                "description": w.get("description", "")
            })

    return {
        "header": {
            "name":     parsed.get("name", ""),
            "email":    parsed.get("email", ""),
            "phone":    parsed.get("phone", ""),
            "linkedin": parsed.get("linkedin", ""),
            "github":   parsed.get("github_link", "") or parsed.get("github_username", "")
        },
        "education":                 parsed.get("education", []),
        "open_source_contributions": parsed.get("open_source", []),
        "projects":                  projects,
        "competitive_programming":   parsed.get("competitive_programming", []),
        "certifications":            parsed.get("certifications", []),
        "achievements":              parsed.get("achievements", []),
        "technologies": {
            "primary_skills":   primary,
            "secondary_skills": secondary
        },
        "volunteering":  parsed.get("volunteering", []),
        "work_experience": work_exp
    }


def layer2_index(parsed: dict, db_path: str) -> int:
    """Convert parsed resume dict into SQLite FTS5 + vector index."""
    _step(2, "Database Indexing  (build_database.py)")
    _info(f"Using device: {DEVICE}")

    from build_database import (
        _connect, setup_database, load_embedding_model,
        _flatten_text, _fts_experience_text, _fts_projects_text,
        _fts_skills_text, _extract_github_link, _extract_name,
        embed_text, _serialize_f32
    )

    adapted        = _adapt_parsed_to_schema(parsed)
    conn           = _connect(db_path)
    setup_database(conn)
    model          = load_embedding_model(device=DEVICE)

    flat_text      = _flatten_text(adapted)
    fts_experience = _fts_experience_text(adapted)
    fts_projects   = _fts_projects_text(adapted)
    fts_skills     = _fts_skills_text(adapted)
    github_link    = adapted.get("header", {}).get("github", "") or ""
    name           = adapted.get("header", {}).get("name", "") or parsed.get("name", "")
    raw_json_str   = json.dumps(adapted, ensure_ascii=False)
    filename       = Path(parsed.get("_source_file", "resume")).name + ".json"

    _info(f"Generating embedding ({len(flat_text)} chars)...")
    vector = embed_text(model, flat_text)

    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO candidates_raw (filename, name, github_link, raw_json)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(filename) DO UPDATE SET
            name        = excluded.name,
            github_link = excluded.github_link,
            raw_json    = excluded.raw_json,
            inserted_at = datetime('now')
        """,
        (filename, name, github_link, raw_json_str),
    )
    candidate_id = cursor.lastrowid
    if not candidate_id:
        row = cursor.execute(
            "SELECT candidate_id FROM candidates_raw WHERE filename = ?", (filename,)
        ).fetchone()
        candidate_id = row["candidate_id"]

    cursor.execute("DELETE FROM candidates_fts WHERE candidate_id = ?", (candidate_id,))
    cursor.execute(
        "INSERT INTO candidates_fts (candidate_id, experience_and_open_source, projects, skills) VALUES (?,?,?,?)",
        (candidate_id, fts_experience, fts_projects, fts_skills),
    )
    cursor.execute("DELETE FROM candidates_vec WHERE candidate_id = ?", (candidate_id,))
    cursor.execute(
        "INSERT INTO candidates_vec (candidate_id, embedding) VALUES (?, ?)",
        (candidate_id, _serialize_f32(vector)),
    )
    conn.commit()
    conn.close()

    _ok(f"Indexed candidate_id={candidate_id} -> '{name}'")
    return candidate_id


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3: Hybrid Search (BM25 + Vector + RRF)
# ─────────────────────────────────────────────────────────────────────────────

def layer3_search(jd_text: str, db_path: str, skip_hyre: bool = True) -> list:
    """Run hybrid search and return the ranked shortlist."""
    _step(3, "Hybrid Search  (hybrid_search.py)")

    from hybrid_search import (
        _connect, load_embedding_model, vectorize_query,
        expand_jd_to_resume, bm25_search, semantic_search,
        reciprocal_rank_fusion, enrich_with_metadata
    )

    conn  = _connect(db_path)
    model = load_embedding_model(device=DEVICE)

    if skip_hyre:
        _info("HyRE skipped -- using raw JD text (set OPENAI_API_KEY to enable).")
        query_text = jd_text
    else:
        query_text = expand_jd_to_resume(jd_text)

    query_vector = vectorize_query(model, query_text)

    bm25_r   = bm25_search(conn, query_text,   top_k=50)
    vector_r = semantic_search(conn, query_vector, top_k=50)

    fused     = reciprocal_rank_fusion(bm25_r, vector_r)
    shortlist = enrich_with_metadata(conn, fused, top_n=10)
    conn.close()

    _ok(f"Hybrid search complete -- {len(fused)} unique candidates fused.")
    for i, r in enumerate(shortlist[:5], 1):
        bm25_tag = f"BM25#{r['bm25_rank']}"   if r.get("bm25_rank")   else "-"
        vec_tag  = f"Vec#{r['vector_rank']}"   if r.get("vector_rank") else "-"
        print(f"  |   #{i}  {r.get('name','?'):<28}  RRF={r['rrf_score']:.5f}  [{bm25_tag}, {vec_tag}]")

    return shortlist


# ─────────────────────────────────────────────────────────────────────────────
# Layer 4: ATS Scoring
# ─────────────────────────────────────────────────────────────────────────────

def layer4_score(parsed: dict, jd_text: str, algo: str = "hybrid_efficient") -> dict:
    """Run the ATS scoring engine against the parsed resume and JD."""
    _step(4, f"ATS Scoring  (ats_engine.py -- algo: {algo})")
    _info(f"Scoring on device: {DEVICE}")

    import ats_engine

    resume_text   = parsed.get("raw_text", "")
    resume_skills = parsed.get("skills", [])

    algo_map = {
        "bm25":             lambda: ats_engine.score_bm25(resume_text, jd_text, resume_skills),
        "neural":           lambda: ats_engine.score_neural(resume_text, jd_text, resume_skills, device=DEVICE),
        "hybrid_efficient": lambda: ats_engine.score_hybrid(resume_text, jd_text, resume_skills, device=DEVICE),
        "colbert":          lambda: ats_engine.score_colbert(resume_text, jd_text, resume_skills),
    }

    scorer_fn = algo_map.get(algo, algo_map["hybrid_efficient"])
    result = scorer_fn()

    # Project & Experience heuristic (max 20 pts, scaled by domain match)
    base_score = result["score"]
    # Scale factor: how well the base skills match. Base is out of 80 max.
    # When no preferred section, base is out of 60 req + 20 semantic.
    # We use /60 so that missing ALL required skills = ~0 scaling.
    domain_match_ratio = min(1.0, base_score / 60.0)
    project_score = min(10.0, len(parsed.get("projects", [])) * 5.0) * domain_match_ratio
    exp_score     = 0.0
    if parsed.get("education"):    exp_score += 5.0
    if len(resume_text) > 1000:    exp_score += 5.0
    exp_score *= domain_match_ratio

    total_score = min(100.0, base_score + project_score + exp_score)

    print()
    print(f"  |   Base skills & semantic score  : {base_score:.1f} / 80")
    print(f"  |   Projects ({len(parsed.get('projects',[]))} found)      : +{project_score:.1f}")
    print(f"  |   Experience / Education        : +{exp_score:.1f}")
    print(f"  |   ============================================")
    print(f"  |   FINAL ATS SCORE               : {total_score:.1f} / 100")

    matched = result.get("matched_skills", [])
    missing = result.get("missing_skills", [])
    if matched:
        print(f"  |   Matched ({len(matched)}): " + ", ".join(matched[:10]) + ("..." if len(matched) > 10 else ""))
    if missing:
        print(f"  |   Missing ({len(missing)}): " + ", ".join(missing[:10]) + ("..." if len(missing) > 10 else ""))

    return {
        **result,
        "base_score":    base_score,
        "project_score": project_score,
        "exp_score":     exp_score,
        "final_score":   total_score
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ATS End-to-End Test Runner (Parse -> Index -> Search -> Score)"
    )
    parser.add_argument("--resume",   required=True, help="Path to resume PDF or DOCX file")
    jd_group = parser.add_mutually_exclusive_group(required=True)
    jd_group.add_argument("--jd",      type=str, help="Raw JD text (in quotes)")
    jd_group.add_argument("--jd-file", type=str, help="Path to a plain-text JD file")
    parser.add_argument("--algo",    default="hybrid_efficient",
                        choices=["bm25", "neural", "hybrid_efficient", "colbert"],
                        help="ATS scoring algorithm (default: hybrid_efficient)")
    parser.add_argument("--db",      default=":memory:",
                        help="SQLite DB path (default: temp file, use 'ats.db' to persist)")
    parser.add_argument("--hyre",    action="store_true",
                        help="Enable HyRE expansion (requires OPENAI_API_KEY)")
    parser.add_argument("--save",    type=str, default=None,
                        help="Save full results to a JSON file")
    parser.add_argument("--cpu",     action="store_true",
                        help="Force CPU even if CUDA is available")
    args = parser.parse_args()

    global DEVICE
    if args.cpu:
        DEVICE = "cpu"
        print("  [INFO] --cpu flag set -- forcing CPU mode.")

    # Load JD
    if args.jd_file:
        with open(args.jd_file, "r", encoding="utf-8") as f:
            jd_text = f.read().strip()
    else:
        jd_text = args.jd.strip()

    _banner("ATS END-TO-END TEST RUNNER")
    print(f"  Resume : {args.resume}")
    print(f"  JD     : {jd_text[:120].strip()}{'...' if len(jd_text) > 120 else ''}")
    print(f"  Algo   : {args.algo}")
    print(f"  DB     : {args.db}")
    print(f"  Device : {DEVICE.upper()}")

    t0 = time.time()

    # sqlite-vec needs a file-based DB
    use_tmp = args.db == ":memory:"
    if use_tmp:
        tmp     = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        db_path = tmp.name
        tmp.close()
    else:
        db_path = args.db

    try:
        # Layer 1: Parse
        parsed = layer1_parse(args.resume)
        parsed["_source_file"] = args.resume

        # Layer 2: Index
        layer2_index(parsed, db_path)

        # Layer 3: Search
        shortlist = layer3_search(jd_text, db_path, skip_hyre=not args.hyre)

        # Layer 4: ATS Score
        score_result = layer4_score(parsed, jd_text, algo=args.algo)

    finally:
        if use_tmp and os.path.exists(db_path):
            os.unlink(db_path)

    elapsed = time.time() - t0

    # ── Final Summary ─────────────────────────────────────────────────────────
    _banner("RESULTS SUMMARY", char="=")
    print(f"  Candidate   : {parsed.get('name', 'Unknown')}")
    print(f"  Final Score : {score_result['final_score']:.1f} / 100")
    print(f"  Algorithm   : {score_result['algo_used']}")
    expl = score_result.get("explanation", "")
    if expl:
        print(f"  Explanation : {expl[:200]}")
    print(f"  Time        : {elapsed:.1f}s")
    print()

    # ── Save output ───────────────────────────────────────────────────────────
    if args.save:
        output = {
            "candidate":       parsed.get("name", ""),
            "resume_path":     args.resume,
            "ats_score":       score_result["final_score"],
            "base_score":      score_result["base_score"],
            "project_score":   score_result["project_score"],
            "exp_score":       score_result["exp_score"],
            "algo":            score_result["algo_used"],
            "matched_skills":  score_result.get("matched_skills", []),
            "missing_skills":  score_result.get("missing_skills", []),
            "hybrid_shortlist": [
                {k: v for k, v in r.items() if k != "raw_json"}
                for r in shortlist
            ]
        }
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"  [SAVED] Results -> {args.save}")

    return score_result["final_score"]


if __name__ == "__main__":
    main()
