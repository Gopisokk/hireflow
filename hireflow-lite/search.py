"""
search.py — HireFlow-Lite: Phase 2 — ATS Search & Ranking Engine
=================================================================
Given a Job Description, this module:
  1. Generates a 384-dim MiniLM embedding of the JD
  2. Runs vector similarity search against resume_embeddings (sqlite-vec)
  3. Runs BM25 keyword search against students_fts (FTS5)
  4. Fuses results with Reciprocal Rank Fusion (RRF)
  5. Scores the shortlist with the 8-component evidence-aggregation engine
  6. Runs GitHub verification on shortlisted candidates (using per-student tokens)
  7. Writes final scores back to the DB
  8. Returns ranked results

Usage (CLI):
    python search.py --jd "We are hiring..." --top 50 --db hireflow.db

Usage (API):
    from search import run_search
    results = run_search(jd_text, db_path, top_k, github_verify, progress_cb)
"""

import re
import sys
import json
import time
import struct
import sqlite3
import warnings
import argparse
from pathlib import Path
from typing import Callable, Optional

import torch

from database import (
    connect, DB_PATH, serialize_vec,
    update_scores, update_github, get_all_students,
)

import os
os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN_WARNING"] = "1"

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


# ── SBERT Setup (CPU mode for instant, crash-free vector encoding) ──────────

DEVICE = "cpu"
_MODEL = None


def _get_model():
    from minilm import get_minilm_model
    return get_minilm_model(device="cpu")


# ── Import ATS engine ────────────────────────────────────────────────────────
from ats_engine import score_candidate, extract_jd_keywords as _extract_jd_keywords


# ── Semantic Search (sqlite-vec) ──────────────────────────────────────────────

def _vector_search(conn: sqlite3.Connection, jd_embedding: list, top_k: int) -> list[tuple[str, float]]:
    """
    KNN vector search via sqlite-vec.
    Returns [(roll_number, distance)] ordered by ascending distance (= descending similarity).
    """
    blob = serialize_vec(jd_embedding)
    rows = conn.execute("""
        SELECT roll_number, distance
        FROM resume_embeddings
        WHERE embedding MATCH ?
        ORDER BY distance
        LIMIT ?
    """, (blob, top_k)).fetchall()
    return [(r["roll_number"], r["distance"]) for r in rows]


# ── BM25 Search (FTS5) ────────────────────────────────────────────────────────

def _bm25_search(conn: sqlite3.Connection, query_text: str, top_k: int) -> list[tuple[str, float]]:
    """
    BM25 full-text search via FTS5.
    Returns [(roll_number, bm25_score)] ordered by descending relevance.
    FTS5 bm25() returns negative values (more negative = better match).
    """
    # Clean query for FTS5 (remove special chars)
    clean_query = re.sub(r'[^\w\s]', ' ', query_text)
    clean_query = " ".join(clean_query.split()[:50])  # max 50 terms

    try:
        rows = conn.execute("""
            SELECT roll_number, bm25(students_fts) as score
            FROM students_fts
            WHERE content MATCH ?
            ORDER BY score
            LIMIT ?
        """, (clean_query, top_k)).fetchall()
        return [(r["roll_number"], abs(r["score"])) for r in rows]
    except sqlite3.OperationalError:
        return []


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def _rrf(
    vec_results:  list[tuple[str, float]],
    bm25_results: list[tuple[str, float]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    Reciprocal Rank Fusion: merges two ranked lists into one.
    RRF(d) = Σ 1/(k + rank_i(d))
    k=60 is the standard constant from the original RRF paper.
    """
    scores: dict[str, float] = {}

    for rank, (roll, _) in enumerate(vec_results, 1):
        scores[roll] = scores.get(roll, 0.0) + 1.0 / (k + rank)

    for rank, (roll, _) in enumerate(bm25_results, 1):
        scores[roll] = scores.get(roll, 0.0) + 1.0 / (k + rank)

    # Normalise to 0–100
    if scores:
        max_s = max(scores.values())
        scores = {r: (s / max_s) * 100 for r, s in scores.items()}

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ── Final Score Combiner (GitHub) ────────────────────────────────────────────

def _compute_composite(ats_score: float, gh_score: float | None) -> float:
    """
    Composite score = 65% ATS + 35% GitHub.

    This is a DISPLAY-ONLY value. It must never overwrite ats_score or
    github_score in the database or in memory — those remain independent
    0-100 scores answering different questions:

      ATS Score    — "How well does this candidate match the JD?"
      GitHub Score — "Is the candidate's technical evidence credible?"
      Composite    — A weighted blend for initial ranking convenience.

    If GitHub verification was skipped, composite == ats_score.
    """
    if gh_score is None:
        return ats_score
    return round(min(100.0, (ats_score * 0.65) + (gh_score * 0.35)), 1)


# Keep _combine_with_github as an alias for backward compatibility
_combine_with_github = _compute_composite

# ── Main Search Pipeline ──────────────────────────────────────────────────────

def run_search(
    jd_text: str,
    db_path: str | Path = DB_PATH,
    top_k: int = 50,
    github_verify: bool = True,
    global_github_token: str = "",
    progress_cb: Optional[Callable[[str, int, int], None]] = None,
) -> list[dict]:
    """
    Full search + ranking pipeline.
    Returns ranked list of candidate dicts.
    """
    t0 = time.time()
    conn = connect(db_path)

    total_in_db = conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]
    if total_in_db == 0:
        conn.close()
        raise ValueError("Database is empty. Run ingestion first.")

    print("\n  ╔══════════════════════════════════════════╗")
    print(  "  ║  Search: JD → Embed                      ║")
    print(  "  ╚══════════════════════════════════════════╝")
    print(f"  → JD: \"{jd_text[:100]}...\"")

    # ── Step 1: Embed JD ──────────────────────────────────────────────────────
    model = _get_model()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        jd_embedding = model.encode(
            jd_text, convert_to_numpy=True,
            normalize_embeddings=True, device=DEVICE,
        ).tolist()

    jd_keywords = _extract_jd_keywords(jd_text)
    print(f"  ✓ JD keywords: {len(jd_keywords['required'])} required, {len(jd_keywords['preferred'])} preferred")
    if progress_cb:
        progress_cb("embed_jd", 1, 1)

    # ── Step 2: Vector Search ─────────────────────────────────────────────────
    print("\n  ╔══════════════════════════════════════════╗")
    print(  "  ║  Search: Vector Similarity (sqlite-vec)  ║")
    print(  "  ╚══════════════════════════════════════════╝")
    retrieve_k = min(total_in_db, top_k * 4)  # over-retrieve then re-rank

    vec_results  = _vector_search(conn, jd_embedding, retrieve_k)
    bm25_results = _bm25_search(conn, " ".join(jd_keywords["all"][:30]), retrieve_k)

    print(f"  ✓ Vector search: {len(vec_results)} candidates retrieved")
    print(f"  ✓ BM25 search:   {len(bm25_results)} candidates retrieved")

    # ── Step 3: RRF Fusion ────────────────────────────────────────────────────
    fused = _rrf(vec_results, bm25_results)
    shortlist_rolls = [r for r, _ in fused[:top_k]]
    print(f"  ✓ RRF fusion → top {len(shortlist_rolls)} candidates shortlisted")
    if progress_cb:
        progress_cb("rrf", len(shortlist_rolls), min(total_in_db, top_k))

    # ── Step 4: Skill Scoring ─────────────────────────────────────────────────
    print("\n  ╔══════════════════════════════════════════╗")
    print(  "  ║  Search: Skill Scoring                   ║")
    print(  "  ╚══════════════════════════════════════════╝")

    rrf_map = dict(fused)
    shortlist = []

    for roll in shortlist_rolls:
        row = conn.execute(
            "SELECT roll_number, name, email, phone, education, skills, projects, "
            "raw_text, github_url, github_token, resume_filename FROM students "
            "WHERE roll_number=?",
            (roll,)
        ).fetchone()
        if not row:
            continue
        d = dict(row)
        d["skills"]   = json.loads(d["skills"]   or "[]")
        d["projects"] = json.loads(d["projects"]  or "[]")

        rrf_score = rrf_map.get(roll, 0.0)

        # ── 8-component evidence aggregation score ────────────────────────
        scored = score_candidate(
            raw_text   = d.get("raw_text", ""),
            skills     = d["skills"],
            projects   = d["projects"],
            education  = d.get("education", ""),
            email      = d.get("email", ""),
            phone      = d.get("phone", ""),
            github_url = d.get("github_url", ""),
            jd_text    = jd_text,
            model      = _get_model(),
            device     = DEVICE,
        )

        ats_score = scored["final_score"]

        db_scores = {
            "ats_score":      ats_score,
            "final_score":    ats_score,   # updated after GitHub
            "matched_skills": scored["matched_skills"],
            "missing_skills": scored["missing_skills"],
            "explanation":    scored["explanation"],
        }
        update_scores(conn, roll, db_scores)

        d.update(db_scores)
        d["rrf_score"]   = round(rrf_score, 2)
        d["skill_score"] = scored["components"]["required_skill_match"]["score"]
        d["components"]  = scored["components"]
        d["penalty"]     = scored["penalty"]
        shortlist.append(d)

    conn.commit()
    print(f"  ✓ Scored {len(shortlist)} candidates")
    if progress_cb:
        progress_cb("score", len(shortlist), len(shortlist))

    # ── Step 5: GitHub Verification ───────────────────────────────────────────
    if github_verify:
        print("\n  ╔══════════════════════════════════════════╗")
        print(  "  ║  Search: GitHub Verification             ║")
        print(  "  ╚══════════════════════════════════════════╝")

        gh_candidates = [
            s for s in shortlist
            if s.get("github_url") and (s.get("github_token") or global_github_token)
        ]
        print(f"  → {len(gh_candidates)}/{len(shortlist)} candidates have GitHub tokens")

        from github_verifier import run_github_verification, fetch_github_profile

        for i, s in enumerate(gh_candidates):
            token    = s.get("github_token") or global_github_token
            username = _extract_github_username(s.get("github_url", ""))
            if not username:
                continue
            try:
                gh_result = run_github_verification(
                    username=username,
                    token=token,
                    resume_projects=[p.get("name","") if isinstance(p, dict) else str(p) for p in s.get("projects",[])],
                    resume_skills=s.get("skills", []),
                    resume_email=s.get("email", ""),
                    jd_text=jd_text,
                )
                gh_score = gh_result.get("score", 0.0)

                # ── Project verification (multi-signal via project_verifier) ─
                try:
                    from project_verifier import verify_projects
                    profile    = fetch_github_profile(username, token)
                    gh_repos   = profile.get("repositories", {}).get("nodes", [])
                    resume_projs = [
                        p if isinstance(p, dict) else {"name": str(p), "description": ""}
                        for p in s.get("projects", [])
                    ]
                    proj_verif = verify_projects(
                        resume_projects=resume_projs,
                        github_repos=gh_repos,
                        device=DEVICE,
                        github_username=username,
                        github_token=token,
                    )
                except Exception as pv_exc:
                    print(f"  ⚠ project_verifier error for @{username}: {pv_exc}")
                    proj_verif = []

                # ── SCORE INDEPENDENCE: never overwrite ats_score ────────────
                # ats_score:      unchanged (measures JD-resume fit)
                # github_score:   independent (measures technical credibility)
                # composite_score: display-only blend (65% ATS + 35% GitHub)
                composite = _compute_composite(s["ats_score"], gh_score)
                gh_data = {
                    "github_score":    gh_score,
                    "composite_score": composite,
                    "final_score":     composite,   # DB final_score = composite
                    "github_details":  gh_result.get("group_scores", gh_result.get("category_scores", {})),
                    "project_verif":   proj_verif,
                }
                update_github(conn, s["roll_number"], gh_data)
                s.update(gh_data)
                # Ensure ats_score is never silently replaced
                s["ats_score"] = s.get("ats_score", 0.0)

                print(
                    f"  ✓ [{i+1}/{len(gh_candidates)}] @{username}: "
                    f"ATS={s['ats_score']:.1f}  GH={gh_score:.1f}  Composite={composite:.1f}"
                )
            except Exception as exc:
                print(f"  ⚠ [{i+1}/{len(gh_candidates)}] @{username}: {exc}")

            if progress_cb:
                progress_cb("github", i + 1, len(gh_candidates))

        conn.commit()

    conn.close()

    # ── Step 6: Final Rank (by composite if available, else ATS) ──────────────
    shortlist.sort(
        key=lambda s: s.get("composite_score", s.get("final_score", s.get("ats_score", 0.0))),
        reverse=True,
    )
    for i, s in enumerate(shortlist, 1):
        s["rank"] = i
        # Clean out raw_text from return payload (too large)
        s.pop("raw_text", None)
        s.pop("github_token", None)  # never return tokens

    elapsed = time.time() - t0
    print(f"\n  ✓ Search complete in {elapsed:.1f}s — {len(shortlist)} candidates ranked")
    return shortlist


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_github_username(url: str) -> str:
    import re
    if not url:
        return ""
    url = url.strip().rstrip("/")
    m = re.search(r"github\.com/([a-zA-Z0-9_-]+)", url)
    if m:
        u = m.group(1)
        if u.lower() not in {"settings","login","signup","explore","trending"}:
            return u
    return url if "/" not in url else ""


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="HireFlow-Lite: Search JD against ingested resumes")
    p.add_argument("--jd",     type=str, help="Job description text")
    p.add_argument("--jd-file",type=str, help="Path to JD text file")
    p.add_argument("--top",    type=int, default=50, help="Shortlist size (default: 50)")
    p.add_argument("--db",     default=str(DB_PATH), help="SQLite DB path")
    p.add_argument("--no-github", action="store_true", help="Skip GitHub verification")
    p.add_argument("--github-token", default="", help="Global fallback GitHub token")
    args = p.parse_args()

    if args.jd_file:
        jd_text = open(args.jd_file, encoding="utf-8").read().strip()
    elif args.jd:
        jd_text = args.jd
    else:
        p.error("Provide --jd or --jd-file")

    results = run_search(
        jd_text=jd_text,
        db_path=args.db,
        top_k=args.top,
        github_verify=not args.no_github,
        global_github_token=args.github_token,
    )

    import csv, sys
    writer = csv.DictWriter(sys.stdout, fieldnames=[
        "rank","roll_number","name","email","ats_score","github_score","final_score",
        "matched_skills","missing_skills",
    ], extrasaction="ignore")
    writer.writeheader()
    for r in results:
        r2 = dict(r)
        r2["matched_skills"] = "; ".join(r2.get("matched_skills",[]))
        r2["missing_skills"] = "; ".join(r2.get("missing_skills",[]))
        writer.writerow(r2)


if __name__ == "__main__":
    main()
