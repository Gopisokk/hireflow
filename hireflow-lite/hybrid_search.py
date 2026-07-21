"""
hybrid_search.py — ATS Layer 3: Hybrid Search & Ranking Engine
===============================================================

Accepts a raw Job Description, expands it via HyRE (Hypothetical Resume
Expansion using GPT-4o-mini), runs dual retrieval (BM25 via FTS5 +
semantic search via sqlite-vec), and merges results with Reciprocal Rank
Fusion (RRF) into a single ranked shortlist.

pip install commands:
    pip install sqlite-vec sentence-transformers torch openai

Environment variables:
    OPENAI_API_KEY    — Required for HyRE query expansion

Usage:
    python hybrid_search.py --db ats.db --jd "path/to/jd.txt" --top 10
    python hybrid_search.py --db ats.db --jd-text "We are hiring a..." --top 10
"""

import os
import struct
import sqlite3
import argparse
import warnings
import json
from typing import List, Tuple, Dict, Optional

import sqlite_vec
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────────────────────────────────────
# Database Connection
# ─────────────────────────────────────────────────────────────────────────────

def _connect(db_path: str) -> sqlite3.Connection:
    """Open a SQLite connection and load the sqlite-vec extension."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"Database not found: '{db_path}'. "
            "Run build_database.py first to ingest candidate JSON files."
        )
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.row_factory = sqlite3.Row
    return conn


def _serialize_f32(vector: List[float]) -> bytes:
    """Serialize a float list into a binary blob for sqlite-vec."""
    return struct.pack(f"{len(vector)}f", *vector)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — HyRE: Query Expansion via LLM
# ─────────────────────────────────────────────────────────────────────────────

HYRE_SYSTEM_PROMPT = (
    "You are an expert technical recruiter. "
    "Given a Job Description, write a realistic, detailed resume for the IDEAL candidate. "
    "Include: a professional summary, 2–3 work experience entries with bullet-point achievements, "
    "a projects section, and a skills list. "
    "Write in resume prose style — concise, achievement-focused, no filler words. "
    "Output ONLY the resume text with NO extra commentary."
)


def expand_jd_to_resume(jd_text: str) -> str:
    """
    Use GPT-4o-mini to generate a Hypothetical ideal candidate Resume (HyRE)
    from the given Job Description. This bridges the vocabulary mismatch
    between JDs and real resumes.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(
            "  ⚠ OPENAI_API_KEY not set. Skipping HyRE expansion. "
            "Using raw JD text as the query instead."
        )
        return jd_text

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        print("  → [HyRE] Generating synthetic ideal candidate resume via GPT-4o-mini...")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": HYRE_SYSTEM_PROMPT},
                {"role": "user", "content": f"Job Description:\n\n{jd_text}"},
            ],
            temperature=0.3,
            max_tokens=1200,
        )
        synthetic_resume = response.choices[0].message.content.strip()
        print(f"  ✓ [HyRE] Synthetic resume generated ({len(synthetic_resume)} chars).")
        return synthetic_resume

    except Exception as e:
        print(f"  ✗ [HyRE] OpenAI call failed: {e}. Falling back to raw JD text.")
        return jd_text


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Vectorization
# ─────────────────────────────────────────────────────────────────────────────

def load_embedding_model(device: str = "cpu") -> SentenceTransformer:
    """Load intfloat/e5-base-v2 (768-dim, < 1GB VRAM)."""
    print("  → Loading intfloat/e5-base-v2...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SentenceTransformer("intfloat/e5-base-v2", device=device)
    print(f"  ✓ Model loaded (device={device}).")
    return model


def vectorize_query(model: SentenceTransformer, text: str) -> List[float]:
    """
    Embed the synthetic resume text using the e5 'query: ' prefix convention.
    (Opposite of build_database.py which uses 'passage: ' for documents.)
    """
    prefixed = f"query: {text}"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vector = model.encode(prefixed, normalize_embeddings=True)
    return vector.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Dual Retrieval
# ─────────────────────────────────────────────────────────────────────────────

def bm25_search(conn: sqlite3.Connection, query_text: str, top_k: int = 100) -> List[Tuple[int, float]]:
    """
    BM25 keyword search against the candidates_fts FTS5 table.
    Returns a list of (candidate_id, bm25_score) tuples sorted best-first.
    Note: SQLite FTS5 bm25() returns negative scores (more negative = better match).
    """
    print(f"  → [BM25]   Searching FTS5 index (top_k={top_k})...")

    # Sanitize query: FTS5 special chars can break the query
    safe_query = _sanitize_fts_query(query_text)

    try:
        rows = conn.execute(
            """
            SELECT
                candidate_id,
                bm25(candidates_fts) AS score
            FROM candidates_fts
            WHERE candidates_fts MATCH ?
            ORDER BY score
            LIMIT ?
            """,
            (safe_query, top_k),
        ).fetchall()

        results = [(int(row["candidate_id"]), float(row["score"])) for row in rows]
        print(f"  ✓ [BM25]   Retrieved {len(results)} candidates.")
        return results

    except sqlite3.OperationalError as e:
        print(f"  ✗ [BM25]   FTS5 query failed: {e}")
        return []


def semantic_search(
    conn: sqlite3.Connection,
    vector: List[float],
    top_k: int = 100,
) -> List[Tuple[int, float]]:
    """
    Dense vector search against the candidates_vec table using sqlite-vec.
    Returns a list of (candidate_id, distance) tuples sorted best-first (lowest distance).
    """
    print(f"  → [Vector] Searching vec index (top_k={top_k})...")

    vec_blob = _serialize_f32(vector)

    rows = conn.execute(
        """
        SELECT
            candidate_id,
            distance
        FROM candidates_vec
        WHERE embedding MATCH ?
              AND k = ?
        ORDER BY distance
        """,
        (vec_blob, top_k),
    ).fetchall()

    results = [(int(row["candidate_id"]), float(row["distance"])) for row in rows]
    print(f"  ✓ [Vector] Retrieved {len(results)} candidates.")
    return results


def _sanitize_fts_query(text: str) -> str:
    """
    Strip FTS5 special characters and take the first 200 content words
    to avoid overly long / malformed FTS queries.
    """
    # Remove characters that break FTS5 query syntax (including sentence punctuation)
    # Use regex to keep only alphanumeric words — guaranteed FTS5-safe
    import re as _re
    text = _re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # Collapse whitespace, take first 200 words
    words = text.split()[:200]
    # FTS5 OR-mode: join with spaces (implicit AND by default, which is fine)
    return " ".join(words)


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Reciprocal Rank Fusion (RRF)
# ─────────────────────────────────────────────────────────────────────────────

RRF_K = 60  # Standard smoothing constant (Cormack et al., 2009)


def reciprocal_rank_fusion(
    bm25_results: List[Tuple[int, float]],
    vector_results: List[Tuple[int, float]],
    k: int = RRF_K,
) -> List[Dict]:
    """
    Merge BM25 and semantic search result lists using Reciprocal Rank Fusion.

    For each candidate:
        rrf_score += 1 / (k + rank)

    Candidates appearing in both lists accumulate scores from both.
    The final list is sorted in descending RRF score order.

    Parameters
    ----------
    bm25_results   : List of (candidate_id, score) from FTS5 — already ranked.
    vector_results : List of (candidate_id, distance) from sqlite-vec — already ranked.
    k              : RRF smoothing constant (default 60).

    Returns
    -------
    List of dicts sorted by rrf_score descending:
        {candidate_id, rrf_score, bm25_rank, vector_rank}
    """
    scores: Dict[int, Dict] = {}

    for rank, (cid, raw_score) in enumerate(bm25_results, start=1):
        if cid not in scores:
            scores[cid] = {"candidate_id": cid, "rrf_score": 0.0, "bm25_rank": None, "vector_rank": None}
        scores[cid]["rrf_score"] += 1.0 / (k + rank)
        scores[cid]["bm25_rank"] = rank
        scores[cid]["bm25_raw_score"] = raw_score

    for rank, (cid, distance) in enumerate(vector_results, start=1):
        if cid not in scores:
            scores[cid] = {"candidate_id": cid, "rrf_score": 0.0, "bm25_rank": None, "vector_rank": None}
        scores[cid]["rrf_score"] += 1.0 / (k + rank)
        scores[cid]["vector_rank"] = rank
        scores[cid]["vector_distance"] = distance

    # Sort descending by RRF score
    ranked = sorted(scores.values(), key=lambda x: x["rrf_score"], reverse=True)
    return ranked


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Candidate Retrieval & Output Formatting
# ─────────────────────────────────────────────────────────────────────────────

def enrich_with_metadata(
    conn: sqlite3.Connection,
    ranked_list: List[Dict],
    top_n: int,
) -> List[Dict]:
    """
    Join each ranked candidate_id back against candidates_raw to pull
    the name, github_link, and filename for the output shortlist.
    """
    enriched = []
    for entry in ranked_list[:top_n]:
        cid = entry["candidate_id"]
        row = conn.execute(
            "SELECT name, github_link, filename FROM candidates_raw WHERE candidate_id = ?",
            (cid,),
        ).fetchone()
        if row:
            entry["name"] = row["name"] or f"Candidate #{cid}"
            entry["github_link"] = row["github_link"] or ""
            entry["filename"] = row["filename"]
        else:
            entry["name"] = f"Candidate #{cid}"
            entry["github_link"] = ""
            entry["filename"] = ""
        enriched.append(entry)
    return enriched


def print_shortlist(results: List[Dict]):
    """Pretty-print the ranked shortlist to stdout."""
    print()
    print("=" * 68)
    print(f"  🏆  HYBRID SEARCH SHORTLIST — Top {len(results)} Candidates")
    print("=" * 68)
    print(f"  {'Rank':<5} {'Candidate':<28} {'RRF Score':<12} {'BM25 Rank':<12} {'Vec Rank':<10}")
    print(f"  {'─' * 5} {'─' * 28} {'─' * 12} {'─' * 12} {'─' * 10}")

    for i, r in enumerate(results, start=1):
        name      = str(r.get("name", ""))[:26]
        rrf       = f"{r['rrf_score']:.6f}"
        bm25_r    = str(r["bm25_rank"]) if r["bm25_rank"] is not None else "—"
        vec_r     = str(r["vector_rank"]) if r["vector_rank"] is not None else "—"
        print(f"  {i:<5} {name:<28} {rrf:<12} {bm25_r:<12} {vec_r:<10}")

    print("=" * 68)
    print()

    for i, r in enumerate(results, start=1):
        print(f"  [{i}] {r.get('name', 'Unknown')}")
        if r.get("github_link"):
            print(f"       GitHub  : {r['github_link']}")
        print(f"       File    : {r.get('filename', '')}")
        print(f"       RRF     : {r['rrf_score']:.6f}")
        bm25_r = r["bm25_rank"]
        vec_r  = r["vector_rank"]
        retrieved_by = []
        if bm25_r is not None:
            retrieved_by.append(f"BM25 (rank #{bm25_r})")
        if vec_r is not None:
            retrieved_by.append(f"Vector (rank #{vec_r})")
        print(f"       Found by: {', '.join(retrieved_by)}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def run_hybrid_search(
    db_path: str,
    jd_text: str,
    top_n: int = 10,
    bm25_k: int = 100,
    vec_k: int = 100,
    device: str = "cpu",
    skip_hyre: bool = False,
    output_json: Optional[str] = None,
) -> List[Dict]:
    """
    End-to-end hybrid search pipeline.

    Returns the enriched ranked shortlist.
    """
    print()
    print("=" * 68)
    print("  ATS Layer 3 — Hybrid Search & Ranking Engine")
    print("=" * 68)

    # ── Connect ──────────────────────────────────────────────────────────────
    conn = _connect(db_path)

    # ── Step 1: HyRE Query Expansion ─────────────────────────────────────────
    if skip_hyre:
        print("  ℹ [HyRE] Skipped (--skip-hyre flag set). Using raw JD text.")
        query_text = jd_text
    else:
        query_text = expand_jd_to_resume(jd_text)

    # ── Step 2: Vectorize expanded query ─────────────────────────────────────
    model = load_embedding_model(device=device)
    query_vector = vectorize_query(model, query_text)

    # ── Step 3: Dual Retrieval ───────────────────────────────────────────────
    print()
    bm25_results   = bm25_search(conn, query_text, top_k=bm25_k)
    vector_results = semantic_search(conn, query_vector, top_k=vec_k)

    # ── Step 4: Reciprocal Rank Fusion ───────────────────────────────────────
    print()
    print("  → [RRF]    Fusing BM25 and Vector results...")
    fused = reciprocal_rank_fusion(bm25_results, vector_results, k=RRF_K)
    print(f"  ✓ [RRF]    Fused {len(fused)} unique candidates.")

    # ── Step 5: Enrich & Display ─────────────────────────────────────────────
    shortlist = enrich_with_metadata(conn, fused, top_n)
    print_shortlist(shortlist)

    # ── Optional: Save JSON output ────────────────────────────────────────────
    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            # Convert to serializable form
            json.dump(shortlist, f, indent=2, ensure_ascii=False)
        print(f"  💾 Results saved to: {output_json}")

    conn.close()
    return shortlist


def main():
    parser = argparse.ArgumentParser(
        description="ATS Layer 3: Hybrid Search & Ranking Engine (HyRE + BM25 + Vector + RRF)"
    )
    jd_group = parser.add_mutually_exclusive_group(required=True)
    jd_group.add_argument(
        "--jd",
        type=str,
        metavar="FILE",
        help="Path to a plain-text Job Description file.",
    )
    jd_group.add_argument(
        "--jd-text",
        type=str,
        metavar="TEXT",
        help='Raw Job Description text (wrap in quotes).',
    )
    parser.add_argument("--db",       type=str, default="ats.db",   help="Path to the SQLite database (default: ats.db)")
    parser.add_argument("--top",      type=int, default=10,          help="Number of top candidates to return (default: 10)")
    parser.add_argument("--bm25-k",   type=int, default=100,         help="BM25 retrieval pool size (default: 100)")
    parser.add_argument("--vec-k",    type=int, default=100,         help="Vector retrieval pool size (default: 100)")
    parser.add_argument("--device",   type=str, default="cpu",       help="Torch device: 'cpu' or 'cuda' (default: cpu)")
    parser.add_argument("--skip-hyre", action="store_true",          help="Skip HyRE expansion and use raw JD text directly")
    parser.add_argument("--output-json", type=str, default=None,     help="Optional: path to save ranked shortlist as JSON")
    args = parser.parse_args()

    # Load JD text
    if args.jd:
        with open(args.jd, "r", encoding="utf-8") as f:
            jd_text = f.read().strip()
    else:
        jd_text = args.jd_text.strip()

    run_hybrid_search(
        db_path=args.db,
        jd_text=jd_text,
        top_n=args.top,
        bm25_k=args.bm25_k,
        vec_k=args.vec_k,
        device=args.device,
        skip_hyre=args.skip_hyre,
        output_json=args.output_json,
    )


if __name__ == "__main__":
    main()
