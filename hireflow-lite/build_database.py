"""
build_database.py — ATS Layer 2: Hybrid Search Database Builder
================================================================

Reads candidate JSON files from a directory, generates dense embeddings,
and stores all data into a multi-table SQLite database configured for
BM25 keyword search (FTS5) and semantic vector search (sqlite-vec).

pip install commands:
    pip install sqlite-vec sentence-transformers torch

Usage:
    python build_database.py --input ./parsed_resumes --db ats.db
"""

import os
import json
import struct
import sqlite3
import argparse
import warnings
from pathlib import Path
from typing import List, Optional

import sqlite_vec
from sentence_transformers import SentenceTransformer


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _serialize_f32(vector: List[float]) -> bytes:
    """Serialize a list of floats into a byte string for sqlite-vec storage."""
    return struct.pack(f"{len(vector)}f", *vector)


def _connect(db_path: str) -> sqlite3.Connection:
    """Open a SQLite connection and load the sqlite-vec extension."""
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.row_factory = sqlite3.Row
    return conn


def _flatten_text(data: dict) -> str:
    """
    Flatten all meaningful text from the candidate JSON into a single
    text block suitable for embedding and FTS5 indexing.

    Covers: Experience + Open Source, Projects, Skills (both primary and secondary).
    """
    parts: List[str] = []

    # ── Experience & Open Source ─────────────────────────────────────────────
    for section_key in ("work_experience", "open_source_contributions"):
        for item in data.get(section_key, []):
            if isinstance(item, dict):
                # Description or any string value in the dict
                for v in item.values():
                    if isinstance(v, str) and v:
                        parts.append(v)
            elif isinstance(item, str):
                parts.append(item)

    # ── Projects ─────────────────────────────────────────────────────────────
    for proj in data.get("projects", []):
        if isinstance(proj, dict):
            for v in proj.values():
                if isinstance(v, str) and v:
                    parts.append(v)
        elif isinstance(proj, str):
            parts.append(proj)

    # ── Technologies / Skills ────────────────────────────────────────────────
    technologies = data.get("technologies", {})
    if isinstance(technologies, dict):
        for skill_group in ("primary_skills", "secondary_skills"):
            for skill_entry in technologies.get(skill_group, []):
                if isinstance(skill_entry, dict):
                    # {"skill_span": "Python", "context": "..."}
                    skill_span = skill_entry.get("skill_span", "")
                    context = skill_entry.get("context", "")
                    if skill_span:
                        parts.append(skill_span)
                    if context:
                        parts.append(context)
                elif isinstance(skill_entry, str):
                    parts.append(skill_entry)
    elif isinstance(technologies, list):
        for item in technologies:
            if isinstance(item, str):
                parts.append(item)

    return " ".join(filter(None, parts))


def _extract_github_link(data: dict) -> str:
    """Pull the GitHub link from the header section."""
    header = data.get("header", {})
    if isinstance(header, dict):
        return header.get("github", "") or ""
    return ""


def _extract_name(data: dict) -> str:
    """Pull candidate name from the header section."""
    header = data.get("header", {})
    if isinstance(header, dict):
        return header.get("name", "") or ""
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Database Setup
# ─────────────────────────────────────────────────────────────────────────────

EMBEDDING_DIM = 768  # intfloat/e5-base-v2 outputs 768-dimensional vectors


def setup_database(conn: sqlite3.Connection):
    """
    Create the three required tables if they do not already exist:
      1. candidates_raw   — full JSON blob + GitHub link for retrieval.
      2. candidates_fts   — FTS5 virtual table for BM25 keyword search.
      3. candidates_vec   — sqlite-vec table for semantic vector search.
    """
    cursor = conn.cursor()

    # 1. Raw document store
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS candidates_raw (
            candidate_id  INTEGER PRIMARY KEY AUTOINCREMENT,
            filename      TEXT    NOT NULL UNIQUE,
            name          TEXT    DEFAULT '',
            github_link   TEXT    DEFAULT '',
            raw_json      TEXT    NOT NULL,
            inserted_at   TEXT    DEFAULT (datetime('now'))
        )
    """)

    # 2. FTS5 full-text search table (BM25 keyword search)
    #    content='' makes it a contentless table; we store the flat text ourselves.
    cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS candidates_fts
        USING fts5(
            candidate_id UNINDEXED,
            experience_and_open_source,
            projects,
            skills,
            tokenize = 'porter ascii'
        )
    """)

    # 3. Vector table for semantic search (sqlite-vec)
    cursor.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS candidates_vec
        USING vec0(
            candidate_id  INTEGER PRIMARY KEY,
            embedding     float[{EMBEDDING_DIM}]
        )
    """)

    conn.commit()
    print("  ✓ Database schema ready.")


# ─────────────────────────────────────────────────────────────────────────────
# Embedding Model
# ─────────────────────────────────────────────────────────────────────────────

def load_embedding_model(device: str = "cpu") -> SentenceTransformer:
    """
    Load intfloat/e5-base-v2.

    e5 models require the prefix 'query: ' for queries and 'passage: ' for
    documents at inference time to achieve best performance.
    """
    print("  → Loading intfloat/e5-base-v2 embedding model...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SentenceTransformer("intfloat/e5-base-v2", device=device)
    print(f"  ✓ Embedding model loaded on device={device}.")
    return model


def embed_text(model: SentenceTransformer, text: str) -> List[float]:
    """Embed a passage using the e5 'passage:' prefix convention."""
    prefixed = f"passage: {text}" if text.strip() else "passage: "
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vector = model.encode(prefixed, normalize_embeddings=True)
    return vector.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# JSON Field Extractors for FTS Columns
# ─────────────────────────────────────────────────────────────────────────────

def _fts_experience_text(data: dict) -> str:
    """
    Build the FTS column: experience_and_open_source.
    Combines work_experience descriptions and open_source_contributions.
    """
    parts = []
    for section in ("work_experience", "open_source_contributions"):
        for item in data.get(section, []):
            if isinstance(item, dict):
                parts.extend(str(v) for v in item.values() if v)
            elif isinstance(item, str):
                parts.append(item)
    return " ".join(filter(None, parts))


def _fts_projects_text(data: dict) -> str:
    """Build the FTS column: projects."""
    parts = []
    for proj in data.get("projects", []):
        if isinstance(proj, dict):
            parts.extend(str(v) for v in proj.values() if v)
        elif isinstance(proj, str):
            parts.append(proj)
    return " ".join(filter(None, parts))


def _fts_skills_text(data: dict) -> str:
    """
    Build the FTS column: skills.
    Flattens primary + secondary skills into plain text.
    """
    parts = []
    technologies = data.get("technologies", {})
    if isinstance(technologies, dict):
        for group in ("primary_skills", "secondary_skills"):
            for skill_entry in technologies.get(group, []):
                if isinstance(skill_entry, dict):
                    parts.append(skill_entry.get("skill_span", ""))
                elif isinstance(skill_entry, str):
                    parts.append(skill_entry)
    return " ".join(filter(None, parts))


# ─────────────────────────────────────────────────────────────────────────────
# Core Data Processing
# ─────────────────────────────────────────────────────────────────────────────

def process_directory(
    input_dir: str,
    conn: sqlite3.Connection,
    model: SentenceTransformer,
):
    """
    Iterate through all JSON files in input_dir.

    For each file:
      1. Parse the JSON.
      2. Extract and flatten all searchable text.
      3. Generate the dense embedding.
      4. Insert into candidates_raw, candidates_fts, and candidates_vec.
    """
    json_files = list(Path(input_dir).glob("*.json"))
    if not json_files:
        print(f"  ⚠ No JSON files found in '{input_dir}'.")
        return

    print(f"  → Found {len(json_files)} JSON file(s) to process.\n")
    cursor = conn.cursor()

    for idx, json_path in enumerate(json_files, start=1):
        filename = json_path.name
        print(f"  [{idx}/{len(json_files)}] Processing: {filename}")

        # ── 1. Parse JSON ────────────────────────────────────────────────────
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"    ✗ Skipped — could not parse JSON: {e}")
            continue

        # ── 2. Extract text columns ──────────────────────────────────────────
        flat_text          = _flatten_text(data)
        fts_experience     = _fts_experience_text(data)
        fts_projects       = _fts_projects_text(data)
        fts_skills         = _fts_skills_text(data)
        github_link        = _extract_github_link(data)
        name               = _extract_name(data)
        raw_json_str       = json.dumps(data, ensure_ascii=False)

        if not flat_text.strip():
            print("    ⚠ No extractable text found — inserting empty embedding.")

        # ── 3. Generate dense embedding ──────────────────────────────────────
        print(f"    → Embedding text ({len(flat_text)} chars)...")
        vector = embed_text(model, flat_text)

        # ── 4. Insert into all three tables (linked by candidate_id) ─────────
        try:
            # 4a. candidates_raw
            cursor.execute(
                """
                INSERT INTO candidates_raw (filename, name, github_link, raw_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(filename) DO UPDATE SET
                    name = excluded.name,
                    github_link = excluded.github_link,
                    raw_json = excluded.raw_json,
                    inserted_at = datetime('now')
                """,
                (filename, name, github_link, raw_json_str),
            )
            candidate_id = cursor.lastrowid

            # If the row already existed, fetch its id
            if not candidate_id:
                row = cursor.execute(
                    "SELECT candidate_id FROM candidates_raw WHERE filename = ?",
                    (filename,),
                ).fetchone()
                candidate_id = row["candidate_id"]

            # 4b. candidates_fts
            # Delete existing entry to avoid duplicates on re-runs
            cursor.execute(
                "DELETE FROM candidates_fts WHERE candidate_id = ?",
                (candidate_id,),
            )
            cursor.execute(
                """
                INSERT INTO candidates_fts
                    (candidate_id, experience_and_open_source, projects, skills)
                VALUES (?, ?, ?, ?)
                """,
                (candidate_id, fts_experience, fts_projects, fts_skills),
            )

            # 4c. candidates_vec
            cursor.execute(
                "DELETE FROM candidates_vec WHERE candidate_id = ?",
                (candidate_id,),
            )
            cursor.execute(
                "INSERT INTO candidates_vec (candidate_id, embedding) VALUES (?, ?)",
                (candidate_id, _serialize_f32(vector)),
            )

            conn.commit()
            print(f"    ✓ Inserted candidate_id={candidate_id} — '{name}'")

        except sqlite3.Error as e:
            conn.rollback()
            print(f"    ✗ Database error for {filename}: {e}")

    print("\n  ✅ All candidates processed.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ATS Layer 2: Hybrid Search Database Builder"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="./parsed_resumes",
        help="Input directory containing candidate JSON files (default: ./parsed_resumes)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="ats.db",
        help="Path to the SQLite database file (default: ats.db)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device for embedding model: 'cpu' or 'cuda' (default: cpu)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  ATS Layer 2 — Hybrid Search Database Builder")
    print("=" * 60)
    print(f"  Input Dir : {args.input}")
    print(f"  Database  : {args.db}")
    print(f"  Device    : {args.device}")
    print()

    # Connect to SQLite and load sqlite-vec
    print("  → Connecting to database...")
    conn = _connect(args.db)
    setup_database(conn)

    # Load embedding model
    model = load_embedding_model(device=args.device)

    # Process all JSON files
    print()
    process_directory(args.input, conn, model)

    conn.close()
    print(f"\n  Database saved to: {os.path.abspath(args.db)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
