"""
database.py — HireFlow-Lite: Database Schema & Connection Manager
=================================================================
Single source of truth for all DB operations.

Schema:
  students          — roll_number PK, metadata, scores, stage
  students_fts      — FTS5 virtual table for BM25 keyword search
  resume_embeddings — sqlite-vec virtual table (384-dim MiniLM)
  jobs              — JD runs with results
"""

import json
import struct
import sqlite3
from pathlib import Path
from typing import List, Optional

import sqlite_vec


DB_PATH = Path(__file__).parent / "hireflow.db"
EMBED_DIM = 384   # all-MiniLM-L6-v2


# ── Connection ────────────────────────────────────────────────────────────────

def connect(db_path: str | Path = DB_PATH) -> sqlite3.Connection:
    """Open SQLite + load sqlite-vec extension."""
    conn = sqlite3.connect(str(db_path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-64000")   # 64 MB page cache
    conn.execute("PRAGMA temp_store=MEMORY")
    return conn


# ── Schema ────────────────────────────────────────────────────────────────────

SCHEMA_SQL = """
-- Student metadata & scores
CREATE TABLE IF NOT EXISTS students (
    roll_number     TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    github_url      TEXT DEFAULT '',
    github_token    TEXT DEFAULT '',
    email           TEXT DEFAULT '',
    phone           TEXT DEFAULT '',
    education       TEXT DEFAULT '',
    skills          TEXT DEFAULT '[]',   -- JSON array
    projects        TEXT DEFAULT '[]',   -- JSON array of {name, description}
    raw_text        TEXT DEFAULT '',
    resume_filename TEXT DEFAULT '',
    ingested_at     TEXT DEFAULT (datetime('now')),
    ats_score       REAL DEFAULT 0.0,
    github_score    REAL,
    composite_score REAL,            -- 65% ATS + 35% GitHub, display-only
    final_score     REAL DEFAULT 0.0,
    matched_skills  TEXT DEFAULT '[]',   -- JSON array
    missing_skills  TEXT DEFAULT '[]',   -- JSON array
    github_details  TEXT DEFAULT '{}',   -- JSON object
    project_verif   TEXT DEFAULT '[]',   -- JSON array
    explanation     TEXT DEFAULT '',
    stage           TEXT DEFAULT 'ingested'
              -- ingested | scored | github_verified | done
);

-- BM25 full-text search (FTS5)
CREATE VIRTUAL TABLE IF NOT EXISTS students_fts USING fts5(
    roll_number UNINDEXED,
    content,
    tokenize='porter ascii'
);

-- Dense vector search (sqlite-vec, 384-dim MiniLM)
CREATE VIRTUAL TABLE IF NOT EXISTS resume_embeddings USING vec0(
    roll_number TEXT,
    embedding   FLOAT[384]
);

-- Job Description runs (cached)
CREATE TABLE IF NOT EXISTS jobs (
    job_id      TEXT PRIMARY KEY,
    jd_text     TEXT NOT NULL,
    created_at  TEXT DEFAULT (datetime('now')),
    top_k       INTEGER DEFAULT 50,
    results     TEXT DEFAULT '[]'  -- JSON array of roll_numbers ranked
);
"""


def init_db(db_path: str | Path = DB_PATH) -> sqlite3.Connection:
    """Create all tables if they don't exist. Returns open connection."""
    conn = connect(db_path)
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    print(f"  ✓ Database ready: {db_path}")
    return conn


def reset_db(db_path: str | Path = DB_PATH) -> sqlite3.Connection:
    """Drop all tables and re-create. Use for fresh ingestion."""
    conn = connect(db_path)
    conn.executescript("""
        DROP TABLE IF EXISTS students;
        DROP TABLE IF EXISTS students_fts;
        DROP TABLE IF EXISTS resume_embeddings;
        DROP TABLE IF EXISTS jobs;
    """)
    conn.commit()
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    print(f"  ✓ Database reset: {db_path}")
    return conn


# ── Serialization ─────────────────────────────────────────────────────────────

def serialize_vec(vector: List[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


def deserialize_vec(blob: bytes) -> List[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


# ── Write Operations ──────────────────────────────────────────────────────────

def upsert_student(conn: sqlite3.Connection, student: dict) -> None:
    """Insert or replace a student record."""
    conn.execute("""
        INSERT OR REPLACE INTO students
          (roll_number, name, github_url, github_token, email, phone,
           education, skills, projects, raw_text, resume_filename,
           ingested_at, stage)
        VALUES (?,?,?,?,?,?,?,?,?,?,?, datetime('now'), 'ingested')
    """, (
        student["roll_number"],
        student["name"],
        student.get("github_url", ""),
        student.get("github_token", ""),
        student.get("email", ""),
        student.get("phone", ""),
        student.get("education", ""),
        json.dumps(student.get("skills", [])),
        json.dumps(student.get("projects", [])),
        student.get("raw_text", ""),
        student.get("resume_filename", ""),
    ))


def upsert_fts(conn: sqlite3.Connection, roll_number: str, content: str) -> None:
    """Insert or replace FTS5 entry."""
    conn.execute("DELETE FROM students_fts WHERE roll_number = ?", (roll_number,))
    conn.execute(
        "INSERT INTO students_fts(roll_number, content) VALUES (?, ?)",
        (roll_number, content),
    )


def upsert_embedding(conn: sqlite3.Connection, roll_number: str, embedding: List[float]) -> None:
    """Insert or replace vector embedding."""
    conn.execute(
        "DELETE FROM resume_embeddings WHERE roll_number = ?", (roll_number,)
    )
    conn.execute(
        "INSERT INTO resume_embeddings(roll_number, embedding) VALUES (?, ?)",
        (roll_number, serialize_vec(embedding)),
    )


def update_scores(conn: sqlite3.Connection, roll_number: str, scores: dict) -> None:
    """Update ATS scoring fields after search."""
    conn.execute("""
        UPDATE students SET
            ats_score      = ?,
            final_score    = ?,
            matched_skills = ?,
            missing_skills = ?,
            explanation    = ?,
            stage          = 'scored'
        WHERE roll_number = ?
    """, (
        scores.get("ats_score", 0.0),
        scores.get("final_score", 0.0),
        json.dumps(scores.get("matched_skills", [])),
        json.dumps(scores.get("missing_skills", [])),
        scores.get("explanation", ""),
        roll_number,
    ))


def update_github(conn: sqlite3.Connection, roll_number: str, gh: dict) -> None:
    """
    Update GitHub verification fields.

    IMPORTANT: ats_score is NEVER modified here. The three score columns
    serve distinct purposes:
      ats_score      — unchanged ATS fit score (set during skill scoring)
      github_score   — independent GitHub credibility score (0-100)
      composite_score — 65% ATS + 35% GitHub, display-only blend
      final_score    — equals composite_score when GitHub verified,
                        equals ats_score otherwise
    """
    conn.execute("""
        UPDATE students SET
            github_score    = ?,
            composite_score = ?,
            final_score     = ?,
            github_details  = ?,
            project_verif   = ?,
            stage           = 'github_verified'
        WHERE roll_number = ?
    """, (
        gh.get("github_score"),
        gh.get("composite_score"),
        gh.get("final_score", 0.0),
        json.dumps(gh.get("github_details", {})),
        json.dumps(gh.get("project_verif", [])),
        roll_number,
    ))


# ── Read Operations ───────────────────────────────────────────────────────────

def get_all_students(conn: sqlite3.Connection) -> List[dict]:
    rows = conn.execute("""
        SELECT roll_number, name, github_url, email, skills, projects,
               ats_score, github_score, final_score, matched_skills,
               missing_skills, resume_filename, stage, ingested_at
        FROM students ORDER BY final_score DESC
    """).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d["skills"] = json.loads(d["skills"] or "[]")
        d["projects"] = json.loads(d["projects"] or "[]")
        d["matched_skills"] = json.loads(d["matched_skills"] or "[]")
        d["missing_skills"] = json.loads(d["missing_skills"] or "[]")
        result.append(d)
    return result


def get_student(conn: sqlite3.Connection, roll_number: str) -> Optional[dict]:
    row = conn.execute(
        "SELECT * FROM students WHERE roll_number = ?", (roll_number,)
    ).fetchone()
    if not row:
        return None
    d = dict(row)
    for field in ("skills", "projects", "matched_skills", "missing_skills",
                  "github_details", "project_verif"):
        d[field] = json.loads(d.get(field) or "[]" if field != "github_details" else "{}")
    return d


def count_students(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]


def get_db_stats(conn: sqlite3.Connection) -> dict:
    total = conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]
    ingested = conn.execute("SELECT COUNT(*) FROM students WHERE stage='ingested'").fetchone()[0]
    scored = conn.execute("SELECT COUNT(*) FROM students WHERE stage IN ('scored','github_verified','done')").fetchone()[0]
    gh_verified = conn.execute("SELECT COUNT(*) FROM students WHERE github_score IS NOT NULL").fetchone()[0]
    avg_score = conn.execute("SELECT AVG(final_score) FROM students WHERE final_score > 0").fetchone()[0]
    return {
        "total": total,
        "ingested": ingested,
        "scored": scored,
        "gh_verified": gh_verified,
        "avg_score": round(avg_score or 0, 1),
    }


# ── Migration Helpers ─────────────────────────────────────────────────────────

def migrate_add_composite_score(db_path: str | Path = DB_PATH) -> None:
    """
    Migration: add composite_score column to existing databases.

    Safe to run multiple times — checks if column already exists.
    The composite_score stores 65% ATS + 35% GitHub as a display-only
    blend. It is computed in search.py and stored here separately from
    both ats_score and github_score, which remain independent.
    """
    conn = connect(db_path)
    try:
        cols = [row[1] for row in conn.execute("PRAGMA table_info(students)").fetchall()]
        if "composite_score" not in cols:
            conn.execute("ALTER TABLE students ADD COLUMN composite_score REAL")
            conn.commit()
            print("  [OK] Migration: added composite_score column to students table")
        else:
            print("  [OK] composite_score column already exists -- no migration needed")
    finally:
        conn.close()

