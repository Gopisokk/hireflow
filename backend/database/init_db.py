"""
HireFlow Database Initialization
---------------------------------
Creates the SQLite schema with sqlite-vec virtual tables.
Provides get_db() for connection management and init_database() for bootstrapping.
"""

import sqlite3
import struct
from pathlib import Path
from typing import Generator

import sqlite_vec

from config import DB_PATH, EMBEDDING_DIM


def _serialize_f32(vector: list[float]) -> bytes:
    """Serialize a list of floats into a compact little-endian binary blob."""
    return struct.pack(f"<{len(vector)}f", *vector)


def get_db() -> sqlite3.Connection:
    """
    Return a new sqlite3 connection with:
      - sqlite-vec extension loaded
      - WAL journal mode for concurrent reads
      - foreign keys enabled
      - Row factory set to sqlite3.Row for dict-like access
    """
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.row_factory = sqlite3.Row
    return conn


def init_database() -> None:
    """
    Create all tables if they do not already exist.
    Safe to call multiple times (uses IF NOT EXISTS).
    """
    conn = get_db()
    cur = conn.cursor()

    # ── students ─────────────────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            roll_number     TEXT    NOT NULL UNIQUE,
            name            TEXT    NOT NULL,
            github_username TEXT,
            resume_filename TEXT,
            resume_text     TEXT,
            ats_score       REAL,
            github_score    REAL,
            final_score     REAL,
            stage           TEXT    NOT NULL DEFAULT 'pending'
                CHECK (stage IN (
                    'pending','parsed','scored',
                    'shortlisted','verified','ranked'
                )),
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # ── vec_resumes (sqlite-vec virtual table) ───────────────────────────────
    cur.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_resumes USING vec0(
            student_id INTEGER PRIMARY KEY,
            embedding  float[{EMBEDDING_DIM}]
        );
    """)

    # ── skills ───────────────────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS skills (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id  INTEGER NOT NULL,
            skill       TEXT    NOT NULL,
            source      TEXT    NOT NULL DEFAULT 'resume',
            FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE
        );
    """)

    # ── projects ─────────────────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id      INTEGER NOT NULL,
            project_name    TEXT    NOT NULL,
            source          TEXT    NOT NULL DEFAULT 'resume',
            github_verified INTEGER NOT NULL DEFAULT 0,
            github_repo_url TEXT,
            is_fork         INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE
        );
    """)

    # ── github_profiles ──────────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS github_profiles (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id          INTEGER NOT NULL UNIQUE,
            active_days         INTEGER NOT NULL DEFAULT 0,
            total_commits       INTEGER NOT NULL DEFAULT 0,
            total_prs           INTEGER NOT NULL DEFAULT 0,
            fork_ratio          REAL    NOT NULL DEFAULT 0.0,
            top_languages       TEXT,
            contribution_score  REAL    NOT NULL DEFAULT 0.0,
            FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE
        );
    """)

    # ── pipeline_runs ────────────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            job_description TEXT    NOT NULL,
            ats_weight      REAL    NOT NULL DEFAULT 0.6,
            github_weight   REAL    NOT NULL DEFAULT 0.4,
            algorithm       TEXT    NOT NULL DEFAULT 'hybrid_efficient',
            total_candidates INTEGER NOT NULL DEFAULT 0,
            shortlisted     INTEGER,
            status          TEXT    NOT NULL DEFAULT 'created'
                CHECK (status IN (
                    'created','running','stage1','stage2','stage3',
                    'completed','failed'
                )),
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at    TIMESTAMP
        );
    """)

    # ── indexes ──────────────────────────────────────────────────────────────
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_students_roll
        ON students(roll_number);
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_students_stage
        ON students(stage);
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_skills_student
        ON skills(student_id);
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_projects_student
        ON projects(student_id);
    """)

    conn.commit()
    conn.close()
