"""
HireFlow Database Models (CRUD)
-------------------------------
Raw-sqlite3 CRUD helpers for every table in the schema.
All functions accept an open sqlite3.Connection as first argument.
"""

import json
import sqlite3
import struct
from typing import Any, Optional

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _row_to_dict(row: Optional[sqlite3.Row]) -> Optional[dict[str, Any]]:
    """Convert a sqlite3.Row to a plain dict, or return None."""
    if row is None:
        return None
    return dict(row)


def _rows_to_list(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
    """Convert a list of sqlite3.Row objects to a list of dicts."""
    return [dict(r) for r in rows]


def _serialize_f32(vector: list[float] | np.ndarray) -> bytes:
    """Pack a float32 vector into bytes for sqlite-vec."""
    if isinstance(vector, np.ndarray):
        vector = vector.astype(np.float32).tolist()
    return struct.pack(f"<{len(vector)}f", *vector)


# ═══════════════════════════════════════════════════════════════════════════════
#  Students
# ═══════════════════════════════════════════════════════════════════════════════

def insert_student(
    db: sqlite3.Connection,
    roll_number: str,
    name: str,
    github_username: Optional[str] = None,
    resume_filename: Optional[str] = None,
) -> int:
    """Insert a new student. Returns the new student id."""
    cur = db.execute(
        """
        INSERT INTO students (roll_number, name, github_username, resume_filename)
        VALUES (?, ?, ?, ?)
        """,
        (roll_number, name, github_username, resume_filename),
    )
    db.commit()
    return cur.lastrowid  # type: ignore[return-value]


def get_student(db: sqlite3.Connection, roll_number: str) -> Optional[dict[str, Any]]:
    """Fetch a single student by roll_number."""
    row = db.execute(
        "SELECT * FROM students WHERE roll_number = ?", (roll_number,)
    ).fetchone()
    return _row_to_dict(row)


def get_student_by_id(db: sqlite3.Connection, student_id: int) -> Optional[dict[str, Any]]:
    """Fetch a single student by id."""
    row = db.execute(
        "SELECT * FROM students WHERE id = ?", (student_id,)
    ).fetchone()
    return _row_to_dict(row)


def get_all_students(
    db: sqlite3.Connection,
    stage: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Return all students, optionally filtered by stage."""
    if stage:
        rows = db.execute(
            "SELECT * FROM students WHERE stage = ? ORDER BY final_score DESC NULLS LAST",
            (stage,),
        ).fetchall()
    else:
        rows = db.execute(
            "SELECT * FROM students ORDER BY final_score DESC NULLS LAST"
        ).fetchall()
    return _rows_to_list(rows)


def update_student_stage(db: sqlite3.Connection, student_id: int, stage: str) -> None:
    """Advance a student to the given pipeline stage."""
    db.execute(
        "UPDATE students SET stage = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (stage, student_id),
    )
    db.commit()


def update_student_resume_text(db: sqlite3.Connection, student_id: int, resume_text: str) -> None:
    """Store the extracted resume text for a student."""
    db.execute(
        "UPDATE students SET resume_text = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (resume_text, student_id),
    )
    db.commit()


def update_student_scores(
    db: sqlite3.Connection,
    student_id: int,
    ats_score: Optional[float] = None,
    github_score: Optional[float] = None,
    final_score: Optional[float] = None,
) -> None:
    """Update one or more score columns for a student."""
    parts: list[str] = []
    params: list[Any] = []
    if ats_score is not None:
        parts.append("ats_score = ?")
        params.append(round(ats_score, 2))
    if github_score is not None:
        parts.append("github_score = ?")
        params.append(round(github_score, 2))
    if final_score is not None:
        parts.append("final_score = ?")
        params.append(round(final_score, 2))
    if not parts:
        return
    parts.append("updated_at = CURRENT_TIMESTAMP")
    params.append(student_id)
    sql = f"UPDATE students SET {', '.join(parts)} WHERE id = ?"
    db.execute(sql, params)
    db.commit()


# ═══════════════════════════════════════════════════════════════════════════════
#  Skills
# ═══════════════════════════════════════════════════════════════════════════════

def insert_skill(
    db: sqlite3.Connection,
    student_id: int,
    skill: str,
    source: str = "resume",
) -> int:
    """Insert a skill for a student. Returns the skill row id."""
    cur = db.execute(
        "INSERT INTO skills (student_id, skill, source) VALUES (?, ?, ?)",
        (student_id, skill, source),
    )
    db.commit()
    return cur.lastrowid  # type: ignore[return-value]


def get_skills(db: sqlite3.Connection, student_id: int) -> list[dict[str, Any]]:
    """Return all skills for a student."""
    rows = db.execute(
        "SELECT * FROM skills WHERE student_id = ?", (student_id,)
    ).fetchall()
    return _rows_to_list(rows)


def delete_skills(db: sqlite3.Connection, student_id: int) -> None:
    """Remove all skills for a student (used before re-extraction)."""
    db.execute("DELETE FROM skills WHERE student_id = ?", (student_id,))
    db.commit()


# ═══════════════════════════════════════════════════════════════════════════════
#  Projects
# ═══════════════════════════════════════════════════════════════════════════════

def insert_project(
    db: sqlite3.Connection,
    student_id: int,
    project_name: str,
    source: str = "resume",
    github_verified: bool = False,
    github_repo_url: Optional[str] = None,
    is_fork: bool = False,
) -> int:
    """Insert a project for a student. Returns the project row id."""
    cur = db.execute(
        """
        INSERT INTO projects
            (student_id, project_name, source, github_verified, github_repo_url, is_fork)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (student_id, project_name, source, int(github_verified), github_repo_url, int(is_fork)),
    )
    db.commit()
    return cur.lastrowid  # type: ignore[return-value]


def get_projects(db: sqlite3.Connection, student_id: int) -> list[dict[str, Any]]:
    """Return all projects for a student."""
    rows = db.execute(
        "SELECT * FROM projects WHERE student_id = ?", (student_id,)
    ).fetchall()
    return _rows_to_list(rows)


def delete_projects(db: sqlite3.Connection, student_id: int) -> None:
    """Remove all projects for a student (used before re-extraction)."""
    db.execute("DELETE FROM projects WHERE student_id = ?", (student_id,))
    db.commit()


def update_project_verification(
    db: sqlite3.Connection,
    project_id: int,
    github_verified: bool,
    github_repo_url: Optional[str],
    is_fork: bool,
) -> None:
    """Mark a project as verified / not verified via GitHub."""
    db.execute(
        """
        UPDATE projects
        SET github_verified = ?, github_repo_url = ?, is_fork = ?
        WHERE id = ?
        """,
        (int(github_verified), github_repo_url, int(is_fork), project_id),
    )
    db.commit()


# ═══════════════════════════════════════════════════════════════════════════════
#  GitHub Profiles
# ═══════════════════════════════════════════════════════════════════════════════

def upsert_github_profile(
    db: sqlite3.Connection,
    student_id: int,
    active_days: int,
    total_commits: int,
    total_prs: int,
    fork_ratio: float,
    top_languages: list[str] | dict[str, Any],
    contribution_score: float,
) -> None:
    """Insert or update the GitHub profile for a student."""
    langs_json = json.dumps(top_languages) if not isinstance(top_languages, str) else top_languages
    db.execute(
        """
        INSERT INTO github_profiles
            (student_id, active_days, total_commits, total_prs,
             fork_ratio, top_languages, contribution_score)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(student_id) DO UPDATE SET
            active_days        = excluded.active_days,
            total_commits      = excluded.total_commits,
            total_prs          = excluded.total_prs,
            fork_ratio         = excluded.fork_ratio,
            top_languages      = excluded.top_languages,
            contribution_score = excluded.contribution_score
        """,
        (student_id, active_days, total_commits, total_prs,
         round(fork_ratio, 4), langs_json, round(contribution_score, 2)),
    )
    db.commit()


def get_github_profile(db: sqlite3.Connection, student_id: int) -> Optional[dict[str, Any]]:
    """Return the GitHub profile for a student, parsing top_languages from JSON."""
    row = db.execute(
        "SELECT * FROM github_profiles WHERE student_id = ?", (student_id,)
    ).fetchone()
    if row is None:
        return None
    d = dict(row)
    if d.get("top_languages"):
        try:
            d["top_languages"] = json.loads(d["top_languages"])
        except (json.JSONDecodeError, TypeError):
            d["top_languages"] = []
    return d


# ═══════════════════════════════════════════════════════════════════════════════
#  Pipeline Runs
# ═══════════════════════════════════════════════════════════════════════════════

def create_pipeline_run(
    db: sqlite3.Connection,
    job_description: str,
    ats_weight: float,
    github_weight: float,
    algorithm: str,
    total_candidates: int,
) -> int:
    """Create a new pipeline run record. Returns the run id."""
    cur = db.execute(
        """
        INSERT INTO pipeline_runs
            (job_description, ats_weight, github_weight, algorithm, total_candidates, status)
        VALUES (?, ?, ?, ?, ?, 'created')
        """,
        (job_description, ats_weight, github_weight, algorithm, total_candidates),
    )
    db.commit()
    return cur.lastrowid  # type: ignore[return-value]


def update_pipeline_run(
    db: sqlite3.Connection,
    run_id: int,
    status: str,
    shortlisted: Optional[int] = None,
    completed_at: Optional[str] = None,
) -> None:
    """Update the status (and optionally shortlisted count / completion time) of a run."""
    parts: list[str] = ["status = ?"]
    params: list[Any] = [status]
    if shortlisted is not None:
        parts.append("shortlisted = ?")
        params.append(shortlisted)
    if completed_at is not None:
        parts.append("completed_at = ?")
        params.append(completed_at)
    params.append(run_id)
    sql = f"UPDATE pipeline_runs SET {', '.join(parts)} WHERE id = ?"
    db.execute(sql, params)
    db.commit()


def get_pipeline_run(db: sqlite3.Connection, run_id: int) -> Optional[dict[str, Any]]:
    """Fetch a single pipeline run."""
    row = db.execute(
        "SELECT * FROM pipeline_runs WHERE id = ?", (run_id,)
    ).fetchone()
    return _row_to_dict(row)


def get_latest_pipeline_run(db: sqlite3.Connection) -> Optional[dict[str, Any]]:
    """Return the most recent pipeline run."""
    row = db.execute(
        "SELECT * FROM pipeline_runs ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return _row_to_dict(row)


# ═══════════════════════════════════════════════════════════════════════════════
#  Embeddings (sqlite-vec)
# ═══════════════════════════════════════════════════════════════════════════════

def store_embedding(
    db: sqlite3.Connection,
    student_id: int,
    embedding_vector: list[float] | np.ndarray,
) -> None:
    """Store (or replace) the resume embedding for a student in the vec table."""
    blob = _serialize_f32(
        embedding_vector.tolist() if isinstance(embedding_vector, np.ndarray) else embedding_vector
    )
    # Delete existing row first (vec0 doesn't support ON CONFLICT)
    db.execute("DELETE FROM vec_resumes WHERE student_id = ?", (student_id,))
    db.execute(
        "INSERT INTO vec_resumes (student_id, embedding) VALUES (?, ?)",
        (student_id, blob),
    )
    db.commit()


def search_similar(
    db: sqlite3.Connection,
    query_vector: list[float] | np.ndarray,
    limit: int = 10,
) -> list[tuple[int, float]]:
    """
    Find the closest resume embeddings to the query vector.
    Returns list of (student_id, distance) ordered by distance ascending.
    """
    blob = _serialize_f32(
        query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector
    )
    rows = db.execute(
        """
        SELECT student_id, distance
        FROM vec_resumes
        WHERE embedding MATCH ?
        ORDER BY distance
        LIMIT ?
        """,
        (blob, limit),
    ).fetchall()
    return [(row[0], row[1]) for row in rows]
