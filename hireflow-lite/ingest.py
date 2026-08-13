"""
ingest.py — HireFlow-Lite: Phase 1 — Ingestion Pipeline
=========================================================
Reads a CSV (roll_number, name, github_url, github_token),
matches each row to a resume file in the resumes folder by roll_number,
parses the resume, generates a 384-dim MiniLM embedding, and stores
everything into the SQLite database.

Resume filenames MUST start with the roll_number:
    resumes/23CS001.pdf
    resumes/23CS002.docx

Usage (CLI):
    python ingest.py --csv responses.csv --resumes ./resumes --db hireflow.db

Usage (API):
    from ingest import run_ingestion
    stats = run_ingestion(csv_path, resumes_dir, db_path, progress_cb)
"""

import csv
import sys
import time
import json
import warnings
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

import torch
from sentence_transformers import SentenceTransformer

from database import init_db, reset_db, upsert_student, upsert_fts, upsert_embedding, DB_PATH, EMBED_DIM

# Force UTF-8
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


# ── GPU Setup ─────────────────────────────────────────────────────────────────

def _setup_device() -> str:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        name = torch.cuda.get_device_name(0)
        vram = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
        print(f"  ✓ GPU: {name} ({vram} GB) — CUDA ENABLED")
        return "cuda"
    print("  → CPU mode (no CUDA GPU found)")
    return "cpu"


DEVICE = _setup_device()

# Shared SBERT model (loaded once, reused for all embeddings)
_MODEL: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        print("  → Loading all-MiniLM-L6-v2 embedding model...")
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
        print("  ✓ Model loaded.")
    return _MODEL


# ── CSV Column Detection ──────────────────────────────────────────────────────

_COL_ALIASES = {
    "roll_number": ["roll_number", "rollnumber", "roll number", "roll no", "roll", "reg_no",
                    "student_id", "id", "enrollment", "registration number"],
    "name":        ["name", "student name", "full name", "candidate name"],
    "github_url":  ["github_url", "github_link", "github", "github profile",
                    "github profile url", "github profile link"],
    "github_token":["github_token", "github_key", "github personal access token",
                    "personal access token", "token", "pat", "github pat", "access token",
                    "github token", "github key to access repo"],
}


def _detect_columns(header: list[str]) -> dict[str, int]:
    """Map internal field names to column indices. Returns {} for missing cols."""
    header_lower = [h.strip().lower() for h in header]
    mapping = {}
    for field, aliases in _COL_ALIASES.items():
        for alias in aliases:
            for idx, col in enumerate(header_lower):
                if alias == col or alias in col:
                    if field not in mapping:
                        mapping[field] = idx
                        break
            if field in mapping:
                break
    # roll_number is critical
    if "roll_number" not in mapping:
        # Positional fallback for simple forms
        if len(header) >= 4:
            mapping = {"roll_number": 0, "name": 1, "github_url": 2, "github_token": 3}
    return mapping


# ── File Resolver ─────────────────────────────────────────────────────────────

def _build_file_index(resumes_dir: Path) -> dict[str, Path]:
    """
    Build a mapping of filename_stem_lower → Path.
    Resume must be named like: 23CS001.pdf or 23CS001_gopi.pdf
    (stem starts with roll_number).
    """
    index: dict[str, Path] = {}
    for ext in ("pdf", "docx", "doc"):
        for fp in resumes_dir.glob(f"**/*.{ext}"):
            if fp.name.startswith("~$"):
                continue
            # Key: stem lowercase (e.g. "23cs001")
            stem = fp.stem.lower()
            index[stem] = fp
    return index


def _resolve_resume(roll_number: str, file_index: dict[str, Path]) -> Optional[Path]:
    """Find a resume file whose stem starts with the roll_number."""
    roll_lower = roll_number.lower().strip()
    # Exact stem match
    if roll_lower in file_index:
        return file_index[roll_lower]
    # Stem starts-with match (e.g. "23cs001_gopi" starts with "23cs001")
    for stem, fp in file_index.items():
        if stem.startswith(roll_lower):
            return fp
    return None


# ── Parse Worker ──────────────────────────────────────────────────────────────

def _parse_resume_safe(student: dict) -> dict:
    """Parse a single resume. Returns student dict enriched with parsed fields."""
    try:
        from resume_parser import parse_resume
        parsed = parse_resume(student["resume_path"])
        student["raw_text"]  = parsed.get("raw_text", "")
        student["email"]     = parsed.get("email", "")
        student["phone"]     = parsed.get("phone", "")
        student["education"] = parsed.get("education", "")
        student["skills"]    = parsed.get("skills", [])
        student["projects"]  = parsed.get("projects", [])
        # If no name in CSV, use parsed name
        if not student.get("name") or student["name"].startswith("Student_"):
            student["name"] = parsed.get("name") or student["name"]
        # If no github in CSV, try from resume
        if not student.get("github_url"):
            gh = parsed.get("github_username", "")
            if gh:
                student["github_url"] = f"https://github.com/{gh}"
        student["_status"] = "ok"
    except Exception as exc:
        student["_status"] = "failed"
        student["_error"]  = str(exc)[:300]
        student["raw_text"] = ""
        student["skills"]   = []
        student["projects"] = []
        student["email"]    = ""
        student["phone"]    = ""
        student["education"] = ""
    return student


# ── Embedding Generation ──────────────────────────────────────────────────────

def _build_embed_text(student: dict) -> str:
    """Combine raw_text + skills into a single string for embedding."""
    skills_str = " ".join(student.get("skills", []))
    projects_str = " ".join(
        p.get("name", "") + " " + p.get("description", "")
        for p in student.get("projects", [])
        if isinstance(p, dict)
    )
    raw = student.get("raw_text", "")
    # Truncate raw_text to 2000 chars to keep embeddings focused
    return f"{raw[:2000]} {skills_str} {projects_str}".strip()


# ── Main Ingestion Pipeline ───────────────────────────────────────────────────

def run_ingestion(
    csv_path: str,
    resumes_dir: str,
    db_path: str | Path = DB_PATH,
    workers: int = 4,
    reset: bool = False,
    progress_cb: Optional[Callable[[str, int, int], None]] = None,
) -> dict:
    """
    Full ingestion pipeline:
      1. Read CSV → build student list
      2. Resolve resume files by roll_number
      3. Parse resumes in parallel (ThreadPoolExecutor)
      4. Batch generate MiniLM embeddings on GPU
      5. Commit all to SQLite

    Returns stats dict.
    """
    t0 = time.time()
    csv_file    = Path(csv_path)
    resume_path = Path(resumes_dir)

    if not csv_file.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not resume_path.exists():
        raise FileNotFoundError(f"Resumes directory not found: {resumes_dir}")

    # ── Step 1: Read CSV ──────────────────────────────────────────────────────
    print("\n  ╔══════════════════════════════════════════╗")
    print(  "  ║  Phase 1: Reading CSV                    ║")
    print(  "  ╚══════════════════════════════════════════╝")

    with open(csv_file, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows   = [r for r in reader if any(c.strip() for c in r)]

    col = _detect_columns(header)
    print(f"  → {len(rows)} student rows, columns: {col}")

    def _get(row, field):
        idx = col.get(field)
        return row[idx].strip() if idx is not None and idx < len(row) else ""

    # ── Step 2: Resolve resume files ──────────────────────────────────────────
    print("\n  ╔══════════════════════════════════════════╗")
    print(  "  ║  Phase 2: Resolving Resume Files         ║")
    print(  "  ╚══════════════════════════════════════════╝")

    file_index = _build_file_index(resume_path)
    print(f"  → {len(file_index)} resume files indexed in {resumes_dir}")

    students = []
    no_file  = []

    for i, row in enumerate(rows):
        roll = _get(row, "roll_number")
        if not roll:
            continue
        resume_fp = _resolve_resume(roll, file_index)
        if resume_fp is None:
            no_file.append(roll)
            if len(no_file) <= 5:
                print(f"  ⚠ No resume found for roll: {roll}")
            continue
        students.append({
            "roll_number":    roll,
            "name":           _get(row, "name") or f"Student_{roll}",
            "github_url":     _get(row, "github_url"),
            "github_token":   _get(row, "github_token"),
            "resume_path":    str(resume_fp),
            "resume_filename": resume_fp.name,
        })

    if len(no_file) > 5:
        print(f"  ⚠ ... and {len(no_file) - 5} more roll numbers with no resume file.")
    print(f"  ✓ Matched {len(students)} students to resume files ({len(no_file)} unmatched)")

    if not students:
        return {"error": "No students matched to resume files.", "total": 0}

    # ── Step 3: Parse resumes in parallel ─────────────────────────────────────
    print("\n  ╔══════════════════════════════════════════╗")
    print(  "  ║  Phase 3: Parsing Resumes (parallel)     ║")
    print(  "  ╚══════════════════════════════════════════╝")

    parsed_students = []
    failed = 0
    done   = 0
    total  = len(students)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_parse_resume_safe, s): s for s in students}
        for future in as_completed(futures):
            result = future.result()
            parsed_students.append(result)
            done += 1
            if result["_status"] == "failed":
                failed += 1
            if progress_cb:
                progress_cb("parse", done, total)
            if done % 50 == 0 or done == total:
                print(f"  → Parsed {done}/{total}  ({failed} failed)")

    valid = [s for s in parsed_students if s.get("raw_text")]
    print(f"  ✓ {len(valid)} valid resumes parsed, {failed} failed")

    # ── Step 4: Batch embedding on GPU ────────────────────────────────────────
    print("\n  ╔══════════════════════════════════════════╗")
    print(  "  ║  Phase 4: Generating Embeddings (GPU)    ║")
    print(  "  ╚══════════════════════════════════════════╝")

    model = _get_model()
    embed_texts = [_build_embed_text(s) for s in valid]

    print(f"  → Embedding {len(embed_texts)} documents on {DEVICE}...")
    t_embed = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        embeddings = model.encode(
            embed_texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            device=DEVICE,
            normalize_embeddings=True,   # cosine via dot product
        )
    embed_time = time.time() - t_embed
    print(f"  ✓ {len(embeddings)} embeddings in {embed_time:.1f}s  ({embed_time/max(1,len(embeddings))*1000:.1f}ms each)")

    if progress_cb:
        progress_cb("embed", len(embeddings), len(embeddings))

    # ── Step 5: Write to DB ───────────────────────────────────────────────────
    print("\n  ╔══════════════════════════════════════════╗")
    print(  "  ║  Phase 5: Writing to Database            ║")
    print(  "  ╚══════════════════════════════════════════╝")

    conn = reset_db(db_path) if reset else init_db(db_path)

    written = 0
    for s, emb in zip(valid, embeddings):
        upsert_student(conn, s)
        fts_content = f"{s['name']} {s['raw_text'][:3000]} {' '.join(s.get('skills',[]))}"
        upsert_fts(conn, s["roll_number"], fts_content)
        upsert_embedding(conn, s["roll_number"], emb.tolist())
        written += 1

    conn.commit()

    # Also write failed students (no embedding) with empty raw_text
    for s in parsed_students:
        if s["_status"] == "failed":
            upsert_student(conn, s)
    conn.commit()
    conn.close()

    elapsed = time.time() - t0
    stats = {
        "total_in_csv":   len(rows),
        "matched":        len(students),
        "parsed_ok":      len(valid),
        "parse_failed":   failed,
        "no_file":        len(no_file),
        "written_to_db":  written,
        "embed_time_s":   round(embed_time, 2),
        "total_time_s":   round(elapsed, 2),
    }

    print(f"\n  ✓ Ingestion complete in {elapsed:.1f}s")
    print(f"    {written} students indexed in DB  |  {failed} failed  |  {len(no_file)} no file")
    return stats


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="HireFlow-Lite: Ingest students from CSV + resumes")
    p.add_argument("--csv",      required=True, help="Google Forms CSV file")
    p.add_argument("--resumes",  required=True, help="Folder containing PDF/DOCX files named by roll number")
    p.add_argument("--db",       default=str(DB_PATH), help="SQLite database path")
    p.add_argument("--workers",  type=int, default=4, help="Parallel parse workers")
    p.add_argument("--reset",    action="store_true", help="Wipe DB before ingesting")
    args = p.parse_args()

    stats = run_ingestion(
        csv_path=args.csv,
        resumes_dir=args.resumes,
        db_path=args.db,
        workers=args.workers,
        reset=args.reset,
    )
    print("\n  Final Stats:", json.dumps(stats, indent=4))


if __name__ == "__main__":
    main()
