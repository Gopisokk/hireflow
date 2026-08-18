"""
ingest.py — HireFlow-Lite: Phase 1 — Ingestion Pipeline
=========================================================
Sequential, GPU-maximized ingestion pipeline.

Ollama runs on GPU and only handles one request at a time.
ThreadPoolExecutor with Ollama causes deadlock/freezing.
So: parse each resume sequentially (one at a time), then batch
embed all at once on GPU with max batch size for maximum VRAM use.

Key fixes:
  - 95s hard timeout per resume (outer) + 75s Ollama-level timeout (inner)
  - Input text truncated BEFORE sending to Ollama (MAX_CHARS=3500)
  - Heartbeat every 8s so UI shows elapsed time during LLM call
  - On timeout/failure: saves raw document text for embedding (PARTIAL status)
  - Name-based file matching for roll-number-less filenames
"""

import csv
import sys
import os
import time
import json
import warnings
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from pathlib import Path
from typing import Callable, Optional

# Environment flags
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN_WARNING"] = "1"
warnings.filterwarnings("ignore")

import torch

from database import init_db, reset_db, upsert_student, upsert_fts, upsert_embedding, DB_PATH, EMBED_DIM

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
        torch.backends.cudnn.benchmark        = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
        torch.backends.cudnn.deterministic    = False
        name = torch.cuda.get_device_name(0)
        vram = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
        free = round(torch.cuda.mem_get_info(0)[0] / 1024**3, 1)
        print(f"  GPU: {name} ({vram} GB total, {free} GB free) — CUDA ENABLED")
        return "cuda"
    print("  CPU mode (no CUDA GPU found)")
    return "cpu"


DEVICE = _setup_device()

_SBERT_MODEL = None


def _get_sbert_model():
    from minilm import get_minilm_model
    return get_minilm_model(device="cpu")


# ── CSV Column Detection ──────────────────────────────────────────────────────

_COL_ALIASES = {
    "roll_number": [
        "roll_number", "rollnumber", "roll number", "roll no", "roll", "reg_no",
        "student_id", "id", "enrollment", "registration number", "regno",
    ],
    "name":        ["name", "student name", "full name", "candidate name"],
    "github_url":  [
        "github_url", "github_link", "github", "github profile",
        "github profile url", "github profile link",
    ],
    "github_token": [
        "github_token", "github_key", "github personal access token",
        "personal access token", "token", "pat", "github pat", "access token",
        "github token", "github key to access repo",
    ],
}


def _detect_columns(header: list[str]) -> dict[str, int]:
    header_lower = [h.strip().lower() for h in header]
    mapping: dict[str, int] = {}
    for field, aliases in _COL_ALIASES.items():
        for alias in aliases:
            for idx, col in enumerate(header_lower):
                if alias == col or alias in col:
                    if field not in mapping:
                        mapping[field] = idx
                        break
            if field in mapping:
                break
    if "roll_number" not in mapping and len(header) >= 1:
        mapping["roll_number"] = 0
    if "name" not in mapping and len(header) >= 2:
        mapping["name"] = 1
    return mapping


# ── File Resolver ─────────────────────────────────────────────────────────────

def _build_file_index(resumes_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for ext in ("pdf", "docx", "doc"):
        for fp in resumes_dir.glob(f"**/*.{ext}"):
            if fp.name.startswith("~$"):
                continue
            index[fp.stem.lower().strip()] = fp
    return index


def _resolve_resume(roll_number: str, file_index: dict[str, Path], name: str = "") -> Optional[Path]:
    """
    Match a roll_number to a resume file using 4 strategies:
      1. Exact stem match
      2. File stem starts with roll  (e.g. '23ad044 - gopi')
      3. Roll starts with file stem  (reversed short stems)
      4. CSV name words appear in file stem (for name-only files)
    """
    roll = roll_number.lower().strip()

    if roll in file_index:
        return file_index[roll]

    for stem, fp in file_index.items():
        if stem.startswith(roll):
            return fp

    for stem, fp in file_index.items():
        if len(stem) >= 5 and roll.startswith(stem):
            return fp

    if name:
        parts = name.lower().strip().split()
        for stem, fp in file_index.items():
            if parts and all(p in stem for p in parts[:2]):
                return fp

    return None


# ── Single Resume Parser ──────────────────────────────────────────────────────

HARD_TIMEOUT_S = 95   # outer timeout per resume (> qwen_parser's inner 75s)


def _parse_one_resume(student: dict, model_name: str, log_fn) -> dict:
    """
    Parse one resume via Qwen2.5 (Ollama) with:
      - Heartbeat every 8s so UI shows progress
      - Hard 95s outer timeout so ingestion never stalls
      - Fallback: saves raw document text on LLM failure for embedding
    """
    path = student["resume_path"]
    roll = student["roll_number"]

    # Heartbeat thread
    _stop  = threading.Event()
    _t0    = time.time()

    def _beat():
        while not _stop.wait(8.0):
            log_fn(f"      ... Ollama processing ({int(time.time()-_t0)}s elapsed)", "info")

    hb = threading.Thread(target=_beat, daemon=True)
    hb.start()

    # Outer timeout wrapper
    parsed = None
    try:
        def _do_parse():
            from qwen_resume_parser import parse_resume as qp
            return qp(path, model_name=model_name)

        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_do_parse)
            try:
                parsed = fut.result(timeout=HARD_TIMEOUT_S)
            except FutureTimeout:
                log_fn(f"      [{roll}] Hard timeout ({HARD_TIMEOUT_S}s) — using raw text fallback", "err")
    except Exception as exc:
        log_fn(f"      [{roll}] Executor error: {str(exc)[:100]}", "err")
    finally:
        _stop.set()

    elapsed = round(time.time() - _t0, 1)

    # ── Extract fields from parsed JSON ──────────────────────────────────
    try:
        if parsed is None:
            raise RuntimeError("LLM parse returned None")

        candidate = parsed.get("candidate", {}) or {}
        student["email"] = candidate.get("email", "") or ""
        student["phone"] = candidate.get("phone", "") or ""

        pname = candidate.get("candidate_name", "")
        if (not student.get("name") or student["name"].startswith("Student_")) and pname:
            student["name"] = pname

        if not student.get("github_url"):
            gh = candidate.get("github_url", "")
            if gh:
                student["github_url"] = gh

        edu_list = parsed.get("education", []) or []
        student["education"] = " | ".join(
            f"{e.get('institution','')} {e.get('degree','')} {e.get('field','')} {e.get('cgpa','')}".strip()
            for e in edu_list if isinstance(e, dict)
        )

        skills_dict = parsed.get("skills", {}) or {}
        all_skills: list[str] = []
        if isinstance(skills_dict, dict):
            for cat in skills_dict.values():
                if isinstance(cat, list):
                    all_skills.extend(s for s in cat if s)
        else:
            all_skills = skills_dict if isinstance(skills_dict, list) else []
        student["skills"] = all_skills

        proj_raw = parsed.get("projects", []) or []
        student["projects"] = [
            {
                "name":         p.get("project_name", ""),
                "description":  p.get("description", ""),
                "technologies": p.get("technologies", []),
                "url":          p.get("url", ""),
            }
            for p in proj_raw if isinstance(p, dict) and p.get("project_name")
        ]

        exp_list = parsed.get("experience", []) or []
        exp_txt  = " ".join(
            f"{e.get('organization','')} {e.get('role','')} {' '.join(e.get('description',[]))}"
            for e in exp_list if isinstance(e, dict)
        )
        prj_txt  = " ".join(
            f"{p.get('project_name','')} {p.get('description','')} {' '.join(p.get('technologies',[]))}"
            for p in proj_raw if isinstance(p, dict)
        )
        student["raw_text"] = (
            f"{student['education']} {exp_txt} {prj_txt} {' '.join(student['skills'])}"
        ).strip()

        # Fallback if LLM returned empty content
        if len(student["raw_text"]) < 150:
            from document_extractor import extract_plain_text
            student["raw_text"] = extract_plain_text(path)[:3000]

        student["_status"] = "ok"
        log_fn(
            f"    [{roll}] DONE in {elapsed}s — "
            f"{len(student['skills'])} skills, {len(student['projects'])} projects",
            "ok"
        )

    except Exception as exc:
        # Graceful degradation: save raw text so this student still gets an embedding
        student["_status"]   = "partial"
        student["_error"]    = str(exc)[:300]
        student["skills"]    = []
        student["projects"]  = []
        student["email"]     = ""
        student["phone"]     = ""
        student["education"] = ""
        try:
            from document_extractor import extract_plain_text
            student["raw_text"] = extract_plain_text(path)[:3000]
            log_fn(
                f"    [{roll}] PARTIAL in {elapsed}s — LLM failed, raw text saved for embedding",
                "err"
            )
        except Exception:
            student["raw_text"] = ""
            log_fn(f"    [{roll}] FAILED in {elapsed}s — no text extracted", "err")

    return student


# ── Embedding text builder ────────────────────────────────────────────────────

def _build_embed_text(student: dict) -> str:
    skills = " ".join(str(s) for s in student.get("skills", []) if s)
    projs  = " ".join(
        f"{p.get('name','')} {p.get('description','')} {' '.join(p.get('technologies',[]))}"
        for p in student.get("projects", []) if isinstance(p, dict)
    )
    raw = student.get("raw_text", "")
    return f"{raw[:2500]} {skills} {projs}".strip()


# ── Main Ingestion Pipeline ───────────────────────────────────────────────────

def run_ingestion(
    csv_path:    str,
    resumes_dir: str,
    db_path:     str | Path  = DB_PATH,
    workers:     int         = 1,       # ignored — always sequential for GPU Ollama
    reset:       bool        = False,
    progress_cb: Optional[Callable[[str, int, int], None]] = None,
    log_cb:      Optional[Callable[[str, str], None]]      = None,
    model_name:  str         = "qwen2.5:1.5b-instruct",
) -> dict:
    """Sequential GPU-maximized ingestion pipeline."""

    def _log(msg: str, level: str = "info"):
        print(msg)
        if log_cb:
            try:
                log_cb(msg.strip(), level)
            except Exception:
                pass

    t0          = time.time()
    csv_file    = Path(csv_path)
    resume_path = Path(resumes_dir)

    if not csv_file.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not resume_path.exists():
        raise FileNotFoundError(f"Resumes folder not found: {resumes_dir}")

    # ── Phase 1: Read CSV ─────────────────────────────────────────────────
    _log("━━━ Phase 1: Reading CSV ━━━")
    with open(csv_file, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows   = [r for r in reader if any(c.strip() for c in r)]

    col = _detect_columns(header)
    _log(f"  {len(rows)} student rows found. Column map: {col}")

    def _get(row, field):
        idx = col.get(field)
        return row[idx].strip() if idx is not None and idx < len(row) else ""

    # ── Phase 2: Match Resume Files ───────────────────────────────────────
    _log("━━━ Phase 2: Matching Resume Files ━━━")
    file_index = _build_file_index(resume_path)
    _log(f"  {len(file_index)} resume files found in ZIP")
    if file_index:
        _log(f"  Sample files: {list(file_index.keys())[:8]}")

    students: list[dict] = []
    no_file:  list[str]  = []

    for row in rows:
        roll = _get(row, "roll_number")
        if not roll:
            continue
        name      = _get(row, "name") or ""
        resume_fp = _resolve_resume(roll, file_index, name)
        if resume_fp is None:
            no_file.append(roll)
            _log(f"  NO FILE for roll: {roll} (name: {name})", "err")
            continue
        students.append({
            "roll_number":     roll,
            "name":            name or f"Student_{roll}",
            "github_url":      _get(row, "github_url"),
            "github_token":    _get(row, "github_token"),
            "resume_path":     str(resume_fp),
            "resume_filename": resume_fp.name,
        })

    _log(
        f"  Matched: {len(students)} students | No file: {len(no_file)}",
        "ok" if students else "err"
    )

    if not students:
        _log("ERROR: No students could be matched to resume files!", "err")
        _log(f"  CSV rolls:   {[_get(r,'roll_number') for r in rows[:6]]}", "err")
        _log(f"  ZIP stems:   {list(file_index.keys())[:10]}", "err")
        return {
            "error":     "No students matched to resume files.",
            "total":     0,
            "csv_rolls": [_get(r, "roll_number") for r in rows[:10]],
            "zip_files": list(file_index.keys())[:10],
        }

    # ── Phase 3: Sequential Ollama Parsing ───────────────────────────────
    _log(f"━━━ Phase 3: Parsing {len(students)} Resumes via {model_name} (GPU/Ollama) ━━━")
    _log("  Sequential mode — Ollama uses full GPU, one resume at a time")
    _log(f"  Each resume: up to {HARD_TIMEOUT_S}s max (LLM timeout: 75s)")

    parsed_students: list[dict] = []
    failed = 0
    total  = len(students)

    for i, student in enumerate(students):
        idx    = i + 1
        roll   = student["roll_number"]
        name   = student["name"]
        fname  = student["resume_filename"]
        _log(f"  [{idx}/{total}] Parsing: {roll} — {name} ({fname})")

        result = _parse_one_resume(student, model_name, _log)
        parsed_students.append(result)

        if result["_status"] in ("failed",):
            failed += 1

        if progress_cb:
            progress_cb("parse", idx, total)

    valid = [s for s in parsed_students if s.get("raw_text")]
    _log(f"  Done: {len(valid)} with text, {failed} fully failed", "ok")

    # ── Phase 4: Batch Embedding ──────────────────────────────────────────
    _log("━━━ Phase 4: Batch Embedding (MiniLM-L6-v2) ━━━")
    model       = _get_sbert_model()
    embed_texts = [_build_embed_text(s) for s in valid]
    _log(f"  Embedding {len(embed_texts)} documents...")

    t_emb = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        embeddings = model.encode(
            embed_texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            device="cpu",
            normalize_embeddings=True,
        )

    emb_time = round(time.time() - t_emb, 2)
    _log(f"  {len(embeddings)} embeddings in {emb_time}s", "ok")

    if progress_cb:
        progress_cb("embed", len(embeddings), len(embeddings))

    # ── Phase 5: Write to DB ──────────────────────────────────────────────
    _log("━━━ Phase 5: Writing to Database ━━━")
    conn    = reset_db(db_path) if reset else init_db(db_path)
    written = 0

    for s, emb in zip(valid, embeddings):
        upsert_student(conn, s)
        fts = f"{s['name']} {s['raw_text'][:3000]} {' '.join(s.get('skills', []))}"
        upsert_fts(conn, s["roll_number"], fts)
        upsert_embedding(conn, s["roll_number"], emb.tolist())
        written += 1

    conn.commit()
    for s in parsed_students:
        if s.get("_status") in ("failed",) and not s.get("raw_text"):
            upsert_student(conn, s)
    conn.commit()
    conn.close()

    elapsed = round(time.time() - t0, 2)
    stats = {
        "total_in_csv":  len(rows),
        "matched":       len(students),
        "parsed_ok":     len(valid),
        "parse_failed":  failed,
        "no_file":       len(no_file),
        "written_to_db": written,
        "embed_time_s":  emb_time,
        "total_time_s":  elapsed,
        "model":         model_name,
        "device":        DEVICE,
    }

    _log(f"Ingestion complete in {elapsed}s — {written} students indexed", "ok")
    return stats


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",     required=True)
    p.add_argument("--resumes", required=True)
    p.add_argument("--db",      default=str(DB_PATH))
    p.add_argument("--reset",   action="store_true")
    p.add_argument("--model",   default="qwen2.5:1.5b-instruct")
    args = p.parse_args()
    stats = run_ingestion(
        csv_path=args.csv, resumes_dir=args.resumes,
        db_path=args.db, reset=args.reset, model_name=args.model,
    )
    print("\nStats:", json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
