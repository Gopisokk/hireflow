"""
server.py — HireFlow-Lite ATS: FastAPI Backend
===============================================
Architecture:
  PDF/DOCX → document_extractor → plain text
             → qwen_resume_parser (Ollama Qwen2.5-1.5B, GPU)
             → qwen_validator (deterministic rules)
             → VERIFIED canonical JSON
             → ATS scoring + GitHub verification

Endpoints:
  POST /api/ingest          Phase 1: CSV + resumes zip → index DB
  POST /api/search          Phase 2: JD → search + rank candidates
  POST /api/quick-score     Score single resume + optional GitHub verify
  POST /api/github-verify   Verify a GitHub profile (standalone)
  GET  /api/jobs/{id}/stream SSE real-time progress
  GET  /api/jobs/{id}/result Poll job result
  GET  /api/health          GPU/DB/Ollama status
  GET  /api/students        All indexed students
  GET  /api/students/{roll} Single student detail
  DELETE /api/reset         Wipe DB
"""

import os
import sys
import re
import warnings

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN_WARNING"] = "1"
warnings.filterwarnings("ignore")

import json
import time
import uuid
import queue
import zipfile
import tempfile
import threading
import traceback
from pathlib import Path
from datetime import datetime
from typing import Optional

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

try:
    from fastapi import FastAPI, UploadFile, File, Form, HTTPException
    from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.middleware.cors import CORSMiddleware
except ImportError as e:
    print(f"ImportError: {e}")
    print("Run: venv\\Scripts\\pip install fastapi uvicorn python-multipart")
    sys.exit(1)

from database import DB_PATH, init_db, get_db_stats, get_all_students, get_student, reset_db

# ── GPU Info ──────────────────────────────────────────────────────────────────
DEVICE = "cpu"
GPU_INFO = "No GPU"
try:
    import torch
    if torch.cuda.is_available():
        DEVICE = "cuda"
        n = torch.cuda.get_device_name(0)
        v = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
        GPU_INFO = f"{n} ({v} GB)"
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="HireFlow-Lite ATS", version="4.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

init_db(DB_PATH)

# ── Job Store ─────────────────────────────────────────────────────────────────
JOBS: dict[str, dict] = {}
JOB_QUEUES: dict[str, queue.Queue] = {}


def _emit(job_id: str, etype: str, data: dict):
    payload = json.dumps({"type": etype, "data": data, "ts": datetime.now().isoformat()}, default=str)
    if job_id in JOB_QUEUES:
        JOB_QUEUES[job_id].put(payload)
    if job_id in JOBS:
        JOBS[job_id].setdefault("logs", []).append({"type": etype, **data})


# ── SSE Stream ────────────────────────────────────────────────────────────────
@app.get("/api/jobs/{job_id}/stream")
async def stream_job(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "Job not found")

    def gen():
        q = JOB_QUEUES.get(job_id)
        if not q:
            yield f"data: {json.dumps({'type':'error','data':{'message':'no stream'}})}\n\n"
            return
        while True:
            try:
                msg = q.get(timeout=30)
            except queue.Empty:
                yield "data: {\"type\":\"heartbeat\"}\n\n"
                continue
            if msg == "__DONE__":
                yield f"data: {json.dumps({'type':'stream_end'})}\n\n"
                break
            yield f"data: {msg}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    try:
        import database as _db
        conn = _db.connect(DB_PATH)
        stats = get_db_stats(conn)
        conn.close()
    except Exception:
        stats = {}

    # Check Ollama status
    ollama_ok = False
    ollama_model = "qwen2.5:1.5b-instruct"
    try:
        import ollama
        models = ollama.list()
        pulled = [m.get("name", m.get("model", "")) for m in (models.get("models") or [])]
        ollama_ok = len(pulled) > 0
        # Prefer qwen2.5:1.5b-instruct, else any available
        for m in pulled:
            if "1.5b" in m.lower() and "qwen" in m.lower():
                ollama_model = m
                break
    except Exception:
        pass

    return {
        "status": "ok",
        "device": DEVICE,
        "gpu": GPU_INFO,
        "db": str(DB_PATH),
        "db_stats": stats,
        "ollama_ok": ollama_ok,
        "ollama_model": ollama_model,
        "jobs_running": sum(1 for j in JOBS.values() if j.get("status") == "running"),
    }


# ── PHASE 1: INGEST ───────────────────────────────────────────────────────────
def _run_ingest_bg(job_id: str, csv_path: str, resumes_dir: str, reset: bool):
    JOBS[job_id]["status"] = "running"
    try:
        from ingest import run_ingestion

        def cb(phase: str, done: int, total: int):
            _emit(job_id, "progress", {
                "phase": phase, "done": done, "total": total,
                "pct": int(done / max(1, total) * 100)
            })

        def log_cb(msg: str, level: str = "info"):
            _emit(job_id, "log", {"level": level, "msg": msg})

        _emit(job_id, "log", {"level": "info", "msg": f"Ingestion started (sequential GPU mode). Reset={reset}"})

        stats = run_ingestion(
            csv_path=csv_path,
            resumes_dir=resumes_dir,
            db_path=DB_PATH,
            reset=reset,
            progress_cb=cb,
            log_cb=log_cb,
        )
        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["result"] = stats
        _emit(job_id, "done", stats)
    except Exception as exc:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(exc)
        _emit(job_id, "error", {"message": str(exc), "traceback": traceback.format_exc()[:1200]})
    finally:
        if job_id in JOB_QUEUES:
            JOB_QUEUES[job_id].put("__DONE__")


@app.post("/api/ingest")
async def ingest(
    csv_file: UploadFile = File(...),
    resumes:  UploadFile = File(...),
    reset:    bool       = Form(default=False),
):
    """Upload CSV + resumes ZIP → parse all resumes sequentially on GPU → index into DB."""
    job_id  = str(uuid.uuid4())[:8]
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV
    csv_path = job_dir / csv_file.filename
    csv_path.write_bytes(await csv_file.read())

    # Save & extract ZIP
    zip_path = job_dir / resumes.filename
    zip_path.write_bytes(await resumes.read())

    resumes_dir = job_dir / "resumes"
    resumes_dir.mkdir(exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(resumes_dir)
    except zipfile.BadZipFile:
        resumes_dir = job_dir

    JOBS[job_id] = {"status": "queued", "created_at": datetime.now().isoformat(), "logs": []}
    JOB_QUEUES[job_id] = queue.Queue()

    t = threading.Thread(
        target=_run_ingest_bg,
        args=(job_id, str(csv_path), str(resumes_dir), reset),
        daemon=True,
    )
    t.start()
    return {"job_id": job_id, "status": "queued"}


# ── PHASE 2: SEARCH / JD RANK ─────────────────────────────────────────────────
def _run_search_bg(job_id: str, jd_text: str, top_k: int, github_verify: bool, global_token: str):
    JOBS[job_id]["status"] = "running"
    try:
        from search import run_search

        def cb(phase: str, done: int, total: int):
            _emit(job_id, "progress", {
                "phase": phase, "done": done, "total": total,
                "pct": int(done / max(1, total) * 100)
            })

        _emit(job_id, "log", {"level": "info", "msg": f"Searching top {top_k}, github_verify={github_verify}"})
        results = run_search(
            jd_text=jd_text,
            db_path=DB_PATH,
            top_k=top_k,
            github_verify=github_verify,
            global_github_token=global_token,
            progress_cb=cb,
        )

        scores = [r.get("final_score", 0) for r in results]
        summary = {
            "total":      len(results),
            "avg_score":  round(sum(scores) / max(1, len(scores)), 1),
            "top_score":  scores[0] if scores else 0,
            "gh_verified": sum(1 for r in results if r.get("github_score") is not None),
            "distribution": {
                "excellent": sum(1 for s in scores if s >= 90),
                "strong":    sum(1 for s in scores if 70 <= s < 90),
                "moderate":  sum(1 for s in scores if 50 <= s < 70),
                "weak":      sum(1 for s in scores if 30 <= s < 50),
                "poor":      sum(1 for s in scores if s < 30),
            },
            "candidates": results,
        }
        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["result"] = summary
        _emit(job_id, "done", summary)
    except Exception as exc:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(exc)
        _emit(job_id, "error", {"message": str(exc), "traceback": traceback.format_exc()[:1200]})
    finally:
        if job_id in JOB_QUEUES:
            JOB_QUEUES[job_id].put("__DONE__")


@app.post("/api/search")
async def search(
    jd:            str  = Form(...),
    top_k:         int  = Form(default=50),
    github_verify: bool = Form(default=False),
    github_token:  str  = Form(default=""),
):
    """Run JD search against indexed DB. Returns job_id for SSE streaming."""
    job_id = str(uuid.uuid4())[:8]
    JOBS[job_id] = {"status": "queued", "created_at": datetime.now().isoformat(), "logs": []}
    JOB_QUEUES[job_id] = queue.Queue()

    t = threading.Thread(
        target=_run_search_bg,
        args=(job_id, jd, top_k, github_verify, github_token),
        daemon=True,
    )
    t.start()
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/jobs/{job_id}/result")
async def job_result(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "Job not found")
    j = JOBS[job_id]
    return {"job_id": job_id, "status": j["status"], "result": j.get("result"), "error": j.get("error")}


# ── STUDENTS ──────────────────────────────────────────────────────────────────
@app.get("/api/students")
async def list_students():
    import database as _db
    conn = _db.connect(DB_PATH)
    students = get_all_students(conn)
    conn.close()
    return {"total": len(students), "students": students}


@app.get("/api/students/{roll_number}")
async def get_student_detail(roll_number: str):
    import database as _db
    conn = _db.connect(DB_PATH)
    s = get_student(conn, roll_number)
    conn.close()
    if not s:
        raise HTTPException(404, "Student not found")
    s.pop("raw_text", None)
    s.pop("github_token", None)
    return s


# ── QUICK SCORE (no DB — uses Qwen parser + ATS engine) ──────────────────────
@app.post("/api/quick-score")
async def quick_score(
    jd:          str        = Form(...),
    resume:      UploadFile = File(...),
    github_url:  str        = Form(default=""),
    github_token: str       = Form(default=""),
):
    """
    Score a single resume against a JD instantly.
    Steps:
      1. Extract plain text from PDF/DOCX (PyMuPDF)
      2. Parse with Qwen2.5-1.5B via Ollama (GPU)
      3. Score with SBERT ATS engine (GPU)
      4. Optionally verify GitHub profile (REST API, no token needed)
    """
    suffix = Path(resume.filename).suffix.lower()
    if suffix not in (".pdf", ".docx"):
        raise HTTPException(400, "Only PDF and DOCX files are supported.")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, dir=UPLOAD_DIR) as tmp:
        tmp.write(await resume.read())
        tmp_path = tmp.name

    try:
        # ── Step 1 + 2: Parse resume via Ollama Qwen ──────────────────────
        from qwen_resume_parser import parse_resume as qwen_parse
        parsed = qwen_parse(tmp_path, model_name="qwen2.5:1.5b-instruct")

        # Extract flat fields from canonical structure
        candidate = parsed.get("candidate", {}) or {}
        name      = candidate.get("candidate_name") or parsed.get("name", "")
        email     = candidate.get("email") or parsed.get("email", "")
        phone     = candidate.get("phone") or parsed.get("phone", "")
        gh_url    = (
            github_url.strip()
            or candidate.get("github_url") or ""
        )

        # Build flat skill list from structured skills dict
        skills_dict = parsed.get("skills", {}) or {}
        if isinstance(skills_dict, dict):
            all_skills = []
            for cat_skills in skills_dict.values():
                if isinstance(cat_skills, list):
                    all_skills.extend(cat_skills)
        else:
            all_skills = skills_dict if isinstance(skills_dict, list) else []

        # Build project list
        projects = parsed.get("projects", []) or []

        # Build raw text from education + experience + projects for ATS
        edu_list  = parsed.get("education", []) or []
        exp_list  = parsed.get("experience", []) or []
        edu_text  = " ".join(
            f"{e.get('institution','')} {e.get('degree','')} {e.get('field','')}"
            for e in edu_list if isinstance(e, dict)
        )
        exp_text  = " ".join(
            f"{e.get('organization','')} {e.get('role','')} {' '.join(e.get('description',[]))}"
            for e in exp_list if isinstance(e, dict)
        )
        proj_text = " ".join(
            f"{p.get('project_name','')} {p.get('description','')} {' '.join(p.get('technologies',[]))}"
            for p in projects if isinstance(p, dict)
        )
        raw_text  = f"{edu_text} {exp_text} {proj_text} {' '.join(all_skills)}".strip()

        # Fallback: use document_extractor if raw_text is sparse
        if len(raw_text) < 200:
            from document_extractor import extract_plain_text
            raw_text = extract_plain_text(tmp_path)

        # ── Step 3: ATS scoring ─────────────────────────────────────────
        from ats_engine import score_candidate
        result = score_candidate(
            raw_text   = raw_text,
            skills     = all_skills,
            projects   = [
                {
                    "name":         p.get("project_name", ""),
                    "description":  p.get("description", ""),
                    "technologies": p.get("technologies", []),
                }
                for p in projects if isinstance(p, dict)
            ],
            education  = edu_text,
            email      = email,
            phone      = phone,
            github_url = gh_url,
            jd_text    = jd,
            device     = DEVICE,
        )

        # ── Step 4: GitHub verification (optional, no token required) ────
        github_result = None
        gh_username   = _extract_github_username(gh_url)

        if gh_username:
            try:
                from github_verifier import run_github_verification
                resume_project_names = [
                    p.get("project_name", "") for p in projects
                    if isinstance(p, dict) and p.get("project_name")
                ]
                github_result = run_github_verification(
                    username        = gh_username,
                    token           = github_token.strip() or "",
                    resume_projects = resume_project_names,
                    resume_skills   = all_skills,
                    resume_email    = email,
                    jd_text         = jd,
                )
            except Exception as gh_exc:
                github_result = {"error": str(gh_exc), "score": None}

        return {
            "name":            name,
            "email":           email,
            "phone":           phone,
            "github_url":      gh_url,
            "github_username": gh_username,
            "ats_score":       result["final_score"],
            "tier":            result.get("tier", ""),
            "recommendation":  result.get("recommendation", ""),
            "tier_note":       result.get("tier_note", ""),
            "semantic_score":  result["components"]["semantic_match"]["score"],
            "skill_score":     result["components"]["required_skill_match"]["score"],
            "components":      result["components"],
            "penalty":         result["penalty"],
            "raw_cosine":      result["raw_cosine"],
            "matched_skills":  result["matched_skills"],
            "missing_skills":  result["missing_skills"],
            "strengths":       result.get("strengths", []),
            "weaknesses":      result.get("weaknesses", []),
            "jd_required":     result["jd_required"],
            "skills":          all_skills,
            "projects":        projects,
            "education":       edu_list,
            "experience":      exp_list,
            "explanation":     result["explanation"],
            "github":          github_result,
            "parser_metadata": parsed.get("parser_metadata", {}),
        }
    except Exception as exc:
        tb = traceback.format_exc()
        raise HTTPException(500, f"Scoring error: {exc}\n\n{tb[:800]}")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def _extract_github_username(url_or_username: str) -> str:
    """Extract GitHub username from a URL or return raw username string."""
    if not url_or_username:
        return ""
    s = url_or_username.strip().rstrip("/")
    # Full URL: https://github.com/username or github.com/username
    m = re.match(r"(?:https?://)?(?:www\.)?github\.com/([A-Za-z0-9._-]+)(?:/.*)?$", s)
    if m:
        return m.group(1)
    # Plain username (no slashes, no dots except in username)
    if re.match(r"^[A-Za-z0-9._-]+$", s) and "/" not in s:
        return s
    return ""


# ── GITHUB VERIFY (standalone) ────────────────────────────────────────────────
@app.post("/api/github-verify")
async def github_verify_standalone(
    github_url:   str = Form(...),
    github_token: str = Form(default=""),
    resume_skills: str = Form(default=""),
    jd_text:      str = Form(default=""),
):
    """
    Verify a GitHub profile standalone.
    Uses the GitHub REST API (unauthenticated: 60 requests/hr free limit).
    Accepts full URL or just username.
    """
    username = _extract_github_username(github_url)
    if not username:
        raise HTTPException(400, f"Could not extract a valid GitHub username from: {github_url!r}")

    try:
        from github_verifier import run_github_verification
        skills = [s.strip() for s in resume_skills.split(",") if s.strip()]
        result = run_github_verification(
            username        = username,
            token           = github_token.strip() or "",
            resume_projects = [],
            resume_skills   = skills,
            resume_email    = "",
            jd_text         = jd_text,
        )
        return result
    except Exception as exc:
        raise HTTPException(500, f"GitHub verification error: {exc}")


# ── RESET DB ──────────────────────────────────────────────────────────────────
@app.delete("/api/reset")
async def reset_database():
    reset_db(DB_PATH)
    return {"status": "ok", "message": "Database reset successfully."}


# ── SERVE UI ──────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    ui = Path(__file__).parent / "ui" / "index.html"
    return HTMLResponse(ui.read_text(encoding="utf-8") if ui.exists() else "<h1>UI not found</h1>")

ui_dir = Path(__file__).parent / "ui"
ui_dir.mkdir(exist_ok=True)
app.mount("/ui", StaticFiles(directory=str(ui_dir)), name="ui")


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("═" * 60)
    print("  HireFlow-Lite ATS Server v4.0")
    print(f"  GPU    : {GPU_INFO}")
    print(f"  Device : {DEVICE}")
    print(f"  DB     : {DB_PATH}")
    print(f"  URL    : http://localhost:8000")
    print("═" * 60)
    config = uvicorn.Config("server:app", host="0.0.0.0", port=8000, log_level="warning")
    server = uvicorn.Server(config)
    server.run()
