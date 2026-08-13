"""
server.py — HireFlow-Lite Enterprise ATS: FastAPI Backend
==========================================================
Two-phase architecture:
  Phase 1  POST /api/ingest   → Upload CSV + resumes zip → index to DB
  Phase 2  POST /api/search   → Upload JD → search DB → rank + GitHub verify

Additional:
  GET  /api/health            → GPU/DB status
  GET  /api/students          → All indexed students
  GET  /api/students/{roll}   → Single student detail
  GET  /api/jobs/{id}/stream  → SSE real-time progress
  POST /api/quick-score       → Single resume instant score (no DB)
  DELETE /api/reset           → Wipe DB
"""

import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
except ImportError:
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
except Exception:
    pass

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="HireFlow-Lite ATS", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Initialise DB on startup
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
        conn = __import__("database").connect(DB_PATH)
        stats = get_db_stats(conn)
        conn.close()
    except Exception:
        stats = {}
    return {
        "status": "ok",
        "device": DEVICE,
        "gpu": GPU_INFO,
        "db": str(DB_PATH),
        "db_stats": stats,
        "jobs_running": sum(1 for j in JOBS.values() if j.get("status") == "running"),
    }


# ── PHASE 1: INGEST ───────────────────────────────────────────────────────────
def _run_ingest_bg(job_id: str, csv_path: str, resumes_dir: str, workers: int, reset: bool):
    JOBS[job_id]["status"] = "running"
    try:
        from ingest import run_ingestion

        def cb(phase: str, done: int, total: int):
            _emit(job_id, "progress", {"phase": phase, "done": done, "total": total,
                                        "pct": int(done / max(1, total) * 100)})

        _emit(job_id, "log", {"level": "info", "msg": f"Ingestion started. Workers={workers}, Reset={reset}"})
        stats = run_ingestion(
            csv_path=csv_path,
            resumes_dir=resumes_dir,
            db_path=DB_PATH,
            workers=workers,
            reset=reset,
            progress_cb=cb,
        )
        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["result"] = stats
        _emit(job_id, "done", stats)
    except Exception as exc:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(exc)
        _emit(job_id, "error", {"message": str(exc), "traceback": traceback.format_exc()[:800]})
    finally:
        if job_id in JOB_QUEUES:
            JOB_QUEUES[job_id].put("__DONE__")


@app.post("/api/ingest")
async def ingest(
    csv_file:  UploadFile = File(...),
    resumes:   UploadFile = File(...),
    workers:   int        = Form(default=4),
    reset:     bool       = Form(default=False),
):
    """Upload CSV + resumes ZIP → index all students into DB."""
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
        # Maybe they uploaded a folder directly — treat job_dir as resumes_dir
        resumes_dir = job_dir

    JOBS[job_id] = {"status": "queued", "created_at": datetime.now().isoformat(), "logs": []}
    JOB_QUEUES[job_id] = queue.Queue()

    t = threading.Thread(
        target=_run_ingest_bg,
        args=(job_id, str(csv_path), str(resumes_dir), workers, reset),
        daemon=True,
    )
    t.start()
    return {"job_id": job_id, "status": "queued"}


# ── PHASE 2: SEARCH ───────────────────────────────────────────────────────────
def _run_search_bg(job_id: str, jd_text: str, top_k: int, github_verify: bool, global_token: str):
    JOBS[job_id]["status"] = "running"
    try:
        from search import run_search

        def cb(phase: str, done: int, total: int):
            _emit(job_id, "progress", {"phase": phase, "done": done, "total": total,
                                        "pct": int(done / max(1, total) * 100)})

        _emit(job_id, "log", {"level": "info", "msg": f"Searching top {top_k}, github_verify={github_verify}"})
        results = run_search(
            jd_text=jd_text,
            db_path=DB_PATH,
            top_k=top_k,
            github_verify=github_verify,
            global_github_token=global_token,
            progress_cb=cb,
        )

        # Build summary
        scores = [r.get("final_score", 0) for r in results]
        summary = {
            "total":     len(results),
            "avg_score": round(sum(scores) / max(1, len(scores)), 1),
            "top_score": scores[0] if scores else 0,
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
        _emit(job_id, "error", {"message": str(exc), "traceback": traceback.format_exc()[:800]})
    finally:
        if job_id in JOB_QUEUES:
            JOB_QUEUES[job_id].put("__DONE__")


@app.post("/api/search")
async def search(
    jd:            str  = Form(...),
    top_k:         int  = Form(default=50),
    github_verify: bool = Form(default=True),
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
    conn = __import__("database").connect(DB_PATH)
    students = get_all_students(conn)
    conn.close()
    return {"total": len(students), "students": students}


@app.get("/api/students/{roll_number}")
async def get_student_detail(roll_number: str):
    conn = __import__("database").connect(DB_PATH)
    s = get_student(conn, roll_number)
    conn.close()
    if not s:
        raise HTTPException(404, "Student not found")
    s.pop("raw_text", None)
    s.pop("github_token", None)
    return s


# ── QUICK SCORE (no DB) ───────────────────────────────────────────────────────
@app.post("/api/quick-score")
async def quick_score(
    jd:     str        = Form(...),
    resume: UploadFile = File(...),
):
    """
    Score a single resume against a JD instantly, without touching the DB.
    Uses the full 8-component evidence-aggregation engine.
    """
    suffix = Path(resume.filename).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, dir=UPLOAD_DIR) as tmp:
        tmp.write(await resume.read())
        tmp_path = tmp.name
    try:
        from resume_parser import parse_resume
        from ats_engine import score_candidate

        parsed = parse_resume(tmp_path)
        result = score_candidate(
            raw_text   = parsed.get("raw_text", ""),
            skills     = parsed.get("skills", []),
            projects   = parsed.get("projects", []),
            education  = parsed.get("education", ""),
            email      = parsed.get("email", ""),
            phone      = parsed.get("phone", ""),
            github_url = parsed.get("github_url", ""),
            jd_text    = jd,
            device     = DEVICE,
        )
        return {
            "name":            parsed.get("name"),
            "email":           parsed.get("email"),
            "github_username": parsed.get("github_username"),
            "phone":           parsed.get("phone"),
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
            "skills":          parsed.get("skills", []),
            "explanation":     result["explanation"],
        }
    except Exception as exc:
        raise HTTPException(500, f"Scoring error: {exc}")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass



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
    print("  HireFlow-Lite ATS Server v3.0")
    print(f"  GPU    : {GPU_INFO}")
    print(f"  DB     : {DB_PATH}")
    print(f"  URL    : http://localhost:8000")
    print("═" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
