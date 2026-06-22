"""
HireFlow Pipeline Status Router
---------------------------------
POST /api/pipeline/start   — Start the full pipeline as a background task.
GET  /api/pipeline/status  — SSE endpoint streaming real-time progress updates.
GET  /api/pipeline/result  — Get the final pipeline run results.

The pipeline runs three stages:
  Stage 1: Parse resumes → extract skills/projects → embed → ATS score → shortlist
  Stage 2: GitHub verification of shortlisted candidates
  Stage 3: Merge scores → compute final ranking
"""

import asyncio
import json
import math
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from config import RESUME_FOLDER
from database.init_db import get_db
from database.models import (
    create_pipeline_run,
    delete_projects,
    delete_skills,
    get_all_students,
    get_github_profile,
    get_latest_pipeline_run,
    get_pipeline_run,
    get_projects,
    get_skills,
    get_student_by_id,
    insert_project,
    insert_skill,
    store_embedding,
    update_pipeline_run,
    update_student_resume_text,
    update_student_scores,
    update_student_stage,
    upsert_github_profile,
)
from pipeline.ats_scorer import ATSScorer
from pipeline.embedder import EmbedderService
from pipeline.github_verifier import GitHubVerifier
from pipeline.parser import extract_text
from pipeline.skill_extractor import extract_projects, extract_skills
from routers.config_router import get_runtime_config

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])


# ═══════════════════════════════════════════════════════════════════════════════
#  Shared state for SSE progress reporting
# ═══════════════════════════════════════════════════════════════════════════════

class PipelineProgress:
    """Thread-safe progress tracker that SSE clients can read."""

    def __init__(self) -> None:
        self.running: bool = False
        self.current_stage: str = "idle"
        self.processed_count: int = 0
        self.total_count: int = 0
        self.current_candidate: str = ""
        self.message: str = ""
        self.run_id: Optional[int] = None
        self.error: Optional[str] = None
        self._events: asyncio.Queue[dict] = asyncio.Queue()

    def update(
        self,
        stage: Optional[str] = None,
        processed: Optional[int] = None,
        total: Optional[int] = None,
        candidate: Optional[str] = None,
        message: Optional[str] = None,
    ) -> None:
        if stage is not None:
            self.current_stage = stage
        if processed is not None:
            self.processed_count = processed
        if total is not None:
            self.total_count = total
        if candidate is not None:
            self.current_candidate = candidate
        if message is not None:
            self.message = message

        event = self.snapshot()
        try:
            self._events.put_nowait(event)
        except asyncio.QueueFull:
            pass  # Drop if backpressured

    def snapshot(self) -> dict:
        return {
            "running": self.running,
            "current_stage": self.current_stage,
            "processed_count": self.processed_count,
            "total_count": self.total_count,
            "current_candidate": self.current_candidate,
            "message": self.message,
            "run_id": self.run_id,
            "error": self.error,
        }

    async def get_event(self, timeout: float = 5.0) -> Optional[dict]:
        try:
            return await asyncio.wait_for(self._events.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    def reset(self) -> None:
        self.running = False
        self.current_stage = "idle"
        self.processed_count = 0
        self.total_count = 0
        self.current_candidate = ""
        self.message = ""
        self.run_id = None
        self.error = None
        # Drain queue
        while not self._events.empty():
            try:
                self._events.get_nowait()
            except asyncio.QueueEmpty:
                break


progress = PipelineProgress()


# ═══════════════════════════════════════════════════════════════════════════════
#  Request / Response models
# ═══════════════════════════════════════════════════════════════════════════════

class PipelineStartRequest(BaseModel):
    job_description: str = Field(..., min_length=10, description="The job description to score against")
    github_token: Optional[str] = Field(None, description="GitHub PAT for API access (optional, skips Stage 2 if absent)")
    resume_folder: Optional[str] = Field(None, description="Override resume folder path")


# ═══════════════════════════════════════════════════════════════════════════════
#  Pipeline orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

async def run_pipeline(
    job_description: str,
    github_token: Optional[str],
    resume_dir: Path,
) -> None:
    """
    Main pipeline orchestrator. Runs sequentially through three stages.
    Updates the shared `progress` object for SSE streaming.
    """
    cfg = get_runtime_config()
    ats_weight = cfg["ats_weight"]
    github_weight = cfg["github_weight"]
    algorithm = cfg["algorithm"]
    shortlist_pct = cfg["shortlist_percent"]

    db = get_db()

    try:
        students = get_all_students(db)
        if not students:
            progress.update(stage="error", message="No candidates in the database")
            progress.error = "No candidates found. Upload a batch first."
            progress.running = False
            return

        total = len(students)
        progress.update(total=total)

        # Create pipeline run record
        run_id = create_pipeline_run(
            db,
            job_description=job_description,
            ats_weight=ats_weight,
            github_weight=github_weight,
            algorithm=algorithm,
            total_candidates=total,
        )
        progress.run_id = run_id
        update_pipeline_run(db, run_id, status="running")

        # ──────────────────────────────────────────────────────────────────
        #  STAGE 1: Parse → Extract Skills/Projects → Embed → ATS Score
        # ──────────────────────────────────────────────────────────────────
        progress.update(stage="stage1", message="Stage 1: Parsing resumes & scoring")
        update_pipeline_run(db, run_id, status="stage1")

        embedder = EmbedderService()
        resumes_for_scoring: dict[int, str] = {}

        for i, student in enumerate(students):
            sid = student["id"]
            name = student["name"]
            progress.update(processed=i, candidate=name, message=f"Parsing resume: {name}")

            # ── Parse resume ─────────────────────────────────────────────
            resume_text = student.get("resume_text") or ""
            if not resume_text and student.get("resume_filename"):
                resume_path = resume_dir / student["resume_filename"]
                try:
                    resume_text = extract_text(str(resume_path))
                    update_student_resume_text(db, sid, resume_text)
                except (FileNotFoundError, ValueError) as exc:
                    progress.update(message=f"Warning: {name} — {exc}")
                    resume_text = ""

            if not resume_text:
                progress.update(message=f"Skipping {name}: no resume text")
                continue

            resumes_for_scoring[sid] = resume_text

            # ── Extract skills ───────────────────────────────────────────
            delete_skills(db, sid)
            skills = extract_skills(resume_text)
            for skill in skills:
                insert_skill(db, sid, skill, source="resume")

            # ── Extract projects ─────────────────────────────────────────
            delete_projects(db, sid)
            projects = extract_projects(resume_text)
            for proj in projects:
                insert_project(db, sid, proj, source="resume")

            # ── Embed resume ─────────────────────────────────────────────
            try:
                embedding = embedder.embed_text(resume_text)
                store_embedding(db, sid, embedding)
            except Exception as exc:
                progress.update(message=f"Embedding warning for {name}: {exc}")

            # ── Update stage ─────────────────────────────────────────────
            update_student_stage(db, sid, "parsed")

            # Yield control to allow SSE events
            await asyncio.sleep(0)

        progress.update(processed=total, message=f"Parsed {len(resumes_for_scoring)}/{total} resumes")

        # ── ATS Scoring ──────────────────────────────────────────────────
        progress.update(message=f"Running ATS scoring ({algorithm})...")
        scorer = ATSScorer()
        ats_scores = scorer.score(job_description, resumes_for_scoring, algorithm)

        for sid, score_val in ats_scores.items():
            update_student_scores(db, sid, ats_score=score_val)
            update_student_stage(db, sid, "scored")

        progress.update(message=f"ATS scoring complete: {len(ats_scores)} candidates scored")
        await asyncio.sleep(0)

        # ── Shortlist top N% ─────────────────────────────────────────────
        sorted_by_ats = sorted(ats_scores.items(), key=lambda x: x[1], reverse=True)
        shortlist_count = max(1, math.ceil(len(sorted_by_ats) * shortlist_pct / 100))
        shortlisted_ids = [sid for sid, _ in sorted_by_ats[:shortlist_count]]

        for sid in shortlisted_ids:
            update_student_stage(db, sid, "shortlisted")

        update_pipeline_run(db, run_id, status="stage1", shortlisted=shortlist_count)
        progress.update(
            message=f"Shortlisted top {shortlist_count} candidates for GitHub verification"
        )

        # ──────────────────────────────────────────────────────────────────
        #  STAGE 2: GitHub Verification (only shortlisted candidates)
        # ──────────────────────────────────────────────────────────────────
        progress.update(stage="stage2", message="Stage 2: GitHub verification")
        update_pipeline_run(db, run_id, status="stage2")

        github_scores: dict[int, float] = {}

        if github_token:
            verifier = GitHubVerifier(requests_per_second=1.0)

            # Extract ALL JD skills for per-project tech alignment
            jd_skills = extract_skills(job_description)

            for i, sid in enumerate(shortlisted_ids):
                student = get_student_by_id(db, sid)
                if not student:
                    continue

                username = student.get("github_username", "")
                name = student["name"]
                progress.update(
                    processed=i,
                    total=shortlist_count,
                    candidate=name,
                    message=f"Verifying GitHub: {username or 'N/A'}",
                )

                if not username:
                    progress.update(message=f"Skipping {name}: no GitHub username")
                    github_scores[sid] = 0.0
                    update_student_stage(db, sid, "verified")
                    continue

                # Get resume projects for matching
                resume_projects = [p["project_name"] for p in get_projects(db, sid)]

                try:
                    result = await verifier.verify_candidate(
                        token=github_token,
                        username=username,
                        resume_projects=resume_projects,
                        jd_skills=jd_skills,
                    )

                    if result.error:
                        progress.update(message=f"GitHub warning for {name}: {result.error}")
                        github_scores[sid] = 0.0
                    else:
                        github_scores[sid] = result.github_score

                        # Store GitHub profile
                        upsert_github_profile(
                            db,
                            student_id=sid,
                            active_days=result.active_days,
                            total_commits=result.total_commits,
                            total_prs=result.total_prs,
                            fork_ratio=result.fork_ratio,
                            top_languages=result.top_languages,
                            contribution_score=result.github_score,
                        )

                        # Update project verification status
                        for matched in result.repos:
                            # Find the project in DB and update it
                            db_projects = get_projects(db, sid)
                            for db_proj in db_projects:
                                if db_proj["project_name"].lower() == matched["resume_project"].lower():
                                    from database.models import update_project_verification
                                    update_project_verification(
                                        db,
                                        project_id=db_proj["id"],
                                        github_verified=True,
                                        github_repo_url=matched.get("url"),
                                        is_fork=matched.get("is_fork", False),
                                    )

                    update_student_scores(db, sid, github_score=github_scores[sid])
                    update_student_stage(db, sid, "verified")

                except Exception as exc:
                    progress.update(message=f"GitHub error for {name}: {exc}")
                    github_scores[sid] = 0.0
                    update_student_stage(db, sid, "verified")

                await asyncio.sleep(0)

            progress.update(
                processed=shortlist_count,
                message=f"GitHub verification complete: {len(github_scores)} verified",
            )
        else:
            progress.update(message="Skipping GitHub verification (no token provided)")
            for sid in shortlisted_ids:
                github_scores[sid] = 0.0
                update_student_stage(db, sid, "verified")

        # ──────────────────────────────────────────────────────────────────
        #  STAGE 3: Merge scores → Final ranking
        # ──────────────────────────────────────────────────────────────────
        progress.update(stage="stage3", message="Stage 3: Computing final rankings")
        update_pipeline_run(db, run_id, status="stage3")

        all_students = get_all_students(db)
        progress.update(total=len(all_students))

        for i, student in enumerate(all_students):
            sid = student["id"]
            ats = student.get("ats_score") or 0.0
            gh = student.get("github_score") or 0.0

            if sid in github_scores:
                # Shortlisted candidate: weighted combination
                final = round(ats_weight * ats + github_weight * gh, 2)
            else:
                # Non-shortlisted: ATS only (penalized slightly)
                final = round(ats * 0.9, 2)

            update_student_scores(db, sid, final_score=final)
            update_student_stage(db, sid, "ranked")
            progress.update(processed=i + 1, candidate=student["name"])
            await asyncio.sleep(0)

        # ── Pipeline complete ────────────────────────────────────────────
        now = datetime.now(timezone.utc).isoformat()
        update_pipeline_run(
            db, run_id,
            status="completed",
            shortlisted=shortlist_count,
            completed_at=now,
        )

        progress.update(
            stage="completed",
            message=f"Pipeline complete! {len(all_students)} candidates ranked.",
            processed=len(all_students),
            total=len(all_students),
        )
        progress.running = False

    except Exception as exc:
        progress.error = str(exc)
        progress.update(stage="error", message=f"Pipeline failed: {exc}")
        progress.running = False
        if progress.run_id:
            try:
                update_pipeline_run(db, progress.run_id, status="failed")
            except Exception:
                pass
        traceback.print_exc()

    finally:
        db.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  Routes
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/start")
async def start_pipeline(body: PipelineStartRequest) -> dict:
    """
    Start the hiring pipeline as a background task.
    Returns immediately with the run status.
    """
    if progress.running:
        raise HTTPException(
            status_code=409,
            detail="A pipeline is already running. Wait for it to complete.",
        )

    resume_dir = Path(body.resume_folder) if body.resume_folder else RESUME_FOLDER

    # Reset progress
    progress.reset()
    progress.running = True
    progress.current_stage = "starting"
    progress.message = "Pipeline starting..."

    # Launch background task
    asyncio.create_task(
        run_pipeline(
            job_description=body.job_description,
            github_token=body.github_token,
            resume_dir=resume_dir,
        )
    )

    return {
        "status": "started",
        "message": "Pipeline started. Use GET /api/pipeline/status for progress (SSE).",
    }


@router.get("/status")
async def pipeline_status_sse() -> EventSourceResponse:
    """
    Server-Sent Events endpoint for real-time pipeline progress.
    Emits JSON events with: current_stage, processed_count, total_count,
    current_candidate, message, running, error.
    """
    async def event_generator() -> AsyncGenerator[dict, None]:
        # Send initial snapshot
        yield {"event": "progress", "data": json.dumps(progress.snapshot())}

        while progress.running:
            event = await progress.get_event(timeout=2.0)
            if event:
                yield {"event": "progress", "data": json.dumps(event)}
            else:
                # Send heartbeat / current state
                yield {"event": "heartbeat", "data": json.dumps(progress.snapshot())}

        # Send final state
        yield {"event": "complete", "data": json.dumps(progress.snapshot())}

    return EventSourceResponse(event_generator())


@router.get("/result")
async def get_pipeline_result(
    run_id: Optional[int] = Query(None, description="Pipeline run ID (latest if omitted)"),
) -> dict:
    """Get the results of a pipeline run."""
    db = get_db()
    try:
        if run_id:
            run = get_pipeline_run(db, run_id)
        else:
            run = get_latest_pipeline_run(db)

        if not run:
            raise HTTPException(status_code=404, detail="No pipeline run found")

        # Get ranked candidates
        students = get_all_students(db, stage="ranked")

        return {
            "run": dict(run),
            "candidates": [
                {
                    "rank": i + 1,
                    "roll_number": s["roll_number"],
                    "name": s["name"],
                    "ats_score": s["ats_score"],
                    "github_score": s["github_score"],
                    "final_score": s["final_score"],
                    "stage": s["stage"],
                }
                for i, s in enumerate(students)
            ],
            "total_ranked": len(students),
        }
    finally:
        db.close()
