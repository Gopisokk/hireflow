"""
HireFlow Candidates Router
----------------------------
GET /api/candidates         — List all candidates (filterable by stage).
GET /api/candidates/{roll}  — Full candidate profile with skills, projects, GitHub data.
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from database.init_db import get_db
from database.models import (
    get_all_students,
    get_github_profile,
    get_projects,
    get_skills,
    get_student,
)

router = APIRouter(prefix="/api/candidates", tags=["candidates"])


@router.get("")
async def list_candidates(
    stage: Optional[str] = Query(None, description="Filter by pipeline stage"),
    limit: int = Query(100, ge=1, le=1000, description="Max results to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
) -> dict:
    """
    List all candidates with their scores.
    Optionally filter by stage: pending, parsed, scored, shortlisted, verified, ranked.
    """
    valid_stages = {"pending", "parsed", "scored", "shortlisted", "verified", "ranked"}
    if stage and stage not in valid_stages:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid stage '{stage}'. Valid stages: {sorted(valid_stages)}",
        )

    db = get_db()
    try:
        students = get_all_students(db, stage=stage)

        # Manual pagination (sqlite could do it, but this keeps the model simple)
        total = len(students)
        paginated = students[offset : offset + limit]

        # Build response with score summary
        candidates = []
        for s in paginated:
            candidates.append({
                "id": s["id"],
                "roll_number": s["roll_number"],
                "name": s["name"],
                "github_username": s["github_username"],
                "stage": s["stage"],
                "ats_score": s["ats_score"],
                "github_score": s["github_score"],
                "final_score": s["final_score"],
                "resume_filename": s["resume_filename"],
                "created_at": s["created_at"],
            })
    finally:
        db.close()

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "candidates": candidates,
    }


@router.get("/{roll_number}")
async def get_candidate_profile(roll_number: str) -> dict:
    """
    Full candidate profile including skills, projects, and GitHub data.
    """
    db = get_db()
    try:
        student = get_student(db, roll_number)
        if not student:
            raise HTTPException(
                status_code=404,
                detail=f"Candidate with roll number '{roll_number}' not found",
            )

        student_id = student["id"]
        skills = get_skills(db, student_id)
        projects = get_projects(db, student_id)
        github_profile = get_github_profile(db, student_id)
    finally:
        db.close()

    return {
        "student": {
            "id": student["id"],
            "roll_number": student["roll_number"],
            "name": student["name"],
            "github_username": student["github_username"],
            "stage": student["stage"],
            "ats_score": student["ats_score"],
            "github_score": student["github_score"],
            "final_score": student["final_score"],
            "resume_filename": student["resume_filename"],
            "created_at": student["created_at"],
            "updated_at": student["updated_at"],
        },
        "skills": [
            {"skill": sk["skill"], "source": sk["source"]}
            for sk in skills
        ],
        "projects": [
            {
                "project_name": p["project_name"],
                "source": p["source"],
                "github_verified": bool(p["github_verified"]),
                "github_repo_url": p["github_repo_url"],
                "is_fork": bool(p["is_fork"]),
            }
            for p in projects
        ],
        "github_profile": github_profile,
    }
