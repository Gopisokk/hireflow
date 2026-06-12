"""
HireFlow Batch Upload Router
------------------------------
POST /api/batch/upload — Accept a CSV + resume_folder + job_description,
parse the CSV, insert students into the DB, and return a summary.
"""

import csv
import io
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from config import RESUME_FOLDER
from database.init_db import get_db
from database.models import get_student, insert_student

router = APIRouter(prefix="/api/batch", tags=["batch"])


@router.post("/upload")
async def batch_upload(
    csv_file: UploadFile = File(..., description="CSV with columns: roll_number, name, github_username, resume_filename"),
    job_description: str = Form(..., description="Job description text for the pipeline"),
    resume_folder: Optional[str] = Form(None, description="Absolute path to the folder with resume files (default: data/resumes)"),
) -> dict:
    """
    Upload a CSV of candidate data to seed the database.

    CSV format (with header row):
        roll_number, name, github_username, resume_filename

    The resume_filename should refer to a file inside `resume_folder`.
    """
    # ── Validate CSV file ────────────────────────────────────────────────────
    if not csv_file.filename or not csv_file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a .csv")

    raw_bytes = await csv_file.read()
    try:
        text = raw_bytes.decode("utf-8-sig")  # handle BOM
    except UnicodeDecodeError:
        text = raw_bytes.decode("latin-1")

    # ── Resolve resume folder ────────────────────────────────────────────────
    resume_dir = Path(resume_folder) if resume_folder else RESUME_FOLDER
    if not resume_dir.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Resume folder does not exist: {resume_dir}",
        )

    # ── Parse CSV ────────────────────────────────────────────────────────────
    reader = csv.DictReader(io.StringIO(text))
    required_cols = {"roll_number", "name"}
    if reader.fieldnames is None:
        raise HTTPException(status_code=400, detail="CSV file appears to be empty")

    # Normalize header names (strip whitespace, lowercase)
    header_map: dict[str, str] = {}
    for col in reader.fieldnames:
        header_map[col.strip().lower().replace(" ", "_")] = col

    for req in required_cols:
        if req not in header_map:
            raise HTTPException(
                status_code=400,
                detail=f"CSV must contain column: {req}. Found: {list(header_map.keys())}",
            )

    # ── Insert students ──────────────────────────────────────────────────────
    db = get_db()
    inserted = 0
    skipped = 0
    errors: list[str] = []

    try:
        # Re-parse with the reader since we already consumed it above
        reader = csv.DictReader(io.StringIO(text))

        for row_num, row in enumerate(reader, start=2):
            # Normalize keys
            normalized: dict[str, str] = {}
            for k, v in row.items():
                normalized[k.strip().lower().replace(" ", "_")] = (v or "").strip()

            roll_number = normalized.get("roll_number", "").strip()
            name = normalized.get("name", "").strip()
            github_username = normalized.get("github_username", "").strip() or None
            resume_filename = normalized.get("resume_filename", "").strip() or None

            if not roll_number or not name:
                errors.append(f"Row {row_num}: missing roll_number or name")
                continue

            # Check if resume file exists
            if resume_filename:
                resume_path = resume_dir / resume_filename
                if not resume_path.exists():
                    errors.append(
                        f"Row {row_num}: resume file not found: {resume_filename}"
                    )
                    # Still insert, just note the warning

            # Check for duplicate roll numbers
            existing = get_student(db, roll_number)
            if existing:
                skipped += 1
                continue

            try:
                insert_student(
                    db,
                    roll_number=roll_number,
                    name=name,
                    github_username=github_username,
                    resume_filename=resume_filename,
                )
                inserted += 1
            except Exception as exc:
                errors.append(f"Row {row_num}: {str(exc)}")

    finally:
        db.close()

    return {
        "status": "success",
        "inserted": inserted,
        "skipped": skipped,
        "errors": errors,
        "total_rows_processed": inserted + skipped + len(errors),
        "job_description_length": len(job_description),
        "resume_folder": str(resume_dir),
    }
