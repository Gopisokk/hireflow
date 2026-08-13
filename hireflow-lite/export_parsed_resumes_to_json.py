"""
export_parsed_resumes_to_json.py — Standalone Resume Parser & JSON Exporter
=============================================================================
Parses every PDF/DOCX resume in the uploads directory using HireFlow-Lite's
upgraded Firecrawl pdf-inspector engine (resume_parser.py) and outputs the
structured JSON representation for every resume.

Usage:
    venv\\Scripts\\python.exe export_parsed_resumes_to_json.py
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path

# Force UTF-8 stdout on Windows
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from resume_parser import parse_resume


def parse_and_export_all_resumes() -> list[dict]:
    # Locate resumes folder
    base_dir = Path(__file__).parent
    resumes_dir = base_dir / "uploads" / "9d3e5586" / "resumes" / "_Resume Upload (Filename must be Roll Number, e.g., 23AD044.pdf) (File responses)"

    if not resumes_dir.exists():
        print(f"Resume directory not found: {resumes_dir}")
        return []

    resume_files = sorted([
        p for p in resumes_dir.glob("*") if p.suffix.lower() in (".pdf", ".docx")
    ])

    print(f"=== Found {len(resume_files)} Resume Files for Parsing ===")
    
    all_parsed_json = []

    for filepath in resume_files:
        print(f"Parsing: {filepath.name}...")
        parsed = parse_resume(str(filepath))
        
        # Add metadata fields
        parsed["resume_filename"] = filepath.name
        parsed["file_size_bytes"] = filepath.stat().st_size
        
        all_parsed_json.append(parsed)

    # Save to standalone JSON file
    output_file = base_dir / "all_parsed_resumes.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_parsed_json, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Successfully parsed {len(all_parsed_json)} resumes.")
    print(f"[OK] Saved output to: {output_file}\n")

    return all_parsed_json


if __name__ == "__main__":
    results = parse_and_export_all_resumes()

    # Print the formatted JSON for every resume to stdout
    print("=" * 80)
    print(" COMPLETE STRUCTURED PARSED RESUMES JSON DATA")
    print("=" * 80)
    print(json.dumps(results, indent=2, ensure_ascii=False))
