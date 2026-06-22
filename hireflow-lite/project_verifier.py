"""
project_verifier.py — HireFlow-Lite Project Verification Module
================================================================

Verifies that resume-claimed projects genuinely exist on the candidate's
GitHub profile using semantic similarity (SBERT) rather than exact string
matching.

Uses the same all-MiniLM-L6-v2 model cached in ats_engine to avoid
reloading.
"""

from __future__ import annotations

import sys
import warnings
from typing import Any

# Force UTF-8 stdout/stderr on Windows
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def _get_sbert_model(device: str = "cpu"):
    """Reuse the SBERT model from ats_engine if already loaded, else load it."""
    try:
        from ats_engine import _get_sbert_model as _ats_get_sbert
        return _ats_get_sbert(device)
    except ImportError:
        print("  \u2192 Loading SBERT model for project verification...")
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2", device=device)


def _readme_text(repo: dict) -> str:
    """Extract README text from a GitHub repo dict (GraphQL response)."""
    obj = repo.get("object")
    if obj and isinstance(obj, dict):
        return obj.get("text", "") or ""
    return ""


def verify_projects(
    resume_projects: list[dict],
    github_repos: list[dict],
    device: str = "cpu",
) -> list[dict]:
    """Verify that resume-claimed projects exist on GitHub using SBERT.

    Parameters
    ----------
    resume_projects : list[dict]
        Each dict has 'name' and 'description' keys
        (from resume_parser.py's _extract_projects).
    github_repos : list[dict]
        Repo dicts from github_verifier.py's GraphQL response.
        Each has 'name', 'description', 'isFork', and optionally
        'object' (README blob).
    device : str, optional
        Torch device string (default 'cpu').

    Returns
    -------
    list[dict]
        For each resume project:
        {
            "claimed_project": str,
            "status": "verified" | "uncertain" | "unverified",
            "matched_repo": str or None,
            "similarity": float (0.0-1.0),
            "is_fork": bool or None,
        }
    """
    if not resume_projects:
        print("  \u2192 No resume projects to verify.")
        return []

    if not github_repos:
        print("  \u2192 No GitHub repos available for verification.")
        return [
            {
                "claimed_project": p.get("name", "Unknown"),
                "status": "unverified",
                "matched_repo": None,
                "similarity": 0.0,
                "is_fork": None,
            }
            for p in resume_projects
        ]

    try:
        from sentence_transformers import util as st_util
    except ImportError:
        print("  \u2717 sentence-transformers not installed, skipping verification.")
        return [
            {
                "claimed_project": p.get("name", "Unknown"),
                "status": "unverified",
                "matched_repo": None,
                "similarity": 0.0,
                "is_fork": None,
            }
            for p in resume_projects
        ]

    print(f"  \u2192 Verifying {len(resume_projects)} resume projects against "
          f"{len(github_repos)} GitHub repos...")

    model = _get_sbert_model(device)

    # Build text strings for resume projects
    resume_texts = []
    for p in resume_projects:
        name = p.get("name", "")
        desc = p.get("description", "")
        resume_texts.append(f"{name}. {desc}".strip())

    # Build text strings for GitHub repos
    repo_texts = []
    for r in github_repos:
        name = r.get("name", "")
        desc = r.get("description", "") or ""
        readme = _readme_text(r)
        # Limit README to first 500 chars to keep embeddings focused
        readme_snippet = readme[:500] if readme else ""
        repo_texts.append(f"{name}. {desc}. {readme_snippet}".strip())

    # Encode all texts
    print("  \u2192 Encoding project and repo texts with SBERT...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        resume_embs = model.encode(resume_texts, convert_to_tensor=True, device=device)
        repo_embs = model.encode(repo_texts, convert_to_tensor=True, device=device)

    # Compute similarity matrix: resume_projects x github_repos
    print("  \u2192 Computing similarity matrix...")
    sim_matrix = st_util.cos_sim(resume_embs, repo_embs)

    results = []
    for i, project in enumerate(resume_projects):
        sims = sim_matrix[i]
        best_idx = int(sims.argmax())
        best_sim = float(sims[best_idx])
        best_repo = github_repos[best_idx]

        if best_sim >= 0.55:
            status = "verified"
        elif best_sim >= 0.35:
            status = "uncertain"
        else:
            status = "unverified"

        results.append({
            "claimed_project": project.get("name", "Unknown"),
            "status": status,
            "matched_repo": best_repo.get("name"),
            "similarity": round(best_sim, 4),
            "is_fork": best_repo.get("isFork"),
        })

    # Print summary table
    print()
    separator = "\u2500"
    print(f"  {'Claimed Project':<30} {'Status':<14} {'Matched Repo':<25} {'Similarity':<10}")
    print(f"  {separator * 30} {separator * 14} {separator * 25} {separator * 10}")
    for r in results:
        proj = r['claimed_project'][:28]
        repo = (r['matched_repo'] or 'None')[:23]
        print(f"  {proj:<30} {r['status']:<14} {repo:<25} {r['similarity']:.4f}")
    print()

    verified_count = sum(1 for r in results if r['status'] == 'verified')
    uncertain_count = sum(1 for r in results if r['status'] == 'uncertain')
    unverified_count = sum(1 for r in results if r['status'] == 'unverified')
    print(f"  \u2713 Project verification complete: "
          f"{verified_count} verified, {uncertain_count} uncertain, "
          f"{unverified_count} unverified")

    return results
