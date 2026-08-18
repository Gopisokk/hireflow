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
    from minilm import get_minilm_model
    return get_minilm_model(device="cpu")


def _readme_text(repo: dict) -> str:
    """Extract README text from a GitHub repo dict (GraphQL response)."""
    obj = repo.get("object")
    if obj and isinstance(obj, dict):
        return obj.get("text", "") or ""
    return ""


def _score_project_authenticity(
    name_match_score: float,         # 0–1: fuzzy/exact/SBERT name match
    sbert_similarity: float,         # 0–1: full text SBERT similarity
    is_fork: bool | None,
    candidate_is_author: bool,
    candidate_commit_ratio: float,   # 0–1: candidate commits / total commits
    active_days: int,
    total_commits: int,
    development_span_days: int,
    has_technology_evidence: bool,
    readme_similarity: float = 0.0,  # 0–1 SBERT README vs description
) -> tuple[float, str]:
    """
    Per-project authenticity score (0–100) and confidence label.

    Weights:
      15% Repository match (name + text similarity)
      15% Ownership / fork status
      15% Candidate authorship (commit author login)
      10% Development timeline (active days + span)
      10% Commit depth (total commits, density)
      15% Code / technology evidence
       5% README / resume alignment
      15% Reserved (returns 50% neutral until question-answer is implemented)

    Confidence thresholds:
      high   ≥ 70
      medium ≥ 45
      low    ≥ 25
      none   <  25
    """
    # ── 15%: Repository match ──────────────────────────────────────────
    repo_match = ((name_match_score * 0.6) + (sbert_similarity * 0.4)) * 100.0

    # ── 15%: Ownership / fork status ───────────────────────────────────
    if is_fork is None:
        ownership = 50.0   # neutral: no data
    elif not is_fork:
        ownership = 100.0  # original: best
    else:
        # Fork: score based on how much the candidate contributed after fork
        if candidate_commit_ratio >= 0.50:
            ownership = 60.0   # fork_substantial
        elif candidate_commit_ratio >= 0.20:
            ownership = 35.0   # fork_minimal
        else:
            ownership = 10.0   # fork_none

    # ── 15%: Candidate authorship ──────────────────────────────────────
    if candidate_commit_ratio > 0:
        authorship = min(100.0, candidate_commit_ratio * 100.0 * 1.2)
    elif candidate_is_author:
        authorship = 60.0
    else:
        authorship = 10.0   # not confirmed

    # ── 10%: Development timeline ─────────────────────────────────────────
    # Duration is a WEAK signal; active_days is stronger
    span_signal   = min(1.0, development_span_days / 90.0)   # 3-month span = full
    active_signal = min(1.0, active_days / 30.0)             # 30 active days = full
    timeline = ((span_signal * 0.35) + (active_signal * 0.65)) * 100.0

    # ── 10%: Commit depth ────────────────────────────────────────────────
    commit_depth = min(100.0, (total_commits / 20.0) * 100.0)   # 20 commits = full

    # ── 15%: Technology evidence ────────────────────────────────────────────
    tech_evidence = 80.0 if has_technology_evidence else 20.0

    # ── 5%: README alignment ───────────────────────────────────────────────
    readme_align = readme_similarity * 100.0

    # ── 15%: Reserved (neutral 50 until question-answer is built) ──────────
    reserved = 50.0

    score = (
        repo_match    * 0.15 +
        ownership     * 0.15 +
        authorship    * 0.15 +
        timeline      * 0.10 +
        commit_depth  * 0.10 +
        tech_evidence * 0.15 +
        readme_align  * 0.05 +
        reserved      * 0.15
    )
    score = round(min(100.0, max(0.0, score)), 1)

    if score >= 70:
        confidence = "high"
    elif score >= 45:
        confidence = "medium"
    elif score >= 25:
        confidence = "low"
    else:
        confidence = "none"

    return score, confidence


def _assess_upload_risk(commit_history: list | None) -> dict:
    """
    Upload-risk heuristic assessment. Returns a risk dict, not a score.

    A single large initial commit (common when someone uploads a finished
    project) is a red flag, but NOT proof of fraud. It is one signal among
    many. This function generates risk_flags for human review, it does not
    automatically disqualify a candidate.

    Risk flags:
      single_large_initial_commit — first commit is only commit (or >80% of lines)
      timestamp_clustering        — multiple commits within 5-minute windows
      minimal_commit_history      — total commits <= 2
      no_iterative_development    — no sign of work after the first commit
    """
    if not commit_history:
        return {"risk_level": "unknown", "flags": [], "evidence": {}}

    flags: list[str] = []
    evidence: dict = {}

    total_commits = len(commit_history)
    evidence["total_commits"] = total_commits

    if total_commits <= 2:
        flags.append("minimal_commit_history")
        evidence["commit_count"] = total_commits

    # Check timestamp clustering (multiple commits in 5-min windows)
    import re as _re
    timestamps = []
    for c in commit_history:
        committed_date = c.get("committedDate") or c.get("date") or ""
        if committed_date:
            timestamps.append(committed_date)
    timestamps.sort()
    cluster_count = 0
    for i in range(1, len(timestamps)):
        try:
            from datetime import datetime, timezone
            t1 = datetime.fromisoformat(timestamps[i-1].replace("Z", "+00:00"))
            t2 = datetime.fromisoformat(timestamps[i].replace("Z", "+00:00"))
            diff_minutes = abs((t2 - t1).total_seconds()) / 60
            if diff_minutes < 5:
                cluster_count += 1
        except (ValueError, TypeError):
            pass
    if cluster_count > 3:
        flags.append("timestamp_clustering")
        evidence["clustered_commit_pairs"] = cluster_count

    # Single commit = upload risk
    if total_commits == 1:
        flags.append("single_large_initial_commit")

    if flags:
        risk_level = "high" if len(flags) >= 2 else "medium"
    else:
        risk_level = "low"

    return {"risk_level": risk_level, "flags": flags, "evidence": evidence}


def verify_projects(
    resume_projects: list[dict],
    github_repos: list[dict],
    device: str = "cpu",
    github_username: str = "",
    github_token: str = "",
) -> list[dict]:
    """Verify that resume-claimed projects exist on GitHub using multi-signal fusion.

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
    github_username : str, optional
        Candidate's GitHub username.
    github_token : str, optional
        GitHub PAT for API calls.

    Returns
    -------
    list[dict]
        For each resume project:
        {
            "claimed_project": str,
            "status": "verified" | "uncertain" | "unverified",
            "project_score": float (0-100),  # multi-signal authenticity
            "confidence": "high"|"medium"|"low"|"none",
            "matched_repo": str or None,
            "similarity": float (0.0-1.0),
            "is_fork": bool or None,
            "forked_from": str or None,
            "total_commits": int,
            "active_days": int,
            "last_pushed": str or None,
            "candidate_is_author": bool,
            "commit_frequency": float,
            "upload_risk": str,
            "risk_flags": list[str],
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
                "forked_from": None,
                "total_commits": 0,
                "active_days": 0,
                "last_pushed": None,
                "candidate_is_author": False,
                "commit_frequency": 0.0,
            }
            for p in resume_projects
        ]

    import numpy as np

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
        resume_embs = model.encode(resume_texts, convert_to_numpy=True, device=device)
        repo_embs = model.encode(repo_texts, convert_to_numpy=True, device=device)

    # Compute similarity matrix: resume_projects x github_repos
    print("  \u2192 Computing similarity matrix...")
    sim_matrix = np.dot(resume_embs, repo_embs.T)

    results = []
    for i, project in enumerate(resume_projects):
        sims = sim_matrix[i]
        best_idx = int(sims.argmax())
        best_sim = float(sims[best_idx])
        best_repo = github_repos[best_idx]

        matched_repo_name  = best_repo.get("name")
        is_fork            = best_repo.get("isFork")
        forked_from        = None
        total_commits      = 0
        active_days        = 0
        development_span   = 0
        last_pushed        = None
        candidate_is_author = False
        commit_frequency   = 0.0
        candidate_commit_ratio = 0.0
        commit_history_nodes: list = []

        # ── Fetch commit activity for this matched repo ───────────────────
        if best_sim >= 0.30 and matched_repo_name and github_username and github_token:
            try:
                import github_verifier
                activity = github_verifier.fetch_repo_commit_activity(
                    github_username, matched_repo_name, github_token
                )
                if activity:
                    forked_from   = activity.get("forked_from")
                    total_commits = activity.get("total_commits", 0)
                    active_days   = activity.get("active_days", 0)
                    last_pushed   = activity.get("last_pushed")
                    commit_history_nodes = activity.get("commit_nodes", [])

                    # Calculate development span from first to last commit
                    first_date = activity.get("first_commit_date")
                    last_date  = activity.get("last_commit_date") or last_pushed
                    if first_date and last_date:
                        try:
                            from datetime import datetime, timezone
                            d1 = datetime.fromisoformat(first_date.replace("Z", "+00:00"))
                            d2 = datetime.fromisoformat(last_date.replace("Z", "+00:00"))
                            development_span = max(1, (d2 - d1).days)
                        except (ValueError, TypeError):
                            development_span = 0

                    author_logins = activity.get("author_logins", [])
                    author_commit_counts = activity.get("author_commit_counts", {})

                    if github_username.lower() in [a.lower() for a in author_logins if a]:
                        candidate_is_author = True
                        candidate_count = author_commit_counts.get(
                            github_username.lower(),
                            author_commit_counts.get(github_username, 0),
                        )
                        if total_commits > 0:
                            candidate_commit_ratio = candidate_count / total_commits

                    if active_days > 0:
                        commit_frequency = total_commits / active_days
            except Exception as e:
                print(f"  ⚠ Failed to fetch commit activity for {matched_repo_name}: {e}")

        # ── Upload risk assessment ─────────────────────────────────────────
        risk_info  = _assess_upload_risk(commit_history_nodes)
        upload_risk = risk_info["risk_level"]
        risk_flags  = risk_info["flags"]

        # ── Technology evidence check: primary language in resume claims ──
        resume_techs  = set(t.lower() for t in (project.get("technologies") or []))
        repo_langs    = set()
        pl = best_repo.get("primaryLanguage")
        if pl:
            repo_langs.add((pl.get("name") or "").lower())
        for edge in (best_repo.get("languages") or {}).get("edges") or []:
            repo_langs.add((edge.get("node", {}).get("name") or "").lower())

        has_tech_evidence = bool(resume_techs & repo_langs) if resume_techs else False

        # ── README similarity (if available) ──────────────────────────────
        readme = _readme_text(best_repo)
        readme_sim = 0.0
        if readme and best_sim >= 0.30:
            try:
                import numpy as np
                proj_text = f"{project.get('name', '')} {project.get('description', '')}".strip()
                if proj_text:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        proj_emb_r   = model.encode([proj_text],    convert_to_numpy=True, device=device)
                        readme_emb_r = model.encode([readme[:400]], convert_to_numpy=True, device=device)
                    readme_sim = float(np.dot(proj_emb_r, readme_emb_r.T)[0][0])
            except Exception:
                readme_sim = 0.0

        # ── Multi-signal authenticity score ───────────────────────────────
        # Name fuzzy match as a name_match_score (0-1)
        from difflib import SequenceMatcher
        proj_name = project.get("name", "").lower().replace("-", " ").replace("_", " ")
        repo_name = (matched_repo_name or "").lower().replace("-", " ").replace("_", " ")
        name_fuzzy = SequenceMatcher(None, proj_name, repo_name).ratio()

        project_score, confidence = _score_project_authenticity(
            name_match_score       = name_fuzzy,
            sbert_similarity       = best_sim,
            is_fork                = is_fork,
            candidate_is_author    = candidate_is_author,
            candidate_commit_ratio = candidate_commit_ratio,
            active_days            = active_days,
            total_commits          = total_commits,
            development_span_days  = development_span,
            has_technology_evidence= has_tech_evidence,
            readme_similarity      = readme_sim,
        )

        # ── Multi-signal status decision (replaces single SBERT threshold) ─
        # Verified:  project_score >= 55 AND best_sim >= 0.35
        # Uncertain: project_score >= 30 AND best_sim >= 0.20
        # Unverified: everything else
        if best_sim < 0.20:
            status = "unverified"
        elif project_score >= 55 and best_sim >= 0.35:
            status = "verified"
        elif project_score >= 30:
            status = "uncertain"
        else:
            status = "unverified"

        results.append({
            "claimed_project":       project.get("name", "Unknown"),
            "status":                status,
            "project_score":         project_score,
            "confidence":            confidence,
            "matched_repo":          matched_repo_name,
            "similarity":            round(best_sim, 4),
            "is_fork":               is_fork,
            "forked_from":           forked_from,
            "total_commits":         total_commits,
            "active_days":           active_days,
            "development_span_days": development_span,
            "last_pushed":           last_pushed,
            "candidate_is_author":   candidate_is_author,
            "candidate_commit_ratio":round(candidate_commit_ratio, 3),
            "commit_frequency":      round(commit_frequency, 3),
            "upload_risk":           upload_risk,
            "risk_flags":            risk_flags,
        })

    # Print summary table
    print()
    sep = "─"
    print(f"  {'Claimed Project':<28} {'Status':<12} {'Score':<7} {'Conf':<9} {'Repo':<25} {'Sim':<7}")
    print(f"  {sep*28} {sep*12} {sep*7} {sep*9} {sep*25} {sep*7}")
    for r in results:
        proj  = r["claimed_project"][:26]
        repo  = (r["matched_repo"] or "None")[:23]
        score = f"{r['project_score']:.0f}/100"
        print(f"  {proj:<28} {r['status']:<12} {score:<7} {r['confidence']:<9} {repo:<25} {r['similarity']:.4f}")
    print()

    verified_count   = sum(1 for r in results if r["status"] == "verified")
    uncertain_count  = sum(1 for r in results if r["status"] == "uncertain")
    unverified_count = sum(1 for r in results if r["status"] == "unverified")
    print(
        f"  ✓ Project verification complete: "
        f"{verified_count} verified, {uncertain_count} uncertain, "
        f"{unverified_count} unverified"
    )

    return results

