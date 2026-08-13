"""
project_authenticator.py — HireFlow-Lite Advanced Project Authenticity Engine (Option B)
======================================================================================
Modular pipeline to verify claimed resume projects against GitHub evidence.

Architecture Components:
  1. HistoryAnalyzer: Analyzes commit density, development span, active days, and upload risk flags.
  2. AuthorshipAnalyzer: Verifies commit author logins, contribution ratio, and post-fork commit depth.
  3. CodeTechnologyAnalyzer: Cross-checks resume tech claims against repository languages, topics, and README.
  4. ProjectAuthenticityEngine: Combines all analyzer signals into a weighted 0-100 authenticity score,
     confidence level, and verification report.
"""

from __future__ import annotations

import sys
import re
from typing import Any, Dict, List, Optional, Tuple, Set

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


# ── 1. HistoryAnalyzer ────────────────────────────────────────────────────────

class HistoryAnalyzer:
    """Analyzes repository development history, timeline, and commit patterns."""

    @staticmethod
    def analyze(activity_data: dict) -> dict:
        total_commits = activity_data.get("total_commits", 0)
        active_days = activity_data.get("active_days", 0)
        pushed_at = activity_data.get("pushed_at") or activity_data.get("last_pushed")
        created_at = activity_data.get("created_at")

        span_days = 0
        if created_at and pushed_at:
            try:
                from datetime import datetime, timezone
                c_dt = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
                p_dt = datetime.fromisoformat(str(pushed_at).replace("Z", "+00:00"))
                span_days = max(1, (p_dt - c_dt).days)
            except Exception:
                span_days = 0

        commit_density = round(total_commits / max(1, active_days), 2) if active_days > 0 else 0.0

        # Upload risk heuristics
        flags: list[str] = []
        if total_commits <= 2 and total_commits > 0:
            flags.append("minimal_commit_history")
        if total_commits == 1:
            flags.append("single_large_initial_commit")
        if active_days == 1 and total_commits > 5:
            flags.append("single_day_bulk_upload")

        if len(flags) >= 2:
            risk_level = "high"
        elif len(flags) == 1:
            risk_level = "medium"
        else:
            risk_level = "low"

        return {
            "total_commits": total_commits,
            "active_days": active_days,
            "span_days": span_days,
            "commit_density": commit_density,
            "upload_risk": risk_level,
            "risk_flags": flags,
        }


# ── 2. AuthorshipAnalyzer ──────────────────────────────────────────────────────

class AuthorshipAnalyzer:
    """Verifies commit authorship logins, fork post-analysis, and contribution share."""

    @staticmethod
    def analyze(activity_data: dict, repo_data: dict, candidate_username: str) -> dict:
        candidate = candidate_username.lower().strip()
        author_logins: set = activity_data.get("author_logins", set())
        if isinstance(author_logins, (list, tuple)):
            author_logins = set(author_logins)

        candidate_is_author = False
        if candidate:
            candidate_is_author = any(a.lower() == candidate for a in author_logins if isinstance(a, str))

        total_commits = activity_data.get("total_commits", 0)
        commits_by_author = activity_data.get("commits_by_author", {})
        
        candidate_commits = 0
        if candidate and isinstance(commits_by_author, dict):
            for login, count in commits_by_author.items():
                if str(login).lower() == candidate:
                    candidate_commits += count

        commit_ratio = round(candidate_commits / max(1, total_commits), 3) if total_commits > 0 else (1.0 if candidate_is_author else 0.0)

        # Fork classification
        is_fork = repo_data.get("isFork", False)
        if not is_fork:
            fork_status = "original"
        elif commit_ratio >= 0.50:
            fork_status = "fork_substantial"
        elif commit_ratio >= 0.20:
            fork_status = "fork_minimal"
        else:
            fork_status = "fork_none"

        return {
            "candidate_is_author": candidate_is_author,
            "candidate_commits": candidate_commits,
            "candidate_commit_ratio": commit_ratio,
            "is_fork": is_fork,
            "fork_status": fork_status,
            "forked_from": activity_data.get("forked_from"),
        }


# ── 3. CodeTechnologyAnalyzer ───────────────────────────────────────────────

class CodeTechnologyAnalyzer:
    """Cross-checks resume technology claims against repository code signals."""

    @staticmethod
    def analyze(project_techs: list[str], repo_data: dict) -> dict:
        claimed = [t.lower().strip() for t in project_techs if t]
        
        repo_techs: set[str] = set()

        # Primary language
        pl = repo_data.get("primaryLanguage")
        if isinstance(pl, dict) and pl.get("name"):
            repo_techs.add(pl["name"].lower())

        # Languages list
        langs = repo_data.get("languages")
        if isinstance(langs, dict):
            for edge in langs.get("edges", []):
                n = edge.get("node", {}).get("name")
                if n:
                    repo_techs.add(n.lower())

        # Topics
        topics = repo_data.get("repositoryTopics") or repo_data.get("topics")
        if isinstance(topics, dict):
            for node in topics.get("nodes", []):
                tname = node.get("topic", {}).get("name")
                if tname:
                    repo_techs.add(tname.lower())

        # Description / README text snippet
        desc = (repo_data.get("description") or "").lower()
        obj = repo_data.get("object")
        readme = obj.get("text", "").lower() if isinstance(obj, dict) else ""
        combined_text = f"{desc} {readme[:1000]}"

        matched: list[str] = []
        for tech in claimed:
            if tech in repo_techs or tech in combined_text:
                matched.append(tech)

        if not claimed:
            tech_score = 50.0   # neutral
        elif matched:
            tech_score = round(min(100.0, (len(matched) / len(claimed)) * 100.0), 1)
        else:
            tech_score = 10.0

        return {
            "claimed_technologies": claimed,
            "matched_technologies": matched,
            "technology_evidence_score": tech_score,
            "repo_detected_technologies": sorted(list(repo_techs)),
        }


# ── 4. ProjectAuthenticityEngine ─────────────────────────────────────────────

class ProjectAuthenticityEngine:
    """Orchestrates modular analyzers to compute final Option B authenticity score."""

    @staticmethod
    def authenticate(
        claimed_project_name: str,
        claimed_description: str,
        claimed_technologies: list[str],
        github_repo: dict,
        commit_activity: dict,
        candidate_username: str = "",
        name_match_score: float = 0.0,
        sbert_similarity: float = 0.0,
    ) -> dict:
        history = HistoryAnalyzer.analyze(commit_activity)
        authorship = AuthorshipAnalyzer.analyze(commit_activity, github_repo, candidate_username)
        code_tech = CodeTechnologyAnalyzer.analyze(claimed_technologies, github_repo)

        # ── Weighted Authenticity Score Calculation ────────────────────────────
        # 15% Repo Match (fuzzy + SBERT)
        match_val = ((name_match_score * 0.6) + (sbert_similarity * 0.4)) * 100.0

        # 15% Ownership / Fork status
        if authorship["fork_status"] == "original":
            ownership_val = 100.0
        elif authorship["fork_status"] == "fork_substantial":
            ownership_val = 65.0
        elif authorship["fork_status"] == "fork_minimal":
            ownership_val = 35.0
        else:
            ownership_val = 10.0

        # 15% Authorship ratio
        auth_val = min(100.0, authorship["candidate_commit_ratio"] * 100.0 * 1.2) if authorship["candidate_commit_ratio"] > 0 else (60.0 if authorship["candidate_is_author"] else 10.0)

        # 10% Timeline & Active Days
        span_sig = min(1.0, history["span_days"] / 90.0)
        active_sig = min(1.0, history["active_days"] / 30.0)
        timeline_val = ((span_sig * 0.35) + (active_sig * 0.65)) * 100.0

        # 10% Commit Depth
        commit_val = min(100.0, (history["total_commits"] / 20.0) * 100.0)

        # 15% Tech Evidence
        tech_val = code_tech["technology_evidence_score"]

        # 5% Quality Signals (README, tests, topics)
        has_readme = bool(github_repo.get("object"))
        has_topics = bool(github_repo.get("repositoryTopics") or github_repo.get("topics"))
        quality_val = (50.0 if has_readme else 0.0) + (50.0 if has_topics else 0.0)

        # 15% Reserved (Question-Answer placeholder, neutral 50)
        reserved_val = 50.0

        final_score = (
            match_val    * 0.15 +
            ownership_val * 0.15 +
            auth_val     * 0.15 +
            timeline_val * 0.10 +
            commit_val   * 0.10 +
            tech_val     * 0.15 +
            quality_val  * 0.05 +
            reserved_val * 0.15
        )
        final_score = round(max(0.0, min(100.0, final_score)), 1)

        # Assign confidence
        if final_score >= 70:
            confidence = "high"
        elif final_score >= 45:
            confidence = "medium"
        elif final_score >= 25:
            confidence = "low"
        else:
            confidence = "none"

        # Determine verification status
        if sbert_similarity < 0.20 and name_match_score < 0.40:
            status = "unverified"
        elif final_score >= 55 and (sbert_similarity >= 0.35 or name_match_score >= 0.70):
            status = "verified"
        elif final_score >= 30:
            status = "uncertain"
        else:
            status = "unverified"

        return {
            "claimed_project": claimed_project_name,
            "status": status,
            "project_score": final_score,
            "confidence": confidence,
            "matched_repo": github_repo.get("name"),
            "similarity": round(sbert_similarity, 4),
            "name_match_score": round(name_match_score, 4),
            "is_fork": authorship["is_fork"],
            "fork_status": authorship["fork_status"],
            "forked_from": authorship["forked_from"],
            "candidate_is_author": authorship["candidate_is_author"],
            "candidate_commit_ratio": authorship["candidate_commit_ratio"],
            "total_commits": history["total_commits"],
            "active_days": history["active_days"],
            "development_span_days": history["span_days"],
            "commit_density": history["commit_density"],
            "upload_risk": history["upload_risk"],
            "risk_flags": history["risk_flags"],
            "technology_evidence": code_tech["matched_technologies"],
            "technology_evidence_score": code_tech["technology_evidence_score"],
        }


# ── Top-level API helper ──────────────────────────────────────────────────────

def authenticate_project(
    project: dict,
    github_repo: dict,
    commit_activity: dict,
    candidate_username: str = "",
    name_match_score: float = 0.0,
    sbert_similarity: float = 0.0,
) -> dict:
    """Convenience wrapper for single project authentication."""
    p_name = project.get("name", "Unknown") if isinstance(project, dict) else str(project)
    p_desc = project.get("description", "") if isinstance(project, dict) else ""
    p_techs = project.get("technologies", []) if isinstance(project, dict) else []

    return ProjectAuthenticityEngine.authenticate(
        claimed_project_name=p_name,
        claimed_description=p_desc,
        claimed_technologies=p_techs,
        github_repo=github_repo,
        commit_activity=commit_activity,
        candidate_username=candidate_username,
        name_match_score=name_match_score,
        sbert_similarity=sbert_similarity,
    )
