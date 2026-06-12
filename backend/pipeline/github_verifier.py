"""
HireFlow GitHub Verifier
--------------------------
Async service that queries the GitHub GraphQL API to:
  - Verify resume projects against real GitHub repos.
  - Compute a composite github_score (0–100) based on:
      • project match score
      • fork penalty
      • active days score
      • language alignment
      • commit volume score
  - Respects rate limits (1 req/sec with retry).
"""

import asyncio
import math
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Optional

import httpx

from config import GITHUB_API_URL


# ═══════════════════════════════════════════════════════════════════════════════
#  Data classes
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GitHubScore:
    """Container for the GitHub verification result."""
    username: str
    project_match_score: float = 0.0
    fork_penalty: float = 0.0
    active_days_score: float = 0.0
    language_alignment: float = 0.0
    commit_volume_score: float = 0.0
    github_score: float = 0.0
    active_days: int = 0
    total_commits: int = 0
    total_prs: int = 0
    fork_ratio: float = 0.0
    top_languages: list[str] = field(default_factory=list)
    repos: list[dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════════
#  GraphQL query (mirrors existing route.js)
# ═══════════════════════════════════════════════════════════════════════════════

GITHUB_GRAPHQL_QUERY = """
query($username: String!) {
  user(login: $username) {
    repositories(first: 100, orderBy: {field: UPDATED_AT, direction: DESC}) {
      totalCount
      nodes {
        name
        isFork
        url
        stargazerCount
        primaryLanguage { name }
        updatedAt
        languages(first: 5) {
          nodes { name }
        }
      }
    }
    contributionsCollection {
      totalCommitContributions
      totalPullRequestContributions
      contributionCalendar {
        totalContributions
        weeks {
          contributionDays {
            contributionCount
            date
          }
        }
      }
    }
    pullRequests(first: 1) { totalCount }
  }
}
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  Helper functions
# ═══════════════════════════════════════════════════════════════════════════════

def _fuzzy_match(a: str, b: str) -> float:
    """Return a similarity ratio between 0 and 1 for two strings."""
    a_clean = re.sub(r"[^a-z0-9]", "", a.lower())
    b_clean = re.sub(r"[^a-z0-9]", "", b.lower())
    if not a_clean or not b_clean:
        return 0.0
    return SequenceMatcher(None, a_clean, b_clean).ratio()


def _jaccard(set_a: set[str], set_b: set[str]) -> float:
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 0.0


def _sigmoid_scale(value: float, midpoint: float, steepness: float = 1.0) -> float:
    """Scale a value to 0-1 using a sigmoid curve."""
    x = steepness * (value - midpoint) / midpoint if midpoint > 0 else 0
    return 1.0 / (1.0 + math.exp(-x))


# ═══════════════════════════════════════════════════════════════════════════════
#  Main verifier class
# ═══════════════════════════════════════════════════════════════════════════════

class GitHubVerifier:
    """
    Async GitHub profile verifier.
    Uses httpx.AsyncClient for HTTP requests with rate limiting.
    """

    def __init__(self, requests_per_second: float = 1.0) -> None:
        self._delay = 1.0 / requests_per_second
        self._last_request_time: float = 0.0

    async def _rate_limit(self) -> None:
        """Enforce minimum delay between requests."""
        now = asyncio.get_event_loop().time()
        elapsed = now - self._last_request_time
        if elapsed < self._delay:
            await asyncio.sleep(self._delay - elapsed)
        self._last_request_time = asyncio.get_event_loop().time()

    async def _query_github(
        self,
        client: httpx.AsyncClient,
        token: str,
        username: str,
        max_retries: int = 3,
    ) -> dict[str, Any]:
        """Execute the GraphQL query with retries."""
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        for attempt in range(max_retries):
            await self._rate_limit()
            try:
                response = await client.post(
                    GITHUB_API_URL,
                    headers=headers,
                    json={"query": GITHUB_GRAPHQL_QUERY, "variables": {"username": username}},
                    timeout=30.0,
                )
                if response.status_code == 200:
                    data = response.json()
                    if "errors" in data:
                        error_msg = data["errors"][0].get("message", "Unknown GraphQL error")
                        raise ValueError(f"GraphQL error: {error_msg}")
                    return data.get("data", {})
                elif response.status_code == 403:
                    # Rate limited — wait and retry
                    retry_after = int(response.headers.get("Retry-After", "60"))
                    await asyncio.sleep(min(retry_after, 60))
                elif response.status_code == 502:
                    await asyncio.sleep(2 ** attempt)
                else:
                    response.raise_for_status()
            except httpx.TimeoutException:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)

        return {}

    def _compute_project_match(
        self,
        resume_projects: list[str],
        github_repos: list[dict[str, Any]],
    ) -> tuple[float, list[dict[str, Any]]]:
        """
        Fuzzy-match resume projects to GitHub repos.
        Returns (score 0-100, list of matched repos with details).
        """
        if not resume_projects or not github_repos:
            return 0.0, []

        repo_names = [r.get("name", "") for r in github_repos]
        matched: list[dict[str, Any]] = []
        total_score = 0.0

        for project in resume_projects:
            best_ratio = 0.0
            best_repo: Optional[dict[str, Any]] = None
            for i, repo_name in enumerate(repo_names):
                ratio = _fuzzy_match(project, repo_name)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_repo = github_repos[i]

            if best_ratio >= 0.5 and best_repo is not None:
                matched.append({
                    "resume_project": project,
                    "github_repo": best_repo.get("name", ""),
                    "url": best_repo.get("url", ""),
                    "is_fork": best_repo.get("isFork", False),
                    "match_ratio": round(best_ratio, 3),
                })
                total_score += best_ratio

        if not resume_projects:
            return 0.0, matched

        match_rate = total_score / len(resume_projects)
        return round(min(match_rate * 100, 100), 2), matched

    def _compute_fork_penalty(self, repos: list[dict[str, Any]]) -> float:
        """Return fork ratio (0 = all original, 1 = all forks)."""
        if not repos:
            return 0.0
        forked = sum(1 for r in repos if r.get("isFork", False))
        return round(forked / len(repos), 4)

    def _compute_active_days(self, contributions_data: dict[str, Any]) -> tuple[int, float]:
        """Count active days in last 12 months, return (days, score 0-100)."""
        calendar = contributions_data.get("contributionCalendar", {})
        weeks = calendar.get("weeks", [])
        active_days = 0
        for week in weeks:
            for day in week.get("contributionDays", []):
                if day.get("contributionCount", 0) > 0:
                    active_days += 1
        # Score: active_days / 365 * 100, capped at 100
        score = min(active_days / 365.0 * 100, 100.0)
        return active_days, round(score, 2)

    def _compute_language_alignment(
        self,
        github_repos: list[dict[str, Any]],
        jd_languages: list[str],
    ) -> tuple[list[str], float]:
        """Jaccard similarity of GitHub languages vs JD languages."""
        github_langs: set[str] = set()
        for repo in github_repos:
            primary = repo.get("primaryLanguage")
            if primary and primary.get("name"):
                github_langs.add(primary["name"].lower())
            for lang_node in repo.get("languages", {}).get("nodes", []):
                if lang_node.get("name"):
                    github_langs.add(lang_node["name"].lower())

        jd_set = {lang.lower() for lang in jd_languages}
        jaccard = _jaccard(github_langs, jd_set)
        top_langs = sorted(github_langs)[:10]
        return top_langs, round(jaccard * 100, 2)

    def _compute_commit_volume(self, contributions_data: dict[str, Any]) -> tuple[int, float]:
        """Score based on total commits in last year."""
        total = contributions_data.get("totalCommitContributions", 0)
        # Sigmoid scaling: 200 commits = ~50 score, 500+ = ~90+
        score = _sigmoid_scale(total, midpoint=200, steepness=3.0) * 100
        return total, round(min(score, 100), 2)

    async def verify_candidate(
        self,
        token: str,
        username: str,
        resume_projects: list[str],
        jd_languages: list[str],
    ) -> GitHubScore:
        """
        Full GitHub verification for a single candidate.

        Parameters
        ----------
        token : str
            GitHub personal access token.
        username : str
            GitHub username.
        resume_projects : list[str]
            Project names extracted from the candidate's resume.
        jd_languages : list[str]
            Programming languages mentioned in the job description.

        Returns
        -------
        GitHubScore
            Composite score and breakdown.
        """
        result = GitHubScore(username=username)

        if not username or not username.strip():
            result.error = "No GitHub username provided"
            return result

        try:
            async with httpx.AsyncClient() as client:
                data = await self._query_github(client, token, username)

            user_data = data.get("user")
            if not user_data:
                result.error = f"GitHub user '{username}' not found"
                return result

            repos_data = user_data.get("repositories", {})
            repos = repos_data.get("nodes", [])
            contributions = user_data.get("contributionsCollection", {})

            # Project match
            result.project_match_score, matched_repos = self._compute_project_match(
                resume_projects, repos
            )
            result.repos = matched_repos

            # Fork penalty
            result.fork_ratio = self._compute_fork_penalty(repos)
            result.fork_penalty = result.fork_ratio * 100

            # Active days
            result.active_days, result.active_days_score = self._compute_active_days(
                contributions
            )

            # Language alignment
            result.top_languages, result.language_alignment = self._compute_language_alignment(
                repos, jd_languages
            )

            # Commit volume
            result.total_commits, result.commit_volume_score = self._compute_commit_volume(
                contributions
            )

            # PRs
            result.total_prs = contributions.get("totalPullRequestContributions", 0)

            # ── Composite score ──────────────────────────────────────────────
            # Weighted combination:
            #   30% project match
            #   10% fork penalty (inverted: less forks = better)
            #   25% active days
            #   15% language alignment
            #   20% commit volume
            composite = (
                0.30 * result.project_match_score
                + 0.10 * (100 - result.fork_penalty)
                + 0.25 * result.active_days_score
                + 0.15 * result.language_alignment
                + 0.20 * result.commit_volume_score
            )
            result.github_score = round(min(max(composite, 0), 100), 2)

        except Exception as exc:
            result.error = str(exc)

        return result
