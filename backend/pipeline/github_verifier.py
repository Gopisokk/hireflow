"""
HireFlow GitHub Verifier — v2 (Project-Focused Scoring)
=========================================================
Scores candidates based on their ACTUAL PROJECT quality on GitHub,
not just overall language stats or activity heatmaps.

Scoring Philosophy:
  - Resume project found on GitHub?  → Core signal
  - Project is forked?               → Heavy penalty (not your work)
  - Commit count on that project     → HIGH weight (real effort)
  - Tech stack in that project       → MEDIUM weight (skill alignment)
  - Active days on that project      → LOW weight (easily gamed)
  - PR contributions                 → Bonus signal

Algorithm per matched project:
  raw_score = commit_score * 0.50
            + tech_match  * 0.35
            + days_score  * 0.15
  final_project_score = raw_score * fork_factor  (0.15 if forked, 1.0 if original)

Final GitHub Score = weighted_mean(project_scores) + pr_bonus
"""

import asyncio
import math
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Optional

import httpx

from config import GITHUB_API_URL


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class ProjectScore:
    """Detailed score breakdown for a single matched GitHub project."""
    resume_name: str           # Name from resume
    repo_name: str             # Matched GitHub repo name
    repo_url: str
    is_fork: bool
    commit_count: int
    active_days_on_repo: int   # unique commit days on THIS repo
    repo_languages: list[str]  # languages used in THIS repo
    tech_match_score: float    # 0-100: how well repo langs match JD
    commit_score: float        # 0-100: commit volume score
    days_score: float          # 0-100: active days score
    raw_score: float           # before fork penalty
    final_score: float         # after fork penalty
    fork_factor: float         # 1.0 original, 0.15 forked


@dataclass
class GitHubScore:
    """Final GitHub verification result for a candidate."""
    username: str
    github_score: float = 0.0          # 0-100 final score
    project_scores: list[ProjectScore] = field(default_factory=list)
    matched_projects: int = 0
    total_resume_projects: int = 0
    total_commits: int = 0
    total_prs: int = 0
    fork_ratio: float = 0.0
    active_days: int = 0               # overall (kept for DB compat)
    top_languages: list[str] = field(default_factory=list)
    repos: list[dict[str, Any]] = field(default_factory=list)  # DB compat
    error: Optional[str] = None


# ─── GraphQL Queries ──────────────────────────────────────────────────────────

# Step 1: Get repo list + overall stats
GQL_USER_REPOS = """
query($username: String!) {
  user(login: $username) {
    repositories(first: 100, orderBy: {field: UPDATED_AT, direction: DESC}, isFork: false) {
      totalCount
      nodes {
        name
        url
        isFork
        stargazerCount
        primaryLanguage { name }
        languages(first: 10) { nodes { name } }
        defaultBranchRef {
          target {
            ... on Commit {
              history(first: 1) { totalCount }
            }
          }
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
  }
}
"""

# Step 2: Per-repo deep dive — get commit dates to calculate active days
GQL_REPO_COMMITS = """
query($owner: String!, $name: String!) {
  repository(owner: $owner, name: $name) {
    isFork
    languages(first: 15) { nodes { name } }
    defaultBranchRef {
      target {
        ... on Commit {
          history(first: 100) {
            totalCount
            nodes {
              committedDate
              author { user { login } }
            }
          }
        }
      }
    }
  }
}
"""


# ─── Scoring helpers ──────────────────────────────────────────────────────────

# Words that carry no signal for matching
_STOP_WORDS = {
    "a", "an", "the", "and", "or", "for", "of", "in", "on", "to", "with",
    "using", "based", "built", "trained", "developed", "created", "implemented",
    "project", "system", "app", "application", "tool", "model", "two", "one",
    "three", "strong", "multiple", "various", "documents", "document",
}

_TECH_ALIASES: dict[str, set[str]] = {
    "ml":        {"machine learning", "machinelearning", "ml"},
    "ai":        {"artificial intelligence", "ai"},
    "nlp":       {"natural language", "nlp", "text"},
    "cv":        {"computer vision", "vision", "facial", "recognition"},
    "langchain": {"langchain", "lang chain", "llm", "rag", "pdf", "chat"},
    "pricing":   {"pricing", "price", "product", "amazon"},
    "chatbot":   {"chatbot", "chat", "csv", "pdf", "document"},
}


def _keywords(text: str) -> set[str]:
    """Extract meaningful keywords from a project/repo name."""
    # Split on non-alphanumeric (handles camelCase, kebab-case, snake_case)
    tokens = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)   # camelCase split
    tokens = re.sub(r"[^a-z0-9]+", " ", tokens.lower())
    return {t for t in tokens.split() if len(t) > 2 and t not in _STOP_WORDS}


def _keyword_overlap(a: str, b: str) -> float:
    """
    Jaccard-like keyword overlap between two name strings.
    Also checks tech alias expansion.
    Returns 0-1.
    """
    ka = _keywords(a)
    kb = _keywords(b)
    if not ka or not kb:
        return 0.0

    # Direct token overlap
    direct = len(ka & kb) / len(ka | kb)

    # Alias expansion: check if any keyword in 'a' is an alias for keywords in 'b'
    alias_bonus = 0.0
    for alias_key, alias_set in _TECH_ALIASES.items():
        a_has = any(t in alias_set for t in ka) or alias_key in ka
        b_has = any(t in alias_set for t in kb) or alias_key in kb
        if a_has and b_has:
            alias_bonus = max(alias_bonus, 0.35)

    return min(1.0, direct + alias_bonus)


def _fuzzy_match(a: str, b: str) -> float:
    """Sequence similarity (fallback for alias matching)."""
    a = re.sub(r"[^a-z0-9 ]", " ", a.lower()).strip()
    b = re.sub(r"[^a-z0-9 ]", " ", b.lower()).strip()
    return SequenceMatcher(None, a, b).ratio()


def _commit_score(count: int) -> float:
    """
    Log-scale commit score (0-100).
    Rationale: going from 0→10 commits is meaningful;
    going from 200→210 is not.
      0 commits  →  0
      5 commits  → 35
      20 commits → 60
      50 commits → 78
     100 commits → 88
     200 commits → 95
    """
    if count <= 0:
        return 0.0
    return min(100.0, math.log(count + 1, 2) * 14.0)


def _days_score(unique_days: int) -> float:
    """
    Active-days score (0-100) — deliberately low-weighted.
    A student can write 1 line per day to inflate this.
    We reward consistency but cap aggressively.
      0  days → 0
      5  days → 40
     10  days → 55
     20  days → 70
     50  days → 85
    """
    if unique_days <= 0:
        return 0.0
    return min(100.0, math.log(unique_days + 1, 2) * 17.0)


def _tech_match_score(repo_languages: list[str], jd_skills: list[str]) -> float:
    """
    How well the LANGUAGES IN THIS SPECIFIC REPO align with JD skills.
    Uses fuzzy matching so 'PyTorch' matches 'pytorch', etc.
    Returns 0-100.
    """
    if not repo_languages or not jd_skills:
        return 30.0  # neutral baseline — can't penalise unknown

    repo_lower = [l.lower() for l in repo_languages]
    jd_lower   = [s.lower() for s in jd_skills]

    matched = 0
    for jd_skill in jd_lower:
        for repo_lang in repo_lower:
            if _fuzzy_match(jd_skill, repo_lang) >= 0.75:
                matched += 1
                break

    # Score = fraction of JD skills covered, capped at 100
    return min(100.0, (matched / len(jd_lower)) * 100.0)


def _active_days_from_history(nodes: list[dict], github_username: str) -> int:
    """Count unique calendar days the candidate committed to THIS repo."""
    dates: set[str] = set()
    for node in nodes:
        author = node.get("author") or {}
        user = author.get("user") or {}
        login = (user.get("login") or "").lower()
        if login == github_username.lower():
            date_str = node.get("committedDate", "")[:10]  # YYYY-MM-DD
            if date_str:
                dates.add(date_str)
    return len(dates)


def _score_project(
    repo: dict,
    commit_nodes: list[dict],
    github_username: str,
    jd_skills: list[str],
    resume_project_name: str,
) -> ProjectScore:
    """
    Compute the full score for a single matched project.

    Weights:
      commits      → 50%  (hardest to fake, best effort proxy)
      tech_match   → 35%  (is this repo relevant to the JD?)
      active_days  → 15%  (consistency signal, but easily gamed)
    Fork factor:
      original    → ×1.00 (full score)
      forked      → ×0.15 (severe penalty — not their original work)
    """
    is_fork = repo.get("isFork", False)
    repo_languages = [n["name"] for n in (repo.get("languages") or {}).get("nodes", [])]
    commit_count = len(commit_nodes)
    active_days = _active_days_from_history(commit_nodes, github_username)

    c_score = _commit_score(commit_count)
    t_score = _tech_match_score(repo_languages, jd_skills)
    d_score = _days_score(active_days)

    raw = c_score * 0.50 + t_score * 0.35 + d_score * 0.15
    fork_factor = 0.15 if is_fork else 1.0
    final = round(raw * fork_factor, 2)

    return ProjectScore(
        resume_name=resume_project_name,
        repo_name=repo["name"],
        repo_url=repo.get("url", ""),
        is_fork=is_fork,
        commit_count=commit_count,
        active_days_on_repo=active_days,
        repo_languages=repo_languages,
        tech_match_score=round(t_score, 2),
        commit_score=round(c_score, 2),
        days_score=round(d_score, 2),
        raw_score=round(raw, 2),
        final_score=final,
        fork_factor=fork_factor,
    )


# ─── Main verifier ────────────────────────────────────────────────────────────

class GitHubVerifier:
    """
    Project-focused GitHub verification engine.

    For each resume project:
      1. Fuzzy-match it against the candidate's GitHub repos
      2. Fetch per-repo commit history (to get author-specific active days)
      3. Score each project individually
      4. Combine project scores into a final GitHub score

    Rate limiting: 1 request/second to respect GitHub API limits.
    """

    def __init__(self, requests_per_second: float = 1.0) -> None:
        self._min_delay = 1.0 / requests_per_second

    async def _gql(
        self,
        client: httpx.AsyncClient,
        token: str,
        query: str,
        variables: dict,
    ) -> dict:
        """Execute a GitHub GraphQL request with rate-limit delay."""
        await asyncio.sleep(self._min_delay)
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        resp = await client.post(
            GITHUB_API_URL,
            json={"query": query, "variables": variables},
            headers=headers,
            timeout=20.0,
        )
        resp.raise_for_status()
        body = resp.json()
        if "errors" in body:
            raise ValueError(f"GraphQL errors: {body['errors']}")
        return body.get("data", {})

    def _match_project_to_repo(
        self,
        resume_project: str,
        repos: list[dict],
        threshold: float = 0.20,
    ) -> Optional[dict]:
        """
        Find the best-matching GitHub repo for a resume project name.

        Strategy (in priority order):
          1. Keyword overlap with tech alias expansion (catches 'csv chatbot'
             matching 'Chat-with-multiple-PDF-Documents-using-Langchain')
          2. Sequence similarity as tiebreaker

        Threshold is intentionally low (0.20) because project names on
        resumes rarely match repo names exactly.
        """
        best_repo = None
        best_score = threshold

        for repo in repos:
            kw_score = _keyword_overlap(resume_project, repo["name"])
            seq_score = _fuzzy_match(resume_project, repo["name"]) * 0.5
            combined = kw_score * 0.7 + seq_score * 0.3
            if combined > best_score:
                best_score = combined
                best_repo = repo

        return best_repo

    async def verify_candidate(
        self,
        token: str,
        username: str,
        resume_projects: list[str],
        jd_skills: list[str],
    ) -> GitHubScore:
        """
        Full verification pipeline for one candidate.

        Steps:
          1. Fetch all repos + overall contribution stats
          2. For each resume project, fuzzy-match to a GitHub repo
          3. Fetch per-repo commit history for matched repos
          4. Score each project (commits, tech, active_days, fork)
          5. Aggregate into final GitHub score
        """
        result = GitHubScore(username=username)

        if not username:
            result.error = "No GitHub username provided"
            return result

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:

                # ── Step 1: User repos + overall stats ────────────────────────
                data = await self._gql(
                    client, token, GQL_USER_REPOS, {"username": username}
                )
                user = data.get("user")
                if not user:
                    result.error = f"GitHub user '{username}' not found"
                    return result

                repos = user["repositories"]["nodes"]
                total_repo_count = user["repositories"]["totalCount"]
                contrib = user["contributionsCollection"]
                result.total_commits = contrib.get("totalCommitContributions", 0)
                result.total_prs = contrib.get("totalPullRequestContributions", 0)

                # Overall active days (kept for DB, low-weighted below)
                cal = contrib.get("contributionCalendar", {})
                active_days = sum(
                    day["contributionCount"] > 0
                    for week in cal.get("weeks", [])
                    for day in week.get("contributionDays", [])
                )
                result.active_days = active_days

                # Fork ratio across ALL repos
                all_repos_data = user["repositories"]["nodes"]
                forked_count = sum(1 for r in all_repos_data if r.get("isFork"))
                result.fork_ratio = round(
                    forked_count / total_repo_count if total_repo_count else 0.0, 4
                )

                # Top languages (global — kept for DB compat)
                lang_freq: dict[str, int] = {}
                for repo in repos:
                    for lang_node in (repo.get("languages") or {}).get("nodes", []):
                        lang_freq[lang_node["name"].lower()] = (
                            lang_freq.get(lang_node["name"].lower(), 0) + 1
                        )
                result.top_languages = sorted(lang_freq, key=lang_freq.get, reverse=True)[:10]  # type: ignore[arg-type]

                # ── Step 2+3+4: Per-project deep scoring ──────────────────────
                result.total_resume_projects = len(resume_projects)
                project_scores: list[ProjectScore] = []

                for resume_proj in resume_projects:
                    matched_repo = self._match_project_to_repo(resume_proj, repos)
                    if not matched_repo:
                        continue  # No GitHub repo found for this project

                    # Fetch per-repo commit history
                    try:
                        repo_data = await self._gql(
                            client, token, GQL_REPO_COMMITS,
                            {"owner": username, "name": matched_repo["name"]}
                        )
                        repo_detail = repo_data.get("repository", {})
                        commit_nodes = (
                            repo_detail
                            .get("defaultBranchRef", {})
                            .get("target", {})
                            .get("history", {})
                            .get("nodes", [])
                        )
                        # Merge language data from detail (more complete)
                        if repo_detail.get("languages"):
                            matched_repo["languages"] = repo_detail["languages"]
                        if repo_detail.get("isFork") is not None:
                            matched_repo["isFork"] = repo_detail["isFork"]
                    except Exception:
                        # Fallback: use summary data (no per-author day counting)
                        commit_nodes = []

                    ps = _score_project(
                        repo=matched_repo,
                        commit_nodes=commit_nodes,
                        github_username=username,
                        jd_skills=jd_skills,
                        resume_project_name=resume_proj,
                    )
                    project_scores.append(ps)

                result.project_scores = project_scores
                result.matched_projects = len(project_scores)

                # Build legacy `repos` list for DB compatibility
                result.repos = [
                    {
                        "resume_project": ps.resume_name,
                        "url": ps.repo_url,
                        "is_fork": ps.is_fork,
                    }
                    for ps in project_scores
                ]

                # ── Step 5: Aggregate final GitHub score ──────────────────────
                result.github_score = _aggregate_score(
                    project_scores=project_scores,
                    total_resume_projects=len(resume_projects),
                    total_prs=result.total_prs,
                    total_commits=result.total_commits,
                )

        except httpx.HTTPStatusError as exc:
            result.error = f"GitHub API HTTP error: {exc.response.status_code}"
        except Exception as exc:
            result.error = str(exc)

        return result


def _aggregate_score(
    project_scores: list[ProjectScore],
    total_resume_projects: int,
    total_prs: int,
    total_commits: int,
) -> float:
    """
    Combine per-project scores into a final GitHub score (0-100).

    Formula:
      base     = weighted average of matched project scores
                 (higher-scoring projects weighted more)
      coverage = fraction of resume projects found on GitHub
      pr_bonus = small bonus for pull request contributions
      commit_bonus = small bonus if overall commits are substantial

    Final = base * coverage_factor + pr_bonus + commit_bonus
    """
    if not project_scores:
        # No projects matched at all — penalise heavily
        # Still give a tiny bonus for having commits/PRs
        pr_bonus = min(8.0, total_prs * 0.5)
        commit_bonus = min(5.0, math.log(total_commits + 1, 2))
        return round(pr_bonus + commit_bonus, 2)

    # Weighted average: weight each project by its own raw score
    # (so a great project counts more than a tiny one)
    weights = [max(ps.raw_score, 1.0) for ps in project_scores]
    total_weight = sum(weights)
    base = sum(ps.final_score * w for ps, w in zip(project_scores, weights)) / total_weight

    # Coverage penalty: if 3 projects on resume but only 1 found on GitHub,
    # the candidate may have inflated their resume
    coverage = len(project_scores) / max(total_resume_projects, 1)
    coverage_factor = 0.5 + 0.5 * coverage  # range [0.5, 1.0]

    # Small bonuses
    pr_bonus = min(8.0, total_prs * 0.4)         # max 8 pts for PRs
    commit_bonus = min(5.0, math.log(total_commits + 1, 2) * 0.6)  # max 5 pts

    final = base * coverage_factor + pr_bonus + commit_bonus
    return round(min(100.0, final), 2)
