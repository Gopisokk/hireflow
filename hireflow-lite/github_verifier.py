"""
github_verifier.py — HireFlow-Lite GitHub Profile Verification Module

Standalone, fully synchronous module that verifies a GitHub profile against
a resume using 40 individually-scored factors organized into 7 categories.
Uses a single GraphQL API call, caches responses for 24 hours, and relies
on httpx synchronous client (no async).
"""

import json
import math
import os
import re
import inspect
import sys
from datetime import datetime, timezone, timedelta
from difflib import SequenceMatcher
from pathlib import Path

import httpx

# Force UTF-8 stdout/stderr on Windows to avoid UnicodeEncodeError
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

GITHUB_GRAPHQL_ENDPOINT = "https://api.github.com/graphql"
CACHE_DIR = Path(__file__).parent / "github_cache"
CACHE_MAX_AGE = timedelta(hours=24)

GRAPHQL_QUERY = """
query GetProfile($login: String!) {
  user(login: $login) {
    name
    email
    createdAt
    bio
    avatarUrl
    location
    websiteUrl
    isHireable
    organizations(first: 5) { nodes { name } }
    followers { totalCount }
    contributionsCollection {
      totalCommitContributions
      totalPullRequestContributions
      totalIssueContributions
      contributionCalendar {
        totalContributions
        weeks {
          contributionDays {
            date
            contributionCount
          }
        }
      }
    }
    pinnedItems(first: 6, types: REPOSITORY) {
      nodes {
        ... on Repository {
          name
          description
          stargazerCount
          forkCount
          isFork
        }
      }
    }
    repositories(first: 30, orderBy: {field: UPDATED_AT, direction: DESC}) {
      nodes {
        name
        description
        isFork
        stargazerCount
        forkCount
        createdAt
        updatedAt
        primaryLanguage { name }
        languages(first: 8) {
          edges { size node { name } }
        }
        repositoryTopics(first: 5) { nodes { topic { name } } }
        defaultBranchRef {
          target {
            ... on Commit {
              history(first: 1) { totalCount }
            }
          }
        }
        object(expression: "HEAD:README.md") {
          ... on Blob { text }
        }
      }
    }
  }
}
"""

# ──────────────────────────────────────────────────────────────────────────────
# Caching Utilities
# ──────────────────────────────────────────────────────────────────────────────


def _cache_path(username: str) -> Path:
    """Return the cache file path for a given GitHub username."""
    return CACHE_DIR / f"{username.lower()}.json"


def _read_cache(username: str) -> dict | None:
    """Return cached user data if it exists and is fresh, else None."""
    path = _cache_path(username)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        cached_at = datetime.fromisoformat(raw.get("_cached_at", ""))
        if datetime.now(timezone.utc) - cached_at < CACHE_MAX_AGE:
            return raw.get("user")
    except (json.JSONDecodeError, ValueError, KeyError):
        pass
    return None


def _write_cache(username: str, user_data: dict) -> None:
    """Persist the GraphQL user payload to disk with a timestamp."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "_cached_at": datetime.now(timezone.utc).isoformat(),
        "user": user_data,
    }
    _cache_path(username).write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )


# ──────────────────────────────────────────────────────────────────────────────
# API Fetching
# ──────────────────────────────────────────────────────────────────────────────


def fetch_github_profile(username: str, token: str) -> dict:
    """
    Fetch a GitHub user profile via the GraphQL API (synchronous).

    Returns the ``data.user`` dict from the GraphQL response.
    Raises ``RuntimeError`` on HTTP or GraphQL-level errors.
    """
    # Check cache first
    cached = _read_cache(username)
    if cached is not None:
        print(f"  → Using cached GitHub data for {username}")
        return cached

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    body = {"query": GRAPHQL_QUERY, "variables": {"login": username}}

    import time
    while True:
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(GITHUB_GRAPHQL_ENDPOINT, headers=headers, json=body)
        except httpx.RequestError as exc:
            raise RuntimeError(f"Network error contacting GitHub API: {exc}") from exc

        # --- HTTP-level errors ---------------------------------------------------
        if resp.status_code == 401:
            raise RuntimeError(
                "GitHub API returned 401 Unauthorized — your token is invalid or expired."
            )
        if resp.status_code == 403:
            remaining = resp.headers.get("X-RateLimit-Remaining")
            # If remaining is 0 or "rate limit" is in response body, wait
            if remaining == "0" or "rate limit exceeded" in resp.text.lower():
                reset = resp.headers.get("X-RateLimit-Reset")
                try:
                    reset_time = int(reset) if reset else int(time.time()) + 60
                except ValueError:
                    reset_time = int(time.time()) + 60
                
                wait_seconds = max(reset_time - int(time.time()) + 5, 10)
                print(f"\n  → [GitHub API] Rate limit exceeded (remaining={remaining}).")
                print(f"  → Resets in {wait_seconds} seconds. Pausing pipeline...")
                time.sleep(wait_seconds)
                print("  → Resuming and retrying query...\n")
                continue
            else:
                raise RuntimeError(
                    f"GitHub API returned 403 Forbidden: {resp.text[:300]}"
                )
        if resp.status_code != 200:
            raise RuntimeError(
                f"GitHub API returned HTTP {resp.status_code}: {resp.text[:300]}"
            )
        break

    # --- GraphQL-level errors ------------------------------------------------
    data = resp.json()
    if "errors" in data:
        msgs = "; ".join(e.get("message", "") for e in data["errors"])
        raise RuntimeError(f"GitHub GraphQL errors: {msgs}")

    user = data.get("data", {}).get("user")
    if user is None:
        raise RuntimeError(
            f"GitHub user '{username}' not found (the GraphQL response returned null)."
        )

    _write_cache(username, user)
    return user


# ──────────────────────────────────────────────────────────────────────────────
# Internal Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _clamp(value: float, lo: float = 0.0, hi: float = 10.0) -> float:
    return max(lo, min(hi, value))


def _repos(data: dict) -> list[dict]:
    return (data.get("repositories") or {}).get("nodes") or []


def _non_fork_repos(data: dict) -> list[dict]:
    return [r for r in _repos(data) if not r.get("isFork")]


def _contribution_days(data: dict) -> list[dict]:
    """Flatten all contribution calendar days into a list."""
    cc = data.get("contributionsCollection") or {}
    cal = cc.get("contributionCalendar") or {}
    days: list[dict] = []
    for week in cal.get("weeks") or []:
        days.extend(week.get("contributionDays") or [])
    return days


def _all_languages(data: dict) -> set[str]:
    langs: set[str] = set()
    for repo in _repos(data):
        pl = repo.get("primaryLanguage")
        if pl:
            langs.add(pl["name"].lower())
        for edge in (repo.get("languages") or {}).get("edges") or []:
            langs.add(edge["node"]["name"].lower())
    return langs


def _extract_jd_languages(jd_text: str) -> set[str]:
    """Simple keyword extraction of programming language names from JD text."""
    known = {
        "python", "javascript", "typescript", "java", "kotlin", "swift",
        "c", "c++", "c#", "go", "rust", "ruby", "php", "scala", "dart",
        "r", "matlab", "perl", "lua", "haskell", "elixir", "erlang",
        "objective-c", "shell", "bash", "powershell", "sql", "html",
        "css", "scss", "sass", "vue", "react", "angular", "node",
        "django", "flask", "spring", "rails", "nextjs", "nuxt",
        "terraform", "docker", "kubernetes",
    }
    words = set(re.findall(r"[a-z#+]+", jd_text.lower()))
    return words & known


def _fuzzy_match_score(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def _repo_commit_count(repo: dict) -> int:
    ref = repo.get("defaultBranchRef") or {}
    target = ref.get("target") or {}
    hist = target.get("history") or {}
    return hist.get("totalCount", 0)


def _readme_text(repo: dict) -> str | None:
    obj = repo.get("object")
    if obj and isinstance(obj, dict):
        return obj.get("text")
    return None


# ──────────────────────────────────────────────────────────────────────────────
# CATEGORY 1 — Profile Credibility (5 factors)
# ──────────────────────────────────────────────────────────────────────────────


def score_account_age(data: dict) -> float:
    """Years since account creation, capped at 10."""
    try:
        created = datetime.fromisoformat(data["createdAt"].replace("Z", "+00:00"))
        days = (datetime.now(timezone.utc) - created).days
        return _clamp(days / 365.0)
    except (KeyError, ValueError):
        return 0.0


def score_profile_completeness(data: dict) -> float:
    """Fraction of filled profile fields (bio, avatarUrl, location, websiteUrl)."""
    fields = ["bio", "avatarUrl", "location", "websiteUrl"]
    filled = sum(1 for f in fields if data.get(f))
    return _clamp(filled / 4.0 * 10.0)


def score_hireable_flag(data: dict) -> float:
    """10 if the user has set isHireable=True."""
    return 10.0 if data.get("isHireable") else 0.0


def score_organization_memberships(data: dict) -> float:
    """Organization count / 2, capped at 10."""
    orgs = (data.get("organizations") or {}).get("nodes") or []
    return _clamp(min(len(orgs) / 2.0, 1.0) * 10.0)


def score_email_verified(data: dict, resume_email: str = "") -> float:
    """
    10 if public email matches resume email (case-insensitive),
    5 if email is present but doesn't match, 0 otherwise.
    """
    gh_email = (data.get("email") or "").strip().lower()
    if not gh_email:
        return 0.0
    if resume_email and gh_email == resume_email.strip().lower():
        return 10.0
    return 5.0


# ──────────────────────────────────────────────────────────────────────────────
# CATEGORY 2 — Contribution Activity (8 factors)
# ──────────────────────────────────────────────────────────────────────────────


def score_active_days(data: dict) -> float:
    """Unique days with at least one contribution / 365 * 10."""
    days = _contribution_days(data)
    active = sum(1 for d in days if d.get("contributionCount", 0) > 0)
    return _clamp(active / 365.0 * 10.0)


def score_total_commits(data: dict) -> float:
    """log10(totalCommitContributions + 1) * 3, capped at 10."""
    cc = data.get("contributionsCollection") or {}
    commits = cc.get("totalCommitContributions", 0)
    return _clamp(math.log10(commits + 1) * 3.0)


def score_longest_streak(data: dict) -> float:
    """Longest consecutive days with contributions / 30, capped at 10."""
    days = _contribution_days(data)
    # Sort by date
    days_sorted = sorted(days, key=lambda d: d.get("date", ""))
    longest = 0
    current = 0
    for d in days_sorted:
        if d.get("contributionCount", 0) > 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return _clamp(longest / 30.0 * 10.0)


def score_current_streak(data: dict) -> float:
    """Current streak backwards from today / 10, capped at 10."""
    days = _contribution_days(data)
    day_map: dict[str, int] = {d["date"]: d["contributionCount"] for d in days}
    streak = 0
    current_date = datetime.now(timezone.utc).date()
    for i in range(366):
        check = (current_date - timedelta(days=i)).isoformat()
        if day_map.get(check, 0) > 0:
            streak += 1
        else:
            # Allow today to have 0 (maybe hasn't pushed yet) — only break
            # after the first real gap past today.
            if i > 0:
                break
    return _clamp(streak / 10.0 * 10.0)


def score_contribution_consistency(data: dict) -> float:
    """Months (out of 12) with ≥1 contribution / 12 * 10."""
    days = _contribution_days(data)
    months_with_activity: set[str] = set()
    for d in days:
        if d.get("contributionCount", 0) > 0:
            # YYYY-MM
            months_with_activity.add(d["date"][:7])
    return _clamp(len(months_with_activity) / 12.0 * 10.0)


def score_weekend_activity(data: dict) -> float:
    """Weekend contribution days / total active days * 10."""
    days = _contribution_days(data)
    active_total = 0
    weekend_active = 0
    for d in days:
        if d.get("contributionCount", 0) > 0:
            active_total += 1
            try:
                dt = datetime.fromisoformat(d["date"])
                if dt.weekday() >= 5:  # Saturday=5, Sunday=6
                    weekend_active += 1
            except ValueError:
                pass
    if active_total == 0:
        return 0.0
    return _clamp(weekend_active / active_total * 10.0)


def score_recent_activity(data: dict) -> float:
    """Contributions in the last 30 days, min(count / 10, 1) * 10."""
    days = _contribution_days(data)
    cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).date().isoformat()
    recent = sum(
        d.get("contributionCount", 0)
        for d in days
        if d.get("date", "") >= cutoff
    )
    return _clamp(min(recent / 10.0, 1.0) * 10.0)


def score_pr_contributions(data: dict) -> float:
    """log10(totalPullRequestContributions + 1) * 3, capped at 10."""
    cc = data.get("contributionsCollection") or {}
    prs = cc.get("totalPullRequestContributions", 0)
    return _clamp(math.log10(prs + 1) * 3.0)


# ──────────────────────────────────────────────────────────────────────────────
# CATEGORY 3 — Repository Authenticity (7 factors)
# ──────────────────────────────────────────────────────────────────────────────


def score_fork_ratio(data: dict) -> float:
    """(1 - forked_repos / total_repos) * 10."""
    repos = _repos(data)
    if not repos:
        return 0.0
    forked = sum(1 for r in repos if r.get("isFork"))
    return _clamp((1.0 - forked / len(repos)) * 10.0)


def score_original_repos(data: dict) -> float:
    """min(non-fork count / 5, 1) * 10."""
    non_fork = len(_non_fork_repos(data))
    return _clamp(min(non_fork / 5.0, 1.0) * 10.0)


def score_fork_detection(data: dict, resume_projects: list[str] | None = None) -> float:
    """
    For each resume project, fuzzy-match against repo names and check isFork.
    Score = (non-forked matches / total matches) * 10.
    """
    if not resume_projects:
        return 5.0  # Neutral when no projects to check
    repos = _repos(data)
    matched = 0
    non_forked_matches = 0
    for proj in resume_projects:
        best_ratio = 0.0
        best_repo: dict | None = None
        for r in repos:
            ratio = _fuzzy_match_score(proj, r.get("name", ""))
            if ratio > best_ratio:
                best_ratio = ratio
                best_repo = r
        if best_ratio > 0.5 and best_repo is not None:
            matched += 1
            if not best_repo.get("isFork"):
                non_forked_matches += 1
    if matched == 0:
        return 5.0  # Neutral — no matches found
    return _clamp(non_forked_matches / matched * 10.0)


def score_commit_authorship(data: dict) -> float:
    """Simplified: 7 if user email is set, 3 otherwise."""
    if data.get("email"):
        return 7.0
    return 3.0


def score_first_commit_date(data: dict) -> float:
    """Average age of repos in months / 12, capped at 10."""
    repos = _repos(data)
    if not repos:
        return 0.0
    now = datetime.now(timezone.utc)
    total_months = 0.0
    count = 0
    for r in repos:
        try:
            created = datetime.fromisoformat(r["createdAt"].replace("Z", "+00:00"))
            months = (now - created).days / 30.0
            total_months += months
            count += 1
        except (KeyError, ValueError):
            pass
    if count == 0:
        return 0.0
    avg_months = total_months / count
    return _clamp(avg_months / 12.0)


def score_sole_contributor(data: dict) -> float:
    """Repos with forkCount==0 as proxy for sole contributor / total * 10."""
    repos = _repos(data)
    if not repos:
        return 0.0
    zero_forks = sum(1 for r in repos if r.get("forkCount", 0) == 0)
    return _clamp(zero_forks / len(repos) * 10.0)


def score_multi_contributor(data: dict) -> float:
    """Repos with forkCount > 0 / total * 10."""
    repos = _repos(data)
    if not repos:
        return 0.0
    has_forks = sum(1 for r in repos if r.get("forkCount", 0) > 0)
    return _clamp(has_forks / len(repos) * 10.0)


# ──────────────────────────────────────────────────────────────────────────────
# CATEGORY 4 — Code Quality Signals (6 factors)
# ──────────────────────────────────────────────────────────────────────────────


def score_readme_quality(data: dict) -> float:
    """Repos with a README longer than 50 chars / total * 10."""
    repos = _repos(data)
    if not repos:
        return 0.0
    good = sum(1 for r in repos if (_readme_text(r) or "") and len(_readme_text(r) or "") > 50)
    return _clamp(good / len(repos) * 10.0)


def score_has_tests(data: dict) -> float:
    """10 if any repo README mentions testing keywords or topics include 'testing'."""
    test_kw = {"test", "testing", "pytest", "jest", "unittest", "mocha", "spec"}
    for r in _repos(data):
        readme = (_readme_text(r) or "").lower()
        if any(kw in readme for kw in test_kw):
            return 10.0
        topics = {
            t["topic"]["name"].lower()
            for t in (r.get("repositoryTopics") or {}).get("nodes") or []
            if t.get("topic")
        }
        if topics & test_kw:
            return 10.0
    return 0.0


def score_ci_cd(data: dict) -> float:
    """10 if any repo README mentions CI/CD keywords or topics include 'ci-cd'."""
    ci_kw = {"github actions", "ci", "workflow", "ci/cd", "ci-cd", "travis", "circleci"}
    for r in _repos(data):
        readme = (_readme_text(r) or "").lower()
        if any(kw in readme for kw in ci_kw):
            return 10.0
        topics = {
            t["topic"]["name"].lower()
            for t in (r.get("repositoryTopics") or {}).get("nodes") or []
            if t.get("topic")
        }
        if topics & ci_kw:
            return 10.0
    return 0.0


def score_repo_topics(data: dict) -> float:
    """Repos with at least 1 topic / total * 10."""
    repos = _repos(data)
    if not repos:
        return 0.0
    with_topic = sum(
        1
        for r in repos
        if len((r.get("repositoryTopics") or {}).get("nodes") or []) > 0
    )
    return _clamp(with_topic / len(repos) * 10.0)


def score_avg_commits_per_repo(data: dict) -> float:
    """Average commits per repo from history.totalCount; min(avg / 20, 1) * 10."""
    repos = _repos(data)
    if not repos:
        return 0.0
    total = sum(_repo_commit_count(r) for r in repos)
    avg = total / len(repos)
    return _clamp(min(avg / 20.0, 1.0) * 10.0)


def score_issue_pr_activity(data: dict) -> float:
    """log10(issues + PRs + 1) * 3, capped at 10."""
    cc = data.get("contributionsCollection") or {}
    issues = cc.get("totalIssueContributions", 0)
    prs = cc.get("totalPullRequestContributions", 0)
    return _clamp(math.log10(issues + prs + 1) * 3.0)


# ──────────────────────────────────────────────────────────────────────────────
# CATEGORY 5 — Skill Verification (5 factors)
# ──────────────────────────────────────────────────────────────────────────────


def score_language_match(data: dict, jd_text: str = "") -> float:
    """Jaccard similarity between JD languages and GitHub languages * 10."""
    if not jd_text:
        return 5.0
    jd_langs = _extract_jd_languages(jd_text)
    gh_langs = _all_languages(data)
    if not jd_langs and not gh_langs:
        return 5.0
    union = jd_langs | gh_langs
    if not union:
        return 5.0
    intersection = jd_langs & gh_langs
    return _clamp(len(intersection) / len(union) * 10.0)


def score_language_depth(data: dict, jd_text: str = "") -> float:
    """Sum bytes for languages matching JD. log10(total + 1) * 1.5, capped at 10."""
    if not jd_text:
        return 5.0
    jd_langs = _extract_jd_languages(jd_text)
    if not jd_langs:
        return 5.0
    total_bytes = 0
    for repo in _repos(data):
        for edge in (repo.get("languages") or {}).get("edges") or []:
            lang_name = edge["node"]["name"].lower()
            if lang_name in jd_langs:
                total_bytes += edge.get("size", 0)
    return _clamp(math.log10(total_bytes + 1) * 1.5)


def score_language_diversity(data: dict) -> float:
    """Unique languages across all repos. min(count / 8, 1) * 10."""
    langs = _all_languages(data)
    return _clamp(min(len(langs) / 8.0, 1.0) * 10.0)


def score_primary_language_match(data: dict, jd_text: str = "") -> float:
    """10 if most common primaryLanguage matches any JD keyword, else 0."""
    if not jd_text:
        return 5.0
    jd_langs = _extract_jd_languages(jd_text)
    if not jd_langs:
        return 5.0
    lang_count: dict[str, int] = {}
    for repo in _repos(data):
        pl = repo.get("primaryLanguage")
        if pl:
            name = pl["name"].lower()
            lang_count[name] = lang_count.get(name, 0) + 1
    if not lang_count:
        return 0.0
    most_common = max(lang_count, key=lang_count.get)  # type: ignore[arg-type]
    return 10.0 if most_common in jd_langs else 0.0


def score_tech_stack_alignment(data: dict, resume_skills: list[str] | None = None) -> float:
    """Repo topics matching resume skills / max(len(resume_skills), 1) * 10."""
    if not resume_skills:
        return 5.0
    skills_lower = {s.lower() for s in resume_skills}
    matched = set()
    for repo in _repos(data):
        for t in (repo.get("repositoryTopics") or {}).get("nodes") or []:
            topic_name = (t.get("topic") or {}).get("name", "").lower()
            if topic_name in skills_lower:
                matched.add(topic_name)
    return _clamp(len(matched) / max(len(resume_skills), 1) * 10.0)


# ──────────────────────────────────────────────────────────────────────────────
# CATEGORY 6 — Resume Cross-Reference (5 factors)
# ──────────────────────────────────────────────────────────────────────────────


def score_project_exists(data: dict, resume_projects: list[str] | None = None) -> float:
    """Fuzzy match resume project names against repo names (threshold 0.4)."""
    if not resume_projects:
        return 5.0
    repos = _repos(data)
    matched = 0
    for proj in resume_projects:
        for r in repos:
            if _fuzzy_match_score(proj, r.get("name", "")) > 0.4:
                matched += 1
                break
    return _clamp(matched / max(len(resume_projects), 1) * 10.0)


def score_project_fuzzy_match(data: dict, resume_projects: list[str] | None = None) -> float:
    """Best fuzzy match ratio across all project-repo pairs * 10."""
    if not resume_projects:
        return 5.0
    repos = _repos(data)
    best = 0.0
    for proj in resume_projects:
        for r in repos:
            ratio = _fuzzy_match_score(proj, r.get("name", ""))
            best = max(best, ratio)
    return _clamp(best * 10.0)


def score_readme_resume_alignment(data: dict, resume_projects: list[str] | None = None) -> float:
    """
    For matched repos, check word overlap between README and project name.
    Returns average overlap ratio * 10.
    """
    if not resume_projects:
        return 5.0
    repos = _repos(data)
    scores: list[float] = []
    for proj in resume_projects:
        best_repo: dict | None = None
        best_ratio = 0.0
        for r in repos:
            ratio = _fuzzy_match_score(proj, r.get("name", ""))
            if ratio > best_ratio:
                best_ratio = ratio
                best_repo = r
        if best_ratio > 0.4 and best_repo is not None:
            readme = (_readme_text(best_repo) or "").lower()
            if not readme:
                scores.append(0.0)
                continue
            proj_words = set(re.findall(r"\w+", proj.lower()))
            readme_words = set(re.findall(r"\w+", readme))
            if not proj_words:
                scores.append(0.0)
                continue
            overlap = len(proj_words & readme_words) / len(proj_words)
            scores.append(overlap)
    if not scores:
        return 5.0
    return _clamp(sum(scores) / len(scores) * 10.0)


def score_project_age_vs_experience(data: dict) -> float:
    """min(oldest_repo_age_months / 24, 1) * 10."""
    repos = _repos(data)
    if not repos:
        return 0.0
    now = datetime.now(timezone.utc)
    oldest_months = 0.0
    for r in repos:
        try:
            created = datetime.fromisoformat(r["createdAt"].replace("Z", "+00:00"))
            months = (now - created).days / 30.0
            oldest_months = max(oldest_months, months)
        except (KeyError, ValueError):
            pass
    return _clamp(min(oldest_months / 24.0, 1.0) * 10.0)


def score_tech_in_repo_matches_resume(data: dict, resume_skills: list[str] | None = None) -> float:
    """Repo languages that appear in resume_skills / total unique langs * 10."""
    if not resume_skills:
        return 5.0
    gh_langs = _all_languages(data)
    if not gh_langs:
        return 0.0
    skills_lower = {s.lower() for s in resume_skills}
    matched = gh_langs & skills_lower
    return _clamp(len(matched) / max(len(gh_langs), 1) * 10.0)


# ──────────────────────────────────────────────────────────────────────────────
# CATEGORY 7 — Social Proof (4 factors)
# ──────────────────────────────────────────────────────────────────────────────


def score_stars_received(data: dict) -> float:
    """log10(total stars + 1) * 2, capped at 10."""
    repos = _repos(data)
    total = sum(r.get("stargazerCount", 0) for r in repos)
    return _clamp(math.log10(total + 1) * 2.0)


def score_forks_received(data: dict) -> float:
    """log10(total forks + 1) * 3, capped at 10."""
    repos = _repos(data)
    total = sum(r.get("forkCount", 0) for r in repos)
    return _clamp(math.log10(total + 1) * 3.0)


def score_open_source_contributions(data: dict) -> float:
    """10 if PRs > 5, 5 if > 0, else 0."""
    cc = data.get("contributionsCollection") or {}
    prs = cc.get("totalPullRequestContributions", 0)
    if prs > 5:
        return 10.0
    if prs > 0:
        return 5.0
    return 0.0


def score_pinned_repo_quality(data: dict) -> float:
    """For pinned repos: avg(has_description + has_stars + not_fork) / 3 * 10."""
    pinned = (data.get("pinnedItems") or {}).get("nodes") or []
    if not pinned:
        return 0.0
    scores: list[float] = []
    for r in pinned:
        has_desc = 1.0 if r.get("description") else 0.0
        has_stars = 1.0 if r.get("stargazerCount", 0) > 0 else 0.0
        not_fork = 1.0 if not r.get("isFork") else 0.0
        scores.append((has_desc + has_stars + not_fork) / 3.0)
    return _clamp(sum(scores) / len(scores) * 10.0)


# ──────────────────────────────────────────────────────────────────────────────
# Factor Registry & Category Map
# ──────────────────────────────────────────────────────────────────────────────

FACTOR_REGISTRY: dict[str, callable] = {
    # Profile Credibility
    "account_age": score_account_age,
    "profile_completeness": score_profile_completeness,
    "hireable_flag": score_hireable_flag,
    "organization_memberships": score_organization_memberships,
    "email_verified": score_email_verified,
    # Contribution Activity
    "active_days": score_active_days,
    "total_commits": score_total_commits,
    "longest_streak": score_longest_streak,
    "current_streak": score_current_streak,
    "contribution_consistency": score_contribution_consistency,
    "weekend_activity": score_weekend_activity,
    "recent_activity": score_recent_activity,
    "pr_contributions": score_pr_contributions,
    # Repository Authenticity
    "fork_ratio": score_fork_ratio,
    "original_repos": score_original_repos,
    "fork_detection": score_fork_detection,
    "commit_authorship": score_commit_authorship,
    "first_commit_date": score_first_commit_date,
    "sole_contributor": score_sole_contributor,
    "multi_contributor": score_multi_contributor,
    # Code Quality Signals
    "readme_quality": score_readme_quality,
    "has_tests": score_has_tests,
    "ci_cd": score_ci_cd,
    "repo_topics": score_repo_topics,
    "avg_commits_per_repo": score_avg_commits_per_repo,
    "issue_pr_activity": score_issue_pr_activity,
    # Skill Verification
    "language_match": score_language_match,
    "language_depth": score_language_depth,
    "language_diversity": score_language_diversity,
    "primary_language_match": score_primary_language_match,
    "tech_stack_alignment": score_tech_stack_alignment,
    # Resume Cross-Reference
    "project_exists": score_project_exists,
    "project_fuzzy_match": score_project_fuzzy_match,
    "readme_resume_alignment": score_readme_resume_alignment,
    "project_age_vs_experience": score_project_age_vs_experience,
    "tech_in_repo_matches_resume": score_tech_in_repo_matches_resume,
    # Social Proof
    "stars_received": score_stars_received,
    "forks_received": score_forks_received,
    "open_source_contributions": score_open_source_contributions,
    "pinned_repo_quality": score_pinned_repo_quality,
}

FACTOR_CATEGORIES: dict[str, list[str]] = {
    "Profile Credibility": [
        "account_age",
        "profile_completeness",
        "hireable_flag",
        "organization_memberships",
        "email_verified",
    ],
    "Contribution Activity": [
        "active_days",
        "total_commits",
        "longest_streak",
        "current_streak",
        "contribution_consistency",
        "weekend_activity",
        "recent_activity",
        "pr_contributions",
    ],
    "Repository Authenticity": [
        "fork_ratio",
        "original_repos",
        "fork_detection",
        "commit_authorship",
        "first_commit_date",
        "sole_contributor",
        "multi_contributor",
    ],
    "Code Quality Signals": [
        "readme_quality",
        "has_tests",
        "ci_cd",
        "repo_topics",
        "avg_commits_per_repo",
        "issue_pr_activity",
    ],
    "Skill Verification": [
        "language_match",
        "language_depth",
        "language_diversity",
        "primary_language_match",
        "tech_stack_alignment",
    ],
    "Resume Cross-Reference": [
        "project_exists",
        "project_fuzzy_match",
        "readme_resume_alignment",
        "project_age_vs_experience",
        "tech_in_repo_matches_resume",
    ],
    "Social Proof": [
        "stars_received",
        "forks_received",
        "open_source_contributions",
        "pinned_repo_quality",
    ],
}

# ──────────────────────────────────────────────────────────────────────────────
# Factor Argument Dispatch
# ──────────────────────────────────────────────────────────────────────────────

# Factors that require extra keyword arguments beyond `data`.
_FACTOR_EXTRA_ARGS: dict[str, list[str]] = {
    "email_verified": ["resume_email"],
    "language_match": ["jd_text"],
    "language_depth": ["jd_text"],
    "primary_language_match": ["jd_text"],
    "fork_detection": ["resume_projects"],
    "project_exists": ["resume_projects"],
    "project_fuzzy_match": ["resume_projects"],
    "readme_resume_alignment": ["resume_projects"],
    "tech_stack_alignment": ["resume_skills"],
    "tech_in_repo_matches_resume": ["resume_skills"],
}


def _call_factor(
    name: str,
    fn: callable,
    data: dict,
    *,
    resume_projects: list[str] | None = None,
    resume_skills: list[str] | None = None,
    resume_email: str = "",
    jd_text: str = "",
) -> float:
    """Dispatch a scoring factor with the correct extra arguments."""
    extras = _FACTOR_EXTRA_ARGS.get(name)
    if not extras:
        return float(fn(data))

    kwargs: dict = {}
    for arg in extras:
        if arg == "resume_email":
            kwargs["resume_email"] = resume_email
        elif arg == "jd_text":
            kwargs["jd_text"] = jd_text
        elif arg == "resume_projects":
            kwargs["resume_projects"] = resume_projects
        elif arg == "resume_skills":
            kwargs["resume_skills"] = resume_skills
    return float(fn(data, **kwargs))


# ──────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────────────────────────────────────


def run_github_verification(
    username: str,
    token: str,
    resume_projects: list[str] | None = None,
    resume_skills: list[str] | None = None,
    resume_email: str = "",
    jd_text: str = "",
    selected_factors: list[str] | None = None,
) -> dict:
    """
    Run GitHub verification for a user against the selected scoring factors.

    Parameters
    ----------
    username : str
        GitHub login handle.
    token : str
        GitHub personal access token (classic or fine-grained with read scope).
    resume_projects : list[str], optional
        Project names extracted from the candidate's resume.
    resume_skills : list[str], optional
        Skills/technologies listed on the candidate's resume.
    resume_email : str, optional
        Email address from the resume for cross-checking.
    jd_text : str, optional
        Full text of the job description for language/skill matching.
    selected_factors : list[str], optional
        Subset of factor names to evaluate. Defaults to *all* factors.

    Returns
    -------
    dict
        {
            "score": float (0-100),
            "factors_checked": list[str],
            "factor_scores": {name: float, ...},
            "category_scores": {category: float, ...},
            "explanation": str,
            "username": str,
        }
    """
    if selected_factors is None:
        selected_factors = list(FACTOR_REGISTRY.keys())

    # Validate factor names
    invalid = [f for f in selected_factors if f not in FACTOR_REGISTRY]
    if invalid:
        raise ValueError(f"Unknown factors: {invalid}")

    # ── Fetch profile ────────────────────────────────────────────────────────
    print(f"  → Fetching GitHub profile for @{username}...")
    data = fetch_github_profile(username, token)
    print(f"  → Profile fetched. Evaluating {len(selected_factors)} factors...")

    # ── Score each factor ────────────────────────────────────────────────────
    factor_scores: dict[str, float] = {}
    for name in selected_factors:
        fn = FACTOR_REGISTRY[name]
        print(f"  → Scoring factor: {name}...")
        score = _call_factor(
            name,
            fn,
            data,
            resume_projects=resume_projects,
            resume_skills=resume_skills,
            resume_email=resume_email,
            jd_text=jd_text,
        )
        factor_scores[name] = round(score, 2)

    # ── Aggregate ────────────────────────────────────────────────────────────
    if factor_scores:
        overall = sum(factor_scores.values()) / len(factor_scores) * 10.0
    else:
        overall = 0.0
    overall = max(0.0, min(100.0, overall))

    # ── Category-level averages ──────────────────────────────────────────────
    category_scores: dict[str, float] = {}
    for cat, members in FACTOR_CATEGORIES.items():
        cat_vals = [factor_scores[m] for m in members if m in factor_scores]
        if cat_vals:
            category_scores[cat] = round(sum(cat_vals) / len(cat_vals) * 10.0, 1)

    # ── Explanation text ─────────────────────────────────────────────────────
    top_factors = sorted(factor_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    bottom_factors = sorted(factor_scores.items(), key=lambda x: x[1])[:5]

    lines = [
        f"GitHub verification for @{username}: {overall:.1f}/100",
        f"Checked {len(factor_scores)} factors across {len(category_scores)} categories.",
        "",
        "Top strengths:",
    ]
    for name, val in top_factors:
        lines.append(f"  • {name}: {val}/10")
    lines.append("")
    lines.append("Areas with lowest scores:")
    for name, val in bottom_factors:
        lines.append(f"  • {name}: {val}/10")

    explanation = "\n".join(lines)

    return {
        "score": round(overall, 1),
        "factors_checked": list(factor_scores.keys()),
        "factor_scores": factor_scores,
        "category_scores": category_scores,
        "explanation": explanation,
        "username": username,
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI Quick-Test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python github_verifier.py <username> <github_token>")
        sys.exit(1)

    _username = sys.argv[1]
    _token = sys.argv[2]

    result = run_github_verification(
        username=_username,
        token=_token,
        resume_projects=["example-project"],
        resume_skills=["python", "javascript"],
        resume_email="test@example.com",
        jd_text="Looking for a Python and JavaScript developer",
    )

    print("\n" + "=" * 60)
    print(result["explanation"])
    print("=" * 60)
    print(f"\nOverall GitHub Score: {result['score']}/100")
    print(f"Factors checked: {len(result['factors_checked'])}")

    print("\nCategory Breakdown:")
    for cat, score in result["category_scores"].items():
        print(f"  {cat}: {score}/100")
