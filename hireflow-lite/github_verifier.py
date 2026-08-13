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
        pushedAt
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


# ── Rate Limit & Security Tracking ───────────────────────────────────────────
_RATE_LIMIT_STATS = {
    "queries_made": 0,
    "total_cost": 0,
    "last_remaining": None,
    "last_reset": None,
}


def mask_github_token(token: str) -> str:
    """Mask a GitHub PAT for safe logging / display (e.g. ghp_1234...abcd)."""
    if not token:
        return ""
    if len(token) <= 8:
        return "********"
    return f"{token[:4]}...{token[-4:]}"


def get_rate_limit_stats() -> dict:
    """Return session-level API budget usage statistics."""
    return dict(_RATE_LIMIT_STATS)


def _track_rate_limit(data: dict) -> None:
    """Extract and track rateLimit info from GraphQL response."""
    rl = data.get("data", {}).get("rateLimit") or data.get("rateLimit")
    if isinstance(rl, dict):
        cost = rl.get("cost", 1)
        remaining = rl.get("remaining")
        reset_at = rl.get("resetAt")

        _RATE_LIMIT_STATS["queries_made"] += 1
        _RATE_LIMIT_STATS["total_cost"] += cost
        if remaining is not None:
            _RATE_LIMIT_STATS["last_remaining"] = remaining
        if reset_at:
            _RATE_LIMIT_STATS["last_reset"] = reset_at


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

    _track_rate_limit(data)

    user = data.get("data", {}).get("user")
    if user is None:
        raise RuntimeError(
            f"GitHub user '{username}' not found (the GraphQL response returned null)."
        )

    _write_cache(username, user)
    return user


# ──────────────────────────────────────────────────────────────────────────────
# Per-Repo Commit Activity
# ──────────────────────────────────────────────────────────────────────────────

_REPO_ACTIVITY_QUERY = """
query GetRepoActivity($owner: String!, $name: String!) {
  rateLimit { cost remaining resetAt }
  repository(owner: $owner, name: $name) {
    isFork
    parent { nameWithOwner url }
    pushedAt
    createdAt
    defaultBranchRef {
      target {
        ... on Commit {
          history(first: 100) {
            totalCount
            nodes {
              committedDate
              author {
                name
                email
                user { login }
              }
            }
          }
        }
      }
    }
  }
}
"""


def fetch_repo_commit_activity(
    owner: str, repo_name: str, token: str
) -> dict:
    """Fetch detailed commit activity for a single repository.

    Returns a dict with:
        - is_fork (bool)
        - forked_from (str | None): 'owner/repo' if forked
        - pushed_at (str | None): ISO datetime of last push
        - created_at (str | None): ISO datetime of repo creation
        - total_commits (int): total commits on default branch
        - active_days (int): unique calendar days with commits
        - commit_dates (list[str]): list of YYYY-MM-DD date strings
        - author_logins (set[str]): set of commit author GitHub logins
        - commits_by_author (dict[str, int]): login -> commit count
    """
    import time as _time

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    body = {
        "query": _REPO_ACTIVITY_QUERY,
        "variables": {"owner": owner, "name": repo_name},
    }

    result = {
        "is_fork": False,
        "forked_from": None,
        "pushed_at": None,
        "created_at": None,
        "last_pushed": None,
        "total_commits": 0,
        "active_days": 0,
        "commit_dates": [],
        "first_commit_date": None,
        "last_commit_date": None,
        "author_logins": [],
        "commits_by_author": {},
        "author_commit_counts": {},
        "commit_nodes": [],
    }

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(GITHUB_GRAPHQL_ENDPOINT, headers=headers, json=body)

        if resp.status_code == 403:
            remaining = resp.headers.get("X-RateLimit-Remaining")
            if remaining == "0" or "rate limit" in resp.text.lower():
                reset = resp.headers.get("X-RateLimit-Reset")
                try:
                    reset_time = int(reset) if reset else int(_time.time()) + 60
                except ValueError:
                    reset_time = int(_time.time()) + 60
                wait_seconds = max(reset_time - int(_time.time()) + 5, 10)
                print(f"  → [GitHub API] Rate limit hit. Sleeping {wait_seconds}s...")
                _time.sleep(wait_seconds)
                # Retry once
                resp = client.post(GITHUB_GRAPHQL_ENDPOINT, headers=headers, json=body)

        if resp.status_code != 200:
            print(f"  → [GitHub API] HTTP {resp.status_code} for {owner}/{repo_name}")
            return result

        data = resp.json()
        if "errors" in data:
            print(f"  → [GitHub API] GraphQL error for {owner}/{repo_name}")
            return result

        _track_rate_limit(data)

        repo = data.get("data", {}).get("repository")
        if not repo:
            return result

        result["is_fork"] = repo.get("isFork", False)
        parent = repo.get("parent")
        if parent:
            result["forked_from"] = parent.get("nameWithOwner")
        result["pushed_at"] = repo.get("pushedAt")
        result["last_pushed"] = repo.get("pushedAt")
        result["created_at"] = repo.get("createdAt")

        branch_ref = repo.get("defaultBranchRef")
        if branch_ref:
            target = branch_ref.get("target", {})
            history = target.get("history", {})
            result["total_commits"] = history.get("totalCount", 0)

            nodes = history.get("nodes", [])
            result["commit_nodes"] = nodes
            dates = set()
            commits_by_author = {}
            author_logins = set()

            for commit in nodes:
                # Extract date
                committed_date = commit.get("committedDate", "")
                if committed_date:
                    day = committed_date[:10]  # YYYY-MM-DD
                    dates.add(day)

                # Extract author
                author = commit.get("author", {})
                user = author.get("user", {})
                login = user.get("login", "") if user else ""
                if login:
                    author_logins.add(login)
                    commits_by_author[login] = commits_by_author.get(login, 0) + 1

            sorted_dates = sorted(dates)
            result["active_days"] = len(sorted_dates)
            result["commit_dates"] = sorted_dates
            if sorted_dates:
                result["first_commit_date"] = sorted_dates[0]
                result["last_commit_date"] = sorted_dates[-1]
            result["author_logins"] = list(author_logins)  # JSON-serializable list!
            result["commits_by_author"] = commits_by_author
            result["author_commit_counts"] = commits_by_author

    except Exception as exc:
        print(f"  → [GitHub API] Error fetching activity for {owner}/{repo_name}: {exc}")

    return result


def verify_project_exists(
    project_name: str,
    github_username: str,
    github_repos: list[dict],
    token: str,
    similarity_threshold: float = 0.5,
) -> dict:
    """Check if a claimed resume project exists on the candidate's GitHub.

    Uses a 3-tier matching strategy:
    1. Exact name match (case-insensitive)
    2. Fuzzy name match (SequenceMatcher ratio >= 0.6)
    3. Falls back to the best fuzzy match if above threshold

    If a match is found, fetches detailed commit activity.

    Returns
    -------
    dict
        {
            "claimed_project": str,
            "status": "verified" | "uncertain" | "unverified",
            "matched_repo": str | None,
            "match_method": "exact" | "fuzzy" | None,
            "match_score": float,
            "is_fork": bool,
            "forked_from": str | None,
            "total_commits": int,
            "active_days": int,
            "last_pushed": str | None,
            "candidate_is_author": bool,
            "commit_frequency": float,
        }
    """
    result = {
        "claimed_project": project_name,
        "status": "unverified",
        "matched_repo": None,
        "match_method": None,
        "match_score": 0.0,
        "is_fork": False,
        "forked_from": None,
        "total_commits": 0,
        "active_days": 0,
        "last_pushed": None,
        "candidate_is_author": False,
        "commit_frequency": 0.0,
    }

    if not github_repos or not project_name:
        return result

    project_lower = project_name.lower().strip()
    best_match = None
    best_score = 0.0
    best_method = None

    for repo in github_repos:
        repo_name = (repo.get("name") or "").lower().strip()
        if not repo_name:
            continue

        # Tier 1: Exact match
        if repo_name == project_lower:
            best_match = repo
            best_score = 1.0
            best_method = "exact"
            break

        # Tier 2: Fuzzy match
        ratio = SequenceMatcher(None, project_lower, repo_name).ratio()
        if ratio > best_score:
            best_score = ratio
            best_match = repo
            best_method = "fuzzy"

    if best_match is None:
        return result

    matched_repo_name = best_match.get("name", "")

    if best_score >= 0.8:
        result["status"] = "verified"
    elif best_score >= similarity_threshold:
        result["status"] = "uncertain"
    else:
        return result  # Below threshold, don't even bother fetching activity

    result["matched_repo"] = matched_repo_name
    result["match_method"] = best_method
    result["match_score"] = round(best_score, 4)
    result["is_fork"] = best_match.get("isFork", False)

    # Fetch detailed commit activity for the matched repo
    if token and github_username:
        print(f"    → Fetching commit activity for {github_username}/{matched_repo_name}...")
        activity = fetch_repo_commit_activity(github_username, matched_repo_name, token)

        result["is_fork"] = activity.get("is_fork", False)
        result["forked_from"] = activity.get("forked_from")
        result["total_commits"] = activity.get("total_commits", 0)
        result["active_days"] = activity.get("active_days", 0)
        result["last_pushed"] = (activity.get("pushed_at") or "")[:10] or None

        # Check if the candidate is actually an author
        author_logins = activity.get("author_logins", set())
        if github_username.lower() in {a.lower() for a in author_logins}:
            result["candidate_is_author"] = True

        # Commit frequency = commits / active days
        if result["active_days"] > 0:
            result["commit_frequency"] = round(
                result["total_commits"] / result["active_days"], 2
            )

    return result


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
    """
    Real authorship signal: checks whether the candidate's GitHub login
    appears as a commit author in their repositories.

    Uses contributorsCollection data from the GraphQL profile query.
    Falls back to checking email field as a very weak proxy if no
    contributor data is available.

    Score:
      10.0 — username confirmed in commit author data
       5.0 — email set but no commit author data to verify
       2.0 — no email and no commit author data
    """
    login = (data.get("login") or "").lower()
    # GitHub contributor data is in repos' defaultBranchRef commit history nodes
    # Check if the candidate's login appears in any repo commit authors
    if login:
        for repo in _repos(data):
            history = (repo.get("defaultBranchRef") or {}).get("target", {}).get("history", {})
            nodes = history.get("nodes") or []
            for commit in nodes:
                author = commit.get("author") or {}
                user   = author.get("user") or {}
                if (user.get("login") or "").lower() == login:
                    return 10.0
                # also check name/email match
                if login in (author.get("email") or "").lower():
                    return 8.0
    # Fallback: at least email field is set (profile completeness signal)
    if data.get("email"):
        return 5.0
    return 2.0


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
    """
    Evidence-based development timeline score.

    Duration ALONE is a weak and misleading signal:
      - A 24-month empty repo scores high under age-only logic
      - A 2-day hackathon with 20 meaningful commits scores near 0

    Correct approach: measure commit density (commits per active day)
    combined with development span as a secondary supporting signal.

    Formula:
      commit_density_signal = min(avg_commits_per_active_day / 3.0, 1.0)
      span_signal           = min(span_days / 90.0, 1.0)  # 3 months = full
      score = (commit_density_signal * 0.65 + span_signal * 0.35) * 10

    A hackathon (2 days, 20 commits): density=10, span=0.022
      -> (1.0 * 0.65 + 0.022 * 0.35) * 10 = 6.58  (decent signal)
    An empty 24-month repo: density=0, span=1.0
      -> (0 * 0.65 + 1.0 * 0.35) * 10 = 3.5  (weak signal, not inflated)
    """
    repos = _non_fork_repos(data)
    if not repos:
        repos = _repos(data)
    if not repos:
        return 0.0

    now = datetime.now(timezone.utc)
    total_commits = 0
    total_active_days = 0
    max_span_days = 0.0

    for r in repos:
        commit_count = _repo_commit_count(r)
        total_commits += commit_count

        try:
            created = datetime.fromisoformat(r["createdAt"].replace("Z", "+00:00"))
            pushed_at = r.get("pushedAt") or r.get("updatedAt")
            if pushed_at:
                last_push = datetime.fromisoformat(pushed_at.replace("Z", "+00:00"))
            else:
                last_push = now
            span_days = (last_push - created).days
            max_span_days = max(max_span_days, span_days)
            # Approximate active days: number of unique calendar weeks with commits
            # Use commit count as proxy (1 commit ~= 1 active day, capped at span)
            active_days = min(commit_count, max(1, span_days))
            total_active_days += active_days
        except (KeyError, ValueError):
            pass

    avg_commit_density = total_commits / max(1, total_active_days)
    density_signal = min(1.0, avg_commit_density / 3.0)   # 3 commits/day = full signal
    span_signal    = min(1.0, max_span_days / 90.0)        # 3-month span = full signal

    score = (density_signal * 0.65 + span_signal * 0.35) * 10.0
    return _clamp(score)


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


def score_collaboration_profile(data: dict) -> float:
    """
    Balanced collaboration signal. Replaces the redundant sole_contributor
    and multi_contributor factors which measured opposite sides of the same
    signal (forkCount == 0 vs forkCount > 0) without combining them.

    A developer with BOTH solo original projects (forkCount == 0) AND
    projects others have forked (forkCount > 0) demonstrates the most
    complete and credible development profile.

    Scoring:
      10.0 — has both types (solo + forked-by-others)
       7.0 — only solo projects (normal for student developers)
       5.0 — only projects that others have forked (rare, may indicate farm)
       0.0 — no repositories
    """
    repos = _repos(data)
    if not repos:
        return 0.0
    has_solo  = any(r.get("forkCount", 0) == 0 for r in repos)
    has_multi = any(r.get("forkCount", 0) >  0 for r in repos)
    if has_solo and has_multi:
        return 10.0
    if has_solo:
        return 7.0
    if has_multi:
        return 5.0
    return 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Factor Registry & Category Map
# ──────────────────────────────────────────────────────────────────────────────

FACTOR_REGISTRY: dict[str, callable] = {
    # Profile Credibility
    "account_age":              score_account_age,
    "profile_completeness":     score_profile_completeness,
    "hireable_flag":            score_hireable_flag,
    "organization_memberships": score_organization_memberships,
    "email_verified":           score_email_verified,
    # Contribution Activity
    "active_days":              score_active_days,
    "total_commits":            score_total_commits,
    "longest_streak":           score_longest_streak,
    "current_streak":           score_current_streak,
    "contribution_consistency": score_contribution_consistency,
    "weekend_activity":         score_weekend_activity,
    "recent_activity":          score_recent_activity,
    "pr_contributions":         score_pr_contributions,
    # Repository Authenticity
    "fork_ratio":               score_fork_ratio,
    "original_repos":           score_original_repos,
    "fork_detection":           score_fork_detection,
    "commit_authorship":        score_commit_authorship,    # FIXED: real login check
    "first_commit_date":        score_first_commit_date,
    "collaboration_profile":    score_collaboration_profile, # MERGED: sole+multi
    # Code Quality Signals
    "readme_quality":           score_readme_quality,
    "has_tests":                score_has_tests,
    "ci_cd":                    score_ci_cd,
    "repo_topics":              score_repo_topics,
    "avg_commits_per_repo":     score_avg_commits_per_repo,
    "issue_pr_activity":        score_issue_pr_activity,
    # Skill Verification
    "language_match":           score_language_match,
    "language_depth":           score_language_depth,
    "language_diversity":       score_language_diversity,
    "primary_language_match":   score_primary_language_match,
    "tech_stack_alignment":     score_tech_stack_alignment,
    # Resume Cross-Reference
    "project_exists":           score_project_exists,
    "project_fuzzy_match":      score_project_fuzzy_match,
    "readme_resume_alignment":  score_readme_resume_alignment,
    "project_age_vs_experience":score_project_age_vs_experience,  # FIXED: commit-density
    "tech_in_repo_matches_resume": score_tech_in_repo_matches_resume,
    # Social Proof
    "stars_received":           score_stars_received,
    "forks_received":           score_forks_received,
    "open_source_contributions":score_open_source_contributions,
    "pinned_repo_quality":      score_pinned_repo_quality,
}

# ──────────────────────────────────────────────────────────────────────────────
# Weighted Group Scoring
# ──────────────────────────────────────────────────────────────────────────────
# The 5 evidence groups and their weights for the final GitHub score.
# Rationale:
#   40% — Project authenticity is the core question: did the candidate
#          actually build what they claim on their resume?
#   25% — Development history proves sustained, real work over time.
#   20% — Technical evidence confirms the claimed tech stack is real.
#   10% — Repository quality shows engineering discipline.
#    5% — Social signals (stars, forks, hireable) are weak proxies;
#          a student with 0 stars and 4 strong original projects is
#          still excellent, so these should barely affect the score.

GITHUB_GROUP_WEIGHTS: dict[str, float] = {
    "Project Authenticity":   0.40,
    "Development History":    0.25,
    "Technical Evidence":     0.20,
    "Repository Quality":     0.10,
    "Social Context":         0.05,
}

GITHUB_FACTOR_GROUPS: dict[str, list[str]] = {
    "Project Authenticity": [
        "project_exists",
        "project_fuzzy_match",
        "readme_resume_alignment",
        "fork_detection",
        "original_repos",
        "fork_ratio",
        "tech_in_repo_matches_resume",
        "collaboration_profile",
    ],
    "Development History": [
        "commit_authorship",
        "total_commits",
        "active_days",
        "avg_commits_per_repo",
        "longest_streak",
        "current_streak",
        "contribution_consistency",
        "project_age_vs_experience",
    ],
    "Technical Evidence": [
        "language_match",
        "language_depth",
        "language_diversity",
        "primary_language_match",
        "tech_stack_alignment",
    ],
    "Repository Quality": [
        "readme_quality",
        "has_tests",
        "ci_cd",
        "repo_topics",
        "issue_pr_activity",
        "first_commit_date",
    ],
    "Social Context": [
        "account_age",
        "profile_completeness",
        "hireable_flag",
        "organization_memberships",
        "email_verified",
        "stars_received",
        "forks_received",
        "open_source_contributions",
        "pinned_repo_quality",
        "recent_activity",
        "weekend_activity",
        "pr_contributions",
    ],
}

# Legacy flat category map (kept for backward-compatible reporting)
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
        "collaboration_profile",
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

    # ── Weighted Group Aggregation (replaces simple equal-weight average) ────
    # Each factor score is on a 0-10 scale. Normalize to 0-100 per factor,
    # average within each group, then weight the groups.
    group_scores: dict[str, float] = {}
    for group_name, group_members in GITHUB_FACTOR_GROUPS.items():
        vals = [factor_scores[m] * 10.0 for m in group_members if m in factor_scores]
        if vals:
            group_scores[group_name] = round(sum(vals) / len(vals), 1)
        else:
            group_scores[group_name] = 0.0

    if factor_scores:
        overall = sum(
            group_scores.get(g, 0.0) * w
            for g, w in GITHUB_GROUP_WEIGHTS.items()
        )
    else:
        overall = 0.0
    overall = round(max(0.0, min(100.0, overall)), 1)

    # ── Legacy flat category averages (for backward-compatible reporting) ────
    category_scores: dict[str, float] = {}
    for cat, members in FACTOR_CATEGORIES.items():
        cat_vals = [factor_scores[m] * 10.0 for m in members if m in factor_scores]
        if cat_vals:
            category_scores[cat] = round(sum(cat_vals) / len(cat_vals), 1)

    # ── Explanation text ─────────────────────────────────────────────────────
    top_factors    = sorted(factor_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    bottom_factors = sorted(factor_scores.items(), key=lambda x: x[1])[:5]

    lines = [
        f"GitHub verification for @{username}: {overall:.1f}/100",
        f"Checked {len(factor_scores)} factors across 5 weighted groups.",
        "",
        "Group scores (weighted):",
    ]
    for g, w in GITHUB_GROUP_WEIGHTS.items():
        gs = group_scores.get(g, 0.0)
        lines.append(f"  • {g}: {gs:.1f}/100  (weight {int(w*100)}%)")
    lines.append("")
    lines.append("Top factor strengths:")
    for fname, val in top_factors:
        lines.append(f"  • {fname}: {val:.1f}/10")
    lines.append("")
    lines.append("Factors with lowest scores:")
    for fname, val in bottom_factors:
        lines.append(f"  • {fname}: {val:.1f}/10")

    explanation = "\n".join(lines)

    return {
        "score":           round(overall, 1),
        "factors_checked": list(factor_scores.keys()),
        "factor_scores":   factor_scores,
        "group_scores":    group_scores,
        "category_scores": category_scores,     # legacy flat categories
        "explanation":     explanation,
        "username":        username,
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
