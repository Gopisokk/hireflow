"""Diagnostic: show actual GitHub repos vs resume project names + matching attempts."""
import asyncio
import httpx
import re
from difflib import SequenceMatcher

TOKEN = "<YOUR_GITHUB_TOKEN>"
USERNAME = "Gopisokk"

RESUME_PROJECTS = [
    "Smart Product Pricing",
    "Built a text",
    "Trained two strong specialists",
    "Competitive programming",
    "CodeChef",
]

GQL = """
query($username: String!) {
  user(login: $username) {
    repositories(first: 100, orderBy: {field: UPDATED_AT, direction: DESC}) {
      nodes {
        name
        url
        isFork
        primaryLanguage { name }
        languages(first: 5) { nodes { name } }
        defaultBranchRef {
          target {
            ... on Commit {
              history(first: 1) { totalCount }
            }
          }
        }
      }
    }
  }
}
"""

def normalize(s):
    return re.sub(r"[^a-z0-9 ]", " ", s.lower()).strip()

def fuzzy(a, b):
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()

async def main():
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(
            "https://api.github.com/graphql",
            json={"query": GQL, "variables": {"username": USERNAME}},
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        repos = resp.json()["data"]["user"]["repositories"]["nodes"]

    print(f"=== {USERNAME} has {len(repos)} repos ===\n")
    print(f"{'Repo Name':<40} {'Fork':^5} {'Lang':<15} {'Commits':>7}")
    print("-" * 75)
    for r in repos:
        lang = (r.get("primaryLanguage") or {}).get("name", "—")
        commits = (
            (r.get("defaultBranchRef") or {})
            .get("target", {})
            .get("history", {})
            .get("totalCount", 0)
        )
        fork = "YES" if r["isFork"] else "no"
        print(f"{r['name']:<40} {fork:^5} {lang:<15} {commits:>7}")

    print("\n\n=== FUZZY MATCH ATTEMPTS ===")
    print("(threshold currently 0.40)\n")
    for proj in RESUME_PROJECTS:
        print(f"Resume project: '{proj}'")
        scores = [(r["name"], fuzzy(proj, r["name"])) for r in repos]
        scores.sort(key=lambda x: x[1], reverse=True)
        for name, score in scores[:5]:
            match = "MATCH" if score >= 0.40 else "no match"
            print(f"  {score:.3f}  {name:<35}  {match}")
        print()

asyncio.run(main())
