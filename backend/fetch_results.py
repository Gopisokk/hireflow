import httpx
import json
import time

BASE = "http://localhost:8000"

print("Waiting for pipeline to complete...")
time.sleep(60)

# Get pipeline result
print("\n--- PIPELINE RESULT ---")
r = httpx.get(f"{BASE}/api/pipeline/result", timeout=15)
if r.status_code == 200:
    data = r.json()
    run = data.get("run", {})
    print(f"Run status  : {run.get('status')}")
    print(f"Algorithm   : {run.get('algorithm')}")
    print(f"Total cands : {run.get('total_candidates')}")
    print(f"Shortlisted : {run.get('shortlisted')}")
    print(f"Completed at: {run.get('completed_at')}")
    print()
    print("--- RANKED CANDIDATES ---")
    for c in data.get("candidates", []):
        print(f"  #{c['rank']}  {c['name']} ({c['roll_number']})")
        print(f"       ATS Score    : {c['ats_score']}")
        print(f"       GitHub Score : {c['github_score']}")
        print(f"       Final Score  : {c['final_score']}")
        print(f"       Stage        : {c['stage']}")
else:
    print(f"Result error {r.status_code}: {r.text[:400]}")

# Get full candidate profile
print()
print("--- FULL CANDIDATE PROFILE (23AD044) ---")
r2 = httpx.get(f"{BASE}/api/candidates/23AD044", timeout=15)
if r2.status_code == 200:
    p = r2.json()
    s = p.get("student", {})
    print(f"Name        : {s.get('name')}")
    print(f"GitHub User : {s.get('github_username')}")
    print(f"ATS Score   : {s.get('ats_score')}")
    print(f"GitHub Score: {s.get('github_score')}")
    print(f"Final Score : {s.get('final_score')}")
    print(f"Stage       : {s.get('stage')}")
    print()
    skills = [sk["skill"] for sk in p.get("skills", [])]
    print(f"Matched Skills ({len(skills)}): {skills}")
    print()
    projects = p.get("projects", [])
    print(f"Projects detected ({len(projects)}):")
    for pr in projects:
        verified = "VERIFIED" if pr["github_verified"] else "unverified"
        fork = " [FORK]" if pr["is_fork"] else ""
        print(f"  - {pr['project_name']}  [{verified}]{fork}")
    print()
    gh = p.get("github_profile")
    if gh:
        print("GitHub Profile:")
        print(f"  Active Days  : {gh.get('active_days')}")
        print(f"  Total Commits: {gh.get('total_commits')}")
        print(f"  Total PRs    : {gh.get('total_prs')}")
        print(f"  Fork Ratio   : {gh.get('fork_ratio')}")
        print(f"  Top Languages: {gh.get('top_languages')}")
        print(f"  GitHub Score : {gh.get('contribution_score')}")
    else:
        print("GitHub profile: not verified")
else:
    print(f"Profile error {r2.status_code}: {r2.text[:300]}")
