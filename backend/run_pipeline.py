"""Run the full HireFlow pipeline and print detailed results."""
import httpx
import json
import time

BASE = "http://localhost:8000"
RESUME_FOLDER = r"C:\Users\radha\Desktop\finalyear_project\code_file\hireflow\backend\data\resumes"
CSV_PATH = r"C:\Users\radha\Desktop\finalyear_project\code_file\hireflow\backend\data\candidates.csv"
GITHUB_TOKEN = "<YOUR_GITHUB_TOKEN>"
JD = """Position: Machine Learning Engineer
Required Skills: Python, NumPy, Pandas, Scikit-learn, TensorFlow, PyTorch, Machine Learning, Deep Learning, Data Preprocessing, Feature Engineering, Model Evaluation, SQL, Git
Preferred Skills: LangChain, LLMs, RAG, FastAPI, Hugging Face Transformers, Vector Databases, AWS, GCP, Azure, MLOps
Responsibilities: Build and train ML/DL models, data cleaning, feature engineering, NLP, Computer Vision, fine-tune transformers, deploy with FastAPI and Docker.
Experience: 0-3 years. Strong knowledge of Statistics, Probability, Linear Algebra."""

# Step 1: Upload
print("STEP 1: Uploading batch...")
with open(CSV_PATH, "rb") as f:
    r = httpx.post(f"{BASE}/api/batch/upload",
        files={"csv_file": ("candidates.csv", f, "text/csv")},
        data={"job_description": JD, "resume_folder": RESUME_FOLDER},
        timeout=30)
print(f"  Upload: {r.status_code} — inserted={r.json().get('inserted')}, skipped={r.json().get('skipped')}")

# Step 2: Config
print("STEP 2: Setting config...")
r = httpx.post(f"{BASE}/api/config",
    json={"ats_weight": 0.6, "github_weight": 0.4, "algorithm": "hybrid_efficient"},
    timeout=10)
print(f"  Config: {r.status_code}")

# Step 3: Start
print("STEP 3: Starting pipeline...")
r = httpx.post(f"{BASE}/api/pipeline/start",
    json={"job_description": JD, "github_token": GITHUB_TOKEN, "resume_folder": RESUME_FOLDER},
    timeout=30)
print(f"  Start: {r.status_code} — {r.json().get('status')}")

# Wait
print("\nWaiting 90s for pipeline (parsing + ATS + GitHub verification)...")
for i in range(9):
    time.sleep(10)
    print(f"  {(i+1)*10}s elapsed...")

# Step 4: Results
print("\n" + "="*60)
print("PIPELINE RESULTS")
print("="*60)
r = httpx.get(f"{BASE}/api/pipeline/result", timeout=15)
data = r.json()
run = data.get("run", {})
print(f"Status     : {run.get('status')}")
print(f"Algorithm  : {run.get('algorithm')}")
print(f"Shortlisted: {run.get('shortlisted')} / {run.get('total_candidates')}")
print()

for c in data.get("candidates", []):
    print(f"#{c['rank']}  {c['name']} ({c['roll_number']})")
    print(f"    ATS Score    : {c['ats_score']}/100")
    print(f"    GitHub Score : {c['github_score']}/100")
    print(f"    Final Score  : {c['final_score']}/100")

# Detailed profile
print("\n" + "="*60)
print("DETAILED PROFILE — 23AD044")
print("="*60)
r2 = httpx.get(f"{BASE}/api/candidates/23AD044", timeout=15)
p = r2.json()
s = p.get("student", {})
skills = [sk["skill"] for sk in p.get("skills", [])]
projects = p.get("projects", [])
gh = p.get("github_profile", {})

print(f"Name        : {s.get('name')}")
print(f"GitHub      : github.com/{s.get('github_username')}")
print(f"\nMatched Skills ({len(skills)}):")
print(f"  {', '.join(skills)}")

print(f"\nProjects detected ({len(projects)}):")
for pr in projects:
    verified = "GITHUB VERIFIED" if pr["github_verified"] else "resume only"
    fork = " [FORK - penalised]" if pr["is_fork"] else ""
    url = f" -> {pr['github_repo_url']}" if pr.get("github_repo_url") else ""
    print(f"  - {pr['project_name']}  [{verified}]{fork}{url}")

if gh:
    print(f"\nGitHub Activity:")
    print(f"  Active Days (overall) : {gh.get('active_days')}  [low weight - easily gamed]")
    print(f"  Total Commits         : {gh.get('total_commits')}  [global count]")
    print(f"  Total PRs             : {gh.get('total_prs')}  [bonus signal]")
    print(f"  Fork Ratio            : {gh.get('fork_ratio')*100:.1f}% repos are forks")
    print(f"  GitHub Score (v2)     : {gh.get('contribution_score')}/100")
