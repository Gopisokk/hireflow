import sys
sys.path.insert(0, '.')
from pipeline.parser import extract_text
from pipeline.skill_extractor import extract_projects, extract_skills
from pipeline.github_verifier import _keyword_overlap

text = extract_text(r'C:\Users\radha\Desktop\gopi__resumesmbc.docx')
print("=== RESUME PROJECTS EXTRACTED (v2) ===")
projects = extract_projects(text)
for i, p in enumerate(projects, 1):
    print(f"  {i}. {p}")

print()
print("=== MATCHING AGAINST GITHUB REPOS ===")
repos = [
    "hireflow", "Funding-bot", "amazon-ml-2025", "Gopisokk",
    "csv_chatbot", "Chat-with-multiple-PDF-Documents-using-Langchain",
    "facial-recognition", "RapidResQ", "portfolio-website",
    "Open-Library-MCP-Server", "interview_practise_questtions",
]

THRESHOLD = 0.20
matched = 0
for proj in projects:
    scores = [(r, _keyword_overlap(proj, r)) for r in repos]
    best = max(scores, key=lambda x: x[1])
    status = "MATCHED" if best[1] >= THRESHOLD else "no match"
    if best[1] >= THRESHOLD:
        matched += 1
    print(f'  "{proj}" -> {best[0]} (score={best[1]:.3f}) [{status}]')

print()
print(f"Projects matched: {matched}/{len(projects)}")
