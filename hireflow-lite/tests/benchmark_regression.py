import os
import json
import glob
from pathlib import Path
import sys

# Add parent to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from resume_parser import parse_resume

RESUMES_DIR = r"C:\Users\radha\Desktop\finalyear_project\code_file\hireflow\hireflow-lite\uploads\9d3e5586\resumes\_Resume Upload (Filename must be Roll Number, e.g., 23AD044.pdf) (File responses)"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "regression")

def run_benchmark():
    files = glob.glob(os.path.join(RESUMES_DIR, "*.*"))
    print(f"Found {len(files)} resumes for regression testing.")
    
    results = {}
    for filepath in files:
        filename = os.path.basename(filepath)
        print(f"\nProcessing {filename}...")
        try:
            parsed = parse_resume(filepath, header_threshold=0.30)
            
            output_path = os.path.join(OUTPUT_DIR, f"{filename}.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(parsed, f, indent=2)
                
            projects = parsed.get("projects", [])
            print(f"  Extracted {len(projects)} projects.")
            for p in projects:
                print(f"    - {p.get('name')} | Tech: {', '.join(p.get('technologies', []))}")
                
            results[filename] = {
                "num_projects": len(projects),
                "projects": [p.get("name") for p in projects]
            }
        except Exception as e:
            print(f"  Failed: {e}")
            
    print("\nBenchmark complete. Inspect the output JSONs in tests/regression for detailed evidence mapping.")

if __name__ == "__main__":
    run_benchmark()
