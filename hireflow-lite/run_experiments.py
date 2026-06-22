import sys
import os
import json
import argparse
from pathlib import Path

# Add current directory to path to import main and ats_engine
current_dir = Path(__file__).parent.resolve()
sys.path.append(str(current_dir))

from main import run_pipeline

# Define the 4 Job Descriptions
JOB_DESCRIPTIONS = {
    "Cloud Engineer": (
        "We are seeking a Cloud Engineer experienced in AWS, Azure, GCP, Docker, "
        "Kubernetes, Terraform, cloud-native deployments, and CI/CD pipelines."
    ),
    "System Engineer (C)": (
        "We are seeking a Systems Software Engineer with strong proficiency in C/C++, "
        "Linux kernel development, systems programming, low-level architecture, "
        "multi-threading, and operating systems concepts."
    ),
    "AI Engineer": (
        "We are seeking an AI/Machine Learning Engineer to develop, train, and deploy "
        "AI/ML solutions, deep learning models, LLMs, and Python-based GenAI applications."
    ),
    "Full Stack Developer (FSD)": (
        "We are seeking a Full Stack Web Developer experienced in React, Next.js, "
        "Node.js, TypeScript, NextJS, databases, and building responsive web applications."
    )
}

ALGORITHMS = ["bm25", "neural", "hybrid_efficient", "colbert"]

def main():
    resume_path = r"C:\Users\radha\Desktop\gopi__resumesmbc.docx"
    token = "<YOUR_GITHUB_TOKEN>"
    factors = ["active_days", "fork_ratio", "language_match", "project_exists", "total_commits"]
    
    print("================================================================================")
    # Force UTF-8 encoding on Windows to prevent output crashes
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print(f"Starting HireFlow-Lite Batch Experimentation for 4 Roles across {len(ALGORITHMS)} Algos")
    print(f"Resume: {resume_path}")
    print("================================================================================")

    results = []

    for role_name, jd_text in JOB_DESCRIPTIONS.items():
        print(f"\n🚀 Evaluating role: {role_name}...")
        for algo in ALGORITHMS:
            print(f"  → Running algorithm: {algo}...")
            
            # Construct a namespace argument object
            args = argparse.Namespace(
                resume=resume_path,
                jd=jd_text,
                token=token,
                algo=algo,
                factors=factors,
                ats_weight=0.6,
                github_weight=0.4,
                output="temp_results.json"
            )
            
            try:
                # Capture standard output temporarily to prevent cluttering console too much
                # or just run it directly so we see progress
                output = run_pipeline(args)
                
                results.append({
                    "role": role_name,
                    "algorithm": algo,
                    "ats_score": output["ats"]["score"],
                    "github_score": output["github"]["score"],
                    "final_score": output["final_score"],
                    "device_used": output["device_used"],
                    "elapsed_seconds": output["elapsed_seconds"],
                    "status": "Success"
                })
            except Exception as e:
                print(f"  ❌ Failed for {role_name} with {algo}: {e}")
                results.append({
                    "role": role_name,
                    "algorithm": algo,
                    "ats_score": 0.0,
                    "github_score": 0.0,
                    "final_score": 0.0,
                    "device_used": "N/A",
                    "elapsed_seconds": 0.0,
                    "status": f"Failed: {e}"
                })

    # Save summary to file
    summary_path = current_dir / "batch_experiment_results.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n\n================================================================================")
    print("                             BATCH EVALUATION RESULTS SUMMARY")
    print("================================================================================")
    print(f"| {'Role':<28} | {'Algo':<18} | {'ATS Score':<10} | {'GH Score':<10} | {'Final Score':<11} | {'Status':<8} |")
    print("|" + "-"*30 + "|" + "-"*20 + "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*13 + "|" + "-"*10 + "|")
    for r in results:
        print(f"| {r['role']:<28} | {r['algorithm']:<18} | {r['ats_score']:<10.1f} | {r['github_score']:<10.1f} | {r['final_score']:<11.1f} | {r['status']:<8} |")
    print("================================================================================")
    print(f"Detailed results saved to {summary_path}")

if __name__ == "__main__":
    main()
