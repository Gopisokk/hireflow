"""
HireFlow-Lite — Main CLI Runner
---------------------------------
Pure Python CLI prototype that scores a single resume against a job description
using ATS matching + GitHub profile verification.

Usage:
  python main.py \
    --resume ./my_resume.pdf \
    --jd "We need a Python backend developer with FastAPI, PostgreSQL, Docker" \
    --token ghp_YOURTOKEN \
    --algo hybrid_efficient \
    --factors active_days fork_ratio language_match project_exists total_commits \
    --ats-weight 0.6 --github-weight 0.4
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Force UTF-8 stdout/stderr on Windows to avoid UnicodeEncodeError
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass



# ═══════════════════════════════════════════════════════════════════════════════
#  GPU Detection
# ═══════════════════════════════════════════════════════════════════════════════

def detect_device() -> str:
    """Detect CUDA availability and print GPU info."""
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            vram_mb = torch.cuda.get_device_properties(0).total_memory // 1024**2
            print(f"  🖥️  Device: {device}")
            print(f"  🖥️  GPU: {gpu_name}")
            print(f"  🖥️  VRAM: {vram_mb} MB")
        else:
            device = "cpu"
            print(f"  🖥️  Device: {device} (no CUDA GPU detected)")
        return device
    except ImportError:
        print("  🖥️  Device: cpu (torch not installed)")
        return "cpu"


# ═══════════════════════════════════════════════════════════════════════════════
#  Available Factors (for --list-factors)
# ═══════════════════════════════════════════════════════════════════════════════

ALL_FACTORS = {
    "Profile Credibility": [
        "account_age", "profile_completeness", "hireable_flag",
        "organization_memberships", "email_verified",
    ],
    "Contribution Activity": [
        "active_days", "total_commits", "longest_streak",
        "current_streak", "contribution_consistency",
        "weekend_activity", "recent_activity", "pr_contributions",
    ],
    "Repository Authenticity": [
        "fork_ratio", "original_repos", "fork_detection",
        "commit_authorship", "first_commit_date",
        "sole_contributor", "multi_contributor",
    ],
    "Code Quality Signals": [
        "readme_quality", "has_tests", "ci_cd",
        "repo_topics", "avg_commits_per_repo", "issue_pr_activity",
    ],
    "Skill Verification": [
        "language_match", "language_depth", "language_diversity",
        "primary_language_match", "tech_stack_alignment",
    ],
    "Resume Cross-Reference": [
        "project_exists", "project_fuzzy_match",
        "readme_resume_alignment", "project_age_vs_experience",
        "tech_in_repo_matches_resume",
    ],
    "Social Proof": [
        "stars_received", "forks_received",
        "open_source_contributions", "pinned_repo_quality",
    ],
}


def print_available_factors():
    """Print all 40 factors organised by category."""
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║           HireFlow-Lite — Available GitHub Factors          ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")
    total = 0
    for category, factors in ALL_FACTORS.items():
        print(f"  📂 {category}:")
        for f in factors:
            print(f"      • {f}")
            total += 1
        print()
    print(f"  Total: {total} factors\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(args) -> dict:
    """Execute the full HireFlow-Lite pipeline."""

    start_time = time.time()

    # ── Banner ────────────────────────────────────────────────────────────────
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                 HireFlow-Lite  v1.0                        ║")
    print("║         Resume Scoring & GitHub Verification CLI           ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    # ── Step 0: Device detection ──────────────────────────────────────────────
    print("━━━ Step 0: Device Detection ━━━")
    device = detect_device()
    print()

    # ── Step 1: Parse resume ──────────────────────────────────────────────────
    print("━━━ Step 1: Resume Parsing ━━━")
    from resume_parser import parse_resume

    resume_data = parse_resume(args.resume)
    print()

    candidate_name = resume_data["name"] or "Unknown"
    candidate_email = resume_data["email"] or ""
    github_username = resume_data["github_username"] or ""
    resume_skills = resume_data["skills"]
    resume_projects = [p["name"] for p in resume_data["projects"]]

    print(f"  📋 Candidate: {candidate_name}")
    print(f"  📧 Email: {candidate_email or 'Not found'}")
    print(f"  🐙 GitHub: {github_username or 'Not found'}")
    print(f"  🔧 Skills ({len(resume_skills)}): {', '.join(resume_skills[:10])}"
          + (f" +{len(resume_skills)-10} more" if len(resume_skills) > 10 else ""))
    print(f"  📁 Projects ({len(resume_projects)}): "
          + ", ".join(resume_projects[:5])
          + (f" +{len(resume_projects)-5} more" if len(resume_projects) > 5 else ""))
    if resume_data["education"]:
        edu_preview = resume_data["education"][:100].replace("\n", " ")
        print(f"  🎓 Education: {edu_preview}...")
    print()

    # ── Step 2: ATS Scoring ───────────────────────────────────────────────────
    print(f"━━━ Step 2: ATS Scoring (algorithm: {args.algo}) ━━━")
    from ats_engine import run_ats, generate_hypothetical_resume, re_rank_candidates_llm

    # SOTA HYRE query expansion
    hyre_text = generate_hypothetical_resume(args.jd)

    ats_result = run_ats(
        resume_text=resume_data["raw_text"],
        jd_text=args.jd,
        resume_skills=resume_skills,
        algo=args.algo,
        device=device,
        hyre_text=hyre_text,
    )

    ats_score = ats_result["score"]
    print()
    print(f"  ✅ ATS Score: {ats_score:.1f}/100")
    print(f"  📊 Algorithm: {ats_result['algo_used']}")
    if ats_result["matched_skills"]:
        print(f"  ✓ Matched Skills ({len(ats_result['matched_skills'])}): "
              + ", ".join(ats_result["matched_skills"][:10]))
    if ats_result["missing_skills"]:
        print(f"  ✗ Missing Skills ({len(ats_result['missing_skills'])}): "
              + ", ".join(ats_result["missing_skills"][:10]))
    print(f"  💡 {ats_result['explanation']}")
    print()

    # ── Step 3: GitHub Verification ───────────────────────────────────────────
    github_result = {
        "score": 0.0,
        "factors_checked": [],
        "factor_scores": {},
        "explanation": "Skipped — no GitHub username found.",
    }

    if github_username and args.token:
        print(f"━━━ Step 3: GitHub Verification ({len(args.factors)} factors) ━━━")
        from github_verifier import run_github_verification

        try:
            github_result = run_github_verification(
                username=github_username,
                token=args.token,
                resume_projects=resume_projects,
                resume_skills=resume_skills,
                resume_email=candidate_email,
                jd_text=args.jd,
                selected_factors=args.factors,
            )

            github_score = github_result["score"]
            print()
            print(f"  ✅ GitHub Score: {github_score:.1f}/100")
            print(f"  📊 Factors checked: {len(github_result['factors_checked'])}")

            # Print per-factor breakdown grouped by category
            for category, cat_factors in ALL_FACTORS.items():
                checked_in_cat = [
                    f for f in cat_factors
                    if f in github_result["factor_scores"]
                ]
                if checked_in_cat:
                    print(f"\n  📂 {category}:")
                    for f in checked_in_cat:
                        score = github_result["factor_scores"][f]
                        bar = "█" * int(score) + "░" * (10 - int(score))
                        print(f"      {f:<32} {bar} {score:.1f}/10")

            print(f"\n  💡 {github_result['explanation']}")

        except Exception as exc:
            print(f"  ❌ GitHub verification failed: {exc}")
            github_result["explanation"] = f"Error: {exc}"
    elif not github_username:
        print("━━━ Step 3: GitHub Verification ━━━")
        print("  ⚠️  Skipped — no GitHub username found in resume.")
    elif not args.token:
        print("━━━ Step 3: GitHub Verification ━━━")
        print("  ⚠️  Skipped — no GitHub token provided (--token).")
        github_result["explanation"] = "Skipped — no GitHub token provided."

    print()

    # ── Step 3.5: Project Verification ────────────────────────────────────────
    project_verification_results = []
    if github_username and args.token and resume_data["projects"]:
        print("━━━ Step 3.5: Project Verification (SBERT semantic matching) ━━━")
        try:
            from project_verifier import verify_projects
            from github_verifier import fetch_github_profile

            # Reuse the cached GitHub profile data (no second API call)
            github_data = fetch_github_profile(github_username, args.token)
            github_repos = (github_data.get("repositories") or {}).get("nodes") or []

            project_verification_results = verify_projects(
                resume_projects=resume_data["projects"],
                github_repos=github_repos,
                device=device,
            )
        except Exception as exc:
            print(f"  ❌ Project verification failed: {exc}")
    elif not resume_data["projects"]:
        print("━━━ Step 3.5: Project Verification ━━━")
        print("  ⚠️  Skipped — no projects extracted from resume.")
    print()

    # ── Step 4: Final Score ───────────────────────────────────────────────────
    print("━━━ Step 4: Final Score ━━━")
    from scorer import compute_final_score

    github_score = github_result["score"]
    final = compute_final_score(
        ats_score=ats_score,
        github_score=github_score,
        ats_weight=args.ats_weight,
        github_weight=args.github_weight,
    )

    # ── Step 4.5: SOTA Stage 2 LLM Re-ranking & Evaluation ───────────────────
    print("━━━ Step 4.5: SOTA Stage 2 LLM Evaluation ━━━")
    candidate_dict = {
        "name": candidate_name,
        "resume_text": resume_data["raw_text"],
        "score": final["final_score"],
        "matched_skills": ats_result["matched_skills"],
        "missing_skills": ats_result["missing_skills"],
    }
    
    import os
    re_ranked = re_rank_candidates_llm([candidate_dict], args.jd)
    updated_candidate = re_ranked[0]
    
    llm_score = updated_candidate["score"]
    llm_explanation = updated_candidate.get("llm_explanation", {
        "strengths": [],
        "gaps": [],
        "fit_justification": "Stage 1 scoring retained (LLM re-ranker bypassed)."
    })
    
    is_llm_active = "llm_explanation" in updated_candidate and len(llm_explanation.get("strengths", [])) > 0
    
    if is_llm_active:
        print(f"  ✅ Adjusted Final Score: {llm_score:.1f}/100")
        print("  ✓ Strengths:")
        for strg in llm_explanation["strengths"]:
            print(f"      - {strg}")
        print("  ✗ Gaps:")
        for gap in llm_explanation["gaps"]:
            print(f"      - {gap}")
        print(f"  💡 Justification: {llm_explanation['fit_justification']}")
    else:
        print("  ⚠️  Skipped — no local Ollama or cloud API keys found, or LLM failed.")
    print()

    elapsed = time.time() - start_time

    print(f"  ATS Score:       {ats_score:6.1f}/100  (weight: {final['ats_weight']:.0%})")
    print(f"  GitHub Score:    {github_score:6.1f}/100  (weight: {final['github_weight']:.0%})")
    print(f"  ─────────────────────────────")
    print(f"  Stage 1 Score:   {final['final_score']:6.1f}/100")
    if is_llm_active:
        print(f"  LLM Adjusted:    {llm_score:6.1f}/100")
    print(f"  Time elapsed:    {elapsed:.1f}s")
    print()

    # ── Build output ──────────────────────────────────────────────────────────
    output = {
        "candidate_name": candidate_name,
        "candidate_email": candidate_email,
        "github_username": github_username,
        "ats": {
            "score": round(ats_score, 2),
            "algo_used": ats_result["algo_used"],
            "matched_skills": ats_result["matched_skills"],
            "missing_skills": ats_result["missing_skills"],
            "explanation": ats_result["explanation"],
        },
        "github": {
            "score": round(github_score, 2),
            "factors_checked": github_result["factors_checked"],
            "factor_scores": {
                k: round(v, 2)
                for k, v in github_result["factor_scores"].items()
            },
            "explanation": github_result["explanation"],
        },
        "project_verification": [
            {
                "claimed_project": pv["claimed_project"],
                "status": pv["status"],
                "matched_repo": pv["matched_repo"],
                "similarity": pv["similarity"],
                "is_fork": pv["is_fork"],
            }
            for pv in project_verification_results
        ],
        "final_score": llm_score if is_llm_active else final["final_score"],
        "stage1_final_score": final["final_score"],
        "llm_evaluation": {
            "is_active": is_llm_active,
            "strengths": llm_explanation["strengths"],
            "gaps": llm_explanation["gaps"],
            "justification": llm_explanation["fit_justification"],
        },
        "weights": {
            "ats_weight": final["ats_weight"],
            "github_weight": final["github_weight"],
        },
        "device_used": device,
        "elapsed_seconds": round(elapsed, 2),
    }

    return output


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        prog="HireFlow-Lite",
        description="Score a resume against a job description using ATS + GitHub verification.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --resume resume.pdf --jd "Python developer with FastAPI" --token ghp_XXX
  python main.py --resume cv.docx --jd "ML Engineer" --token ghp_XXX --algo neural
  python main.py --list-factors
        """,
    )

    parser.add_argument(
        "--resume", type=str,
        help="Path to resume file (PDF or DOCX)",
    )
    parser.add_argument(
        "--jd", type=str,
        help="Job description text (or path to .txt file)",
    )
    parser.add_argument(
        "--token", type=str, default="",
        help="GitHub personal access token for profile verification",
    )
    parser.add_argument(
        "--algo", type=str, default="hybrid_efficient",
        choices=["bm25", "neural", "hybrid_efficient", "colbert"],
        help="ATS scoring algorithm (default: hybrid_efficient)",
    )
    parser.add_argument(
        "--factors", nargs="+",
        default=[
            "active_days", "fork_ratio", "language_match",
            "project_exists", "total_commits",
        ],
        help="GitHub factors to evaluate (space-separated)",
    )
    parser.add_argument(
        "--ats-weight", type=float, default=0.6,
        help="Weight for ATS score (default: 0.6)",
    )
    parser.add_argument(
        "--github-weight", type=float, default=0.4,
        help="Weight for GitHub score (default: 0.4)",
    )
    parser.add_argument(
        "--list-factors", action="store_true",
        help="Print all available GitHub factors and exit",
    )
    parser.add_argument(
        "--output", type=str, default="results.json",
        help="Output JSON file path (default: results.json)",
    )

    args = parser.parse_args()

    # Handle --list-factors
    if args.list_factors:
        print_available_factors()
        sys.exit(0)

    # Validate required args
    if not args.resume:
        parser.error("--resume is required (or use --list-factors)")
    if not args.jd:
        parser.error("--jd is required")

    # If --jd is a file path, read its contents
    jd_path = Path(args.jd)
    if jd_path.exists() and jd_path.suffix in (".txt", ".md"):
        print(f"  📄 Reading JD from file: {jd_path}")
        args.jd = jd_path.read_text(encoding="utf-8")

    # Validate factors
    all_factor_names = []
    for factors in ALL_FACTORS.values():
        all_factor_names.extend(factors)

    invalid_factors = [f for f in args.factors if f not in all_factor_names]
    if invalid_factors:
        print(f"  ⚠️  Unknown factors (will be skipped): {invalid_factors}")
        args.factors = [f for f in args.factors if f in all_factor_names]

    # Run the pipeline
    try:
        result = run_pipeline(args)

        # Save results
        output_path = Path(args.output)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"  💾 Results saved to: {output_path.resolve()}")
        print()
        print("═" * 62)
        print("  Done! Run with --list-factors to see all 40 GitHub factors.")
        print("═" * 62)
        print()

    except FileNotFoundError as exc:
        print(f"\n  ❌ File not found: {exc}")
        sys.exit(1)
    except ValueError as exc:
        print(f"\n  ❌ Error: {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n  ⚠️  Interrupted by user.")
        sys.exit(130)
    except Exception as exc:
        print(f"\n  ❌ Unexpected error: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
