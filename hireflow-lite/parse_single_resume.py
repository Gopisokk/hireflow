"""
parse_single_resume.py — Single Resume to Canonical JSON Converter
===================================================================
Converts any given PDF or DOCX resume into canonical structured JSON using
Qwen via local Ollama.

Usage:
    venv\\Scripts\\python.exe parse_single_resume.py <path_to_resume> [output_file.json] [--model MODEL_NAME]

Examples:
    venv\\Scripts\\python.exe parse_single_resume.py "path/to/resume.pdf"
    venv\\Scripts\\python.exe parse_single_resume.py "path/to/resume.pdf" --model qwen2.5:1.5b-instruct
"""

from __future__ import annotations

import os
import sys
import json
import argparse

# Force UTF-8 encoding for Windows console
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from qwen_resume_parser import parse_resume, DEFAULT_MODEL


def main():
    parser = argparse.ArgumentParser(description="Convert a PDF or DOCX resume into Canonical JSON using Qwen via Ollama.")
    parser.add_argument("resume_path", help="Path to the PDF or DOCX resume file")
    parser.add_argument("output_path", nargs="?", default=None, help="Optional output JSON filepath. If omitted, saves to <resume_name>.json")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model name (default: {DEFAULT_MODEL})")

    args = parser.parse_args()

    resume_path = os.path.abspath(args.resume_path)
    if not os.path.exists(resume_path):
        print(f"Error: File not found at '{resume_path}'", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print(f"HIREFLOW-LITE: RESUME TO CANONICAL JSON CONVERTER")
    print("=" * 70)
    print(f"Input Resume : {resume_path}")
    print(f"Model Engine : {args.model} (Ollama API)")
    print("Processing... Please wait a few seconds...\n")

    # Run extraction & evidence validation
    result = parse_resume(resume_path, model_name=args.model)

    # Determine output filepath
    if args.output_path:
        out_file = os.path.abspath(args.output_path)
    else:
        base_name = os.path.basename(resume_path)
        out_file = os.path.abspath(f"{os.path.splitext(base_name)[0]}_parsed.json")

    # Save formatted JSON to file
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("Extraction Complete!")
    print(f"Status        : {result.get('status', 'unknown').upper()}")
    print(f"Projects Count: {len(result.get('projects', []))}")
    print(f"Latency       : {result.get('parser_metadata', {}).get('latency_ms', 0) / 1000:.2f}s")
    print(f"Saved Output  : {out_file}\n")
    print("=" * 70)
    print("EXTRACTED CANONICAL JSON:")
    print("=" * 70)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
