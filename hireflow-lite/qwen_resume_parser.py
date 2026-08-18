"""
qwen_resume_parser.py — Ollama Qwen2.5 Resume Parser (GPU-optimised, timeout-safe)
====================================================================================
Key fixes:
  - Hard 90-second timeout on every Ollama call (via threading)
  - Input text truncated to MAX_CHARS before sending to LLM
  - num_ctx=4096 + num_predict=1536 so the model ALWAYS finishes
  - PDF extracted only ONCE (not twice)
  - Retry on JSON decode error with a tighter prompt
  - Falls back to document extractor + empty JSON on timeout/error
"""

from __future__ import annotations

import os
import sys
import json
import time
import logging
import threading
from typing import Dict, Any, List, Optional

import ollama
from pydantic import ValidationError

from document_extractor import extract_numbered_lines, extract_plain_text
from qwen_schema import CanonicalResume
from qwen_validator import validate_and_score_resume

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "qwen2.5:1.5b-instruct"

# Max chars sent to LLM — keeps prompt under ~1 000 tokens leaving room for JSON
MAX_CHARS = 3500

# Hard timeout per resume (seconds). If Ollama hasn't replied, we give up.
OLLAMA_TIMEOUT_S = 75

# ── Ollama options ─────────────────────────────────────────────────────────────
# num_ctx  = prompt context window. 4096 is safe for 1.5B model on 4GB VRAM.
# num_predict = max output tokens. 1536 is enough for a full resume JSON.
OLLAMA_OPTIONS = {
    "temperature": 0.0,
    "num_ctx":     4096,
    "num_predict": 1536,
    "top_k":       1,       # greedy decode = fastest + most deterministic
    "top_p":       1.0,
    "repeat_penalty": 1.0,
}

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert resume information extraction engine for an ATS platform.
Analyse the resume text and extract structured JSON matching this EXACT schema.
Return ONLY valid JSON, nothing else.

{
  "candidate": {"candidate_name": "string|null", "email": "string|null", "phone": "string|null", "github_url": "string|null", "linkedin_url": "string|null"},
  "education": [{"institution": "string|null", "degree": "string|null", "field": "string|null", "cgpa": "string|null", "start_date": "string|null", "end_date": "string|null"}],
  "experience": [{"organization": "string|null", "role": "string|null", "type": "employment", "description": ["string"], "technologies": ["string"], "start_date": "string|null", "end_date": "string|null"}],
  "projects": [{"project_name": "string", "description": "string", "technologies": ["string"], "start_date": "string|null", "end_date": "string|null", "url": "string|null", "confidence": "High"}],
  "skills": {"programming_languages": ["string"], "frameworks": ["string"], "libraries": ["string"], "databases": ["string"], "cloud": ["string"], "tools": ["string"], "other": ["string"]},
  "certifications": ["string"],
  "achievements": ["string"],
  "competitive_programming": ["string"],
  "volunteering": ["string"]
}

RULES:
1. Extract ONLY information explicitly in the text. Never invent data.
2. Certifications/courses (e.g. NPTEL, AWS Certified) go in certifications[], NOT projects[].
3. LeetCode/CodeChef ratings go in competitive_programming[], NOT projects[].
4. Section headings like "Skills" are NOT projects.
5. If information is missing return empty arrays or null.
"""


# ── Timeout-safe Ollama caller ────────────────────────────────────────────────

def _call_ollama_with_timeout(
    model_name: str,
    messages: list[dict],
    fmt,
    timeout: float = OLLAMA_TIMEOUT_S,
) -> Optional[str]:
    """
    Call ollama.chat() in a daemon thread. Returns the content string,
    or None if the call times out.
    """
    result: list[Optional[str]] = [None]
    error:  list[Optional[str]] = [None]

    def _worker():
        try:
            resp = ollama.chat(
                model=model_name,
                messages=messages,
                format=fmt,
                options=OLLAMA_OPTIONS,
            )
            result[0] = resp.get("message", {}).get("content", "")
        except Exception as exc:
            error[0] = str(exc)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=timeout)

    if t.is_alive():
        # Thread is still running → timeout
        return None
    if error[0]:
        raise RuntimeError(error[0])
    return result[0]


# ── Main parser ───────────────────────────────────────────────────────────────

def parse_resume_text(
    plain_text: str,
    original_lines: Optional[List[Any]] = None,
    model_name: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """
    Send (truncated) resume text to Ollama → validate → return canonical dict.
    """
    t0 = time.time()

    # ── 1. Truncate input ──────────────────────────────────────────────────
    if len(plain_text) > MAX_CHARS:
        plain_text = plain_text[:MAX_CHARS]
        logger.debug(f"Truncated resume text to {MAX_CHARS} chars")

    # ── 2. Choose format arg ───────────────────────────────────────────────
    if "gemma" in model_name.lower() or "3.5" in model_name.lower():
        fmt_arg = "json"
    else:
        fmt_arg = CanonicalResume.model_json_schema()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Extract resume information into JSON:\n\n{plain_text}"},
    ]

    # ── 3. First Ollama call (with timeout) ────────────────────────────────
    raw_content = _call_ollama_with_timeout(model_name, messages, fmt_arg, OLLAMA_TIMEOUT_S)
    latency_ms  = (time.time() - t0) * 1000

    if raw_content is None:
        logger.warning(f"Ollama timed out after {OLLAMA_TIMEOUT_S}s — returning empty result")
        return _empty_result(model_name, latency_ms, timed_out=True)

    # ── 4. Parse JSON ──────────────────────────────────────────────────────
    parsed_data = _try_parse_json(raw_content)
    if parsed_data is None:
        # Retry with a simpler prompt
        logger.warning("JSON decode failed — retrying with tighter prompt")
        retry_msgs = [
            {"role": "system", "content": "Return ONLY valid JSON. No explanation."},
            {"role": "user",   "content": f"Convert this resume to JSON:\n\n{plain_text}"},
        ]
        raw2 = _call_ollama_with_timeout(model_name, retry_msgs, "json", 45)
        if raw2:
            parsed_data = _try_parse_json(raw2)
        if parsed_data is None:
            logger.warning("Retry also failed — returning empty result")
            return _empty_result(model_name, latency_ms)

    # ── 5. Pydantic validation ─────────────────────────────────────────────
    try:
        resume_obj    = CanonicalResume(**parsed_data)
        canonical_dict = resume_obj.model_dump()
    except (ValidationError, Exception):
        canonical_dict = parsed_data

    # ── 6. Deterministic evidence validation ──────────────────────────────
    lines    = original_lines or []
    validated = validate_and_score_resume(canonical_dict, lines, plain_text=plain_text)
    validated["parser_metadata"] = {
        "model":      model_name,
        "latency_ms": round(latency_ms, 2),
        "engine":     "Ollama-API",
        "chars_sent": len(plain_text),
    }
    return validated


# ── Helpers ───────────────────────────────────────────────────────────────────

def _try_parse_json(text: str) -> Optional[dict]:
    """Try to parse JSON; strip markdown fences if present."""
    text = text.strip()
    # Strip ```json ... ``` fences
    if text.startswith("```"):
        lines = text.splitlines()
        text  = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find first '{' ... last '}'
        start = text.find("{")
        end   = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end+1])
            except json.JSONDecodeError:
                pass
    return None


def _empty_result(model_name: str, latency_ms: float, timed_out: bool = False) -> dict:
    """Return an empty-but-valid canonical dict when Ollama fails/times out."""
    return {
        "candidate":              {"candidate_name": None, "email": None, "phone": None,
                                   "github_url": None, "linkedin_url": None},
        "education":              [],
        "experience":             [],
        "projects":               [],
        "skills":                 {"programming_languages": [], "frameworks": [],
                                   "libraries": [], "databases": [], "cloud": [],
                                   "tools": [], "other": []},
        "certifications":         [],
        "achievements":           [],
        "competitive_programming":[],
        "volunteering":           [],
        "status":                 "timed_out" if timed_out else "needs_review",
        "parser_metadata": {
            "model":      model_name,
            "latency_ms": round(latency_ms, 2),
            "engine":     "Ollama-API",
            "timed_out":  timed_out,
        },
    }


# ── High-level entry point ────────────────────────────────────────────────────

def parse_resume(filepath: str, model_name: str = DEFAULT_MODEL) -> Dict[str, Any]:
    """
    High-level entry point:
      1. Extract plain text from PDF/DOCX (once)
      2. Send truncated text to Ollama with timeout
      3. Validate + return canonical dict
    """
    # Extract text ONCE — pass both lines and plain_text from same call
    lines, plain_text = extract_numbered_lines(filepath)

    # Truncate NOW before any processing so we never pass huge text anywhere
    if len(plain_text) > MAX_CHARS:
        logger.debug(f"Pre-truncating {filepath}: {len(plain_text)} → {MAX_CHARS} chars")
        plain_text = plain_text[:MAX_CHARS]
        # Keep only lines whose text appears in truncated range
        lines = [l for l in lines if plain_text.find(l.text) != -1]

    return parse_resume_text(plain_text, lines, model_name=model_name)
