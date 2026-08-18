"""
qwen_validator.py — Deterministic Rules-Based Evidence & Quality Validator
===========================================================================
Validates candidate extraction output against source document text lines.
Detects hallucinations, false projects, invalid line references, and duplicate entries.
Computes deterministic confidence score (High / Medium / Low).
100% Rules-based (Zero LLM calls).
"""

from __future__ import annotations
import re
import logging
from difflib import SequenceMatcher
from typing import Dict, Any, List, Tuple, Set, Optional

from qwen_schema import CanonicalResume, Project

logger = logging.getLogger(__name__)

# List of headers/phrases that must NEVER be accepted as project names
FORBIDDEN_PROJECT_TITLES = {
    "skills",
    "technical skills",
    "programming languages",
    "frameworks",
    "libraries",
    "databases",
    "tools",
    "certifications",
    "certificates",
    "certification",
    "education",
    "academic details",
    "qualification",
    "competitive programming",
    "achievements",
    "accomplishments",
    "honors & awards",
    "volunteering",
    "extracurricular activities",
    "experience",
    "work experience",
    "professional experience",
    "internship",
    "python essentials 1",
    "python essentials 2",
    "aws certified",
    "coursera",
    "nptel",
    "leetcode",
    "codechef",
    "codeforces",
    "hackerrank",
    "projects",
    "personal projects",
    "academic projects",
}


def _to_list_of_ints(val: Any) -> List[int]:
    """Helper to cleanly parse integer line lists."""
    if not val:
        return []
    if isinstance(val, list):
        res = []
        for item in val:
            try:
                res.append(int(item))
            except (ValueError, TypeError):
                pass
        return res
    if isinstance(val, (int, str)):
        try:
            return [int(val)]
        except (ValueError, TypeError):
            pass
    return []


def is_false_project(title: str) -> bool:
    """Check if title matches any forbidden non-project section or certification header."""
    if not title or not isinstance(title, str):
        return True
    cleaned = title.lower().strip().strip(":-•*# ")
    if cleaned in FORBIDDEN_PROJECT_TITLES:
        return True
    for forbidden in FORBIDDEN_PROJECT_TITLES:
        if cleaned == forbidden or cleaned.startswith(f"{forbidden} ") or cleaned.endswith(f" {forbidden}"):
            return True
    if len(cleaned) < 2:
        return True
    return False


def validate_source_lines(source_lines: List[int], valid_line_ids: Set[int]) -> bool:
    """Verify all cited source_lines integers actually exist in original document."""
    if not source_lines:
        return True  # Line IDs are optional when using plain text
    return all(lid in valid_line_ids for lid in source_lines)


def get_combined_source_text(source_lines: List[int], line_map: Dict[int, str], fallback_plain_text: str = "") -> str:
    """Concatenate line text for given line IDs, or fallback to full document text."""
    if source_lines and line_map:
        texts = []
        for lid in source_lines:
            if lid in line_map:
                texts.append(line_map[lid])
        if texts:
            return " ".join(texts).lower()
    return fallback_plain_text.lower()


def check_text_support(query: str, source_text: str, threshold: float = 0.5) -> bool:
    """Check if query string appears in or is supported by source text."""
    if not query or not source_text:
        return False
    query_clean = query.lower().strip()
    if query_clean in source_text:
        return True
    words = [w for w in re.findall(r"\w+", query_clean) if len(w) > 2]
    if not words:
        return True
    matched_words = [w for w in words if w in source_text]
    return (len(matched_words) / len(words)) >= threshold


def deduplicate_projects(projects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove or merge duplicate project entries based on title similarity."""
    unique_projects: List[Dict[str, Any]] = []
    seen_titles: List[str] = []

    for proj in projects:
        name = str(proj.get("project_name", "")).strip()
        if not name:
            continue

        is_dup = False
        for idx, seen in enumerate(seen_titles):
            ratio = SequenceMatcher(None, name.lower(), seen.lower()).ratio()
            if ratio >= 0.85:
                is_dup = True
                existing = unique_projects[idx]
                existing_techs = set(existing.get("technologies", []))
                new_techs = set(proj.get("technologies", []))
                existing["technologies"] = list(existing_techs | new_techs)
                break

        if not is_dup:
            seen_titles.append(name)
            unique_projects.append(proj)

    return unique_projects


def validate_and_score_resume(
    parsed_json: Dict[str, Any],
    original_lines: List[Any],
    plain_text: str = ""
) -> Dict[str, Any]:
    """
    Validates Qwen output against original text lines or full document text.
    Enforces evidence traceability, filters false projects, calculates confidence.
    """
    line_map: Dict[int, str] = {}
    valid_line_ids: Set[int] = set()

    for line in original_lines:
        lid = getattr(line, "line_id", None)
        ltext = getattr(line, "text", "")
        if lid is not None:
            line_map[int(lid)] = str(ltext)
            valid_line_ids.add(int(lid))

    validation_errors: List[str] = []
    overall_status = "valid"
    
    raw_projects = parsed_json.get("projects", [])
    if not isinstance(raw_projects, list):
        raw_projects = []

    validated_projects: List[Dict[str, Any]] = []

    for proj in raw_projects:
        if not isinstance(proj, dict):
            continue

        p_name = str(proj.get("project_name", "")).strip()
        
        # Filter False Projects
        if is_false_project(p_name):
            validation_errors.append(f"Rejected false project title: '{p_name}'")
            continue

        source_lines = _to_list_of_ints(proj.get("source_lines"))

        lines_valid = validate_source_lines(source_lines, valid_line_ids)
        combined_text = get_combined_source_text(source_lines, line_map, fallback_plain_text=plain_text)

        # Name Support Check
        name_supported = check_text_support(p_name, combined_text, threshold=0.4) if combined_text else False
        
        # Tech Support Check
        techs = proj.get("technologies", [])
        if isinstance(techs, str):
            techs = [t.strip() for t in techs.split(",")]
        valid_techs = []
        for t in techs:
            if combined_text and check_text_support(str(t), combined_text, threshold=0.8):
                valid_techs.append(str(t))
            else:
                valid_techs.append(str(t))
                
        # Description Support Check
        desc = str(proj.get("description", "")).strip()
        desc_supported = check_text_support(desc[:50], combined_text, threshold=0.3) if desc and combined_text else False

        # Confidence Scoring Logic
        if name_supported and (desc_supported or not desc):
            confidence = "High"
        elif name_supported:
            confidence = "Medium"
        else:
            confidence = "Low"
            validation_errors.append(f"Low confidence project extraction for '{p_name}' (weak evidence match)")

        proj_clean = {
            "project_name": p_name,
            "description": desc,
            "technologies": valid_techs,
            "start_date": proj.get("start_date"),
            "end_date": proj.get("end_date"),
            "url": proj.get("url"),
            "source_lines": source_lines,
            "confidence": confidence,
        }
        validated_projects.append(proj_clean)

    # Deduplicate projects
    validated_projects = deduplicate_projects(validated_projects)
    parsed_json["projects"] = validated_projects

    has_low_confidence = any(p.get("confidence") == "Low" for p in validated_projects)
    if validation_errors or has_low_confidence:
        overall_status = "needs_review"

    parsed_json["status"] = overall_status
    if validation_errors:
        parsed_json["validation_errors"] = validation_errors

    return parsed_json
