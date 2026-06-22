"""
HireFlow-Lite — Resume Parser
-------------------------------
Extracts text from PDF/DOCX resume files and parses structured fields
(name, email, phone, GitHub username, skills, projects, education)
using regex + heuristics.

Supported formats: .pdf, .docx
"""

import re
from pathlib import Path
import sys

# Force UTF-8 stdout/stderr on Windows to avoid UnicodeEncodeError
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import fitz  # PyMuPDF
from docx import Document


# ═══════════════════════════════════════════════════════════════════════════════
#  Text Extraction
# ═══════════════════════════════════════════════════════════════════════════════

def _clean_text(raw: str) -> str:
    """Normalise whitespace, preserve paragraph boundaries."""
    text = raw.replace("\t", " ").replace("\f", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[^\S\n]+", " ", text)
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def _extract_pdf(filepath: str) -> str:
    """Extract text from a PDF using PyMuPDF in a layout-aware manner (handling columns)."""
    doc = fitz.open(filepath)
    pages = []
    for page in doc:
        rect = page.rect
        width = rect.width
        
        # Retrieve text blocks
        blocks = page.get_text("blocks")
        if not blocks:
            continue
            
        # Group blocks into left column, right column, and full-width blocks
        left_col = []
        right_col = []
        full_width = []
        
        midpoint = width / 2.0
        
        for b in blocks:
            x0, y0, x1, y1, text, block_no, block_type = b
            if block_type != 0:  # Skip image blocks
                continue
            text = text.strip()
            if not text:
                continue
                
            # Classify blocks by horizontal coordinates (with a 20px tolerance)
            if x1 <= midpoint + 20:
                left_col.append((y0, text))
            elif x0 >= midpoint - 20:
                right_col.append((y0, text))
            else:
                full_width.append((y0, text))
                
        # Sort blocks vertically within columns
        left_col.sort(key=lambda x: x[0])
        right_col.sort(key=lambda x: x[0])
        
        # Reconstruct page text column-by-column if multi-column layout is detected
        if len(left_col) > 1 or len(right_col) > 1:
            page_blocks = []
            first_col_y = min([y for y, _ in left_col + right_col]) if (left_col or right_col) else 0
            
            # Header full-width blocks
            top_blocks = [text for y, text in full_width if y < first_col_y]
            # Footer/bottom full-width blocks
            bottom_blocks = [text for y, text in full_width if y >= first_col_y]
            
            page_blocks.extend(top_blocks)
            page_blocks.extend([text for y, text in left_col])
            page_blocks.extend([text for y, text in right_col])
            page_blocks.extend(bottom_blocks)
            page_text = "\n\n".join(page_blocks)
        else:
            # Otherwise, perform standard vertical sorting of all text blocks
            all_blocks = [(y0, text) for x0, y0, x1, y1, text, b_no, b_type in blocks if b_type == 0 and text.strip()]
            all_blocks.sort(key=lambda x: x[0])
            page_text = "\n\n".join([text for y, text in all_blocks])
            
        pages.append(page_text)
    doc.close()
    return "\n\n".join(pages)


def _extract_pdf_first_line_font(filepath: str) -> str | None:
    """Try to get the largest-font text on the first page (likely the name)."""
    try:
        doc = fitz.open(filepath)
        page = doc[0]
        blocks = page.get_text("dict")["blocks"]
        best_text = ""
        best_size = 0.0
        for block in blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if span["size"] > best_size and span["text"].strip():
                        best_size = span["size"]
                        best_text = span["text"].strip()
        doc.close()
        return best_text if best_text else None
    except Exception:
        return None


def _extract_docx(filepath: str) -> str:
    """Extract text from a DOCX using python-docx."""
    doc = Document(filepath)
    paragraphs = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            paragraphs.append(text)
    for table in doc.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                paragraphs.append(" | ".join(cells))
    return "\n\n".join(paragraphs)


def extract_text(filepath: str) -> str:
    """
    Detect file type and extract cleaned text.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file type is unsupported or no text can be extracted.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Resume file not found: {filepath}")

    ext = path.suffix.lower()
    if ext == ".pdf":
        raw = _extract_pdf(filepath)
    elif ext in (".docx", ".doc"):
        raw = _extract_docx(filepath)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Expected .pdf or .docx")

    cleaned = _clean_text(raw)
    if not cleaned:
        raise ValueError(f"No text could be extracted from: {filepath}")
    return cleaned


# =====================================================================
#  Helper — lazy spaCy loader
# =====================================================================
_spacy_nlp = None

def _get_spacy_nlp():
    """Return the cached spaCy ``en_core_web_sm`` pipeline, loading it once."""
    global _spacy_nlp
    if _spacy_nlp is None:
        print("  → Loading spaCy en_core_web_sm model for parsing...")
        import spacy
        try:
            _spacy_nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("  → en_core_web_sm not found; downloading…")
            from spacy.cli import download as spacy_download
            spacy_download("en_core_web_sm")
            _spacy_nlp = spacy.load("en_core_web_sm")
        print("  → spaCy model loaded.")
    return _spacy_nlp


# ═══════════════════════════════════════════════════════════════════════════════
#  Field Extraction — Regex + Heuristics
# ═══════════════════════════════════════════════════════════════════════════════

# ── Email ─────────────────────────────────────────────────────────────────────

_EMAIL_RE = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", re.IGNORECASE
)


def _extract_email(text: str) -> str | None:
    """Extract the first email address found in the text."""
    match = _EMAIL_RE.search(text)
    return match.group(0) if match else None


# ── GitHub Username ───────────────────────────────────────────────────────────

_GITHUB_RE = re.compile(
    r"github\.com/([a-zA-Z0-9\-]+)", re.IGNORECASE
)


def _extract_github_username(text: str) -> str | None:
    """Extract GitHub username from a github.com/USERNAME URL."""
    match = _GITHUB_RE.search(text)
    if match:
        username = match.group(1)
        # Filter out common non-username paths
        if username.lower() not in ("settings", "login", "signup", "explore",
                                     "marketplace", "notifications", "new",
                                     "organizations", "topics", "trending"):
            return username
    return None


# ── Phone ─────────────────────────────────────────────────────────────────────

_PHONE_RE = re.compile(
    r"(?:\+?\d{1,3}[\s\-]?)?\(?\d{2,4}\)?[\s\-]?\d{3,5}[\s\-]?\d{3,5}"
)


def _extract_phone(text: str) -> str | None:
    """Extract the first phone number found."""
    # Only search the top portion of the resume (contact section)
    top_section = text[:800]
    match = _PHONE_RE.search(top_section)
    if match:
        phone = match.group(0).strip()
        # Ensure it looks like a phone (at least 7 digits)
        digits = re.sub(r"\D", "", phone)
        if 7 <= len(digits) <= 15:
            return phone
    return None


# ── Name ──────────────────────────────────────────────────────────────────────

_NOISE_STARTS = {
    "resume", "curriculum", "cv", "vitae", "objective", "summary",
    "phone", "email", "address", "linkedin", "github", "http",
    "https", "www", "skills", "experience", "education",
}


def _extract_name(text: str, filepath: str) -> str | None:
    """
    Extract candidate name using heuristics:
    1. If PDF, try the largest font on page 1.
    2. Otherwise, use the first non-trivial line (2–5 words, starts with uppercase).
    """
    # Strategy 1: Largest font (PDF only)
    if filepath.lower().endswith(".pdf"):
        big_text = _extract_pdf_first_line_font(filepath)
        if big_text:
            # Clean and validate
            name = big_text.strip()
            words = name.split()
            if 1 <= len(words) <= 5 and words[0][0].isupper():
                first_word_lower = words[0].lower().rstrip(".:,")
                if first_word_lower not in _NOISE_STARTS:
                    return name

    # Strategy 2: First meaningful line
    for line in text.split("\n")[:10]:
        line = line.strip()
        if not line:
            continue
        words = line.split()
        if len(words) < 1 or len(words) > 5:
            continue
        first_word_lower = words[0].lower().rstrip(".:,")
        if first_word_lower in _NOISE_STARTS:
            continue
        # Must start with an uppercase letter
        if not line[0].isupper():
            continue
        # Should not look like an email or URL
        if "@" in line or "http" in line.lower() or "github.com" in line.lower():
            continue
        # Should not be mostly digits
        digit_ratio = sum(c.isdigit() for c in line) / max(len(line), 1)
        if digit_ratio > 0.4:
            continue
        return line

    return None


# ── Section Extraction ────────────────────────────────────────────────────────

def _find_section(text: str, header_keywords: list[str]) -> str:
    """
    Extract the content of a resume section by its header.
    Returns everything between the matching header and the next header.
    """
    # Build header pattern
    headers_pattern = "|".join(re.escape(kw) for kw in header_keywords)
    section_re = re.compile(
        rf"(?:^|\n)\s*(?:{headers_pattern})\s*[:\-]?\s*\n",
        re.IGNORECASE,
    )
    # Generic next-section pattern
    next_section_re = re.compile(
        r"(?:^|\n)\s*(?:experience|education|skills?|certifications?|"
        r"achievements?|awards?|publications?|interests?|hobbies|"
        r"references?|summary|objective|contact|languages|technologies|"
        r"projects?|personal projects?|academic projects?|"
        r"open source|competitive|work history|professional experience|"
        r"technical skills|core competencies|extracurricular)\s*[:\-]?\s*\n",
        re.IGNORECASE,
    )

    match = section_re.search(text)
    if not match:
        return ""

    start = match.end()
    # Find the next section header after this one
    remaining = text[start:]
    next_match = next_section_re.search(remaining)
    end = start + next_match.start() if next_match else len(text)

    return text[start:end].strip()


def _extract_skills(text: str) -> list[str]:
    """Extract skills from the Skills/Technologies section and fallback NLP."""
    section = _find_section(text, [
        "Skills", "Technical Skills", "Technologies", "Tech Stack",
        "Core Competencies", "Tools & Technologies", "Programming Languages",
        "Key Skills",
    ])
    
    skills = []
    if section:
        # Clean up malformed delimiters BEFORE splitting
        # Collapse multiple consecutive commas: "Python,, FastAPI" -> "Python, FastAPI"
        section = re.sub(r"[,]{2,}", ",", section)
        # Remove commas surrounded by spaces: " , " -> " "
        section = re.sub(r"\s*,\s*,\s*", ",", section)
        # Strip leading/trailing commas per line
        section = re.sub(r"(?m)^\s*,|,\s*$", "", section)

        # Split by common delimiters: comma, bullet, pipe, newline
        raw_items = re.split(r"[,\u2022\u00b7|\u25cf\u25cb\u25ba\u25aa\u25b8\n]", section)
        for item in raw_items:
            cleaned = item.strip().strip("-\u2013\u2014*:.")
            # Remove parenthetical explanations
            cleaned = re.sub(r"\s*\([^)]*\)", "", cleaned).strip()
            if cleaned and 1 <= len(cleaned) <= 50 and not cleaned[0].isdigit():
                skills.append(cleaned)
                
    # NLP Fallback for missing skills
    print("  → Running NLP fallback for skill extraction...")
    nlp = _get_spacy_nlp()
    doc = nlp(text)
    
    # Common tech keywords to look for if they weren't in a standard section
    tech_keywords = {
        "python", "java", "c++", "c", "c#", "javascript", "typescript", "react", "angular",
        "vue", "node.js", "express", "django", "flask", "fastapi", "spring", "aws", "azure",
        "gcp", "docker", "kubernetes", "terraform", "ci/cd", "linux", "git", "sql", "mysql",
        "postgresql", "mongodb", "redis", "elasticsearch", "machine learning", "deep learning",
        "ai", "nlp", "computer vision", "pytorch", "tensorflow", "keras", "pandas", "numpy",
        "scikit-learn", "data science", "data engineering", "spark", "hadoop", "kafka",
        "devops", "agile", "scrum", "html", "css", "next.js", "nextjs", "graphql", "rest api",
        "microservices", "kubernetes", "bash", "shell", "powershell", "golang", "rust", "ruby"
    }
    
    nlp_skills = set()
    for token in doc:
        if token.is_stop or token.is_punct or token.is_space:
            continue
        lower_token = token.lemma_.lower()
        if lower_token in tech_keywords:
            nlp_skills.add(lower_token)
            
    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.lower().strip()
        if chunk_text in tech_keywords or any(kw in chunk_text for kw in ["development", "engineering", "machine learning", "deep learning"]):
            nlp_skills.add(chunk_text)
            
    # Combine and deduplicate preserving order
    all_skills = skills + list(nlp_skills)
    seen = set()
    final_skills = []
    for s in all_skills:
        s_lower = s.lower()
        if s_lower not in seen and len(s) > 1:
            seen.add(s_lower)
            final_skills.append(s)

    return final_skills


def _extract_projects(text: str) -> list[dict]:
    """
    Extract projects from the Projects section.
    Returns list of dicts with 'name' and 'description' keys.
    """
    section = _find_section(text, [
        "Projects", "Personal Projects", "Academic Projects",
        "Key Projects", "Notable Projects", "Selected Projects",
        "Side Projects",
    ])
    if not section:
        return []

    projects = []
    lines = [ln.strip() for ln in section.splitlines() if ln.strip()]

    _ACTION_WORDS = {
        "built", "trained", "developed", "implemented", "created",
        "designed", "wrote", "used", "applied", "worked", "fixed",
        "added", "achieved", "engineered", "consolidated", "refactored",
        "deployed", "integrated", "optimized", "improved", "automated",
        "configured", "managed", "led", "contributed", "maintained",
    }

    i = 0
    while i < len(lines):
        line = lines[i]

        # Skip bullet description lines
        if re.match(r"^[•\-\*\u2022\u25CF\u25CB\u2023\d]", line):
            i += 1
            continue

        # Skip action-verb fragments
        first_word = line.split()[0].lower().rstrip(".,:")
        if first_word in _ACTION_WORDS:
            i += 1
            continue

        # Must start with uppercase and have >= 2 words
        words = line.split()
        if len(words) >= 2 and line[0].isupper():
            # Clean title: split on em-dash/pipe, remove parens/links
            raw_title = line
            raw_title = raw_title.replace("\ufffd", " – ")
            title = re.split(r"\s*[–—|]\s*", raw_title)[0]
            title = re.sub(r"\s*\([^)]{0,80}\)", "", title)
            title = re.sub(r"\s*\[.*?\]", "", title)
            title = title.strip().rstrip(":.,–—-")

            # Collect description from next lines (bullets or indented)
            desc_parts = []
            j = i + 1
            while j < len(lines):
                next_line = lines[j]
                if re.match(r"^[•\-\*\u2022]", next_line):
                    desc_parts.append(next_line.lstrip("•-*\u2022 ").strip())
                    j += 1
                elif next_line[0].islower():
                    desc_parts.append(next_line)
                    j += 1
                else:
                    break

            description = " ".join(desc_parts[:2])  # First 2 bullet points
            if len(title) >= 4:
                projects.append({
                    "name": title,
                    "description": description[:200] if description else "",
                })

            i = j
        else:
            i += 1

    return projects[:12]


def _extract_education(text: str) -> str:
    """Extract the Education section as raw text."""
    return _find_section(text, [
        "Education", "Academic Background", "Qualifications",
        "Academic Qualifications", "Educational Background",
    ])


# ═══════════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════════

def parse_resume(filepath: str) -> dict:
    """
    Parse a resume file and extract all structured fields.

    Parameters
    ----------
    filepath : str
        Path to a PDF or DOCX resume file.

    Returns
    -------
    dict
        Keys: name, email, github_username, phone, skills, projects,
              education, raw_text
    """
    print(f"  → Extracting text from: {Path(filepath).name}")
    raw_text = extract_text(filepath)
    print(f"  → Extracted {len(raw_text)} characters")

    print("  → Parsing candidate name...")
    name = _extract_name(raw_text, filepath)

    print("  → Extracting email...")
    email = _extract_email(raw_text)

    print("  → Looking for GitHub username...")
    github_username = _extract_github_username(raw_text)

    print("  → Extracting phone...")
    phone = _extract_phone(raw_text)

    print("  → Extracting skills...")
    skills = _extract_skills(raw_text)

    print("  → Extracting projects...")
    projects = _extract_projects(raw_text)

    print("  → Extracting education...")
    education = _extract_education(raw_text)

    result = {
        "name": name,
        "email": email,
        "github_username": github_username,
        "phone": phone,
        "skills": skills,
        "projects": projects,
        "education": education,
        "raw_text": raw_text,
    }

    print(f"  ✓ Found: name={name}, email={email}, "
          f"github={github_username}, "
          f"{len(skills)} skills, {len(projects)} projects")

    return result
