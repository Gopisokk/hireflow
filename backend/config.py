"""
HireFlow Backend Configuration
-------------------------------
Central settings for database paths, model names, API URLs, and pipeline defaults.
"""

import os
from pathlib import Path

# ── Base paths ────────────────────────────────────────────────────────────────
BASE_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = BASE_DIR / "data"
DB_PATH: Path = DATA_DIR / "hireflow.db"
RESUME_FOLDER: Path = DATA_DIR / "resumes"

# ── Embedding model ──────────────────────────────────────────────────────────
MODEL_NAME: str = "all-MiniLM-L6-v2"
EMBEDDING_DIM: int = 384

# ── GitHub API ───────────────────────────────────────────────────────────────
GITHUB_API_URL: str = "https://api.github.com/graphql"
GITHUB_REST_API_URL: str = "https://api.github.com"

# ── Pipeline defaults ────────────────────────────────────────────────────────
DEFAULT_ATS_WEIGHT: float = 0.6
DEFAULT_GITHUB_WEIGHT: float = 0.4
DEFAULT_ALGORITHM: str = "hybrid_efficient"
DEFAULT_SHORTLIST_PERCENT: float = 10.0  # top 10% go to GitHub verification

# ── CORS ─────────────────────────────────────────────────────────────────────
ALLOWED_ORIGINS: list[str] = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

# ── Ensure data directories exist ────────────────────────────────────────────
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESUME_FOLDER.mkdir(parents=True, exist_ok=True)
