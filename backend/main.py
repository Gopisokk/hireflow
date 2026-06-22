"""
HireFlow Backend — FastAPI Application Entry Point
====================================================
Run with:
    cd backend
    uvicorn main:app --reload --port 8000

Or from the hireflow root:
    uvicorn backend.main:app --reload --port 8000
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import ALLOWED_ORIGINS
from database.init_db import init_database
from routers.batch import router as batch_router
from routers.candidates import router as candidates_router
from routers.config_router import router as config_router
from routers.pipeline_status import router as pipeline_router


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup tasks before the app starts accepting requests."""
    print("[HireFlow] Backend starting up...")
    init_database()
    print("[HireFlow] Database initialized.")
    yield
    print("[HireFlow] Backend shutting down.")


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="HireFlow API",
    description=(
        "Automated developer hiring platform. "
        "Parses resumes, scores against JD via BM25/SBERT, "
        "verifies GitHub profiles, and produces ranked candidate lists."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ─── CORS ─────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Routers ──────────────────────────────────────────────────────────────────

app.include_router(batch_router)
app.include_router(pipeline_router)
app.include_router(candidates_router)
app.include_router(config_router)


# ─── Health check ─────────────────────────────────────────────────────────────

@app.get("/", tags=["health"])
async def root() -> dict:
    """Health check endpoint."""
    return {
        "service": "HireFlow API",
        "version": "1.0.0",
        "status": "ok",
        "docs": "/docs",
    }


@app.get("/health", tags=["health"])
async def health() -> dict:
    """Detailed health check."""
    return {
        "status": "healthy",
        "database": "sqlite + sqlite-vec",
        "embedding_model": "all-MiniLM-L6-v2 (sentence-transformers)",
        "algorithms": ["classic_bm25", "neural_fast", "hybrid_efficient"],
        "endpoints": {
            "batch_upload": "POST /api/batch/upload",
            "pipeline_start": "POST /api/pipeline/start",
            "pipeline_status": "GET /api/pipeline/status (SSE)",
            "candidates": "GET /api/candidates",
            "candidate_profile": "GET /api/candidates/{roll_number}",
            "config": "GET/POST /api/config",
        },
    }
