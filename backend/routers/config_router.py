"""
HireFlow Configuration Router
-------------------------------
GET  /api/config — Retrieve current pipeline configuration.
POST /api/config — Update weights, algorithm selection, shortlist percent.
"""

from pydantic import BaseModel, Field
from fastapi import APIRouter

import config as cfg

router = APIRouter(prefix="/api/config", tags=["config"])


# ── In-memory runtime config (mutable) ──────────────────────────────────────
# These start from the defaults in config.py and can be changed at runtime.

_runtime_config: dict = {
    "ats_weight": cfg.DEFAULT_ATS_WEIGHT,
    "github_weight": cfg.DEFAULT_GITHUB_WEIGHT,
    "algorithm": cfg.DEFAULT_ALGORITHM,
    "shortlist_percent": cfg.DEFAULT_SHORTLIST_PERCENT,
    "model_name": cfg.MODEL_NAME,
    "embedding_dim": cfg.EMBEDDING_DIM,
}


class ConfigUpdate(BaseModel):
    """Request body for updating pipeline configuration."""
    ats_weight: float = Field(
        default=None,
        ge=0.0, le=1.0,
        description="Weight for ATS score in final ranking (0–1)",
    )
    github_weight: float = Field(
        default=None,
        ge=0.0, le=1.0,
        description="Weight for GitHub score in final ranking (0–1)",
    )
    algorithm: str = Field(
        default=None,
        description="Scoring algorithm: classic_bm25 | neural_fast | hybrid_efficient",
    )
    shortlist_percent: float = Field(
        default=None,
        ge=1.0, le=100.0,
        description="Top N% of candidates to send to GitHub verification",
    )


def get_runtime_config() -> dict:
    """Return the current runtime configuration (used by pipeline)."""
    return dict(_runtime_config)


@router.get("")
async def get_config() -> dict:
    """Return the current pipeline configuration."""
    return {
        "status": "ok",
        "config": dict(_runtime_config),
    }


@router.post("")
async def update_config(body: ConfigUpdate) -> dict:
    """
    Update pipeline configuration.
    Only provided fields are updated; omitted fields keep their current values.
    ats_weight and github_weight must sum to 1.0 (when both are provided).
    """
    updates: dict = {}

    if body.ats_weight is not None:
        updates["ats_weight"] = body.ats_weight
    if body.github_weight is not None:
        updates["github_weight"] = body.github_weight
    if body.algorithm is not None:
        valid_algorithms = {"classic_bm25", "neural_fast", "hybrid_efficient"}
        if body.algorithm not in valid_algorithms:
            return {
                "status": "error",
                "detail": f"Invalid algorithm '{body.algorithm}'. Choose from: {sorted(valid_algorithms)}",
            }
        updates["algorithm"] = body.algorithm
    if body.shortlist_percent is not None:
        updates["shortlist_percent"] = body.shortlist_percent

    # Validate weights sum when both are being set or one changes
    new_ats = updates.get("ats_weight", _runtime_config["ats_weight"])
    new_gh = updates.get("github_weight", _runtime_config["github_weight"])
    weight_sum = round(new_ats + new_gh, 4)
    if weight_sum != 1.0:
        return {
            "status": "error",
            "detail": f"ats_weight ({new_ats}) + github_weight ({new_gh}) = {weight_sum}, must equal 1.0",
        }

    # Apply updates
    _runtime_config.update(updates)

    return {
        "status": "ok",
        "updated": updates,
        "config": dict(_runtime_config),
    }
