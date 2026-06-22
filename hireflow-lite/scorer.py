"""
HireFlow-Lite — Scorer
-----------------------
Combines ATS and GitHub scores into a final candidate score.
Simple weighted average with configurable weights.
"""


def compute_final_score(
    ats_score: float,
    github_score: float,
    ats_weight: float = 0.6,
    github_weight: float = 0.4,
) -> dict:
    """
    Compute the final candidate score as a weighted average.

    Parameters
    ----------
    ats_score : float
        ATS score (0-100).
    github_score : float
        GitHub verification score (0-100).
    ats_weight : float
        Weight for ATS score (default 0.6).
    github_weight : float
        Weight for GitHub score (default 0.4).

    Returns
    -------
    dict
        Contains final_score, ats_contribution, github_contribution, and weights.
    """
    # Normalise weights so they sum to 1.0
    total_weight = ats_weight + github_weight
    if total_weight == 0:
        ats_weight = 0.5
        github_weight = 0.5
        total_weight = 1.0

    ats_w = ats_weight / total_weight
    github_w = github_weight / total_weight

    ats_contribution = ats_score * ats_w
    github_contribution = github_score * github_w
    final_score = ats_contribution + github_contribution

    return {
        "final_score": round(final_score, 2),
        "ats_contribution": round(ats_contribution, 2),
        "github_contribution": round(github_contribution, 2),
        "ats_weight": round(ats_w, 2),
        "github_weight": round(github_w, 2),
    }
