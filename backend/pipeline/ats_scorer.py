"""
HireFlow ATS Scorer
---------------------
Scores resumes against a job description using four algorithms:
  - classic_bm25:      BM25 lexical matching
  - neural_fast:       FastEmbed cosine similarity
  - hybrid_efficient:  BM25 pre-filter → SBERT re-rank
  - score():           dispatcher that calls the right algorithm

All scores are normalized to 0–100.
"""

import re
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

from pipeline.embedder import EmbedderService


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split into tokens."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\+\#\.\-]", " ", text)
    tokens = text.split()
    # Remove very short tokens (single chars except 'c', 'r')
    tokens = [t for t in tokens if len(t) > 1 or t in ("c", "r")]
    return tokens


def _normalize_scores(scores: dict[int, float]) -> dict[int, float]:
    """Normalize raw scores to 0–100 range using min-max scaling."""
    if not scores:
        return {}
    values = list(scores.values())
    min_val = min(values)
    max_val = max(values)
    span = max_val - min_val
    if span == 0:
        # All scores are equal; give them all 50
        return {sid: 50.0 for sid in scores}
    return {
        sid: round(((v - min_val) / span) * 100, 2)
        for sid, v in scores.items()
    }


class ATSScorer:
    """
    ATS scoring engine supporting multiple algorithms.
    Instantiate once per pipeline run and call score() with the chosen algorithm.
    """

    def __init__(self) -> None:
        self._embedder: Optional[EmbedderService] = None

    @property
    def embedder(self) -> EmbedderService:
        """Lazy-load the embedder service."""
        if self._embedder is None:
            self._embedder = EmbedderService()
        return self._embedder

    # ──────────────────────────────────────────────────────────────────────────
    #  Algorithm 1: Classic BM25
    # ──────────────────────────────────────────────────────────────────────────

    def classic_bm25(
        self,
        jd: str,
        resumes: dict[int, str],
    ) -> dict[int, float]:
        """
        Score resumes against the JD using BM25Okapi.

        Parameters
        ----------
        jd : str
            Job description text.
        resumes : dict[int, str]
            Mapping of student_id → resume_text.

        Returns
        -------
        dict[int, float]
            student_id → normalized score (0–100).
        """
        if not resumes:
            return {}

        ids = list(resumes.keys())
        corpus = [_tokenize(resumes[sid]) for sid in ids]
        query_tokens = _tokenize(jd)

        bm25 = BM25Okapi(corpus)
        raw_scores = bm25.get_scores(query_tokens)

        score_map = {sid: float(raw_scores[i]) for i, sid in enumerate(ids)}
        return _normalize_scores(score_map)

    # ──────────────────────────────────────────────────────────────────────────
    #  Algorithm 2: Neural Fast (fastembed cosine)
    # ──────────────────────────────────────────────────────────────────────────

    def neural_fast(
        self,
        jd: str,
        resumes: dict[int, str],
    ) -> dict[int, float]:
        """
        Score resumes against the JD using sentence-transformer cosine similarity.

        Uses the same all-MiniLM-L6-v2 model for consistency.
        """
        if not resumes:
            return {}

        ids = list(resumes.keys())
        texts = [resumes[sid] for sid in ids]

        jd_embedding = self.embedder.embed_text(jd)  # (384,)
        resume_embeddings = self.embedder.embed_batch(texts)  # (N, 384)

        # Cosine similarity (embeddings are already normalized)
        similarities = resume_embeddings @ jd_embedding  # (N,)

        # Convert from [-1, 1] cosine to [0, 100]
        score_map = {
            sid: float(max(0.0, similarities[i]) * 100)
            for i, sid in enumerate(ids)
        }
        return _normalize_scores(score_map)

    # ──────────────────────────────────────────────────────────────────────────
    #  Algorithm 3: Hybrid Efficient (BM25 → SBERT re-rank)
    # ──────────────────────────────────────────────────────────────────────────

    def hybrid_efficient(
        self,
        jd: str,
        resumes: dict[int, str],
    ) -> dict[int, float]:
        """
        Two-pass scoring:
          Pass 1 – BM25 selects top 50% of candidates.
          Pass 2 – SBERT re-ranks the BM25 shortlist.
        Final score = 0.4 * BM25_normalized + 0.6 * SBERT_normalized.
        """
        if not resumes:
            return {}

        # Pass 1: BM25
        bm25_scores = self.classic_bm25(jd, resumes)

        # Take top 50 % (at least 5, at most all)
        cutoff = max(5, len(resumes) // 2)
        sorted_ids = sorted(bm25_scores, key=bm25_scores.get, reverse=True)  # type: ignore[arg-type]
        shortlist_ids = sorted_ids[:cutoff]
        shortlist_resumes = {sid: resumes[sid] for sid in shortlist_ids}

        # Pass 2: SBERT on shortlist
        sbert_scores = self.neural_fast(jd, shortlist_resumes)

        # Merge: candidates not in the shortlist get their BM25 score only
        merged: dict[int, float] = {}
        for sid in resumes:
            bm25_val = bm25_scores.get(sid, 0.0)
            if sid in sbert_scores:
                sbert_val = sbert_scores[sid]
                merged[sid] = round(0.4 * bm25_val + 0.6 * sbert_val, 2)
            else:
                # Non-shortlisted: scale down to reflect lower relevance
                merged[sid] = round(0.4 * bm25_val, 2)

        return _normalize_scores(merged)

    # ──────────────────────────────────────────────────────────────────────────
    #  Dispatcher
    # ──────────────────────────────────────────────────────────────────────────

    def score(
        self,
        jd: str,
        resumes: dict[int, str],
        algorithm: str = "hybrid_efficient",
    ) -> dict[int, float]:
        """
        Dispatch to the chosen scoring algorithm.

        Parameters
        ----------
        jd : str
            Job description text.
        resumes : dict[int, str]
            student_id → resume_text mapping.
        algorithm : str
            One of 'classic_bm25', 'neural_fast', 'hybrid_efficient'.

        Returns
        -------
        dict[int, float]
            student_id → score (0–100).
        """
        dispatch = {
            "classic_bm25": self.classic_bm25,
            "neural_fast": self.neural_fast,
            "hybrid_efficient": self.hybrid_efficient,
        }
        fn = dispatch.get(algorithm)
        if fn is None:
            raise ValueError(
                f"Unknown algorithm '{algorithm}'. "
                f"Choose from: {list(dispatch.keys())}"
            )
        return fn(jd, resumes)
