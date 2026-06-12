"""
HireFlow Embedding Service
----------------------------
Lazy-loading singleton wrapper around sentence-transformers (all-MiniLM-L6-v2).
Produces 384-dimensional float32 embeddings for resume and JD text.
"""

import threading
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from config import MODEL_NAME, EMBEDDING_DIM


class EmbedderService:
    """
    Singleton embedding service.
    The model is loaded on first use and then reused across calls.
    Thread-safe via a lock on initialization.
    """

    _instance: Optional["EmbedderService"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "EmbedderService":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    obj = super().__new__(cls)
                    obj._model = None  # type: ignore[attr-defined]
                    obj._init_lock = threading.Lock()  # type: ignore[attr-defined]
                    cls._instance = obj
        return cls._instance

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the SentenceTransformer model."""
        if self._model is None:  # type: ignore[has-type]
            with self._init_lock:  # type: ignore[has-type]
                if self._model is None:
                    self._model = SentenceTransformer(MODEL_NAME)
        return self._model  # type: ignore[return-value]

    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text string.

        Parameters
        ----------
        text : str
            Input text to embed.

        Returns
        -------
        np.ndarray
            384-dimensional float32 vector.
        """
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding.astype(np.float32).flatten()

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Embed a batch of text strings.

        Parameters
        ----------
        texts : list[str]
            List of input texts.
        batch_size : int
            Encoding batch size (default 32).

        Returns
        -------
        np.ndarray
            Array of shape (N, 384) with float32 vectors.
        """
        if not texts:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=batch_size,
        )
        return embeddings.astype(np.float32)

    def warmup(self) -> None:
        """
        Force-load the model (useful at application startup).
        Encodes a dummy string to verify the model works.
        """
        _ = self.embed_text("warmup")


# Module-level convenience instance
embedder = EmbedderService()
