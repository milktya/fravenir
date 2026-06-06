"""Embedding wrapper for ruri-v3-310m (sentence-transformers)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from fravenir.schemas.config import EmbeddingConfig


class Embedder:
    """Lazy-loading wrapper around SentenceTransformer.

    The model is loaded on first use to avoid paying startup cost
    when only CLI metadata commands are run.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config
        self._model: SentenceTransformer | None = None

    def _load(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(
                self._config.model,
                device=self._resolve_device(),
            )
        return self._model

    def _resolve_device(self) -> str:
        if self._config.device != "auto":
            return self._config.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _encode(self, texts: list[str]) -> NDArray[np.float32]:
        model = self._load()
        vecs: NDArray[np.float32] = model.encode(
            texts,
            batch_size=self._config.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        if self._config.normalize:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            vecs = (vecs / norms).astype(np.float32)
        return vecs

    def encode_query(self, text: str) -> NDArray[np.float32]:
        prefixed = self._config.prefixes.query + text
        result: NDArray[np.float32] = self._encode([prefixed])[0]
        return result

    def encode_document(self, text: str) -> NDArray[np.float32]:
        prefixed = self._config.prefixes.document + text
        result: NDArray[np.float32] = self._encode([prefixed])[0]
        return result

    def encode_topic(self, text: str) -> NDArray[np.float32]:
        prefixed = self._config.prefixes.topic + text
        result: NDArray[np.float32] = self._encode([prefixed])[0]
        return result

    def encode_general(self, text: str) -> NDArray[np.float32]:
        prefixed = self._config.prefixes.general + text
        result: NDArray[np.float32] = self._encode([prefixed])[0]
        return result

    def encode_documents_batch(self, texts: list[str]) -> NDArray[np.float32]:
        prefixed = [self._config.prefixes.document + t for t in texts]
        return self._encode(prefixed)
