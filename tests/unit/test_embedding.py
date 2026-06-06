"""Unit tests for Embedder (model mocked)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from fravenir.embedding import Embedder
from fravenir.schemas.config import EmbeddingConfig


@pytest.fixture
def config():
    return EmbeddingConfig()


@pytest.fixture
def mock_model():
    model = MagicMock()

    def fake_encode(texts, **_kwargs):
        n = len(texts)
        vecs = np.random.default_rng(0).random((n, 768), dtype=np.float64).astype(np.float32)
        return vecs

    model.encode.side_effect = fake_encode
    return model


@pytest.fixture
def embedder(config, mock_model):
    e = Embedder(config)
    e._model = mock_model
    return e


class TestEmbedder:
    def test_encode_query_dim(self, embedder):
        vec = embedder.encode_query("テスト")
        assert vec.shape == (768,)

    def test_encode_document_dim(self, embedder):
        vec = embedder.encode_document("テスト文書")
        assert vec.shape == (768,)

    def test_encode_topic_dim(self, embedder):
        vec = embedder.encode_topic("トピック")
        assert vec.shape == (768,)

    def test_encode_general_dim(self, embedder):
        vec = embedder.encode_general("汎用テキスト")
        assert vec.shape == (768,)

    def test_encode_documents_batch_shape(self, embedder):
        vecs = embedder.encode_documents_batch(["a", "b", "c"])
        assert vecs.shape == (3, 768)

    def test_normalize_unit_length(self, embedder):
        vec = embedder.encode_query("正規化テスト")
        norm = float(np.linalg.norm(vec))
        assert abs(norm - 1.0) < 1e-5

    def test_query_prefix_applied(self, embedder):
        embedder.encode_query("こんにちは")
        call_args = embedder._model.encode.call_args[0][0]
        assert call_args[0].startswith("検索クエリ: ")

    def test_document_prefix_applied(self, embedder):
        embedder.encode_document("こんにちは")
        call_args = embedder._model.encode.call_args[0][0]
        assert call_args[0].startswith("検索文書: ")

    def test_topic_prefix_applied(self, embedder):
        embedder.encode_topic("AI")
        call_args = embedder._model.encode.call_args[0][0]
        assert call_args[0].startswith("トピック: ")

    def test_general_prefix_empty(self, embedder):
        embedder.encode_general("hello")
        call_args = embedder._model.encode.call_args[0][0]
        assert call_args[0] == "hello"

    def test_lazy_load(self, config):
        e = Embedder(config)
        assert e._model is None

    def test_normalize_false_no_unit_length(self, mock_model):
        cfg = EmbeddingConfig(normalize=False)
        e = Embedder(cfg)
        e._model = mock_model
        vec = e.encode_query("test")
        norm = float(np.linalg.norm(vec))
        assert norm != pytest.approx(1.0, abs=1e-5)
        assert vec.shape == (768,)

    def test_custom_prefix(self, mock_model):
        from fravenir.schemas.config import EmbeddingPrefixes
        cfg = EmbeddingConfig(prefixes=EmbeddingPrefixes(query="Q: ", document="D: "))
        e = Embedder(cfg)
        e._model = mock_model
        e.encode_query("テスト")
        assert mock_model.encode.call_args[0][0][0].startswith("Q: ")
