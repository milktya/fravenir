"""Latency test for ruri-v3-310m embedding (requires model download).

Run with: uv run pytest -m slow tests/integration/test_embedding_latency.py -v
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from fravenir.embedding import Embedder
from fravenir.schemas.config import EmbeddingConfig

LATENCY_LIMIT_MS = 200
WARMUP_RUNS = 3
BENCH_RUNS = 10


@pytest.mark.slow
class TestEmbeddingLatency:
    @pytest.fixture(scope="class")
    def embedder(self):
        cfg = EmbeddingConfig()
        e = Embedder(cfg)
        # trigger model load
        e.encode_query("warmup")
        return e

    def test_encode_query_latency(self, embedder):
        for _ in range(WARMUP_RUNS):
            embedder.encode_query("ウォームアップ")

        times = []
        for _ in range(BENCH_RUNS):
            t0 = time.perf_counter()
            embedder.encode_query("瑠璃色はどんな色？")
            times.append((time.perf_counter() - t0) * 1000)

        median_ms = float(np.median(times))
        assert median_ms < LATENCY_LIMIT_MS, (
            f"encode_query median latency {median_ms:.1f}ms exceeds {LATENCY_LIMIT_MS}ms"
        )

    def test_encode_document_latency(self, embedder):
        for _ in range(WARMUP_RUNS):
            embedder.encode_document("ウォームアップ文書")

        times = []
        for _ in range(BENCH_RUNS):
            t0 = time.perf_counter()
            embedder.encode_document("瑠璃色（るりいろ）は、紫みを帯びた濃い青色のこと。" * 5)
            times.append((time.perf_counter() - t0) * 1000)

        median_ms = float(np.median(times))
        assert median_ms < LATENCY_LIMIT_MS, (
            f"encode_document median latency {median_ms:.1f}ms exceeds {LATENCY_LIMIT_MS}ms"
        )

    def test_output_dim_is_768(self, embedder):
        vec = embedder.encode_query("次元チェック")
        assert vec.shape == (768,)

    def test_output_is_normalized(self, embedder):
        vec = embedder.encode_query("正規化チェック")
        norm = float(np.linalg.norm(vec))
        assert abs(norm - 1.0) < 1e-4
