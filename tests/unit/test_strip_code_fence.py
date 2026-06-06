"""Unit tests for _strip_code_fence language identifier handling."""

from __future__ import annotations

from fravenir.core.extraction import _strip_code_fence


class TestStripCodeFenceLang:
    def test_no_fence(self) -> None:
        """Fence なしの素の JSON はそのまま返す。"""
        result = _strip_code_fence('{"key": "value"}')
        assert result == '{"key": "value"}'

    def test_fence_with_inline_lang(self) -> None:
        """```json\n{...}\n``` パターン（既存挙動）。"""
        result = _strip_code_fence('```json\n{"key": "value"}\n```')
        assert result == '{"key": "value"}'

    def test_fence_lang_on_separate_line(self) -> None:
        """```\njson\n{...}\n``` パターン（言語識別子が独立行の保険動作）。"""
        result = _strip_code_fence('```\njson\n{"key": "value"}\n```')
        assert result == '{"key": "value"}'

    def test_fence_no_lang(self) -> None:
        """```\n{...}\n``` パターン（言語識別子なし）。"""
        result = _strip_code_fence('```\n{"key": "value"}\n```')
        assert result == '{"key": "value"}'
