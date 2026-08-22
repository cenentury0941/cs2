#!/usr/bin/env python3
"""Unit tests for OpenRouter summarizer helpers (no network)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_TESTS = Path(__file__).resolve().parent
for p in (_ROOT, _TESTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from openrouter import (  # noqa: E402
    _read_key_file,
    normalize_openrouter_model,
    resolve_api_key,
)


class TestNormalizeModel(unittest.TestCase):
    def test_dot_form(self) -> None:
        self.assertEqual(
            normalize_openrouter_model("openai.gpt-5.6-luna"),
            "openai/gpt-5.6-luna",
        )

    def test_slash_unchanged(self) -> None:
        self.assertEqual(
            normalize_openrouter_model("openai/gpt-5.6-luna"),
            "openai/gpt-5.6-luna",
        )


class TestResolveApiKey(unittest.TestCase):
    def test_explicit(self) -> None:
        self.assertEqual(resolve_api_key("sk-or-v1-abc"), "sk-or-v1-abc")

    def test_file_with_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "openrouter_api_key.txt"
            path.write_text(
                "Team: SimulCrowd\nAPI Key: sk-or-v1-deadbeefcafe\n",
                encoding="utf-8",
            )
            self.assertEqual(_read_key_file(path), "sk-or-v1-deadbeefcafe")


if __name__ == "__main__":
    unittest.main()
