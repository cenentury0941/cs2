#!/usr/bin/env python3
"""Unit tests for /v1/summarize prompt + parse helpers (no GPU, no HTTP)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_TESTS = Path(__file__).resolve().parent
for p in (_ROOT, _TESTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from prompts import build_summarize_user_prompt  # noqa: E402
from schemas import SummarizeRequest, SummarizeResponse  # noqa: E402
from summarize_parse import (  # noqa: E402
    clean_summary,
    fallback_summary,
    summarize_or_fallback,
)


class TestSummarizeSchema(unittest.TestCase):
    def test_request_defaults(self) -> None:
        req = SummarizeRequest(request_id="r1", agent_id="Agent_001")
        self.assertEqual(req.snapshot, {})
        self.assertEqual(req.events, [])

    def test_response_roundtrip(self) -> None:
        body = SummarizeResponse(
            request_id="r1",
            agent_id="Agent_001",
            summary="Agent_001 is heading to Seat_012.",
        )
        self.assertEqual(
            body.model_dump(),
            {
                "request_id": "r1",
                "agent_id": "Agent_001",
                "summary": "Agent_001 is heading to Seat_012.",
            },
        )


class TestBuildSummarizePrompt(unittest.TestCase):
    def test_includes_snapshot_and_events(self) -> None:
        req = SummarizeRequest(
            request_id="r1",
            agent_id="Agent_007",
            snapshot={
                "id": "Agent_007",
                "intent": "EXIT",
                "goal": "Exit_A",
                "fire_perceived": "yes",
            },
            events=[
                "t=12.3s  Agent_007: saw fire",
                "t=10.0s  Agent_007: intent — GO_TO_SEAT → EXIT",
            ],
        )
        text = build_summarize_user_prompt(req)
        self.assertIn("AGENT_ID: Agent_007", text)
        self.assertIn('"intent":"EXIT"', text)
        self.assertIn("saw fire", text)
        self.assertIn("EVENTS (newest first)", text)

    def test_empty_events_placeholder(self) -> None:
        req = SummarizeRequest(request_id="r1", agent_id="Agent_001")
        text = build_summarize_user_prompt(req)
        self.assertIn("(no events yet)", text)
        self.assertIn('"id":"Agent_001"', text)


class TestCleanSummary(unittest.TestCase):
    def test_plain(self) -> None:
        self.assertEqual(clean_summary("  Hello.  "), "Hello.")

    def test_fenced(self) -> None:
        raw = "```\nAgent_001 fled toward Exit_A.\n```"
        self.assertEqual(clean_summary(raw), "Agent_001 fled toward Exit_A.")

    def test_quoted(self) -> None:
        self.assertEqual(clean_summary('"Hello world."'), "Hello world.")


class TestFallbackSummary(unittest.TestCase):
    def test_with_fire_and_events(self) -> None:
        text = fallback_summary(
            "Agent_003",
            {
                "intent": "EXIT",
                "goal": "Exit_B",
                "fire_perceived": "yes",
                "smoke_perceived": "no",
            },
            ["t=5.0s  Agent_003: saw fire", "t=1.0s  Agent_003: spawned"],
        )
        self.assertIn("Agent_003", text)
        self.assertIn("EXIT", text)
        self.assertIn("fire", text)
        self.assertIn("saw fire", text)

    def test_no_events(self) -> None:
        text = fallback_summary("Agent_001", {"intent": "IDLE", "goal": "—"}, [])
        self.assertIn("No discrete state changes", text)

    def test_summarize_or_fallback_prefers_model(self) -> None:
        req = SummarizeRequest(request_id="r", agent_id="Agent_001")
        self.assertEqual(
            summarize_or_fallback("  Model wrote this. ", req),
            "Model wrote this.",
        )

    def test_summarize_or_fallback_empty_uses_fallback(self) -> None:
        req = SummarizeRequest(
            request_id="r",
            agent_id="Agent_001",
            snapshot={"intent": "IDLE", "goal": "—"},
        )
        text = summarize_or_fallback("   ", req)
        self.assertIn("Agent_001", text)
        self.assertIn("IDLE", text)


if __name__ == "__main__":
    unittest.main()
