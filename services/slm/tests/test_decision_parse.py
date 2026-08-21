#!/usr/bin/env python3
"""Unit tests for parse / fallback (no GPU, no HTTP)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_TESTS = Path(__file__).resolve().parent
for p in (_ROOT, _TESTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from decision_parse import extract_json, fallback_fields, parse_output  # noqa: E402
from schemas import AgentInput  # noqa: E402
from scenarios import SCENARIOS  # noqa: E402


def _input(agent: dict) -> AgentInput:
    return AgentInput.model_validate(agent)


class TestFallbackFields(unittest.TestCase):
    def test_scenarios_match_rule_priority(self) -> None:
        for case in SCENARIOS:
            with self.subTest(case["name"]):
                intent, dest = fallback_fields(_input(case["agent"]))
                self.assertEqual(intent, case["expect_intent"])
                if case.get("expect_destination_null"):
                    self.assertIsNone(dest)
                elif "expect_destination" in case:
                    self.assertEqual(dest, case["expect_destination"])


class TestExtractJson(unittest.TestCase):
    def test_plain_object(self) -> None:
        self.assertEqual(
            extract_json('{"intent":"IDLE","destination":null}'),
            {"intent": "IDLE", "destination": None},
        )

    def test_fenced_markdown(self) -> None:
        raw = '```json\n{"intent": "EXIT", "destination": "Exit_A"}\n```'
        self.assertEqual(
            extract_json(raw),
            {"intent": "EXIT", "destination": "Exit_A"},
        )

    def test_embedded_noise(self) -> None:
        raw = 'Sure. {"intent":"GO_TO_SEAT","destination":"Seat_001"} done.'
        self.assertEqual(
            extract_json(raw),
            {"intent": "GO_TO_SEAT", "destination": "Seat_001"},
        )

    def test_invalid(self) -> None:
        self.assertIsNone(extract_json("no json here"))


class TestParseOutput(unittest.TestCase):
    def test_valid_exit(self) -> None:
        agent = _input(SCENARIOS[3]["agent"])  # fire_visible_evacuate
        intent, dest = parse_output(
            '{"intent":"EXIT","destination":"Exit_A"}', agent
        )
        self.assertEqual(intent, "EXIT")
        self.assertEqual(dest, "Exit_A")

    def test_idle_clears_destination(self) -> None:
        agent = _input(SCENARIOS[2]["agent"])
        intent, dest = parse_output(
            '{"intent":"IDLE","destination":"Seat_005"}', agent
        )
        self.assertEqual(intent, "IDLE")
        self.assertIsNone(dest)

    def test_exit_fills_nearest_exit(self) -> None:
        agent = _input(SCENARIOS[3]["agent"])
        intent, dest = parse_output('{"intent":"EXIT","destination":null}', agent)
        self.assertEqual(intent, "EXIT")
        self.assertEqual(dest, "Exit_A")

    def test_go_to_seat_fills_assigned(self) -> None:
        agent = _input(SCENARIOS[0]["agent"])
        intent, dest = parse_output(
            '{"intent":"GO_TO_SEAT","destination":null}', agent
        )
        self.assertEqual(intent, "GO_TO_SEAT")
        self.assertEqual(dest, "Seat_003")

    def test_invalid_intent_falls_back(self) -> None:
        agent = _input(SCENARIOS[3]["agent"])
        intent, dest = parse_output('{"intent":"RUN","destination":"x"}', agent)
        self.assertEqual(intent, "EXIT")
        self.assertEqual(dest, "Exit_A")

    def test_null_string_destination(self) -> None:
        agent = _input(SCENARIOS[2]["agent"])
        intent, dest = parse_output(
            '{"intent":"IDLE","destination":"null"}', agent
        )
        self.assertEqual(intent, "IDLE")
        self.assertIsNone(dest)


if __name__ == "__main__":
    unittest.main()
