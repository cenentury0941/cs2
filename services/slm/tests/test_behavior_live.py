#!/usr/bin/env python3
"""Live / offline behavior checks for many agent-state scenarios.

Unit path (no server):
  cd services/slm && python -m unittest tests.test_decision_parse -v

Live path (model must be up):
  cd services/slm && python tests/test_behavior_live.py --base http://127.0.0.1:8080

Sends each scenario alone and also one mixed batch, asserting:
  - contract (HTTP 200, intents allowed, 1:1 ids)
  - expected intent/destination from the system prompt rules
"""

from __future__ import annotations

import argparse
import sys
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_TESTS = Path(__file__).resolve().parent
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

import httpx  # noqa: E402

from scenarios import SCENARIOS, scenarios_as_agents  # noqa: E402

ALLOWED = frozenset({"GO_TO_SEAT", "EXIT", "IDLE"})
_BASE = "http://127.0.0.1:8080"
_TIMEOUT = 180.0


def _wait_ready(base: str, timeout_s: float = 600.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = httpx.get(f"{base}/ready", timeout=5.0)
            if r.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(1.0)
    raise RuntimeError(f"Timed out waiting for {base}/ready")


def _decide(base: str, request_id: str, agents: List[Dict[str, Any]]) -> Dict[str, Any]:
    resp = httpx.post(
        f"{base}/v1/decide",
        json={"request_id": request_id, "agents": agents},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    body = resp.json()
    if body.get("request_id") != request_id:
        raise AssertionError(f"request_id mismatch: {body.get('request_id')}")
    decisions = body.get("decisions") or []
    if len(decisions) != len(agents):
        raise AssertionError(
            f"expected {len(agents)} decisions, got {len(decisions)}: {decisions}"
        )
    by_id = {d["id"]: d for d in decisions}
    for a in agents:
        if a["id"] not in by_id:
            raise AssertionError(f"missing decision for {a['id']}")
        intent = by_id[a["id"]].get("intent")
        if intent not in ALLOWED:
            raise AssertionError(f"bad intent for {a['id']}: {intent}")
    return by_id


def _check_case(by_id: Dict[str, Any], case: Dict[str, Any]) -> None:
    agent_id = case["agent"]["id"]
    d = by_id[agent_id]
    intent = d.get("intent")
    dest = d.get("destination")
    if intent != case["expect_intent"]:
        raise AssertionError(
            f"{case['name']}: intent want {case['expect_intent']!r} got {intent!r} "
            f"(dest={dest!r}) — {case.get('notes', '')}"
        )
    if case.get("expect_destination_null"):
        if dest is not None:
            raise AssertionError(
                f"{case['name']}: destination should be null, got {dest!r}"
            )
    elif "expect_destination" in case:
        if dest != case["expect_destination"]:
            raise AssertionError(
                f"{case['name']}: destination want {case['expect_destination']!r} "
                f"got {dest!r}"
            )


class LiveBehaviorTests(unittest.TestCase):
    """Hit a running SLM; skipped if --base unreachable unless run as __main__."""

    base = _BASE

    @classmethod
    def setUpClass(cls) -> None:
        try:
            httpx.get(f"{cls.base}/health", timeout=3.0).raise_for_status()
            _wait_ready(cls.base, timeout_s=30.0)
        except Exception as exc:  # noqa: BLE001
            raise unittest.SkipTest(f"SLM not reachable at {cls.base}: {exc}") from exc

    def test_each_scenario_solo(self) -> None:
        for case in SCENARIOS:
            with self.subTest(case["name"]):
                by_id = _decide(
                    self.base,
                    f"solo-{case['name']}",
                    [case["agent"]],
                )
                _check_case(by_id, case)

    def test_mixed_batch_all_scenarios(self) -> None:
        agents = scenarios_as_agents()
        # Ensure unique ids (scenarios already unique)
        by_id = _decide(self.base, "mixed-all-scenarios", agents)
        for case in SCENARIOS:
            with self.subTest(case["name"]):
                _check_case(by_id, case)

    def test_empty_batch(self) -> None:
        resp = httpx.post(
            f"{self.base}/v1/decide",
            json={"request_id": "empty", "agents": []},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        body = resp.json()
        self.assertEqual(body.get("request_id"), "empty")
        self.assertEqual(body.get("decisions"), [])


def _run_cli(base: str) -> int:
    print(f"base={base}")
    health = httpx.get(f"{base}/health", timeout=5.0)
    health.raise_for_status()
    print("health:", health.json())
    _wait_ready(base)
    print(f"running {len(SCENARIOS)} solo scenarios + mixed batch…")

    failed = 0
    for case in SCENARIOS:
        try:
            t0 = time.perf_counter()
            by_id = _decide(base, f"solo-{case['name']}", [case["agent"]])
            _check_case(by_id, case)
            ms = (time.perf_counter() - t0) * 1000
            d = by_id[case["agent"]["id"]]
            print(
                f"  PASS  {case['name']:<28} "
                f"{d['intent']:<10} dest={d.get('destination')!s:<10} {ms:.0f}ms"
            )
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {case['name']:<28} {exc}")

    try:
        t0 = time.perf_counter()
        by_id = _decide(base, "mixed-all-scenarios", scenarios_as_agents())
        for case in SCENARIOS:
            _check_case(by_id, case)
        ms = (time.perf_counter() - t0) * 1000
        print(f"  PASS  mixed_batch ({len(SCENARIOS)} agents) {ms:.0f}ms")
    except Exception as exc:  # noqa: BLE001
        failed += 1
        print(f"  FAIL  mixed_batch {exc}")

    try:
        resp = httpx.post(
            f"{base}/v1/decide",
            json={"request_id": "empty", "agents": []},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        assert resp.json().get("decisions") == []
        print("  PASS  empty_batch")
    except Exception as exc:  # noqa: BLE001
        failed += 1
        print(f"  FAIL  empty_batch {exc}")

    print(f"\n{len(SCENARIOS) + 2 - failed} passed, {failed} failed")
    return 1 if failed else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="SLM agent-state behavior tests")
    parser.add_argument("--base", default=_BASE)
    parser.add_argument(
        "--unittest",
        action="store_true",
        help="Run as unittest.TestCase (skips if server down)",
    )
    args = parser.parse_args()
    if args.unittest:
        LiveBehaviorTests.base = args.base
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(LiveBehaviorTests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)
    try:
        raise SystemExit(_run_cli(args.base))
    except httpx.HTTPError as exc:
        print("HTTP error:", exc, file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
