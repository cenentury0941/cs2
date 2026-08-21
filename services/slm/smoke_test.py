#!/usr/bin/env python3
"""Smoke-test the local SLM /v1/decide API."""

from __future__ import annotations

import argparse
import sys
import time

import httpx

ALLOWED = {"GO_TO_SEAT", "EXIT", "IDLE"}


def wait_ready(base: str, timeout_s: float = 600.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = httpx.get(f"{base}/ready", timeout=5.0)
            if r.status_code == 200:
                print("ready:", r.json())
                return
            print("waiting…", r.status_code, r.text)
        except httpx.HTTPError as exc:
            print("waiting…", exc)
        time.sleep(2.0)
    raise SystemExit(f"Timed out waiting for {base}/ready")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8080")
    args = parser.parse_args()

    health = httpx.get(f"{args.base}/health", timeout=5.0)
    health.raise_for_status()
    print("health:", health.json())

    wait_ready(args.base)

    payload = {
        "request_id": "smoke-1",
        "agents": [
            {
                "id": "Agent_001",
                "goal": "Seat_003",
                "assigned_seat": "Seat_003",
                "intent": "WALK",
                "perception": {
                    "fire_visible": False,
                    "smoke_visible": False,
                    "seat_known": True,
                    "at_destination": False,
                    "nearest_exit": "Exit_A",
                    "exits": ["Exit_A", "Exit_B"],
                },
            },
            {
                "id": "Agent_002",
                "goal": "Exit_A",
                "assigned_seat": "Seat_001",
                "intent": "WALK",
                "perception": {
                    "fire_visible": True,
                    "smoke_visible": True,
                    "seat_known": True,
                    "at_destination": False,
                    "nearest_exit": "Exit_A",
                    "exits": ["Exit_A", "Exit_B"],
                },
            },
        ],
    }

    resp = httpx.post(f"{args.base}/v1/decide", json=payload, timeout=120.0)
    resp.raise_for_status()
    body = resp.json()
    print("decide:", body)

    if body.get("request_id") != "smoke-1":
        raise SystemExit("request_id mismatch")
    decisions = body.get("decisions") or []
    if len(decisions) != 2:
        raise SystemExit(f"expected 2 decisions, got {len(decisions)}")
    for d in decisions:
        if d.get("intent") not in ALLOWED:
            raise SystemExit(f"invalid intent: {d}")
    print("smoke test OK")


if __name__ == "__main__":
    try:
        main()
    except httpx.HTTPError as exc:
        print("HTTP error:", exc, file=sys.stderr)
        raise SystemExit(1) from exc
