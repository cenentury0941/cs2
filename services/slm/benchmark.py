#!/usr/bin/env python3
"""Benchmark hosted SLM /v1/decide latency and throughput.

Works against local Mac, AWS via SSM tunnel, or any SLM_BASE_URL.

Examples:
  python benchmark.py --base http://127.0.0.1:8080
  python benchmark.py --base http://127.0.0.1:8080 --requests 40 --concurrency 4 --agents 8
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

import httpx

ALLOWED = {"GO_TO_SEAT", "EXIT", "IDLE"}


def wait_ready(base: str, timeout_s: float = 600.0) -> Dict[str, Any]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = httpx.get(f"{base}/ready", timeout=5.0)
            if r.status_code == 200:
                return r.json()
        except httpx.HTTPError:
            pass
        time.sleep(1.0)
    raise SystemExit(f"Timed out waiting for {base}/ready")


def make_payload(request_id: str, n_agents: int) -> Dict[str, Any]:
    agents = []
    for i in range(n_agents):
        fire = i % 3 == 0
        agents.append(
            {
                "id": f"Agent_{i:03d}",
                "goal": "Seat_003" if not fire else "Exit_A",
                "assigned_seat": f"Seat_{i % 25:03d}",
                "intent": "WALK",
                "perception": {
                    "fire_visible": fire,
                    "smoke_visible": fire,
                    "seat_known": True,
                    "at_destination": False,
                    "nearest_exit": "Exit_A",
                    "exits": ["Exit_A", "Exit_B"],
                },
            }
        )
    return {"request_id": request_id, "agents": agents}


def one_decide(
    client: httpx.Client, base: str, request_id: str, n_agents: int
) -> Tuple[float, int]:
    """Return (latency_s, n_decisions). Raises on HTTP/contract errors."""
    payload = make_payload(request_id, n_agents)
    t0 = time.perf_counter()
    resp = client.post(f"{base}/v1/decide", json=payload)
    elapsed = time.perf_counter() - t0
    resp.raise_for_status()
    body = resp.json()
    decisions = body.get("decisions") or []
    if len(decisions) != n_agents:
        raise RuntimeError(f"{request_id}: expected {n_agents} decisions, got {len(decisions)}")
    for d in decisions:
        if d.get("intent") not in ALLOWED:
            raise RuntimeError(f"{request_id}: bad intent {d}")
    return elapsed, len(decisions)


def percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def summarize(label: str, latencies: List[float], total_agents: int, wall_s: float) -> None:
    xs = sorted(latencies)
    print(f"\n=== {label} ===")
    print(f"requests:     {len(xs)}")
    print(f"wall_s:       {wall_s:.3f}")
    print(f"req/s:        {len(xs) / wall_s:.2f}" if wall_s else "req/s:        n/a")
    print(f"agents/s:     {total_agents / wall_s:.2f}" if wall_s else "agents/s:     n/a")
    print(f"latency_ms:   min={xs[0]*1000:.0f}  p50={percentile(xs,50)*1000:.0f}  "
          f"p90={percentile(xs,90)*1000:.0f}  p99={percentile(xs,99)*1000:.0f}  "
          f"max={xs[-1]*1000:.0f}  mean={statistics.mean(xs)*1000:.0f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark SimulCrowd SLM /v1/decide")
    parser.add_argument("--base", default="http://127.0.0.1:8080")
    parser.add_argument("--requests", type=int, default=20, help="Timed decide calls")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--agents", type=int, default=8, help="Agents per decide batch")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Parallel in-flight decide calls (use 4 to exercise the LB fleet)",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    if args.requests < 1 or args.agents < 1 or args.concurrency < 1:
        raise SystemExit("requests, agents, and concurrency must be >= 1")

    health = httpx.get(f"{args.base}/health", timeout=5.0)
    health.raise_for_status()
    print("health:", health.json())
    print("ready:", wait_ready(args.base))

    with httpx.Client(timeout=args.timeout) as client:
        for i in range(args.warmup):
            one_decide(client, args.base, f"warmup-{i}", args.agents)
        print(f"warmup done ({args.warmup})")

        # Sequential
        seq: List[float] = []
        t0 = time.perf_counter()
        for i in range(args.requests):
            lat, _ = one_decide(client, args.base, f"seq-{i}", args.agents)
            seq.append(lat)
        summarize(
            f"sequential agents={args.agents}",
            seq,
            args.requests * args.agents,
            time.perf_counter() - t0,
        )

        if args.concurrency == 1:
            return

        # Concurrent requests against the same process
        conc: List[float] = []
        errors = 0

        def _job(i: int) -> float:
            # One client per thread — httpx Client is not fully thread-safe for all cases.
            with httpx.Client(timeout=args.timeout) as c:
                lat, _ = one_decide(c, args.base, f"conc-{i}", args.agents)
                return lat

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futs = [pool.submit(_job, i) for i in range(args.requests)]
            for fut in as_completed(futs):
                try:
                    conc.append(fut.result())
                except Exception as exc:
                    errors += 1
                    print("error:", exc, file=sys.stderr)
        wall = time.perf_counter() - t0
        if not conc:
            raise SystemExit("all concurrent requests failed")
        summarize(
            f"concurrent={args.concurrency} agents={args.agents} errors={errors}",
            conc,
            len(conc) * args.agents,
            wall,
        )


if __name__ == "__main__":
    try:
        main()
    except httpx.HTTPError as exc:
        print("HTTP error:", exc, file=sys.stderr)
        raise SystemExit(1) from exc
