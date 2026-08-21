# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""Pluggable decision engine: rule-based now, remote batched SLM for Phase 6b."""

from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

if TYPE_CHECKING:
    from .agent import Agent

ALLOWED_INTENTS = frozenset({"GO_TO_SEAT", "EXIT", "IDLE"})
DEFAULT_SLM_BASE_URL = os.environ.get("SLM_BASE_URL", "http://172.19.190.197:8080")
DEFAULT_SLM_TIMEOUT_S = float(os.environ.get("SLM_TIMEOUT_S", "30"))

_LOG_LOCK = threading.Lock()


def _find_repo_root() -> str:
    """Locate the kit-app-template repo even when Kit loads from ``_build/``."""
    here = os.path.abspath(os.path.dirname(__file__))
    cur = here
    for _ in range(12):
        context_dir = os.path.join(cur, "context")
        source_ext = os.path.join(cur, "source", "extensions")
        # Prefer the real checkout, not ``_build/windows-x86_64``.
        if (
            os.path.isdir(context_dir)
            and os.path.isdir(source_ext)
            and os.path.basename(cur) != "windows-x86_64"
            and "_build" not in os.path.normpath(cur).split(os.sep)[-2:]
        ):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent

    # Fallback: walk until we see repo markers regardless of _build.
    cur = here
    for _ in range(12):
        if os.path.isdir(os.path.join(cur, "context")) and os.path.isdir(
            os.path.join(cur, "source", "extensions")
        ):
            # If we're inside _build/<platform>, step up to the checkout.
            parts = os.path.normpath(cur).split(os.sep)
            if "_build" in parts:
                idx = parts.index("_build")
                if idx > 0:
                    return os.sep.join(parts[:idx])
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent

    # Last resort: five levels up from ai.py (source tree layout).
    return os.path.abspath(os.path.join(here, "..", "..", "..", "..", ".."))


_REPO_ROOT = _find_repo_root()
DEFAULT_SLM_LOG_PATH = os.environ.get(
    "SLM_LOG_PATH",
    os.path.join(_REPO_ROOT, "context", "slm_exchange_log.txt"),
)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def append_slm_exchange_log(
    kind: str,
    *,
    url: str,
    request_body: Any = None,
    response_body: Any = None,
    status: Optional[int] = None,
    error: Optional[str] = None,
    log_path: Optional[str] = None,
) -> None:
    """Append one SLM HTTP exchange (request + response/error) to a text file."""
    path = log_path or DEFAULT_SLM_LOG_PATH
    lines = [
        "=" * 72,
        f"time: {_utc_stamp()}",
        f"kind: {kind}",
        f"url:  {url}",
    ]
    if status is not None:
        lines.append(f"http_status: {status}")
    if error:
        lines.append(f"error: {error}")
    if request_body is not None:
        lines.append("--- request ---")
        if isinstance(request_body, (dict, list)):
            lines.append(json.dumps(request_body, indent=2, ensure_ascii=False))
        else:
            lines.append(str(request_body))
    if response_body is not None:
        lines.append("--- response ---")
        if isinstance(response_body, (dict, list)):
            lines.append(json.dumps(response_body, indent=2, ensure_ascii=False))
        else:
            lines.append(str(response_body))
    lines.append("")
    text = "\n".join(lines)

    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with _LOG_LOCK:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(text)
    except Exception as exc:  # noqa: BLE001
        try:
            import carb

            carb.log_warn(f"[crowd.agents] SLM exchange log failed ({path}): {exc}")
        except Exception:  # noqa: BLE001
            pass


@dataclass(frozen=True)
class AgentDecision:
    """One row from POST /v1/decide ``decisions[]``."""

    agent_id: str
    intent: str
    destination: Optional[str] = None


class DecisionEngine(ABC):
    """High-level intent only — navigation owns movement."""

    @abstractmethod
    def decide(self, agent: "Agent") -> str:
        """Return an intent string such as GO_TO_SEAT, EXIT, or IDLE."""


class RuleBasedDecisionEngine(DecisionEngine):
    """Hackathon-ready brain; also used as SLM fallback."""

    def decide(self, agent: "Agent") -> str:
        perception = agent.perception or {}
        if perception.get("fire_visible") or perception.get("smoke_visible"):
            return "EXIT"
        if perception.get("at_destination"):
            return "IDLE"
        goal = agent.goal or ""
        if goal.startswith("Exit"):
            return "EXIT"
        if agent.destination is not None or agent.assigned_seat or goal.startswith("Seat"):
            return "GO_TO_SEAT"
        return "IDLE"

    def decide_many(self, agents: Sequence["Agent"]) -> List[AgentDecision]:
        out: List[AgentDecision] = []
        for agent in agents:
            intent = self.decide(agent)
            dest: Optional[str] = None
            if intent == "EXIT":
                dest = (agent.perception or {}).get("nearest_exit")
                if dest is not None:
                    dest = str(dest)
            elif intent == "GO_TO_SEAT":
                dest = agent.assigned_seat or agent.goal
            out.append(AgentDecision(agent.agent_id, intent, dest))
        return out


def agent_to_slm_dict(agent: "Agent") -> Dict[str, Any]:
    """Tiny JSON row for POST /v1/decide (no mesh / transforms)."""
    perception = agent.perception or {}
    exits = perception.get("exits") or []
    if not isinstance(exits, list):
        exits = list(exits)
    nearest = perception.get("nearest_exit")
    return {
        "id": agent.agent_id,
        "goal": agent.goal,
        "assigned_seat": agent.assigned_seat,
        "intent": agent.intent,
        "perception": {
            "fire_visible": bool(perception.get("fire_visible")),
            "smoke_visible": bool(perception.get("smoke_visible")),
            "seat_known": bool(
                perception.get("seat_known", bool(agent.assigned_seat or agent.goal))
            ),
            "at_destination": bool(perception.get("at_destination")),
            "nearest_exit": str(nearest) if nearest else None,
            "exits": [str(e) for e in exits],
        },
    }


def normalize_decision(raw: Dict[str, Any], fallback: RuleBasedDecisionEngine, agent: "Agent") -> AgentDecision:
    """Validate one response row; invalid → rule fallback for that agent only."""
    agent_id = str(raw.get("id") or agent.agent_id)
    intent = str(raw.get("intent") or "").strip().upper()
    destination = raw.get("destination")
    if destination is not None:
        destination = str(destination).strip() or None
        if destination and destination.lower() in ("null", "none"):
            destination = None

    if intent not in ALLOWED_INTENTS:
        fb = fallback.decide_many([agent])[0]
        return AgentDecision(agent_id, fb.intent, fb.destination)

    if intent == "IDLE":
        destination = None
    elif intent == "EXIT" and not destination:
        nearest = (agent.perception or {}).get("nearest_exit")
        destination = str(nearest) if nearest else None
    elif intent == "GO_TO_SEAT" and not destination:
        destination = agent.assigned_seat or agent.goal

    return AgentDecision(agent_id, intent, destination)


class RemoteSLMDecisionEngine(DecisionEngine):
    """HTTP client for hosted SLM — batch only; never loads a model in Kit."""

    def __init__(
        self,
        base_url: str = DEFAULT_SLM_BASE_URL,
        timeout_s: float = DEFAULT_SLM_TIMEOUT_S,
        fallback: Optional[RuleBasedDecisionEngine] = None,
        log_path: Optional[str] = None,
    ) -> None:
        self.base_url = (base_url or DEFAULT_SLM_BASE_URL).rstrip("/")
        self.timeout_s = float(timeout_s)
        self._fallback = fallback or RuleBasedDecisionEngine()
        self.log_path = log_path or DEFAULT_SLM_LOG_PATH
        self._ready_checked = False
        self._last_ready_ok = False
        self._last_error: Optional[str] = None

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def set_base_url(self, base_url: str) -> None:
        self.base_url = (base_url or DEFAULT_SLM_BASE_URL).rstrip("/")
        self._ready_checked = False
        self._last_ready_ok = False

    def decide(self, agent: "Agent") -> str:
        """ABC compatibility — prefer ``decide_batch`` from the sim loop."""
        return self._fallback.decide(agent)

    def check_ready(self) -> bool:
        """GET /ready — 200 only after a worker has the model."""
        url = f"{self.base_url}/ready"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=min(self.timeout_s, 10.0)) as resp:
                status = int(resp.status)
                raw = resp.read().decode("utf-8", errors="replace")
            ok = 200 <= status < 300
            self._ready_checked = True
            self._last_ready_ok = ok
            self._last_error = None if ok else f"/ready HTTP {status}"
            try:
                body: Any = json.loads(raw) if raw.strip() else raw
            except json.JSONDecodeError:
                body = raw
            append_slm_exchange_log(
                "GET /ready",
                url=url,
                status=status,
                response_body=body,
                error=None if ok else self._last_error,
                log_path=self.log_path,
            )
            return ok
        except Exception as exc:  # noqa: BLE001
            self._ready_checked = True
            self._last_ready_ok = False
            self._last_error = f"/ready failed: {exc}"
            append_slm_exchange_log(
                "GET /ready",
                url=url,
                error=self._last_error,
                log_path=self.log_path,
            )
            return False

    def decide_batch(
        self,
        agents: Sequence["Agent"],
        *,
        request_id: str,
        require_ready: bool = True,
        agent_payloads: Optional[List[Dict[str, Any]]] = None,
        max_batch_size: Optional[int] = None,
    ) -> List[AgentDecision]:
        """POST /v1/decide for the agent list. Raises on transport/HTTP failure.

        Pass ``agent_payloads`` (from the Kit thread) to avoid racing live Agent
        mutation while the HTTP call runs on a worker thread.

        If ``max_batch_size`` is set and the list is larger, the call is split
        into sequential POSTs of at most that many agents (one GPU batch each).
        """
        if not agents:
            return []

        rows = agent_payloads if agent_payloads is not None else [
            agent_to_slm_dict(a) for a in agents
        ]
        if len(rows) != len(agents):
            rows = [agent_to_slm_dict(a) for a in agents]

        chunk = int(max_batch_size) if max_batch_size is not None else 0
        if chunk > 0 and len(agents) > chunk:
            out: List[AgentDecision] = []
            part = 0
            for i in range(0, len(agents), chunk):
                part += 1
                out.extend(
                    self.decide_batch(
                        agents[i : i + chunk],
                        request_id=f"{request_id}-b{part}",
                        require_ready=require_ready if part == 1 else False,
                        agent_payloads=rows[i : i + chunk],
                        max_batch_size=None,
                    )
                )
            return out

        if require_ready and not self._last_ready_ok:
            if not self.check_ready():
                raise RuntimeError(self._last_error or "SLM not ready")

        payload = {
            "request_id": request_id,
            "agents": rows,
        }
        body = json.dumps(payload).encode("utf-8")
        url = f"{self.base_url}/v1/decide"
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read().decode("utf-8")
                status = int(resp.status)
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:200]
            except Exception:  # noqa: BLE001
                pass
            self._last_error = f"/v1/decide HTTP {exc.code}: {detail}"
            append_slm_exchange_log(
                "POST /v1/decide",
                url=url,
                request_body=payload,
                status=int(exc.code),
                response_body=detail or None,
                error=self._last_error,
                log_path=self.log_path,
            )
            raise RuntimeError(self._last_error) from exc
        except Exception as exc:  # noqa: BLE001
            self._last_error = f"/v1/decide failed: {exc}"
            append_slm_exchange_log(
                "POST /v1/decide",
                url=url,
                request_body=payload,
                error=self._last_error,
                log_path=self.log_path,
            )
            raise RuntimeError(self._last_error) from exc

        if not (200 <= status < 300):
            self._last_error = f"/v1/decide HTTP {status}"
            append_slm_exchange_log(
                "POST /v1/decide",
                url=url,
                request_body=payload,
                status=status,
                response_body=raw,
                error=self._last_error,
                log_path=self.log_path,
            )
            raise RuntimeError(self._last_error)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            self._last_error = "invalid JSON from /v1/decide"
            append_slm_exchange_log(
                "POST /v1/decide",
                url=url,
                request_body=payload,
                status=status,
                response_body=raw,
                error=self._last_error,
                log_path=self.log_path,
            )
            raise RuntimeError(self._last_error) from exc

        append_slm_exchange_log(
            "POST /v1/decide",
            url=url,
            request_body=payload,
            status=status,
            response_body=data,
            log_path=self.log_path,
        )

        by_id = {a.agent_id: a for a in agents}
        decisions_raw = data.get("decisions") or []
        if not isinstance(decisions_raw, list):
            self._last_error = "decisions is not a list"
            raise RuntimeError(self._last_error)

        got: Dict[str, AgentDecision] = {}
        for row in decisions_raw:
            if not isinstance(row, dict):
                continue
            aid = str(row.get("id") or "")
            agent = by_id.get(aid)
            if agent is None:
                continue
            got[aid] = normalize_decision(row, self._fallback, agent)

        # Any agent missing from the response → rule fallback for that row only
        out = []
        for agent in agents:
            if agent.agent_id in got:
                out.append(got[agent.agent_id])
            else:
                fb = self._fallback.decide_many([agent])[0]
                out.append(fb)

        self._last_error = None
        return out
