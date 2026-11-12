"""Deterministic evidence gate for a continuously working Worker.

At each Worker boundary the Host asks one question without calling a model:
does the measured evidence change the choice enough to spend Controller
attention? Ordinary boundaries answer "continue" at zero inference cost.

The gate reads typed measurement facts and the remaining Turn budget. It owns
no Goal, Todo, quota or settlement state, and it never interprets Worker prose.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

GATE_MODES = ("every-return", "evidence", "never")
EVIDENCE_SCHEMA = "turn_worker_evidence_v0"
_MAX_MEASUREMENTS = 512
_MAX_REF_CHARS = 200
_DIGEST_RESULTS = 8
_REVERSED_FIDELITY = ("order_reversed", "frontier_bias_cliff")
_PROFILE_FIELDS = {"unit": "value", "comparison_scope": "turn", "input_revision": "current"}


@dataclass(frozen=True)
class GatePolicy:
    """Thresholds a caller may tune; defaults favour uninterrupted work."""

    plateau_results: int = 3
    relative_epsilon: float = 0.0
    idle_returns: int = 2
    idle_seconds: float = 120.0
    final_reserve_seconds: float = 180.0
    min_consult_remaining_seconds: float = 900.0

    def __post_init__(self) -> None:
        if self.plateau_results < 2 or self.idle_returns < 2:
            raise ValueError("gate windows need at least two observations")
        for value in (self.relative_epsilon, self.idle_seconds, self.final_reserve_seconds,
                      self.min_consult_remaining_seconds):
            if not math.isfinite(value) or value < 0:
                raise ValueError("gate thresholds must be finite and non-negative")


@dataclass(frozen=True)
class GateDecision:
    action: str  # continue | consult | finalize
    reason: str
    facts: dict[str, Any]


def normalize_evidence(payload: Any) -> dict[str, Any]:
    """Validate one evidence snapshot; reject anything the gate cannot rank."""
    if not isinstance(payload, Mapping) or payload.get("schema_version") != EVIDENCE_SCHEMA:
        raise ValueError("unsupported Worker evidence snapshot")
    direction = payload.get("higher_is_better")
    rows = payload.get("measurements")
    if type(direction) is not bool or not isinstance(rows, list) or len(rows) > _MAX_MEASUREMENTS:
        raise ValueError("invalid Worker evidence snapshot")
    measurements = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("invalid Worker measurement")
        ref, value, sequence = row.get("candidate_ref"), row.get("value"), row.get("sequence")
        if (not isinstance(ref, str) or not ref or len(ref) > _MAX_REF_CHARS
                or type(row.get("valid")) is not bool or type(row.get("authoritative")) is not bool
                or type(sequence) is not int or sequence < 0):
            raise ValueError("invalid Worker measurement")
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not math.isfinite(value)):
            raise ValueError("invalid Worker measurement value")
        measurements.append({"candidate_ref": ref, "value": None if value is None else float(value),
                             "valid": row["valid"], "authoritative": row["authoritative"], "sequence": sequence})
    measurements.sort(key=lambda item: (item["sequence"], item["authoritative"], item["candidate_ref"]))
    return {"higher_is_better": direction, "measurements": measurements}


def _improves(value: float, best: float | None, *, higher: bool, epsilon: float) -> bool:
    if best is None:
        return True
    gain = value - best if higher else best - value
    return gain > epsilon * abs(best)


def summarize_authoritative(evidence: Mapping[str, Any], *, epsilon: float = 0.0) -> dict[str, Any]:
    """Incumbent and how many authoritative results arrived since it was set."""
    higher = evidence["higher_is_better"]
    rows = [row for row in evidence["measurements"] if row["authoritative"]]
    best: dict[str, Any] | None = None
    since_best = 0
    for row in rows:
        scored = row["valid"] and row["value"] is not None
        if scored and _improves(row["value"], None if best is None else best["value"],
                                higher=higher, epsilon=epsilon):
            best, since_best = row, 0
        else:
            since_best += 1
    return {
        "count": len(rows),
        "best": None if best is None else {key: best[key] for key in ("candidate_ref", "value", "sequence")},
        "results_since_best": since_best,
        "recent": [{key: row[key] for key in ("candidate_ref", "value", "valid", "sequence")}
                   for row in rows[-_DIGEST_RESULTS:]],
    }


def fidelity_request(evidence: Mapping[str, Any]) -> dict[str, Any] | None:
    """Shape proxy/authoritative pairs for the shared scoped-fidelity operator."""
    rows = evidence["measurements"]
    paired = ({row["candidate_ref"] for row in rows if row["authoritative"]}
              & {row["candidate_ref"] for row in rows if not row["authoritative"]})
    if len(paired) < 3:
        return None

    def profile(kind: str) -> dict[str, Any]:
        return {"metric": kind, "evaluation_ref": kind, "higher_is_better": evidence["higher_is_better"],
                **_PROFILE_FIELDS}

    findings = []
    for order, row in enumerate(rows):
        if row["candidate_ref"] not in paired or not row["valid"] or row["value"] is None:
            continue
        kind = "authoritative" if row["authoritative"] else "proxy"
        findings.append({"node_id": "turn", "finding_id": f"m{order}", "status": "confirmed",
                         "recorded_order": order, "measurements": [{
                             **profile(kind), "value": row["value"], "candidate_ref": row["candidate_ref"],
                             "evidence_ref": f"m{order}"}]})
    return {"node_id": "turn", "authoritative": profile("authoritative"), "proxy": profile("proxy"),
            "findings": findings}


def _reversed_fidelity(evidence: Mapping[str, Any],
                       evaluate: Callable[[dict[str, Any]], Mapping[str, Any]] | None) -> dict[str, Any] | None:
    request = None if evaluate is None else fidelity_request(evidence)
    if request is None:
        return None
    report = evaluate(request)
    if report.get("calibration_tier") != "multi_point" or report.get("fidelity_status") not in _REVERSED_FIDELITY:
        return None
    points = report.get("paired_points")
    return {"fidelity_status": report["fidelity_status"],
            "paired_points": len(points) if isinstance(points, list) else None}


def decide(
    evidence: Mapping[str, Any] | None, *, mode: str, policy: GatePolicy, remaining_seconds: float,
    worker_returns: Sequence[Mapping[str, Any]], state: dict[str, Any],
    evaluate_fidelity: Callable[[dict[str, Any]], Mapping[str, Any]] | None = None,
) -> GateDecision:
    """Decide the next step and remember which evidence a consult consumed.

    ``state`` is the caller's persisted gate memory. A trigger fires once per
    piece of new evidence, so a standing plateau cannot re-interrupt the Worker
    until more results arrive.
    """
    if mode not in GATE_MODES:
        raise ValueError("unknown Controller gate mode")
    facts: dict[str, Any] = {"remaining_seconds": round(max(0.0, remaining_seconds), 1),
                             "worker_returns": len(worker_returns)}
    if remaining_seconds <= policy.final_reserve_seconds:
        return GateDecision("finalize", "deadline_reserve", facts)
    if mode == "every-return":
        return GateDecision("consult", "every_return", facts)
    summary = None
    if evidence is not None:
        summary = summarize_authoritative(evidence, epsilon=policy.relative_epsilon)
        facts["authoritative"] = summary
    if mode == "never" or remaining_seconds < policy.min_consult_remaining_seconds:
        return GateDecision("continue", "no_consult_window" if mode != "never" else "gate_disabled", facts)

    if summary is not None:
        best_sequence = None if summary["best"] is None else summary["best"]["sequence"]
        if state.get("plateau_best_sequence") != best_sequence:
            state["plateau_best_sequence"], state["plateau_consulted_at"] = best_sequence, 0
        stalled = summary["results_since_best"]
        if (summary["best"] is not None
                and stalled - state.get("plateau_consulted_at", 0) >= policy.plateau_results):
            state["plateau_consulted_at"] = stalled
            return GateDecision("consult", "authoritative_plateau", facts)
        reversed_proxy = _reversed_fidelity(evidence, evaluate_fidelity)  # type: ignore[arg-type]
        if reversed_proxy is not None and state.get("fidelity_consulted") != reversed_proxy:
            state["fidelity_consulted"] = reversed_proxy
            return GateDecision("consult", "proxy_order_reversed", {**facts, "yardstick": reversed_proxy})

    recent = list(worker_returns)[-policy.idle_returns:]
    idle = (len(recent) == policy.idle_returns
            and all(item.get("elapsed_seconds", math.inf) < policy.idle_seconds
                    and not item.get("new_measurements") for item in recent))
    if not idle:
        state.pop("idle_consulted", None)
    elif not state.get("idle_consulted"):
        state["idle_consulted"] = True
        return GateDecision("consult", "worker_idle", facts)
    return GateDecision("continue", "no_decision_relevant_evidence", facts)
