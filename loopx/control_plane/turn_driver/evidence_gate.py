"""Host attention gate for a continuously working Worker.

At each Worker boundary the Host asks one question without calling a model:
is anything new worth a Controller look? If not, the Worker continues at zero
inference cost. The Controller, when asked, makes the judgment: continue,
change direction, or ask for help.

The gate is generic. It reads typed signals from three sources and never
interprets Worker prose:

- the Worker's own typed report state (blocked, or its assumption refuted);
- independent checks that keep failing;
- measured results, for tasks that have a score (a stall, no valid result, or
  a proxy that ranks against the authoritative measure).

A task with none of these simply lets the Worker continue. The gate owns no
Goal, Todo, quota or settlement state.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

GATE_MODES = ("every-return", "evidence", "never")
# The Worker states its own situation as a typed value in each report.
WORKER_STATES = ("working", "blocked", "assumption_refuted")
# The Controller's answer to a consult. `needs_help` ends the Turn for an owner.
CONTROLLER_VERDICTS = ("continue", "redirect", "needs_help")
EVIDENCE_SCHEMA = "turn_worker_evidence_v0"
_MAX_ROWS = 512
_MAX_REF_CHARS = 200
_DIGEST_RESULTS = 8
_REVERSED_FIDELITY = ("order_reversed", "frontier_bias_cliff")
_PROFILE_FIELDS = {"unit": "value", "comparison_scope": "turn", "input_revision": "current"}
_REPORTED_REASONS = {"blocked": "worker_blocked", "assumption_refuted": "assumption_refuted"}


@dataclass(frozen=True)
class GatePolicy:
    """Thresholds a caller may tune; defaults favour uninterrupted work."""

    # A run is stalled when its last `plateau_results` authoritative results
    # added less than `min_gain_share` of everything it has gained so far.
    # Counting only strict new bests misses a run that keeps adding marginal
    # gains inside one approach while a different approach has far more room.
    plateau_results: int = 3
    min_gain_share: float = 0.05
    # An independent check that fails this many times in a row is not noise.
    failing_checks: int = 2
    idle_returns: int = 2
    idle_seconds: float = 120.0
    final_reserve_seconds: float = 180.0
    min_consult_remaining_seconds: float = 900.0

    def __post_init__(self) -> None:
        if self.plateau_results < 2 or self.idle_returns < 2 or self.failing_checks < 2:
            raise ValueError("gate windows need at least two observations")
        for value in (self.min_gain_share, self.idle_seconds, self.final_reserve_seconds,
                      self.min_consult_remaining_seconds):
            if not math.isfinite(value) or value < 0:
                raise ValueError("gate thresholds must be finite and non-negative")
        if self.min_gain_share >= 1:
            raise ValueError("the stalled-gain share must be below one")


@dataclass(frozen=True)
class GateDecision:
    action: str  # continue | consult | finalize
    reason: str
    facts: dict[str, Any]


def _ref(row: Mapping[str, Any], key: str) -> str:
    ref, sequence = row.get(key), row.get("sequence")
    if (not isinstance(ref, str) or not ref or len(ref) > _MAX_REF_CHARS
            or type(sequence) is not int or sequence < 0):
        raise ValueError("invalid Worker evidence row")
    return ref


def normalize_evidence(payload: Any) -> dict[str, Any]:
    """Validate one evidence snapshot. Measurements and checks are both optional."""
    if not isinstance(payload, Mapping) or payload.get("schema_version") != EVIDENCE_SCHEMA:
        raise ValueError("unsupported Worker evidence snapshot")
    rows, checked = payload.get("measurements", []), payload.get("checks", [])
    direction = payload.get("higher_is_better")
    if (not isinstance(rows, list) or not isinstance(checked, list)
            or len(rows) > _MAX_ROWS or len(checked) > _MAX_ROWS
            or (rows and type(direction) is not bool) or (direction is not None and type(direction) is not bool)):
        raise ValueError("invalid Worker evidence snapshot")
    measurements = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("invalid Worker measurement")
        value = row.get("value")
        if type(row.get("valid")) is not bool or type(row.get("authoritative")) is not bool:
            raise ValueError("invalid Worker measurement")
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not math.isfinite(value)):
            raise ValueError("invalid Worker measurement value")
        measurements.append({"candidate_ref": _ref(row, "candidate_ref"),
                             "value": None if value is None else float(value), "valid": row["valid"],
                             "authoritative": row["authoritative"], "sequence": row["sequence"]})
    checks = []
    for row in checked:
        if not isinstance(row, Mapping) or type(row.get("passed")) is not bool:
            raise ValueError("invalid Worker check")
        checks.append({"check_ref": _ref(row, "check_ref"), "passed": row["passed"], "sequence": row["sequence"]})
    measurements.sort(key=lambda item: (item["sequence"], item["authoritative"], item["candidate_ref"]))
    checks.sort(key=lambda item: (item["sequence"], item["check_ref"]))
    return {"higher_is_better": bool(direction), "measurements": measurements, "checks": checks}


def evidence_size(evidence: Mapping[str, Any] | None) -> int:
    """How many evidence rows a snapshot carries, to tell new evidence from none."""
    return 0 if evidence is None else len(evidence["measurements"]) + len(evidence["checks"])


def summarize_authoritative(evidence: Mapping[str, Any], *, window: int = 3) -> dict[str, Any]:
    """Incumbent, and how much of the run's total gain its last `window` results added.

    ``recent_gain_share`` is None until `window` results follow the first valid
    one. It is 0.0 for a flat run and 1.0 when all the gain is recent.
    """
    higher = evidence["higher_is_better"]
    rows = [row for row in evidence["measurements"] if row["authoritative"]]
    best: dict[str, Any] | None = None
    first: float | None = None
    since_best, incumbents = 0, []
    for row in rows:
        scored = row["valid"] and row["value"] is not None
        if scored and (best is None or (row["value"] > best["value"] if higher else row["value"] < best["value"])):
            best, since_best = row, 0
            first = row["value"] if first is None else first
        else:
            since_best += 1
        incumbents.append(None if best is None else best["value"])
    share = None
    if best is not None and len(incumbents) > window and incumbents[-window - 1] is not None:
        total = abs(best["value"] - first)
        share = 0.0 if total == 0 else abs(best["value"] - incumbents[-window - 1]) / total
    return {
        "count": len(rows),
        "best": None if best is None else {key: best[key] for key in ("candidate_ref", "value", "sequence")},
        "results_since_best": since_best,
        "recent_gain_share": None if share is None else round(share, 4),
        "recent": [{key: row[key] for key in ("candidate_ref", "value", "valid", "sequence")}
                   for row in rows[-_DIGEST_RESULTS:]],
    }


def fidelity_request(evidence: Mapping[str, Any]) -> dict[str, Any] | None:
    """Shape proxy/authoritative pairs for the shared scoped-fidelity operator."""
    rows = evidence["measurements"]
    paired = ({row["candidate_ref"] for row in rows if row["authoritative"]}
              & {row["candidate_ref"] for row in rows if not row["authoritative"]})
    # Three points give three ordered pairs: one noisy comparison flips the sign.
    if len(paired) < 4:
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


# Each source answers "is there something new here worth a Controller look?"
# and records in `state` what it consumed, so the same fact asks only once.

def _reported_signal(worker_returns: Sequence[Mapping[str, Any]], policy: GatePolicy,
                     state: dict[str, Any]) -> str | None:
    """The Worker said, as a typed state, that it is blocked or that its assumption fell."""
    current = worker_returns[-1].get("state") if worker_returns else None
    if current not in _REPORTED_REASONS:
        state.pop("reported_state", None)
        return None
    # Ask on entering the state, then again only if it persists across more returns.
    persisted = len(worker_returns) - state.get("reported_at", 0)
    if state.get("reported_state") == current and persisted < policy.idle_returns:
        return None
    state["reported_state"], state["reported_at"] = current, len(worker_returns)
    return _REPORTED_REASONS[current]


def _check_signal(evidence: Mapping[str, Any], policy: GatePolicy, state: dict[str, Any],
                  facts: dict[str, Any]) -> str | None:
    """An independent check has failed several times in a row."""
    checks = evidence["checks"]
    if not checks:
        return None
    failing = 0
    for row in reversed(checks):
        if row["passed"]:
            break
        failing += 1
    facts["checks"] = {"count": len(checks), "failing_in_a_row": failing,
                       "recent": checks[-_DIGEST_RESULTS:]}
    if failing < policy.failing_checks or len(checks) - state.get("checks_consulted_count", 0) < policy.failing_checks:
        return None
    state["checks_consulted_count"] = len(checks)
    return "independent_check_failing"


def _measured_signal(evidence: Mapping[str, Any], policy: GatePolicy, state: dict[str, Any],
                     facts: dict[str, Any],
                     evaluate_fidelity: Callable[[dict[str, Any]], Mapping[str, Any]] | None) -> str | None:
    """Scored tasks: results stalled, none is valid, or a proxy ranks against the authoritative measure."""
    summary = facts.get("authoritative")
    if summary is None or not summary["count"]:
        return None
    # Wait for a full window of new results before asking about the same stall again.
    fresh = summary["count"] - state.get("stall_consulted_count", 0) >= policy.plateau_results
    share = summary["recent_gain_share"]
    if summary["best"] is None:
        # With no incumbent at all, every authoritative attempt so far failed.
        stalled, reason = summary["count"] >= policy.plateau_results, "no_valid_authoritative_result"
    else:
        stalled, reason = share is not None and share < policy.min_gain_share, "authoritative_plateau"
    if stalled and fresh:
        state["stall_consulted_count"] = summary["count"]
        return reason
    reversed_proxy = _reversed_fidelity(evidence, evaluate_fidelity)
    if reversed_proxy is not None and state.get("fidelity_consulted") != reversed_proxy:
        state["fidelity_consulted"] = reversed_proxy
        facts["yardstick"] = reversed_proxy
        return "proxy_order_reversed"
    return None


def _idle_signal(worker_returns: Sequence[Mapping[str, Any]], policy: GatePolicy,
                 state: dict[str, Any]) -> str | None:
    """The Worker keeps returning quickly with nothing new."""
    recent = list(worker_returns)[-policy.idle_returns:]
    idle = (len(recent) == policy.idle_returns
            and all(item.get("elapsed_seconds", math.inf) < policy.idle_seconds
                    and not item.get("new_evidence") for item in recent))
    if not idle:
        state.pop("idle_consulted", None)
        return None
    if state.get("idle_consulted"):
        return None
    state["idle_consulted"] = True
    return "worker_idle"


def decide(
    evidence: Mapping[str, Any] | None, *, mode: str, policy: GatePolicy, remaining_seconds: float,
    worker_returns: Sequence[Mapping[str, Any]], state: dict[str, Any],
    evaluate_fidelity: Callable[[dict[str, Any]], Mapping[str, Any]] | None = None,
) -> GateDecision:
    """Decide whether this boundary deserves Controller attention.

    ``state`` is the caller's persisted gate memory. A signal asks once per
    piece of new information, so a standing condition cannot re-interrupt the
    Worker until something new arrives. With no signal the Worker continues.
    """
    if mode not in GATE_MODES:
        raise ValueError("unknown Controller gate mode")
    facts: dict[str, Any] = {"remaining_seconds": round(max(0.0, remaining_seconds), 1),
                             "worker_returns": len(worker_returns)}
    if remaining_seconds <= policy.final_reserve_seconds:
        return GateDecision("finalize", "deadline_reserve", facts)
    if mode == "every-return":
        return GateDecision("consult", "every_return", facts)
    if worker_returns and worker_returns[-1].get("state") in WORKER_STATES:
        facts["worker_state"] = worker_returns[-1]["state"]
    if evidence is not None and evidence["measurements"]:
        facts["authoritative"] = summarize_authoritative(evidence, window=policy.plateau_results)
    if mode == "never" or remaining_seconds < policy.min_consult_remaining_seconds:
        return GateDecision("continue", "no_consult_window" if mode != "never" else "gate_disabled", facts)

    # Explicit statements first, then independent checks, then measured trends.
    reason = _reported_signal(worker_returns, policy, state)
    if reason is None and evidence is not None:
        reason = (_check_signal(evidence, policy, state, facts)
                  or _measured_signal(evidence, policy, state, facts, evaluate_fidelity))
    if reason is None:
        reason = _idle_signal(worker_returns, policy, state)
    if reason is None:
        return GateDecision("continue", "no_decision_relevant_evidence", facts)
    return GateDecision("consult", reason, facts)
