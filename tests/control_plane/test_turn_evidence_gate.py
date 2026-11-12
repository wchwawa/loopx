"""The evidence gate interrupts a working Worker only when measurements change the choice."""
import pytest

from loopx.control_plane.turn_driver import evidence_gate as gate

POLICY = gate.GatePolicy(plateau_results=3, final_reserve_seconds=60, min_consult_remaining_seconds=300)


def snapshot(values, *, higher=True, proxies=()):
    rows = [{"candidate_ref": f"c{i}", "value": value, "valid": value is not None, "authoritative": True,
             "sequence": i} for i, value in enumerate(values)]
    rows += [{"candidate_ref": ref, "value": value, "valid": True, "authoritative": False, "sequence": 100 + i}
             for i, (ref, value) in enumerate(proxies)]
    return gate.normalize_evidence({"schema_version": gate.EVIDENCE_SCHEMA, "higher_is_better": higher,
                                    "measurements": rows})


def decide(evidence, state, *, mode="evidence", remaining=3600.0, returns=(), fidelity=None):
    return gate.decide(evidence, mode=mode, policy=POLICY, remaining_seconds=remaining,
                       worker_returns=list(returns), state=state, evaluate_fidelity=fidelity)


def test_improving_results_never_interrupt_the_worker():
    state = {}
    for upto in range(1, 7):
        decision = decide(snapshot([1.0, 2.0, 3.0, 4.0, 5.0, 6.0][:upto]), state)
        assert (decision.action, decision.reason) == ("continue", "no_decision_relevant_evidence")
    assert decision.facts["authoritative"]["best"] == {"candidate_ref": "c5", "value": 6.0, "sequence": 5}


def test_plateau_consults_once_until_more_results_arrive():
    state = {}
    assert decide(snapshot([5.0, 4.0, 5.0]), state).action == "continue"
    first = decide(snapshot([5.0, 4.0, 5.0, None]), state)
    assert (first.action, first.reason) == ("consult", "authoritative_plateau")
    assert first.facts["authoritative"]["results_since_best"] == 3
    # The same standing plateau must not interrupt the Worker again.
    assert decide(snapshot([5.0, 4.0, 5.0, None]), state).action == "continue"
    assert decide(snapshot([5.0, 4.0, 5.0, None, 4.5, 4.6]), state).action == "continue"
    assert decide(snapshot([5.0, 4.0, 5.0, None, 4.5, 4.6, 4.7]), state).reason == "authoritative_plateau"
    # A new best starts a fresh window.
    assert decide(snapshot([5.0, 4.0, 5.0, None, 4.5, 4.6, 4.7, 9.0]), state).action == "continue"
    assert decide(snapshot([5.0, 4.0, 5.0, None, 4.5, 4.6, 4.7, 9.0, 1.0, 1.0]), state).action == "continue"
    assert decide(snapshot([5.0, 4.0, 5.0, None, 4.5, 4.6, 4.7, 9.0, 1.0, 1.0, 1.0]), state).action == "consult"


def test_lower_is_better_direction_and_no_valid_incumbent():
    state = {}
    assert decide(snapshot([None, None]), state).action == "continue"
    # Every authoritative attempt failed: decision-relevant even without an incumbent.
    failed = decide(snapshot([None, None, None]), state)
    assert (failed.action, failed.reason) == ("consult", "no_valid_authoritative_result")
    assert failed.facts["authoritative"]["best"] is None
    assert decide(snapshot([None, None, None, None]), state).action == "continue"
    assert decide(snapshot([10.0, 9.0, 8.0, 7.0], higher=False), {}).action == "continue"
    assert decide(snapshot([10.0, 9.0, 9.5, 9.0, 12.0], higher=False), {}).reason == "authoritative_plateau"


def test_modes_bound_controller_attention():
    plateau = snapshot([5.0, 1.0, 1.0, 1.0])
    assert decide(plateau, {}, mode="never").action == "continue"
    assert decide(snapshot([1.0, 2.0]), {}, mode="every-return").action == "consult"
    # Too little budget remains for a redirection to pay off.
    assert decide(plateau, {}, remaining=200).reason == "no_consult_window"
    for mode in gate.GATE_MODES:
        assert decide(plateau, {}, mode=mode, remaining=30).action == "finalize"
    with pytest.raises(ValueError, match="gate mode"):
        decide(plateau, {}, mode="always")


def test_idle_returns_without_evidence_consult_once_per_streak():
    state = {}
    quick = {"elapsed_seconds": 20.0, "new_measurements": 0}
    busy = {"elapsed_seconds": 900.0, "new_measurements": 0}
    assert decide(None, state, returns=[quick]).action == "continue"
    assert decide(None, state, returns=[quick, quick]).reason == "worker_idle"
    assert decide(None, state, returns=[quick, quick, quick]).action == "continue"
    assert decide(None, state, returns=[quick, quick, quick, busy]).action == "continue"
    assert decide(None, state, returns=[quick, quick, quick, busy, quick, quick]).reason == "worker_idle"
    measured = {"elapsed_seconds": 20.0, "new_measurements": 1}
    assert decide(None, {}, returns=[measured, measured]).action == "continue"


def test_reversed_proxy_reuses_the_shared_fidelity_operator():
    proxies = [("c0", 9.0), ("c1", 8.0), ("c2", 7.0), ("c3", 6.0)]
    evidence = snapshot([1.0, 2.0, 3.0, 4.0], proxies=proxies)
    request = gate.fidelity_request(evidence)
    assert request["authoritative"]["metric"] == "authoritative" and request["proxy"]["metric"] == "proxy"
    assert {m["candidate_ref"] for f in request["findings"] for m in f["measurements"]} == {"c0", "c1", "c2", "c3"}
    assert gate.fidelity_request(snapshot([1.0, 2.0], proxies=proxies[:2])) is None

    calls = []

    def reversed_report(params):
        calls.append(params)
        return {"calibration_tier": "multi_point", "fidelity_status": "order_reversed", "paired_points": [1, 2, 3, 4]}

    state = {}
    first = decide(evidence, state, fidelity=reversed_report)
    assert (first.action, first.reason) == ("consult", "proxy_order_reversed")
    assert first.facts["yardstick"] == {"fidelity_status": "order_reversed", "paired_points": 4}
    assert decide(evidence, state, fidelity=reversed_report).action == "continue"
    aligned = {"calibration_tier": "multi_point", "fidelity_status": "aligned_in_tested_scope"}
    assert decide(evidence, {}, fidelity=lambda _: aligned).action == "continue"
    two_point = {"calibration_tier": "two_point", "fidelity_status": "order_reversed"}
    assert decide(evidence, {}, fidelity=lambda _: two_point).action == "continue"
    assert len(calls) == 2


@pytest.mark.parametrize("payload", [
    {"schema_version": "other", "higher_is_better": True, "measurements": []},
    {"schema_version": gate.EVIDENCE_SCHEMA, "higher_is_better": "yes", "measurements": []},
    {"schema_version": gate.EVIDENCE_SCHEMA, "higher_is_better": True, "measurements": [
        {"candidate_ref": "", "value": 1.0, "valid": True, "authoritative": True, "sequence": 0}]},
    {"schema_version": gate.EVIDENCE_SCHEMA, "higher_is_better": True, "measurements": [
        {"candidate_ref": "a", "value": float("nan"), "valid": True, "authoritative": True, "sequence": 0}]},
    {"schema_version": gate.EVIDENCE_SCHEMA, "higher_is_better": True, "measurements": [
        {"candidate_ref": "a", "value": True, "valid": True, "authoritative": True, "sequence": 0}]},
])
def test_malformed_evidence_is_rejected(payload):
    with pytest.raises(ValueError, match="Worker"):
        gate.normalize_evidence(payload)
