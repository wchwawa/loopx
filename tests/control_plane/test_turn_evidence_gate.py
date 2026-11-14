"""The Host asks for Controller attention only when something new could change the choice."""
import pytest

from loopx.control_plane.turn_driver import evidence_gate as gate

POLICY = gate.GatePolicy(plateau_results=3, final_reserve_seconds=60, min_consult_remaining_seconds=300)


def snapshot(values, *, higher=True, proxies=(), checks=()):
    rows = [{"candidate_ref": f"c{i}", "value": value, "valid": value is not None, "authoritative": True,
             "sequence": i} for i, value in enumerate(values)]
    rows += [{"candidate_ref": ref, "value": value, "valid": True, "authoritative": False, "sequence": 100 + i}
             for i, (ref, value) in enumerate(proxies)]
    return gate.normalize_evidence({
        "schema_version": gate.EVIDENCE_SCHEMA, "higher_is_better": higher, "measurements": rows,
        "checks": [{"check_ref": "independent", "passed": passed, "sequence": i} for i, passed in enumerate(checks)]})


def checked(*results):
    """A task with no score: only independent pass/fail checks."""
    return gate.normalize_evidence({"schema_version": gate.EVIDENCE_SCHEMA, "checks": [
        {"check_ref": "independent", "passed": passed, "sequence": i} for i, passed in enumerate(results)]})


def returned(state="working", *, seconds=900.0, new=1):
    return {"elapsed_seconds": seconds, "new_evidence": new, "state": state}


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


def test_marginal_gains_inside_one_approach_count_as_a_stall():
    # Every result is a strict new best, yet the last three add under 5% of the gain so far.
    creeping = [10.0, 20.0, 30.0, 30.2, 30.3, 30.4]
    state = {}
    for upto in range(1, 6):
        assert decide(snapshot(creeping[:upto]), state).action == "continue"
    stalled = decide(snapshot(creeping), state)
    assert (stalled.action, stalled.reason) == ("consult", "authoritative_plateau")
    assert stalled.facts["authoritative"]["results_since_best"] == 0
    assert stalled.facts["authoritative"]["recent_gain_share"] == pytest.approx(0.4 / 20.4, abs=1e-4)
    # A run that still converts results into comparable gains is left alone.
    climbing = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
    state = {}
    assert {decide(snapshot(climbing[:upto]), state).action for upto in range(1, 9)} == {"continue"}
    # The share needs a full window of results after the first valid one.
    assert gate.summarize_authoritative(snapshot([10.0, 10.0, 10.0]))["recent_gain_share"] is None
    assert gate.summarize_authoritative(snapshot([10.0, 10.0, 10.0, 10.0]))["recent_gain_share"] == 0.0


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
    quick = {"elapsed_seconds": 20.0, "new_evidence": 0}
    busy = {"elapsed_seconds": 900.0, "new_evidence": 0}
    assert decide(None, state, returns=[quick]).action == "continue"
    assert decide(None, state, returns=[quick, quick]).reason == "worker_idle"
    assert decide(None, state, returns=[quick, quick, quick]).action == "continue"
    assert decide(None, state, returns=[quick, quick, quick, busy]).action == "continue"
    assert decide(None, state, returns=[quick, quick, quick, busy, quick, quick]).reason == "worker_idle"
    measured = {"elapsed_seconds": 20.0, "new_evidence": 1}
    assert decide(None, {}, returns=[measured, measured]).action == "continue"


def test_reversed_proxy_reuses_the_shared_fidelity_operator():
    proxies = [("c0", 9.0), ("c1", 8.0), ("c2", 7.0), ("c3", 6.0)]
    evidence = snapshot([1.0, 2.0, 3.0, 4.0], proxies=proxies)
    request = gate.fidelity_request(evidence)
    assert request["authoritative"]["metric"] == "authoritative" and request["proxy"]["metric"] == "proxy"
    assert {m["candidate_ref"] for f in request["findings"] for m in f["measurements"]} == {"c0", "c1", "c2", "c3"}
    # Three pairs give three comparisons; one noisy one would flip the verdict.
    assert gate.fidelity_request(snapshot([1.0, 2.0, 3.0], proxies=proxies[:3])) is None

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


def test_a_task_without_score_or_news_lets_the_worker_continue():
    state = {}
    working = [returned() for _ in range(6)]
    for upto in range(1, 7):
        for evidence in (None, checked(), checked(*[True] * upto)):
            decision = decide(evidence, state, returns=working[:upto])
            assert (decision.action, decision.reason) == ("continue", "no_decision_relevant_evidence")
    assert "authoritative" not in decision.facts and decision.facts["worker_state"] == "working"


def test_worker_reported_state_asks_once_then_only_if_it_persists():
    state = {}
    history = [returned()]
    assert decide(None, state, returns=history).action == "continue"
    history.append(returned("blocked"))
    first = decide(None, state, returns=history)
    assert (first.action, first.reason, first.facts["worker_state"]) == ("consult", "worker_blocked", "blocked")
    history.append(returned("blocked"))
    assert decide(None, state, returns=history).action == "continue"
    history.append(returned("blocked"))
    assert decide(None, state, returns=history).reason == "worker_blocked"
    # A different statement is new information at once.
    history.append(returned("assumption_refuted"))
    assert decide(None, state, returns=history).reason == "assumption_refuted"
    history += [returned(), returned("assumption_refuted")]
    assert decide(None, state, returns=history[:-1]).action == "continue"
    assert decide(None, state, returns=history).reason == "assumption_refuted"
    # The Worker's own statement outranks a measured stall seen at the same boundary.
    assert decide(snapshot([5.0, 1.0, 1.0, 1.0]), {}, returns=[returned("blocked")]).reason == "worker_blocked"
    # An unknown state is not a signal, and the disabled gate ignores every signal.
    assert decide(None, {}, returns=[{"elapsed_seconds": 900.0, "state": "confused"}]).action == "continue"
    assert decide(None, {}, mode="never", returns=[returned("blocked")]).reason == "gate_disabled"


def test_independent_check_failing_repeatedly_asks_for_attention():
    state = {}
    assert decide(checked(True, False), state, returns=[returned()]).action == "continue"
    failing = decide(checked(True, False, False), state, returns=[returned()])
    assert (failing.action, failing.reason) == ("consult", "independent_check_failing")
    assert failing.facts["checks"]["failing_in_a_row"] == 2 and failing.facts["checks"]["count"] == 3
    # The same failures do not ask again; two more do.
    assert decide(checked(True, False, False), state, returns=[returned()]).action == "continue"
    assert decide(checked(True, False, False, False), state, returns=[returned()]).action == "continue"
    assert decide(checked(True, False, False, False, False), state, returns=[returned()]).action == "consult"
    # A pass ends the run of failures.
    assert decide(checked(False, False, True), {}, returns=[returned()]).action == "continue"
    # Checks and measurements can arrive together; failing checks are judged first.
    both = decide(snapshot([5.0, 1.0, 1.0, 1.0], checks=[False, False]), {}, returns=[returned()])
    assert both.reason == "independent_check_failing" and both.facts["authoritative"]["count"] == 4
    assert gate.evidence_size(snapshot([1.0, 2.0], checks=[True])) == 3 and gate.evidence_size(None) == 0


@pytest.mark.parametrize("payload", [
    {"schema_version": "other", "higher_is_better": True, "measurements": []},
    {"schema_version": gate.EVIDENCE_SCHEMA, "higher_is_better": "yes", "measurements": []},
    {"schema_version": gate.EVIDENCE_SCHEMA, "higher_is_better": True, "measurements": [
        {"candidate_ref": "", "value": 1.0, "valid": True, "authoritative": True, "sequence": 0}]},
    {"schema_version": gate.EVIDENCE_SCHEMA, "higher_is_better": True, "measurements": [
        {"candidate_ref": "a", "value": float("nan"), "valid": True, "authoritative": True, "sequence": 0}]},
    {"schema_version": gate.EVIDENCE_SCHEMA, "higher_is_better": True, "measurements": [
        {"candidate_ref": "a", "value": True, "valid": True, "authoritative": True, "sequence": 0}]},
    # Measurements need a direction; checks need a typed outcome.
    {"schema_version": gate.EVIDENCE_SCHEMA, "measurements": [
        {"candidate_ref": "a", "value": 1.0, "valid": True, "authoritative": True, "sequence": 0}]},
    {"schema_version": gate.EVIDENCE_SCHEMA, "checks": [{"check_ref": "c", "passed": "no", "sequence": 0}]},
    {"schema_version": gate.EVIDENCE_SCHEMA, "checks": [{"check_ref": "", "passed": True, "sequence": 0}]},
])
def test_malformed_evidence_is_rejected(payload):
    with pytest.raises(ValueError, match="Worker"):
        gate.normalize_evidence(payload)
