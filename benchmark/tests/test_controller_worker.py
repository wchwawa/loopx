"""The benchmark selects the product Turn provider and pins its pending topology."""
import json
from pathlib import Path

import pytest

from benchmark.runtime import native_evidence
from benchmark.runtime.codex import CONTROLLER_GATES, Execution
from benchmark.runtime.worker import pending_turn, turn_command
from loopx.control_plane.turn_driver import evidence_gate as gate


@pytest.mark.parametrize("original", [False, True])
def test_recovery_pins_original_topology(tmp_path, original):
    marker = tmp_path / "pending.json"
    def execution(dual):
        return Execution(mode="turn", controller_worker=dual, validation_command=["check"])

    first = pending_turn(marker, "first", execution(original))
    with pytest.raises(ValueError, match="original worker topology"):
        pending_turn(marker, "retry", execution(not original))
    assert pending_turn(marker, "retry", execution(original)) == first


def test_recovery_pins_original_controller_gate(tmp_path):
    marker = tmp_path / "pending.json"

    def execution(mode):
        return Execution(mode="turn", controller_worker=True, controller_gate=mode, validation_command=["check"])

    first = pending_turn(marker, "first", execution("evidence"))
    for other in ("every-return", "never"):
        with pytest.raises(ValueError, match="original worker topology"):
            pending_turn(marker, "retry", execution(other))
    assert pending_turn(marker, "retry", execution("evidence")) == first


def test_native_history_becomes_ordered_authoritative_evidence():
    def entry(sid, score, *, status="completed", valid=True, rate=1.0):
        return {"type": "submission", "submission_id": sid, "round": "agent-" + sid, "status": status,
                "score": score, "valid": valid, "pass_rate": rate}

    history = {"best_score": 1937, "best_pass_rate": 1.0, "entries": [
        entry("a", 2288), entry("b", 2055), {"type": "note"}, entry("c", "Infinity"),
        entry("d", 1937), entry("e", 1990, valid=False), entry("f", None, status="error"),
        entry("g", 1800, status="running")]}
    snapshot = native_evidence.snapshot(history)
    # The judge selected the smallest score as best, so smaller is better here.
    assert snapshot["higher_is_better"] is False
    assert [(m["candidate_ref"], m["value"], m["valid"]) for m in snapshot["measurements"]] == [
        ("a", 2288.0, True), ("b", 2055.0, True), ("c", None, False), ("d", 1937.0, True),
        ("e", None, False), ("f", None, False)]
    summary = gate.summarize_authoritative(gate.normalize_evidence(snapshot))
    assert summary["best"]["candidate_ref"] == "d" and summary["results_since_best"] == 2

    rising = {"best_score": 7159, "best_pass_rate": 1.0, "entries": [entry("x", 5753), entry("y", 7159)]}
    assert native_evidence.snapshot(rising)["higher_is_better"] is True
    assert native_evidence.snapshot(rising, direction="minimize")["higher_is_better"] is False
    assert native_evidence.snapshot({"entries": []}) == {
        "schema_version": "turn_worker_evidence_v0", "higher_is_better": True, "measurements": []}
    with pytest.raises(ValueError, match="direction"):
        native_evidence.snapshot(rising, direction="up")
    with pytest.raises(ValueError, match="malformed"):
        native_evidence.snapshot({})


def test_benchmark_turn_command_selects_the_gate_once():
    env = {"REASONING_EFFORT": "xhigh", "CODEX_BIN": "codex", "MODEL_NAME": "m", "LOOPX_TASK_DOC": "/task.md",
           "LOOPX_PROJECT": "/work", "LOOPX_GOAL_ID": "g", "LOOPX_AGENT_ID": "a", "LOOPX_CLI": "loopx",
           "LOOPX_REGISTRY": "/reg.json", "LOOPX_RUNTIME_ROOT": "/rt"}
    check = ("check-work",)
    legacy = turn_command(env, Execution(mode="turn", controller_worker=True, validation_command=check), "wake-1")
    assert "--controller-gate" not in legacy and "--role-isolation" not in legacy
    assert "--unattended" not in legacy and "--unattended" in turn_command(
        env, Execution(mode="turn", controller_worker=True, controller_gate="never", validation_command=check),
        "wake-1")
    gated = turn_command(env, Execution(mode="turn", controller_worker=True, controller_gate="evidence",
                                        validation_command=check), "wake-1")
    at = gated.index("--controller-gate")
    assert gated[at:at + 4] == ["--controller-gate", "evidence", "--role-isolation", "caller"]
    argv = json.loads(gated[gated.index("--evidence-command-json") + 1])
    assert Path(argv[1]).name == "native_evidence.py" and Path(argv[1]).is_file()
    with pytest.raises(ValueError, match="controller-worker topology"):
        Execution(mode="turn", controller_gate="evidence", validation_command=check)
    assert Execution(mode="turn", controller_worker=True, controller_gate="never",
                     validation_command=check).validation_receipt()["controller_gate"] == "never"
    assert CONTROLLER_GATES == gate.GATE_MODES
    # The gated Worker starts from the original task; the default profile still plans first.
    entries = {mode: Execution(mode="turn", controller_worker=True, controller_gate=mode,
                               validation_command=check).task_entry for mode in CONTROLLER_GATES}
    assert entries == {"every-return": "loopx-planned", "evidence": "seeded-todo", "never": "seeded-todo"}
