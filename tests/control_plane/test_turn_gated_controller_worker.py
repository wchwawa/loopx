"""A continuously working Worker; Controller attention only where evidence changes the choice."""
import json
import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from loopx.control_plane.turn_driver import controller_worker as roles
from loopx.control_plane.turn_driver import evidence_gate as gate

TASK = "Optimise the solver.\nSubmit when it improves."
RECEIPT = {"result_kind": "validated_progress"}


def evidence_of(values):
    return gate.normalize_evidence({"schema_version": gate.EVIDENCE_SCHEMA, "higher_is_better": True,
                                    "measurements": [{"candidate_ref": f"s{i}", "value": value, "valid": True,
                                                      "authoritative": True, "sequence": i}
                                                     for i, value in enumerate(values)]})


class Transport:
    """Charge each Worker turn against one Turn deadline, as the real transport does."""

    def __init__(self, *, budget, turn_seconds, controller=(), fail_worker_calls=()):
        self.now, self.deadline, self.turn_seconds = 0.0, budget, turn_seconds
        self.controller, self.fail = iter(controller), set(fail_worker_calls)
        self.evidence = {"threads": {}}
        self.worker_prompts, self.controller_prompts, self.schemas = [], [], []

    def remaining(self):
        if self.deadline - self.now <= 0:
            raise TimeoutError("controller-worker Turn deadline")
        return self.deadline - self.now

    def send(self, role, prompt, schema, *, reserve=0.0):
        if role == "controller":
            self.controller_prompts.append(prompt)
            self.schemas.append(schema)
            return next(self.controller)
        budget = self.remaining() - reserve
        if budget <= 0:
            return None
        self.worker_prompts.append(prompt)
        if len(self.worker_prompts) in self.fail:
            self.evidence["threads"].setdefault("worker", "worker-thread")
            raise RuntimeError("native role Turn did not complete")
        self.evidence["threads"]["worker"] = "worker-thread"
        if self.turn_seconds > budget:
            self.now += budget
            return None
        self.now += self.turn_seconds
        return {"finding": f"candidate {len(self.worker_prompts)}", "artifacts": ["solver.cc"]}


def run(native, tmp_path, *, mode, snapshots, timeout=10_000.0, sleeps=None):
    host = roles.RoleHost(tmp_path, tmp_path / "rt", tmp_path / "reg", tmp_path / "home", "codex", None, "low",
                          "danger-full-access", timeout, dispatch_task_material=True, gate=mode,
                          role_isolation="caller")
    feed = iter(snapshots)
    trail = {}
    result = roles.run_gated_roles(
        native, {}, TASK, host=host, save=lambda: None, trail=trail,
        read_evidence=lambda _: (next(feed, None) or (None, "evidence_command_failed")),
        sleep=(sleeps.append if sleeps is not None else lambda _: None), clock=lambda: native.now)
    return result, trail


@pytest.fixture(autouse=True)
def plain_contract(monkeypatch):
    monkeypatch.setattr(roles, "prepare_codex_controller_input", lambda _: ("C1 CONTRACT", {"type": "object"}))
    monkeypatch.setattr(roles, "normalize_codex_result", lambda _, result: result)
    monkeypatch.setattr(roles, "evaluate_fidelity", lambda params: pytest.fail("no proxy pairs were supplied"))


def test_worker_advances_without_controller_until_results_stop_improving(tmp_path):
    scores = [1.0, 2.0, 3.0, 3.0, 2.5, 2.9, 4.0, 5.0]
    snapshots = [(evidence_of(scores[:n]), None) for n in range(1, len(scores) + 1)]
    native = Transport(budget=10_000, turn_seconds=1_000, controller=[
        {"instruction": "Replace the greedy pass with beam search.", "result": None},
        {"instruction": None, "result": RECEIPT}])
    result, trail = run(native, tmp_path, mode="evidence", snapshots=snapshots)

    assert result == RECEIPT
    # The Worker starts from the caller's exact task with no Controller turn in front of it.
    assert native.worker_prompts[0] == TASK
    assert native.worker_prompts[1:6] == [roles.CONTINUE_INSTRUCTION] * 5
    # Three results without a new best fired exactly one consult; its instruction reached the Worker verbatim.
    assert native.worker_prompts[6] == "Replace the greedy pass with beam search."
    assert native.worker_prompts[7:] == [roles.CONTINUE_INSTRUCTION] * 3
    assert len(native.controller_prompts) == 2
    consult, settle = native.controller_prompts
    assert consult.startswith("C1 CONTRACT") and TASK in consult and "Trigger: authoritative_plateau" in consult
    assert '"results_since_best": 3' in consult and "For your Core receipt:" in consult
    assert "C1 CONTRACT" not in settle and "Trigger: deadline_reserve" in settle
    assert native.schemas[1]["properties"]["instruction"] == {"type": "null"}
    assert [(d["action"], d["reason"]) for d in trail["gate_decisions"]] == [
        *[("continue", "no_decision_relevant_evidence")] * 5, ("consult", "authoritative_plateau"),
        *[("continue", "no_decision_relevant_evidence")] * 3, ("finalize", "deadline_reserve")]
    assert trail["gate_decisions"][5]["controller_redirected"] is True
    assert [r["outcome"] for r in trail["worker_returns"]][-1] == "reserve_reached"


def test_never_mode_spends_one_controller_turn_for_the_receipt(tmp_path):
    flat = [(evidence_of([5.0] + [1.0] * n), None) for n in range(12)]
    native = Transport(budget=5_000, turn_seconds=700, controller=[{"instruction": None, "result": RECEIPT}])
    result, trail = run(native, tmp_path, mode="never", snapshots=flat, timeout=5_000.0)
    assert result == RECEIPT and len(native.controller_prompts) == 1
    assert native.controller_prompts[0].startswith("C1 CONTRACT")
    assert {d["reason"] for d in trail["gate_decisions"][:-1]} == {"gate_disabled"}
    assert native.worker_prompts == [TASK] + [roles.CONTINUE_INSTRUCTION] * (len(native.worker_prompts) - 1)


def test_controller_may_let_the_worker_proceed_unchanged(tmp_path):
    flat = [(evidence_of([5.0, 1.0, 1.0][:n] + [1.0] * max(0, n - 3)), None) for n in range(1, 9)]
    native = Transport(budget=10_000, turn_seconds=1_500, controller=[
        {"instruction": "continue", "result": None}, {"instruction": None, "result": RECEIPT}])
    _, trail = run(native, tmp_path, mode="evidence", snapshots=flat)
    consult = next(d for d in trail["gate_decisions"] if d["action"] == "consult")
    assert consult["controller_redirected"] is False
    assert set(native.worker_prompts[1:]) == {roles.CONTINUE_INSTRUCTION}


def test_transient_worker_failure_resumes_the_same_thread(tmp_path):
    sleeps = []
    native = Transport(budget=10_000, turn_seconds=2_000, fail_worker_calls={2, 3},
                       controller=[{"instruction": None, "result": RECEIPT}])
    result, trail = run(native, tmp_path, mode="never", snapshots=[], sleeps=sleeps)
    assert result == RECEIPT and sleeps == list(roles._WORKER_RETRY_BACKOFF_SECONDS[:2])
    assert [r["outcome"] for r in trail["worker_returns"]][:3] == ["returned", "retried_after_failure",
                                                                  "retried_after_failure"]
    assert native.worker_prompts[1:4] == [roles.CONTINUE_INSTRUCTION] * 3
    assert all("evidence_error" in d for d in trail["gate_decisions"])

    exhausted = Transport(budget=10_000, turn_seconds=10, fail_worker_calls=set(range(2, 40)))
    with pytest.raises(RuntimeError, match="did not complete"):
        run(exhausted, tmp_path, mode="never", snapshots=[])
    never_started = Transport(budget=10_000, turn_seconds=10, fail_worker_calls={1})
    never_started.evidence["threads"].clear()
    never_started.send = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("native role Turn did not complete"))
    with pytest.raises(RuntimeError, match="did not complete"):
        run(never_started, tmp_path, mode="never", snapshots=[])


def test_settlement_requires_a_typed_result(tmp_path):
    native = Transport(budget=1_000, turn_seconds=2_000, controller=[{"instruction": "keep going", "result": None}])
    with pytest.raises(ValueError, match="invalid task"):
        run(native, tmp_path, mode="never", snapshots=[], timeout=1_000.0)


def test_gated_roles_need_worker_audience_material(tmp_path):
    def host(**options):
        return roles.RoleHost(tmp_path, tmp_path, tmp_path, tmp_path, "codex", None, None, "danger-full-access",
                              10, **options)

    for options in ({"gate": "evidence"}, {"gate": "never"}, {"role_isolation": "caller"}):
        with pytest.raises(ValueError, match="Worker-audience task material"):
            host(**options)
    with pytest.raises(ValueError, match="unknown Controller gate"):
        host(gate="sometimes", dispatch_task_material=True)
    assert host().continuous is False and host(gate="never", dispatch_task_material=True).continuous is True


def test_caller_isolation_drops_the_nested_sandbox_for_both_roles(tmp_path, monkeypatch):
    listed = type("Done", (), {"returncode": 0, "stdout": b"[]"})()
    monkeypatch.setattr(roles.subprocess, "run", lambda *a, **k: listed)
    monkeypatch.setattr(roles, "role_policy", lambda *a: pytest.fail("no nested profile under the caller grant"))
    host = roles.RoleHost(tmp_path, tmp_path / "rt", tmp_path / "reg", tmp_path / "home", "codex", None, None,
                          "danger-full-access", 10, dispatch_task_material=True, role_isolation="caller")
    for role in ("controller", "worker"):
        assert roles.role_sandbox(role, host) == "danger-full-access"
        options = roles.role_cli_options(role, host, 10)
        assert not any(part.startswith(("permissions=", "default_permissions=", "shell_environment_policy="))
                       for part in options)
        assert 'approval_policy="never"' in options
    confined = roles.RoleHost(tmp_path, tmp_path / "rt", tmp_path / "reg", tmp_path / "home", "codex", None, None,
                              "danger-full-access", 10)
    assert roles.role_sandbox("worker", confined) is None


def test_evidence_command_is_trusted_argv_and_fails_soft(tmp_path):
    script = tmp_path / "evidence.py"
    script.write_text("import json,sys\nprint(json.dumps({'schema_version':'turn_worker_evidence_v0',"
                      "'higher_is_better':False,'measurements':[{'candidate_ref':'a','value':3,'valid':True,"
                      "'authoritative':True,'sequence':0}]}))\n")

    def host(argv):
        return roles.RoleHost(tmp_path, tmp_path, tmp_path, tmp_path, "codex", None, None, "read-only", 10,
                              evidence_command=roles.parse_evidence_command(json.dumps(argv)))

    evidence, error = roles.read_worker_evidence(host([sys.executable, str(script)]))
    assert error is None and evidence["higher_is_better"] is False and evidence["measurements"][0]["value"] == 3.0
    assert roles.read_worker_evidence(host([sys.executable, "-c", "raise SystemExit(3)"])) == (
        None, "evidence_command_failed")
    assert roles.read_worker_evidence(host([sys.executable, "-c", "print('not json')"])) == (
        None, "evidence_unreadable")
    assert roles.read_worker_evidence(host([str(tmp_path / "missing-binary")])) == (None, "evidence_unreadable")
    assert roles.parse_evidence_command(None) == ()
    for bad in ("[]", '"sforge-submit --list"', '["ok", 3]'):
        with pytest.raises(ValueError, match="JSON argv"):
            roles.parse_evidence_command(bad)


@pytest.mark.skipif(os.environ.get("LOOPX_TEST_NATIVE_ROLE_HOST") != "1",
                    reason="opt-in installed Codex CLI against a local inference fixture")
def test_real_native_gated_turn_consults_only_on_measured_plateau(tmp_path):
    from tests.control_plane.canonical_authority_fixture import initialize_canonical_source_fixture
    from tests.control_plane.codex_checkpoint_fixture import local_inference_provider

    src = Path(__file__).resolve().parents[2]
    fixture = runpy.run_path(str(src / "examples/loopx-turn-codex-cli-e2e-smoke.py"))
    project, runtime, _, registry = fixture["_write_fixture"](tmp_path, turn_count=2)
    goal, agent = fixture["GOAL_ID"], fixture["AGENT_ID"]
    initialize_canonical_source_fixture(runtime, goal,
        state_path=project / ".codex" / "goals" / goal / "ACTIVE_GOAL_STATE.md", provider="sqlite")
    marker, expected = fixture["MARKER_NAME"], fixture["_marker_value"](1)
    (project / "observation").write_text(expected)
    task = tmp_path / "worker-task.txt"
    task.write_text(TASK)
    # Each read adds three results that fail to beat the first one: a standing plateau.
    calls = tmp_path / "evidence-calls"
    evidence = tmp_path / "evidence.py"
    evidence.write_text(
        "import json, pathlib\n"
        f"p = pathlib.Path({str(calls)!r})\n"
        "n = int(p.read_text()) + 1 if p.exists() else 1\n"
        "p.write_text(str(n))\n"
        "rows = [{'candidate_ref': f's{i}', 'value': 5.0 if i == 0 else 1.0, 'valid': True,\n"
        "         'authoritative': True, 'sequence': i} for i in range(1 + 3 * n)]\n"
        "print(json.dumps({'schema_version': 'turn_worker_evidence_v0', 'higher_is_better': True,\n"
        "                  'measurements': rows}))\n")
    inputs, key = [], None

    def respond(request, index, message):
        nonlocal key
        text = [part["text"] for item in request["input"] if item.get("role") == "user"
                for part in item["content"] if "text" in part][-1]
        if index in (1, 4):
            inputs.append(text)
            command = "cat observation" if index == 1 else (
                "tr -d '\\n' > " + marker + " <<EOF\n$(cat observation)\nEOF")
            return {"type": "function_call", "id": f"fc_{index}", "call_id": f"call_{index}",
                    "name": "exec_command", "arguments": json.dumps({"cmd": command})}
        if index == 2:
            answer = {"finding": "Observed " + expected, "artifacts": []}
        elif index == 3:
            inputs.append(text)
            key = json.JSONDecoder().raw_decode(text.split("Turn request:\n", 1)[1])[0]["turn_key"]
            answer = {"instruction": "Copy the checked observation into the output file.", "result": None}
        elif index == 5:
            answer = {"finding": "Written", "artifacts": [marker]}
        else:
            inputs.append(text)
            answer = {"instruction": None, "result": {
                "schema_version": "loopx_turn_result_v0", "turn_key": key, "result_kind": "validated_progress",
                "completed_phases": ["host_execute", "typed_result"], "classification": "fixture_progress",
                "recommended_action": "Check the output.", "next_action": "Continue the work.",
                "delivery_batch_scale": "single_surface", "delivery_outcome": "outcome_progress",
                "vision_unchanged_reason": "The objective is unchanged.", "summary": "Updated the artifact.",
                "work_evidence": ["artifact:marker"], "validation_evidence": ["check:marker-readback"]}}
        message["content"][0]["text"] = json.dumps(answer)
        return message

    binary = shutil.which("codex")
    assert binary, "Install the Codex CLI before enabling native role qualification"
    invocations = tmp_path / "cli-invocations"
    wrapper = tmp_path / "observed-codex"
    wrapper.write_text(f"#!{sys.executable}\nimport os,sys\nargs = sys.argv[1:]\n"
                       f"with open({str(invocations)!r}, 'a') as f:\n"
                       "    print('exec resume ' + args[-2] if 'resume' in args else args[0], file=f)\n"
                       f"os.execv({binary!r}, [{binary!r}, *args])\n")
    wrapper.chmod(0o700)
    with local_inference_provider(respond) as (url, _, _):
        home = tmp_path / "role-home"
        home.mkdir()
        (home / "config.toml").write_text(
            f'model="gpt-5.4"\nmodel_reasoning_effort="low"\nmodel_provider="local"\n[model_providers.local]\n'
            f'name="local"\nbase_url="{url}"\nwire_api="responses"\nenv_key="OPENAI_API_KEY"\n')
        env = {k: v for k, v in os.environ.items() if not k.startswith("LOOPX_")}
        env.update(CODEX_HOME=str(home), OPENAI_API_KEY="fixture", PYTHONPATH=str(src), LOOPX_USAGE_PING="0")
        cmd = [sys.executable, "-m", "loopx.cli", "--registry", str(registry), "--runtime-root", str(runtime),
               "--format", "json", "turn", "run-once", "--execute", "--no-global-sync",
               "--goal-id", goal, "--agent-id", agent, "--project", str(project),
               "--host", "codex-cli", "--execution-mode", "isolated-headless", "--topology", "controller-worker",
               "--scheduler-owner", "outer_controller", "--prepare-selected-todo",
               "--turn-instance-id", "gated-roles", "--timeout-seconds", "120", "--codex-bin", str(wrapper),
               "--task-material", str(task), "--codex-sandbox", "workspace-write",
               "--controller-gate", "evidence", "--role-isolation", "caller",
               "--evidence-command-json", json.dumps([sys.executable, str(evidence)]),
               "--validation-command-json", json.dumps(fixture["_validator_command"](expected))]
        completed = subprocess.run(cmd, cwd=project, env=env, timeout=150, capture_output=True, text=True)
        assert completed.returncode == 0, completed.stdout[-2200:] + completed.stderr[-500:]
        result = json.loads(completed.stdout)
        assert result["status"] == "committed" and result["host"]["topology"] == "controller-worker"
        # A different gate is a different admitted configuration.
        start = cmd.index("--turn-instance-id")
        changed = cmd[:start] + ["--resume-turn-key", result["resume_turn_key"]] + cmd[start + 2:]
        changed[changed.index("--controller-gate") + 1] = "never"
        held = subprocess.run(changed, cwd=project, env=env, capture_output=True, text=True, timeout=30)
        assert held.returncode != 0 and "original Turn topology" in held.stdout + held.stderr

    # The Worker received the caller's task first, verbatim, with no Controller turn before it.
    assert inputs[0] == TASK and inputs[2] == "Copy the checked observation into the output file."
    assert "Trigger: authoritative_plateau" in inputs[1] and "For your Core receipt:" in inputs[1]
    assert "Trigger: authoritative_plateau" in inputs[3] and "Turn request:" not in inputs[3]
    assert (project / marker).read_text() == expected
    record, = (runtime / "goals" / goal / "turns/controller-worker").glob("*.json")
    journal = json.loads(record.read_text())
    assert [step["role"] for step in journal["steps"]] == ["worker", "controller", "worker", "controller"]
    assert [(d["action"], d["reason"]) for d in journal["gate_decisions"]] == [
        ("consult", "authoritative_plateau")] * 2
    assert journal["gate_decisions"][0]["controller_redirected"] is True
    actual = [line for line in invocations.read_text().splitlines() if line.startswith("exec")]
    assert actual == ["exec", "exec", "exec resume " + journal["threads"]["worker"],
                      "exec resume " + journal["threads"]["controller"]]
    assert "sandbox" not in invocations.read_text().splitlines()
    assert fixture["_quota_spend_count"](runtime) == 1
