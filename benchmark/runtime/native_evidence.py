"""Project the solver-visible native submission history as Worker evidence.

Run by the Turn Host between Worker returns, never by a model. It reads only
the history the solver itself can read with ``sforge-submit --list`` and
prints a ``turn_worker_evidence_v0`` snapshot. It submits nothing and spends no
evaluation slot. Standard library only: the Host runs it by file path.
"""
from __future__ import annotations

import json
import math
import os
import sys
import urllib.request

SCHEMA = "turn_worker_evidence_v0"
DIRECTIONS = {"maximize": True, "minimize": False}


def _score(entry: dict) -> float | None:
    value = entry.get("score")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def infer_higher_is_better(history: dict, scored: list[tuple[dict, float]]) -> bool:
    """Read the direction from the judge's own best selection when it is decisive."""
    best = history.get("best_score")
    top_rate = history.get("best_pass_rate")
    peers = [value for entry, value in scored if entry.get("pass_rate") == top_rate] or [v for _, v in scored]
    if isinstance(best, (int, float)) and not isinstance(best, bool) and peers and min(peers) != max(peers):
        if best == min(peers):
            return False
    return True


def snapshot(history: dict, *, direction: str | None = None) -> dict:
    """Completed agent submissions in arrival order; unfinished ones are not evidence yet."""
    entries = history.get("entries")
    if not isinstance(entries, list):
        raise ValueError("malformed native submission history")
    completed = [entry for entry in entries if isinstance(entry, dict) and entry.get("type") == "submission"
                 and entry.get("status") in {"completed", "error"}]
    scored = [(entry, value) for entry in completed if (value := _score(entry)) is not None
              and entry.get("valid") is not False and entry.get("status") == "completed"]
    if direction is not None and direction not in DIRECTIONS:
        raise ValueError("native score direction must be maximize or minimize")
    higher = DIRECTIONS[direction] if direction else infer_higher_is_better(history, scored)
    valid_ids = {id(entry) for entry, _ in scored}
    measurements = []
    for sequence, entry in enumerate(completed):
        ref = entry.get("submission_id") or entry.get("round")
        if not isinstance(ref, str) or not ref:
            continue
        valid = id(entry) in valid_ids
        measurements.append({"candidate_ref": ref[:200], "value": _score(entry) if valid else None,
                             "valid": valid, "authoritative": True, "sequence": sequence})
    return {"schema_version": SCHEMA, "higher_is_better": higher, "measurements": measurements}


def main() -> int:
    url = os.environ.get("SFORGE_FEEDBACK_URL")
    if not url:
        print("native feedback endpoint unavailable", file=sys.stderr)
        return 2
    with urllib.request.urlopen(url.rstrip("/") + "/history", timeout=20) as response:  # noqa: S310
        history = json.load(response)
    print(json.dumps(snapshot(history, direction=os.environ.get("LOOPX_NATIVE_SCORE_DIRECTION") or None)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
