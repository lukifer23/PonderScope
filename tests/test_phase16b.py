"""Phase 1.6B tests: frozen population locks, canonical draw identity, paired
draw alignment, and draw-unit statistical consistency.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from ponderscope.config.schema import ExperimentSpec
from ponderscope.population import (
    build_lock,
    enforce_population_lock,
    expected_executions,
    load_lock,
    population_key,
    verify_lock,
)
from ponderscope.tasks import generate_pack

ROOT = Path(__file__).resolve().parents[1]


def _base(**overrides) -> ExperimentSpec:
    base = {
        "name": "lock-base",
        "task_pack": "tasks-v1",
        "families": ["arith", "logic"],
        "n_per_family": 2,
        "split": "dev",
        "task_seed": 0,
        "prompt_policy": "pp-v1",
        "greedy_repeats": 0,
        "sampled_seeds": [0, 1],
    }
    base.update(overrides)
    return ExperimentSpec(**base)


def _tasks(spec: ExperimentSpec):
    return generate_pack(
        families=spec.families or None,
        n_per_family=spec.n_per_family,
        pack=spec.task_pack,
        split=spec.split,
        seed=spec.task_seed,
        prompt_policy=spec.prompt_policy,
    )


# --------------------------------------------------------------------------- #
# Population lock
# --------------------------------------------------------------------------- #
def test_identical_regeneration_passes():
    spec = _base()
    lock = build_lock(spec)
    assert verify_lock(spec, lock)["ok"] is True


def test_modified_prompt_text_fails_lock():
    spec = _base()
    lock = build_lock(spec)
    other = _base(prompt_policy="pp-v2")  # different rendered prompt text
    result = verify_lock(other, lock)
    assert result["ok"] is False


def test_modified_structural_parameters_fail_lock():
    spec = _base()
    lock = build_lock(spec)
    tasks = _tasks(spec)
    mutated = list(tasks)
    first = mutated[0]
    mutated[0] = dataclasses.replace(first, structural={**first.structural, "operations": 999})
    result = verify_lock(spec, lock, tasks=mutated)
    assert result["ok"] is False
    assert any("structural-signature" in m for m in result["mismatches"])


def test_changed_generator_behaviour_fails_lock():
    spec = _base()
    lock = build_lock(spec)
    tasks = [
        dataclasses.replace(t, difficulty={**t.difficulty, "level": "hard"}) for t in _tasks(spec)
    ]
    # structural mutation on every task simulates a generator change
    tasks = [dataclasses.replace(t, structural={**t.structural, "_gen": 2}) for t in tasks]
    result = verify_lock(spec, lock, tasks=tasks)
    assert result["ok"] is False


def test_changed_seed_fails_lock():
    spec = _base()
    lock = build_lock(spec)
    result = verify_lock(_base(task_seed=1), lock)
    assert result["ok"] is False
    assert any("population key" in m for m in result["mismatches"])


def test_changed_split_fails_lock():
    spec = _base()
    lock = build_lock(spec)
    assert verify_lock(_base(split="test"), lock)["ok"] is False


def test_changed_family_selection_fails_lock():
    spec = _base()
    lock = build_lock(spec)
    assert verify_lock(_base(families=["arith"]), lock)["ok"] is False


def test_changed_task_count_fails_lock():
    spec = _base()
    lock = build_lock(spec)
    assert verify_lock(_base(n_per_family=3), lock)["ok"] is False


def test_bf16_and_q4_populations_are_identical():
    bf = ExperimentSpec.from_dict(
        json.loads((ROOT / "specs" / "phase1_6-replication-bf16.json").read_text())
    )
    q4 = ExperimentSpec.from_dict(
        json.loads((ROOT / "specs" / "phase1_6-replication-q4.json").read_text())
    )
    assert population_key(bf) == population_key(q4)
    lock = build_lock(bf)
    assert verify_lock(q4, lock)["ok"] is True


def test_hash_or_lock_corruption_fails():
    spec = _base()
    lock = build_lock(spec)
    lock["task_ids_sha256"] = "0" * 64
    assert verify_lock(spec, lock)["ok"] is False
    lock2 = build_lock(spec)
    lock2["structural_signature_sha256"] = "0" * 64
    assert verify_lock(spec, lock2)["ok"] is False


def test_missing_lock_fails_for_locked_experiment():
    spec = _base()
    result = enforce_population_lock(spec, lock_path="/nonexistent/lock.json")
    assert result["locked"] is True and result["ok"] is False


def test_historical_experiments_remain_supported_without_lock(tmp_path):
    spec = _base()
    result = enforce_population_lock(spec, specs_dir=tmp_path)  # no lock present
    assert result["locked"] is False and result["ok"] is True


def test_replication_lock_matches_declared_population():
    bf = ExperimentSpec.from_dict(
        json.loads((ROOT / "specs" / "phase1_6-replication-bf16.json").read_text())
    )
    lock = load_lock(ROOT / "specs" / "phase1_6-population-lock.json")
    assert verify_lock(bf, lock)["ok"] is True
    assert lock["family_counts"] == {"arith": 6, "logic": 6, "order": 6, "path": 6, "sm": 6}
    assert lock["expected_executions"]["sampled_draws"] == 60


def test_expected_executions_separate_draws_from_repeats():
    spec = _base(sampled_seeds=[0, 1], sampled_repeats_per_seed=1)
    e1 = expected_executions(spec)
    assert e1["sampled_draws"] == 8 and e1["sampled_executions"] == 8
    spec2 = _base(sampled_seeds=[0, 1], sampled_repeats_per_seed=2)
    e2 = expected_executions(spec2)
    assert e2["sampled_draws"] == 8 and e2["sampled_executions"] == 16
    assert e2["total_draws"] == 8 and e2["total_executions"] == 16
