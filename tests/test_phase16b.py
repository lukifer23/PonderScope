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


# --------------------------------------------------------------------------- #
# Canonical draw identity (condition-aware) and paired alignment
# --------------------------------------------------------------------------- #
from ponderscope.analysis.analyze import _config_summary  # noqa: E402
from ponderscope.analysis.compare import (  # noqa: E402
    _collapsed_draws,
    _draw_key,
    _key,
    _paired_draw_population,
)
from ponderscope.evidence.run import RunStore  # noqa: E402


def _rec(pres, task, cid, seed, repeat, tokens, correct=True, mode="sampled"):
    return {
        "family": "arith",
        "deployment": {
            "weight_variant": {"precision": "float32", "quantization": None},
            "source": {"repo_id": "fake/model", "revision": "deadbeef"},
            "runtime": {"runtime": "fake", "runtime_version": "0", "hardware": "h"},
            "decoding": {"mode": mode},
        },
        "presentation_id": pres,
        "task_id": task,
        "condition_id": cid,
        "condition": {"mode": mode, "seed": seed, "repeat": repeat},
        "trace": {
            "token_ids": list(tokens),
            "wall_ms": 10.0,
            "tokens_per_sec": 50.0,
            "ttft_ms": 1.0,
        },
        "correct": correct,
        "reasoning_tokens": len(tokens),
        "answer_tokens": 1,
        "total_tokens": len(tokens),
        "answer_normalized": "1" if correct else None,
        "natural_final_status": "correct" if correct else "censored",
        "termination": {
            "think_end_reached": True,
            "terminated_by_eos": True,
            "capped": False,
            "missing_answer": False,
        },
        "metrics": {
            "unique_token_ratio": 1.0,
            "repeated_ngram_fraction_4": 0.0,
            "text_repeat_ratio": 0.0,
        },
        "loop": {"longest_run_length": 1},
    }


def _store(tmp_path):
    return RunStore(path=tmp_path, manifest={})


def test_two_conditions_do_not_collapse(tmp_path):
    a = _rec("pres0", "task0", "cond-A", 0, 0, [1, 2, 3])
    b = _rec("pres0", "task0", "cond-B", 0, 0, [4, 5, 6])
    draws = _collapsed_draws(_store(tmp_path), {_key(a), _key(b)}, traces=[a, b])
    assert len(draws) == 2
    assert all(not d["ambiguous"] for d in draws.values())


def test_technical_repeats_collapse_within_condition(tmp_path):
    a0 = _rec("pres0", "task0", "cond-A", 0, 0, [1, 2, 3])
    a1 = _rec("pres0", "task0", "cond-A", 0, 1, [1, 2, 3])
    draws = _collapsed_draws(_store(tmp_path), {_key(a0), _key(a1)}, traces=[a0, a1])
    assert len(draws) == 1
    assert draws[_draw_key(a0)]["n_executions"] == 2
    assert draws[_draw_key(a0)]["ambiguous"] is False


def test_ambiguous_repeats_flagged(tmp_path):
    a0 = _rec("pres0", "task0", "cond-A", 0, 0, [1, 2, 3])
    a1 = _rec("pres0", "task0", "cond-A", 0, 1, [9, 9, 9])
    draws = _collapsed_draws(_store(tmp_path), {_key(a0), _key(a1)}, traces=[a0, a1])
    assert draws[_draw_key(a0)]["ambiguous"] is True
    assert draws[_draw_key(a0)]["record"] is None


def _pair(tmp_path, a_traces, b_traces):
    keys = {_key(r) for r in a_traces} | {_key(r) for r in b_traces}
    a = _store(tmp_path / "a")
    b = _store(tmp_path / "b")
    return _paired_draw_population(a, b, keys, traces_a=a_traces, traces_b=b_traces)


def test_ambiguous_in_a_excludes_from_both(tmp_path):
    # task0 seed0: A clean, B clean; task0 seed1: A ambiguous, B clean.
    A = [
        _rec("pres0", "task0", "cond-x", 0, 0, [1]),
        _rec("pres0", "task0", "cond-x", 1, 0, [2]),
        _rec("pres0", "task0", "cond-x", 1, 1, [8]),  # divergent -> ambiguous
    ]
    B = [
        _rec("pres0", "task0", "cond-x", 0, 0, [1]),
        _rec("pres0", "task0", "cond-x", 1, 0, [2]),
    ]
    # match by canonical trial key (seed1 repeat1 only in A, so B must declare it too)
    a = _store(tmp_path / "a")
    b = _store(tmp_path / "b")
    keys = {_key(r) for r in A} | {_key(r) for r in B}
    a_by, b_by, pop = _paired_draw_population(a, b, keys, traces_a=A, traces_b=B)
    assert pop["n_ambiguous_a"] == 1
    assert pop["n_paired_draws"] == 1  # only seed0
    assert len(a_by["task0"]) == 1 and len(b_by["task0"]) == 1


def test_ambiguous_in_b_excludes_from_both(tmp_path):
    A = [_rec("pres0", "task0", "cond-x", 0, 0, [1]), _rec("pres0", "task0", "cond-x", 1, 0, [2])]
    B = [
        _rec("pres0", "task0", "cond-x", 0, 0, [1]),
        _rec("pres0", "task0", "cond-x", 1, 0, [2]),
        _rec("pres0", "task0", "cond-x", 1, 1, [7]),
    ]
    a_by, b_by, pop = _pair(tmp_path, A, B)
    assert pop["n_ambiguous_b"] == 1
    assert pop["n_paired_draws"] == 1


def test_zero_valid_paired_draws(tmp_path):
    A = [_rec("pres0", "task0", "cond-x", 0, 0, [1])]
    B = [_rec("pres0", "task0", "cond-x", 1, 0, [2])]  # no shared draw key (different seed)
    a_by, b_by, pop = _pair(tmp_path, A, B)
    assert pop["n_paired_draws"] == 0
    assert a_by == {} and b_by == {}


def test_fully_matched_population_pairs_everything(tmp_path):
    A = [_rec("pres0", "task0", "cond-x", s, 0, [s]) for s in (0, 1)]
    B = [_rec("pres0", "task0", "cond-x", s, 0, [s]) for s in (0, 1)]
    a_by, b_by, pop = _pair(tmp_path, A, B)
    assert pop["n_paired_draws"] == 2
    assert len(a_by["task0"]) == len(b_by["task0"]) == 2


# --------------------------------------------------------------------------- #
# Draw-unit consistency (standalone vs paired)
# --------------------------------------------------------------------------- #
def _run(tmp_path, fake_backend, monkeypatch, name, seeds, repeats=1):
    from ponderscope.config.schema import ExperimentSpec
    from ponderscope.experiment import run_experiment

    monkeypatch.setattr("ponderscope.experiment.get_backend", lambda _n: fake_backend)
    spec = ExperimentSpec(
        name=name,
        task_pack="tasks-v1",
        families=["arith"],
        n_per_family=2,
        greedy_repeats=0,
        sampled_seeds=list(seeds),
        sampled_repeats_per_seed=repeats,
        max_tokens=16,
    )
    return run_experiment(
        spec,
        model_repo="fake/model",
        model_revision="deadbeef",
        runs_dir=tmp_path,
        code_state={"version": "test", "git_sha": "0" * 40, "tracked_dirty": False},
    )


def test_standalone_and_paired_units_agree(tmp_path, fake_backend, monkeypatch):
    from ponderscope.analysis import analyze_run, compare_configs

    a = _run(tmp_path, fake_backend, monkeypatch, "a", [0, 1], repeats=3)
    b = _run(tmp_path, fake_backend, monkeypatch, "b", [0, 1], repeats=3)
    analysis = analyze_run(a.store)
    c = next(iter(analysis["configs"].values()))
    assert c["n"] == 4  # primary draws, not 12 executions
    assert c["n_executions"] == 12
    assert c["execution_diagnostics"]["n_executions"] == 12
    cmp = compare_configs(a.store, b.store, mode="sampled", n_resamples=100)
    assert cmp["analysis_population"]["n_draws_used_a"] == c["n"] == 4


def test_ambiguous_draws_excluded_from_primary_but_visible(tmp_path):
    records = [
        _rec("pres0", "task0", "cond-x", 0, 0, [1]),
        _rec("pres0", "task0", "cond-x", 0, 1, [1]),
        _rec("pres1", "task1", "cond-x", 0, 0, [2]),
        _rec("pres1", "task1", "cond-x", 0, 1, [9]),  # ambiguous
    ]
    summary = _config_summary("cond-x", records)
    assert summary["n"] == 1  # one primary draw (task0); task1 ambiguous excluded
    assert summary["n_ambiguous_draws"] == 1
    assert summary["categories"]["n"] == 1
    assert summary["execution_diagnostics"]["n_executions"] == 4


# --------------------------------------------------------------------------- #
# Compact derived analysis (no raw evidence duplication)
# --------------------------------------------------------------------------- #
from ponderscope.analysis.draws import compact_draw, draw_summary  # noqa: E402


def test_compact_draw_has_no_raw_evidence():
    draw = {
        "stochastic_draw_id": "draw-x",
        "key": ["pres0", "cond-x", 0],
        "task_id": "task0",
        "presentation_id": "pres0",
        "family": "arith",
        "condition_id": "cond-x",
        "mode": "sampled",
        "seed": 0,
        "n_executions": 1,
        "token_identical": True,
        "ambiguous": False,
        "first_token_divergence": None,
        "execution_trial_ids": ["trial-x"],
        "record": _rec("pres0", "task0", "cond-x", 0, 0, [1, 2, 3]),
    }
    entry = compact_draw(draw)
    blob = json.dumps(entry)
    for forbidden in ("token_ids", '"text"', '"prompt"', "raw_text", "final_raw", "record"):
        assert forbidden not in blob
    assert entry["scalars"]["reasoning_tokens"] == 3


def test_draw_summary_is_compact():
    from ponderscope.analysis.draws import collapse_to_stochastic_draws

    records = [_rec("pres0", "task0", "cond-x", 0, 0, [1, 2])]
    summary = draw_summary(collapse_to_stochastic_draws(records))
    assert "record" not in json.dumps(summary)
    assert summary["draws"][0]["scalars"]["reasoning_tokens"] == 2


def test_analysis_json_has_no_raw_evidence(tmp_path, fake_backend, monkeypatch):
    from ponderscope.analysis import analyze_run

    a = _run(tmp_path, fake_backend, monkeypatch, "compact", [0, 1])
    analyze_run(a.store)
    text = (a.store.path / "analysis.json").read_text()
    for forbidden in ("token_ids", '"raw_text"', '"final_raw"', '"prompt"'):
        assert forbidden not in text
    assert '"interpretation": "phase1.6"' in text


def test_analysis_is_idempotent(tmp_path, fake_backend, monkeypatch):
    from ponderscope.analysis import analyze_run

    a = _run(tmp_path, fake_backend, monkeypatch, "idem", [0, 1])
    first = analyze_run(a.store)
    second = analyze_run(a.store)
    c1 = next(iter(first["configs"].values()))
    c2 = next(iter(second["configs"].values()))
    assert c1["reasoning_tokens"] == c2["reasoning_tokens"]
    assert c1["categories"] == c2["categories"]


# --------------------------------------------------------------------------- #
# Preflight gates and design-planning precision
# --------------------------------------------------------------------------- #
from ponderscope.design import design_plan  # noqa: E402
from ponderscope.preflight import run_preflight  # noqa: E402


def test_preflight_gates_block_on_missing_source(tmp_path):
    spec = _base(families=["arith"], n_per_family=1, sampled_seeds=[0])
    report = run_preflight(
        spec, model_repo="fake/model", model_revision="deadbeef", runs_dir=tmp_path
    )
    assert report["gates"]["source_cache"] == "BLOCK"
    assert report["ok"] is False
    assert any("source_cache" in b for b in report["blocks"])
    assert "expected_generations" in report["gates"]


def test_preflight_lock_block(tmp_path):
    spec = _base()
    lock = build_lock(spec)
    lock["task_ids_sha256"] = "0" * 64
    lock_path = tmp_path / "bad-lock.json"
    lock_path.write_text(json.dumps(lock))
    report = run_preflight(
        spec,
        model_repo="fake/model",
        model_revision="deadbeef",
        runs_dir=tmp_path,
        population_lock=str(lock_path),
    )
    assert report["gates"]["population_lock"] == "BLOCK"
    assert report["ok"] is False


def test_design_plan_is_deterministic_and_sensitive():
    a = design_plan(seed=7, n_monte_carlo=40, n_resamples=200, task_sizes=(30, 60))
    b = design_plan(seed=7, n_monte_carlo=40, n_resamples=200, task_sizes=(30, 60))
    assert a == b
    widths = {row["n_tasks"]: row["closure_ci_half_width"] for row in a["sensitivity"]}
    assert widths[30] > widths[60]  # more tasks -> tighter interval
    assert a["assumptions"]["simulation_seed"] == 7
    assert a["kind"].startswith("design-planning")
