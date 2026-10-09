"""Phase 1.5 tests: matched-trial RMST, independent termination event time,
competing-event reporting, and evidence-verification gating.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ponderscope.analysis import compare_configs
from ponderscope.analysis.analyze import (
    _is_generation_termination,
    _is_reasoning_closure,
    _time_to_closure,
    _time_to_termination,
    analyze_run,
)
from ponderscope.analysis.compare import _trial_population
from ponderscope.config.schema import ExperimentSpec
from ponderscope.evidence.run import RunStore
from ponderscope.experiment import run_experiment

CLEAN = {"version": "test", "git_sha": "0" * 40, "tracked_dirty": False}
ALL5 = ["arith", "path", "order", "logic", "sm"]


def _run(
    tmp_path: Path,
    fake_backend,
    monkeypatch,
    name,
    seeds,
    *,
    n_per_family=2,
    families=("arith",),
    repeats=1,
    greedy=0,
    task_seed=0,
):
    monkeypatch.setattr("ponderscope.experiment.get_backend", lambda _n: fake_backend)
    spec = ExperimentSpec(
        name=name,
        task_pack="tasks-v1",
        families=list(families),
        n_per_family=n_per_family,
        greedy_repeats=greedy,
        sampled_seeds=list(seeds),
        sampled_repeats_per_seed=repeats,
        max_tokens=16,
        task_seed=task_seed,
    )
    return run_experiment(
        spec,
        model_repo="fake/model",
        model_revision="deadbeef",
        runs_dir=tmp_path,
        code_state=CLEAN,
    )


# --------------------------------------------------------------------------- #
# Issue A: paired RMST restricted to matched stochastic draws
# --------------------------------------------------------------------------- #
def test_seed0_pair_ignores_extra_bf16_seeds(tmp_path, fake_backend, monkeypatch):
    a = _run(tmp_path, fake_backend, monkeypatch, "a", [0])
    b_full = _run(tmp_path, fake_backend, monkeypatch, "bfull", [0, 1, 2])
    b_seed0 = _run(tmp_path, fake_backend, monkeypatch, "b0", [0])
    cmp_full = compare_configs(a.store, b_full.store, mode="sampled", n_resamples=200)
    cmp_b0 = compare_configs(a.store, b_seed0.store, mode="sampled", n_resamples=200)
    assert cmp_full["refused"] is False
    ep_full = cmp_full["rmst"]["endpoints"]["reasoning_closure"]
    ep_b0 = cmp_b0["rmst"]["endpoints"]["reasoning_closure"]
    # The extra BF16 seeds must not change a seed-0 paired estimate.
    assert ep_full["mean"] == ep_b0["mean"]
    assert ep_full["n_clusters"] == ep_b0["n_clusters"] == 2
    pop = cmp_full["trial_population"]
    assert pop["matched_pairs"] == 2
    assert pop["unmatched_b"] == 4  # seeds 1 and 2 x 2 tasks
    assert pop["complete"] is False


def test_duplicate_technical_repeats_do_not_change_estimate(tmp_path, fake_backend, monkeypatch):
    a = _run(tmp_path, fake_backend, monkeypatch, "r1", [0, 1], repeats=1)
    b = _run(tmp_path, fake_backend, monkeypatch, "r2", [0, 1], repeats=2)
    cmp = compare_configs(a.store, b.store, mode="sampled", n_resamples=200)
    assert cmp["refused"] is False
    ep = cmp["rmst"]["endpoints"]["reasoning_closure"]
    # Repeats collapse to one draw per (presentation, task, seed): no change.
    assert ep["mean"] == 0.0
    assert ep["n_clusters"] == 2
    assert cmp["trial_population"]["matched_pairs"] == 4  # repeat 0 intersection


def test_full_thirty_draw_comparison_is_complete(tmp_path, fake_backend, monkeypatch):
    a = _run(tmp_path, fake_backend, monkeypatch, "fa", [0, 1, 2], n_per_family=2, families=ALL5)
    b = _run(tmp_path, fake_backend, monkeypatch, "fb", [0, 1, 2], n_per_family=2, families=ALL5)
    cmp = compare_configs(a.store, b.store, mode="sampled", n_resamples=200, require_complete=True)
    assert cmp["refused"] is False
    pop = cmp["trial_population"]
    assert pop["matched_pairs"] == 30
    assert pop["observed_executions_a"] == 30
    assert pop["complete"] is True
    assert cmp["rmst"]["endpoints"]["reasoning_closure"]["mean"] == 0.0


def test_require_complete_refuses_unbalanced_population(tmp_path, fake_backend, monkeypatch):
    a = _run(tmp_path, fake_backend, monkeypatch, "ca", [0])
    b = _run(tmp_path, fake_backend, monkeypatch, "cb", [0, 1])
    refused = compare_configs(a.store, b.store, mode="sampled", require_complete=True)
    assert refused["refused"] is True
    assert any("not complete" in r for r in refused["refusal_reasons"])


def test_identical_runs_zero_rmst_and_zero_effect(tmp_path, fake_backend, monkeypatch):
    a = _run(tmp_path, fake_backend, monkeypatch, "ia", [0, 1])
    b = _run(tmp_path, fake_backend, monkeypatch, "ib", [0, 1])
    cmp = compare_configs(a.store, b.store, mode="sampled", n_resamples=200)
    for ep in cmp["rmst"]["endpoints"].values():
        assert ep["mean"] == 0.0
    assert cmp["metrics"]["success_at_budget"]["delta"]["mean"] == 0.0


def test_multiple_conditions_do_not_collide(tmp_path, fake_backend, monkeypatch):
    a = _run(tmp_path, fake_backend, monkeypatch, "ma", [0], greedy=1)
    b = _run(tmp_path, fake_backend, monkeypatch, "mb", [0], greedy=1)
    cmp = compare_configs(a.store, b.store, mode="sampled", n_resamples=200)
    # mode filter keeps greedy out of the sampled population.
    assert cmp["trial_population"]["observed_executions_a"] == 2  # 2 tasks x 1 seed
    assert cmp["trial_population"]["matched_pairs"] == 2


def test_presentation_mismatch_is_refused(tmp_path, fake_backend, monkeypatch):
    a = _run(tmp_path, fake_backend, monkeypatch, "pa", [0])
    b = _run(tmp_path, fake_backend, monkeypatch, "pb", [0])
    b.store.manifest["spec"] = {**b.store.manifest["spec"], "prompt_policy": "pp-v2"}
    cmp = compare_configs(a.store, b.store, mode="sampled")
    assert cmp["refused"] is True
    assert any("prompt policy" in r for r in cmp["refusal_reasons"])


def test_nonmatching_task_populations_refused(tmp_path, fake_backend, monkeypatch):
    a = _run(tmp_path, fake_backend, monkeypatch, "ta", [0], task_seed=0)
    b = _run(tmp_path, fake_backend, monkeypatch, "tb", [0], task_seed=77)
    cmp = compare_configs(a.store, b.store, mode="sampled")
    assert cmp["refused"] is True
    assert any("task populations differ" in r for r in cmp["refusal_reasons"])


def test_trial_population_reports_duplicates_and_missing(tmp_path, fake_backend, monkeypatch):
    a = _run(tmp_path, fake_backend, monkeypatch, "da", [0])
    b = _run(tmp_path, fake_backend, monkeypatch, "db", [0])
    base = a.store.read_traces()
    dup = list(base) + [dict(base[0])]  # duplicate logical identity
    pop = _trial_population(a.store, b.store, dup, b.store.read_traces(), [], "sampled")
    assert pop["duplicate_keys_a"] == 1
    assert pop["complete"] is False


# --------------------------------------------------------------------------- #
# Issue B: independent event times
# --------------------------------------------------------------------------- #
def _rec(*, think_end, eos, capped, reasoning_tokens, total_tokens):
    return {
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": total_tokens,
        "termination": {
            "think_end_reached": think_end,
            "terminated_by_eos": eos,
            "capped": capped,
        },
    }


EVENT_CASES = [
    # (think_end, eos, capped, reasoning_tokens, total_tokens, close_time, term_time, close_ev, term_ev)
    (True, True, False, 800, 850, 801.0, 850.0, True, True),  # close then EOS
    (True, False, True, 800, 850, 801.0, 850.0, True, False),  # close, capped answer
    (False, True, False, 850, 850, 850.0, 850.0, False, True),  # EOS no close (competing)
    (False, False, True, 2048, 2048, 2048.0, 2048.0, False, False),  # neither
]


@pytest.mark.parametrize("case", EVENT_CASES)
def test_event_time_matrix(case):
    think_end, eos, capped, rtok, total, close_t, term_t, close_ev, term_ev = case
    r = _rec(think_end=think_end, eos=eos, capped=capped, reasoning_tokens=rtok, total_tokens=total)
    assert _time_to_closure(r) == close_t
    assert _time_to_termination(r) == term_t
    assert _is_reasoning_closure(r) is close_ev
    assert _is_generation_termination(r) is term_ev


def test_termination_time_is_not_closure_time():
    r = _rec(think_end=True, eos=True, capped=False, reasoning_tokens=800, total_tokens=850)
    assert _time_to_closure(r) == 801.0
    assert _time_to_termination(r) == 850.0
    assert _time_to_closure(r) != _time_to_termination(r)


# --------------------------------------------------------------------------- #
# Stage 2: evidence-verification gating
# --------------------------------------------------------------------------- #
def _tamper_manifest(store: RunStore) -> None:
    p = store.path / "manifest.json"
    p.write_text(p.read_text() + " ")


def test_compare_refuses_unverified_by_default(tmp_path, fake_backend, monkeypatch):
    a = _run(tmp_path, fake_backend, monkeypatch, "va", [0])
    b = _run(tmp_path, fake_backend, monkeypatch, "vb", [0])
    _tamper_manifest(b.store)
    cmp = compare_configs(a.store, b.store, mode="sampled")
    assert cmp["refused"] is True
    assert any("does not verify" in r for r in cmp["refusal_reasons"])
    # explicit override computes but is labelled exploratory.
    cmp2 = compare_configs(
        a.store, b.store, mode="sampled", allow_unverified=True, allow_confounded=True
    )
    assert cmp2["refused"] is False
    assert cmp2["exploratory"] is True
    assert cmp2["verification"]["b"]["pass"] is False


def test_analyze_refuses_unverified_by_default(tmp_path, fake_backend, monkeypatch):
    a = _run(tmp_path, fake_backend, monkeypatch, "aa", [0])
    _tamper_manifest(a.store)
    with pytest.raises(RuntimeError, match="does not verify"):
        analyze_run(a.store)
    analysis = analyze_run(a.store, allow_unverified=True)
    assert analysis["evidence_verified"] is False
    assert analysis["publication_grade"] is False


def test_analyze_verified_run_is_publication_grade(tmp_path, fake_backend, monkeypatch):
    a = _run(tmp_path, fake_backend, monkeypatch, "pg", [0])
    analysis = analyze_run(a.store)
    assert analysis["evidence_verified"] is True
    assert analysis["publication_grade"] is True
    assert analysis["interpretation"] == "phase1.6"
