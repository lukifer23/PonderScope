"""Phase 1.7 tests: corrected answer/censoring outcome classification and the
offline paired-outcome analysis.
"""

from __future__ import annotations

import pytest

from ponderscope.analysis.analyze import _category_counts, natural_final_status_of
from ponderscope.analysis.paired import paired_outcomes, repetition_analysis
from ponderscope.config.schema import ExperimentSpec
from ponderscope.experiment import run_experiment
from ponderscope.reasoning.transitions import (
    budget_exhausted_after_closure,
    natural_final_status_from_termination,
)

CLEAN = {"version": "test", "git_sha": "0" * 40, "tracked_dirty": False}


# --------------------------------------------------------------------------- #
# Outcome classification matrix (Stage 1)
# --------------------------------------------------------------------------- #
# (correct, scorable, raw_present, think_end, capped, finish, expected)
CASES = [
    (True, True, True, True, False, "stop", "correct"),  # EOS correct
    (False, True, True, True, False, "stop", "incorrect"),  # EOS scorable wrong
    (False, False, True, True, False, "stop", "unparseable"),  # EOS malformed
    (True, True, True, True, False, "stop", "correct"),  # close, no EOS
    (False, False, False, False, True, "length", "censored"),  # capped inside reasoning
    (False, False, True, True, True, "length", "unparseable"),  # capped after close, partial
    (False, True, True, True, True, "length", "incorrect"),  # capped after close, scorable
    (False, False, False, False, False, "stop", "terminated_no_closure"),  # EOS no close
    (False, False, False, False, False, "error", "error"),  # backend error
    (False, False, True, False, True, "length", "unparseable"),  # malformed capped
]


@pytest.mark.parametrize("case", CASES)
def test_outcome_matrix(case):
    correct, scorable, raw, think_end, capped, finish, expected = case
    status = natural_final_status_from_termination(
        correct=correct,
        answer_scorable=scorable,
        raw_answer_present=raw,
        think_end_reached=think_end,
        capped=capped,
        finish_reason=finish,
    )
    assert status == expected


def test_raw_but_unscorable_is_never_incorrect():
    status = natural_final_status_from_termination(
        correct=False,
        answer_scorable=False,
        raw_answer_present=True,
        think_end_reached=True,
        capped=False,
        finish_reason="stop",
    )
    assert status == "unparseable"


def test_capped_after_closure_is_distinct():
    closed_then_capped = {"termination": {"think_end_reached": True, "capped": True}}
    capped_in_reasoning = {"termination": {"think_end_reached": False, "capped": True}}
    assert budget_exhausted_after_closure(closed_then_capped) is True
    assert budget_exhausted_after_closure(capped_in_reasoning) is False


def test_status_of_recomputes_from_record_not_stored_field():
    # A record saved under Phase 1.6 with a raw-but-unscorable answer stored as
    # "incorrect" must be recomputed as "unparseable".
    record = {
        "correct": False,
        "answer_normalized": None,
        "natural_final_status": "incorrect",  # stale stored value
        "termination": {
            "think_end_reached": True,
            "terminated_by_eos": False,
            "capped": False,
            "missing_answer": False,  # raw text existed
        },
        "trace": {"error": None},
    }
    assert natural_final_status_of(record) == "unparseable"


def test_category_counts_are_non_overlapping():
    def rec(status, correct, scorable, think_end, eos, capped, raw):
        return {
            "correct": correct,
            "answer_normalized": "1" if scorable else None,
            "natural_final_status": status,
            "termination": {
                "think_end_reached": think_end,
                "terminated_by_eos": eos,
                "capped": capped,
                "missing_answer": not raw,
            },
            "trace": {"error": None},
        }

    records = [
        rec("correct", True, True, True, True, False, True),
        rec("incorrect", False, True, True, True, False, True),
        rec("unparseable", False, False, True, False, True, True),
        rec("censored", False, False, False, False, True, False),
    ]
    cats = _category_counts(records)
    assert cats["n"] == 4
    assert cats["native_reasoning_closures"] == 3
    assert cats["eos_terminations"] == 2
    assert cats["censored"] == 1
    assert cats["unparseable"] == 1
    assert cats["scorable_incorrect"] == 1
    assert cats["correct_answers"] == 1
    assert cats["observed_answers"] == 2
    assert cats["budget_exhausted_after_closure"] == 1
    assert cats["raw_answer_present"] == 3


# --------------------------------------------------------------------------- #
# Offline paired-outcome analysis (Stage 2)
# --------------------------------------------------------------------------- #
def _run(tmp_path, fake_backend, monkeypatch, name, seeds=(0, 1)):
    monkeypatch.setattr("ponderscope.experiment.get_backend", lambda _n: fake_backend)
    spec = ExperimentSpec(
        name=name,
        task_pack="tasks-v1",
        families=["arith"],
        n_per_family=2,
        greedy_repeats=0,
        sampled_seeds=list(seeds),
        max_tokens=16,
    )
    return run_experiment(
        spec,
        model_repo="fake/model",
        model_revision="deadbeef",
        runs_dir=tmp_path,
        code_state=CLEAN,
    )


def test_paired_outcomes_identical_runs(tmp_path, fake_backend, monkeypatch):
    a = _run(tmp_path, fake_backend, monkeypatch, "a")
    b = _run(tmp_path, fake_backend, monkeypatch, "b")
    result = paired_outcomes(a.store, b.store, mode="sampled")
    assert result["n_matched_draws"] == 4  # 2 tasks x 2 seeds
    assert result["closure_transitions"]["both"] == 4
    assert sum(result["correctness_transitions"].values()) == 4
    assert all(r["closure_transition"] == "both" for r in result["draws"])


def test_repetition_analysis_orientation(tmp_path, fake_backend, monkeypatch):
    a = _run(tmp_path, fake_backend, monkeypatch, "a")
    b = _run(tmp_path, fake_backend, monkeypatch, "b")
    rep = repetition_analysis(a.store, b.store, mode="sampled")
    # identical runs -> zero delta
    assert rep["overall_mean_delta"] == 0.0
    assert rep["n_paired_draws"] == 4


# --------------------------------------------------------------------------- #
# Stage 4: fixed prefix checkpoints and population-lock design independence
# --------------------------------------------------------------------------- #
from ponderscope.backends.base import CaptureSpec  # noqa: E402
from ponderscope.config.identity import DecodingPolicy  # noqa: E402
from ponderscope.population import build_lock, verify_lock  # noqa: E402
from ponderscope.reasoning.probes import run_prefix_probes  # noqa: E402
from ponderscope.reasoning.transitions import prefix_lengths  # noqa: E402


def test_prefix_lengths_fixed_checkpoints():
    assert prefix_lengths(1000, 4, checkpoints=[256, 512, 1024, 1536]) == [0, 256, 512, 1000]
    # trajectory shorter than the checkpoints: only 0 and the full prefix
    assert prefix_lengths(100, 4, checkpoints=[256, 512]) == [0, 100]


def test_probe_checkpoints_spec_roundtrip():
    spec = ExperimentSpec.from_dict(
        {
            "name": "p",
            "task_pack": "tasks-v1",
            "probe": True,
            "probe_checkpoints": [256, 512],
        }
    )
    assert spec.probe_checkpoints == [256, 512]
    assert spec.to_dict()["probe_checkpoints"] == [256, 512]


def test_run_prefix_probes_uses_checkpoints(fake_backend):
    prompt_ids = [10, 20, 30]
    prefix = list(range(1000))
    results = run_prefix_probes(
        fake_backend,
        family="arith",
        answer="3",
        prompt_token_ids=prompt_ids,
        prefix_token_ids=prefix,
        n_probes=4,
        probe_decoding=DecodingPolicy(mode="greedy", max_tokens=8),
        capture=CaptureSpec.minimal(),
        eos_token_ids={42},
        checkpoints=[256, 512, 1024, 1536],
    )
    assert [r["reasoning_prefix_tokens"] for r in results] == [0, 256, 512, 1000]


def test_population_lock_is_design_independent(tmp_path):
    base = ExperimentSpec(
        name="base",
        task_pack="tasks-v1",
        families=["arith", "logic"],
        n_per_family=2,
        split="dev",
        task_seed=0,
        greedy_repeats=0,
        sampled_seeds=[0, 1],
    )
    lock = build_lock(base)
    probe = ExperimentSpec(
        name="probe",
        task_pack="tasks-v1",
        families=["arith", "logic"],
        n_per_family=2,
        split="dev",
        task_seed=0,
        greedy_repeats=1,  # different sampling design
        sampled_seeds=[],
        probe=True,
        probe_checkpoints=[256, 512],
    )
    # Same frozen population, different design -> lock still verifies.
    assert verify_lock(probe, lock)["ok"] is True
