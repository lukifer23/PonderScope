"""Phase 1.4 tests: corrected time-to-closure endpoint, a table-driven
termination-state matrix, distinct no-closure termination, and analytically
solvable survival fixtures.

These fixtures are mathematical validation cases, never model measurements.
"""

from __future__ import annotations

import pytest

from ponderscope.analysis.analyze import _time_to_closure
from ponderscope.analysis.survival import kaplan_meier, rmst, survival_summary
from ponderscope.reasoning.transitions import (
    NATURAL_FINAL_STATUSES,
    natural_final_correctness,
    natural_final_status_from_termination,
)

# --------------------------------------------------------------------------- #
# Termination-state matrix (Phase 2.1)
# --------------------------------------------------------------------------- #
# (correct, answer_observed, think_end_reached, capped, finish_reason, error_type)
TERMINATION_CASES = [
    # An observed parseable answer wins regardless of stop cause.
    ((True, True, True, False, "stop", None), "correct"),
    ((False, True, True, False, "stop", None), "incorrect"),
    ((True, True, False, False, "stop", None), "correct"),
    # Horizon reached without a native close -> censored, never "wrong".
    ((False, False, False, True, "length", None), "censored"),
    ((False, False, False, False, "length", None), "censored"),
    # Native close observed but no answer parsed -> unparseable.
    ((False, False, True, True, "length", None), "unparseable"),
    ((False, False, True, False, "stop", None), "unparseable"),
    # EOS without a native close -> a distinct no-closure termination.
    ((False, False, False, False, "stop", None), "terminated_no_closure"),
    # Non-horizon, non-EOS termination with no answer -> unparseable.
    ((False, False, False, False, "other", None), "unparseable"),
    # Backend failures.
    ((False, False, False, False, "error", None), "error"),
    ((False, False, False, False, "stop", "ValueError"), "error"),
]


@pytest.mark.parametrize("args,expected", TERMINATION_CASES)
def test_termination_state_matrix(args, expected):
    correct, answer_observed, think_end, capped, finish_reason, error_type = args
    status = natural_final_status_from_termination(
        correct=correct,
        answer_observed=answer_observed,
        think_end_reached=think_end,
        capped=capped,
        finish_reason=finish_reason,
        error_type=error_type,
    )
    assert status == expected
    assert status in NATURAL_FINAL_STATUSES


def test_no_closure_is_distinct_and_has_no_correctness():
    assert "terminated_no_closure" in NATURAL_FINAL_STATUSES
    assert natural_final_correctness("terminated_no_closure") is None
    assert natural_final_correctness("censored") is None
    # An EOS-without-close stop is NOT evidence of a wrong answer.
    assert natural_final_correctness("correct") is True
    assert natural_final_correctness("incorrect") is False


# --------------------------------------------------------------------------- #
# Corrected time-to-closure endpoint (Phase 2.4 / F3)
# --------------------------------------------------------------------------- #
def _rec(*, reasoning_tokens: int, total_tokens: int, think_end: bool) -> dict:
    return {
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": total_tokens,
        "termination": {"think_end_reached": think_end},
    }


def test_time_to_closure_is_think_end_index_not_total_tokens():
    # Closure at the native think-end: time is reasoning_tokens + 1, and the
    # post-closure final-answer channel must NOT be counted.
    closed = _rec(reasoning_tokens=100, total_tokens=140, think_end=True)
    assert _time_to_closure(closed) == 101.0
    # Censored record is observed at the horizon.
    censored = _rec(reasoning_tokens=2048, total_tokens=2048, think_end=False)
    assert _time_to_closure(censored) == 2048.0


def test_time_to_closure_ignores_answer_length():
    a = _rec(reasoning_tokens=500, total_tokens=510, think_end=True)
    b = _rec(reasoning_tokens=500, total_tokens=900, think_end=True)
    assert _time_to_closure(a) == _time_to_closure(b) == 501.0


# --------------------------------------------------------------------------- #
# Survival analytic fixtures (Phase 2.4, F7)
# --------------------------------------------------------------------------- #
def test_km_all_events_known_survival_and_median():
    km = kaplan_meier([1, 2, 3, 4], [True, True, True, True])
    assert km["n_events"] == 4 and km["n_censored"] == 0
    assert km["survival"] == pytest.approx([0.75, 0.5, 0.25, 0.0])
    assert km["median"] == 2
    # RMST up to 4 equals the mean of the event times here.
    assert rmst([1, 2, 3, 4], [True, True, True, True], tau=4) == pytest.approx(2.5)


def test_km_all_censored_no_event():
    km = kaplan_meier([5, 5], [False, False])
    assert km["n_events"] == 0 and km["n_censored"] == 2
    assert km["survival"] == pytest.approx([1.0])
    assert km["median"] is None
    assert rmst([5, 5], [False, False], tau=5) == pytest.approx(5.0)


def test_km_mixed_event_and_censoring():
    km = kaplan_meier([2, 4], [True, False])
    assert km["survival"] == pytest.approx([0.5, 0.5])
    assert km["median"] == 2
    assert rmst([2, 4], [True, False], tau=4) == pytest.approx(3.0)


def test_km_tied_event_and_censor_at_same_time():
    km = kaplan_meier([3, 3], [True, False])
    assert km["n_events"] == 1 and km["n_censored"] == 1
    assert km["survival"] == pytest.approx([0.5])
    assert km["median"] == 3
    assert rmst([3, 3], [True, False], tau=3) == pytest.approx(3.0)


def test_rmst_horizon_is_not_a_natural_threshold():
    times = [1.0, 10.0]
    events = [True, True]
    # A shorter observation horizon changes RMST; it must never be silently
    # treated as equivalent to a longer one.
    assert rmst(times, events, tau=2) == pytest.approx(1.0 + 0.5 * 1.0)
    assert rmst(times, events, tau=10) != rmst(times, events, tau=2)


def test_rmst_none_for_nonpositive_tau_and_empty():
    assert rmst([1, 2], [True, True], tau=0) is None
    assert rmst([], [], tau=10) is None


def test_survival_summary_reports_counts():
    s = survival_summary([10, 20, 30], [True, False, True], tau=30)
    assert s["n_events"] == 2 and s["n_censored"] == 1
    assert s["n_at_risk_initial"] == 3


# --------------------------------------------------------------------------- #
# Sampler fidelity (Phase 2.2)
# --------------------------------------------------------------------------- #
def _policy(**kw):
    from ponderscope.config.identity import DecodingPolicy

    base = {"mode": "sampled", "max_tokens": 16, "temperature": 1.0, "top_p": 0.95, "top_k": 20}
    base.update(kw)
    return DecodingPolicy(**base)


def test_greedy_uses_no_sampler_and_sampled_requires_seed():
    from ponderscope.backends.mlx_backend import MlxBackend
    from ponderscope.config.identity import DecodingPolicy

    backend = MlxBackend()
    sampler, seed = backend._prepare_sampler(DecodingPolicy(mode="greedy", max_tokens=8))
    assert sampler is None and seed is None
    with pytest.raises(ValueError):
        backend._prepare_sampler(_policy(seed=None))
    sampler, seed = backend._prepare_sampler(_policy(seed=3))
    assert callable(sampler) and seed == 3


def test_inactive_penalties_build_no_processors():
    from ponderscope.backends.mlx_backend import MlxBackend

    backend = MlxBackend()
    processors, meta = backend._prepare_logits_processors(
        _policy(seed=0, presence_penalty=0.0, repetition_penalty=1.0, frequency_penalty=0.0),
        prompt_len=4,
    )
    assert processors == []
    assert meta["active_processors"] == []
    assert meta["n_processors"] == 0


def test_active_generated_history_presence_builds_one_processor():
    from ponderscope.backends.mlx_backend import MlxBackend

    backend = MlxBackend()
    processors, meta = backend._prepare_logits_processors(
        _policy(seed=0, presence_penalty=1.5, presence_scope="generated_history"),
        prompt_len=7,
    )
    assert len(processors) == 1
    assert meta["presence_semantics"] == "generated_history"
    assert meta["equivalence_label"] == "qwen-generated-history-presence-v1"
    assert meta["base_prompt_tokens"] == 7
