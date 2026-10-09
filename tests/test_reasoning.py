from __future__ import annotations

from ponderscope.reasoning.metrics import compute_repetition
from ponderscope.reasoning.parse import extract_answer, parse_reasoning, split_channels
from ponderscope.reasoning.transitions import (
    classify_transitions,
    natural_final_status_from_termination,
    prefix_lengths,
)


def test_split_channels_closed():
    reasoning, final, closed = split_channels("abc</think>\nAnswer: 5", True)
    assert reasoning == "abc"
    assert "Answer: 5" in final
    assert closed


def test_split_channels_censored():
    reasoning, final, closed = split_channels("still thinking", False)
    assert reasoning == "still thinking"
    assert final == ""
    assert not closed


def test_extract_answer_after_cue():
    assert extract_answer("\nblah\nAnswer: 42") == "42"
    assert extract_answer("no cue\nlast line") == "last line"


def test_parse_reasoning_censored_has_no_answer():
    parsed = parse_reasoning("thinking without end", False)
    assert parsed.answer_raw is None
    assert not parsed.closed


def test_parse_reasoning_closed():
    parsed = parse_reasoning("step 1\n</think>\nAnswer: 7", True)
    assert parsed.closed
    assert parsed.answer_raw == "7"


def test_repetition_metrics_detect_loop():
    looped = list(range(10)) * 5
    varied = list(range(50))
    rep_loop = compute_repetition(looped, "x " * 50)
    rep_var = compute_repetition(varied, " ".join(str(i) for i in range(50)))
    assert rep_loop.unique_token_ratio < rep_var.unique_token_ratio
    assert rep_loop.distinct_2 < rep_var.distinct_2
    assert rep_loop.max_token_run == 1


def test_repetition_metrics_constant_run():
    rep = compute_repetition([5] * 20, "a")
    assert rep.max_token_run == 20
    assert rep.unique_token_ratio == 1 / 20


def test_natural_final_status_categories():
    censored = natural_final_status_from_termination(
        correct=False,
        answer_scorable=False,
        think_end_reached=False,
        capped=True,
        finish_reason="length",
    )
    assert censored == "censored"
    assert (
        natural_final_status_from_termination(
            correct=True,
            answer_scorable=True,
            think_end_reached=True,
            capped=False,
            finish_reason="stop",
        )
        == "correct"
    )
    assert (
        natural_final_status_from_termination(
            correct=False,
            answer_scorable=True,
            think_end_reached=True,
            capped=False,
            finish_reason="stop",
        )
        == "incorrect"
    )
    assert (
        natural_final_status_from_termination(
            correct=False,
            answer_scorable=False,
            think_end_reached=True,
            capped=False,
            finish_reason="stop",
        )
        == "unparseable"
    )
    assert (
        natural_final_status_from_termination(
            correct=False,
            answer_scorable=False,
            think_end_reached=False,
            capped=False,
            finish_reason="error",
            error_type="RuntimeError",
        )
        == "error"
    )


def test_classify_stable_correct_observed():
    state = classify_transitions([True, True, True], "correct")
    assert state.prefix_state == "stable_correct"
    assert state.natural_final_correct is True
    assert state.stable_sufficient_with_natural_final_tokens is not None
    assert state.observed_probe_stable_from_tokens is not None


def test_classify_wrong_to_correct():
    state = classify_transitions([False, False, True], "correct")
    assert state.prefix_state == "wrong_to_correct"
    assert state.prefix_flips == 1
    assert state.harmful_overthinking_observed is False


def test_classify_correct_to_wrong_requires_observed_final():
    state = classify_transitions([True, True, False], "incorrect")
    assert state.prefix_state == "correct_to_wrong"
    assert state.harmful_overthinking_observed is True
    assert state.stable_sufficient_with_natural_final_tokens is None


def test_classify_multiple_flips_observed():
    state = classify_transitions([True, False, True], "incorrect")
    assert state.prefix_state == "multiple_flips"
    assert state.harmful_overthinking_observed is True


def test_classify_never_correct_observed():
    assert classify_transitions([False, False, False], "incorrect").prefix_state == "never_correct"


def test_censored_final_does_not_create_correctness_flip():
    # The exact saved-smoke pattern: [False, False, True, True, True] with a
    # capped (censored) natural final that produced no answer.
    state = classify_transitions(
        [False, False, True, True, True], "censored", prefix_token_lengths=[0, 16, 224, 432, 640]
    )
    assert state.prefix_state == "wrong_to_correct"
    assert state.prefix_state != "multiple_flips"
    assert state.prefix_state != "correct_to_wrong"
    assert state.prefix_flips == 1
    assert state.natural_final_status == "censored"
    assert state.natural_final_correct is None
    assert state.harmful_overthinking_observed is False
    assert state.stable_sufficient_with_natural_final_tokens is None
    # but the prefix itself did become stably correct over the observed horizon
    assert state.observed_probe_stable_from_tokens is not None


def test_censored_final_cannot_be_harmful_overthinking():
    state = classify_transitions([True, True, True], "censored")
    assert state.harmful_overthinking_observed is False
    assert state.stable_sufficient_with_natural_final_tokens is None
    assert state.observed_probe_stable_from_tokens is not None


def test_observed_stability_separated_from_natural_sufficiency():
    state = classify_transitions([False, True, True], "censored", prefix_token_lengths=[0, 10, 20])
    assert state.observed_probe_stable_from_tokens == 10
    assert state.stable_sufficient_with_natural_final_tokens is None
    state2 = classify_transitions([False, True, True], "correct", prefix_token_lengths=[0, 10, 20])
    assert state2.stable_sufficient_with_natural_final_tokens == 10


def test_prefix_lengths_monotonic():
    lengths = prefix_lengths(100, 4)
    assert lengths[0] == 0
    assert lengths == sorted(lengths)
    assert len(set(lengths)) == len(lengths)
    assert all(0 <= x <= 100 for x in lengths)
    assert prefix_lengths(10, 4) == [0]
