from __future__ import annotations

from ponderscope.reasoning.metrics import compute_repetition
from ponderscope.reasoning.parse import extract_answer, parse_reasoning, split_channels
from ponderscope.reasoning.transitions import classify_transitions, prefix_lengths


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


def test_classify_stable_correct():
    state = classify_transitions(True, [True, True, True])
    assert state.primary == "stable_correct"
    assert state.flags["stable_correct"]


def test_classify_wrong_to_correct():
    state = classify_transitions(True, [False, False, True])
    assert state.primary == "wrong_to_correct"
    assert state.flips == 1


def test_classify_correct_to_wrong():
    state = classify_transitions(False, [True, True, False])
    assert state.primary == "correct_to_wrong"
    assert state.flags["correct_to_wrong"]


def test_classify_multiple_flips():
    state = classify_transitions(False, [True, False, True])
    assert state.primary == "multiple_flips"
    assert state.flags["multiple_flips"]


def test_classify_never_correct():
    assert classify_transitions(False, [False, False, False]).primary == "never_correct"


def test_prefix_lengths_monotonic():
    lengths = prefix_lengths(100, 4)
    assert lengths[0] == 0
    assert lengths == sorted(lengths)
    assert len(set(lengths)) == len(lengths)
    assert all(0 <= x <= 99 for x in lengths)
    assert prefix_lengths(10, 4) == [0]
