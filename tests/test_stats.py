from __future__ import annotations

from ponderscope.analysis.stats import (
    bootstrap_ci,
    classify_effect,
    paired_bootstrap_delta,
    summarize,
)


def test_summarize_basic():
    s = summarize([1.0, 2.0, 3.0, 4.0])
    assert s["n"] == 4
    assert abs(s["mean"] - 2.5) < 1e-9
    assert s["min"] == 1.0 and s["max"] == 4.0


def test_summarize_empty():
    assert summarize([])["n"] == 0
    assert summarize([])["mean"] is None


def test_bootstrap_ci_brackets_mean():
    values = [float(i) for i in range(50)]
    lo, hi = bootstrap_ci(values, n_resamples=500, seed=1)
    assert lo <= sum(values) / len(values) <= hi


def test_paired_delta_identical_is_zero():
    a = [1.0, 2.0, 3.0]
    d = paired_bootstrap_delta(a, a)
    assert d["mean"] == 0.0
    assert d["excludes_zero"] is False


def test_paired_delta_positive():
    a = [10.0, 11.0, 12.0, 13.0]
    b = [1.0, 2.0, 3.0, 4.0]
    d = paired_bootstrap_delta(a, b, seed=0)
    assert d["mean"] == 9.0
    assert d["excludes_zero"] is True


def test_paired_delta_length_mismatch():
    import pytest

    with pytest.raises(ValueError):
        paired_bootstrap_delta([1.0], [1.0, 2.0])


def test_classify_effect_below_noise():
    delta = {"mean": 0.01, "lo": -0.1, "hi": 0.1, "excludes_zero": False}
    assert classify_effect(delta, 0.5) == "below_noise"


def test_classify_effect_clearly_larger():
    delta = {"mean": 5.0, "lo": 1.0, "hi": 9.0, "excludes_zero": True}
    assert classify_effect(delta, 1.0) == "clearly_larger"


def test_classify_effect_comparable():
    delta = {"mean": 1.2, "lo": 0.1, "hi": 2.3, "excludes_zero": True}
    assert classify_effect(delta, 1.0) == "comparable"
