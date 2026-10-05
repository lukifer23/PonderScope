from __future__ import annotations

import numpy as np
import pytest

from ponderscope.backends.distribution import top_k_from_logprobs


def test_top_k_returns_actual_top_values():
    logprobs = np.array([0.1, 0.9, 0.5, 0.2, 0.8], dtype=np.float32)
    ids, vals = top_k_from_logprobs(logprobs, 2)
    # The old buggy implementation used argpartition(...)[-k:], which returns the
    # *smallest* logprobs (and ids [3, 0]). The correct answer is the two largest.
    assert ids == [1, 4]
    assert vals == [pytest.approx(0.9), pytest.approx(0.8)]


def test_top_k_descending_and_bounded():
    rng = np.random.default_rng(0)
    logprobs = rng.normal(size=50).astype(np.float32)
    ids, vals = top_k_from_logprobs(logprobs, 5)
    assert len(ids) == 5
    assert vals == sorted(vals, reverse=True)
    # every returned value really is >= every non-returned value
    returned = set(ids)
    best_other = max(v for i, v in enumerate(logprobs) if i not in returned)
    assert min(vals) >= best_other


def test_top_k_handles_k_larger_than_vocab_and_zero():
    logprobs = np.array([0.3, 0.1, 0.2], dtype=np.float32)
    ids, vals = top_k_from_logprobs(logprobs, 10)
    assert set(ids) == {0, 1, 2}
    assert top_k_from_logprobs(logprobs, 0) == ([], [])
