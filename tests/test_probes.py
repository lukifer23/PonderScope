from __future__ import annotations

import pytest

from ponderscope.backends.base import CaptureSpec, PrefixProbeUnsupported
from ponderscope.config.identity import DecodingPolicy
from ponderscope.reasoning.parse import parse_trace, reasoning_prefix_ids, strip_special_tokens
from ponderscope.reasoning.probes import run_prefix_probes
from ponderscope.reasoning.transitions import classify_transitions


def test_reasoning_prefix_excludes_final_answer_and_eos():
    think_end = 99
    token_ids = [100, 101, 102, think_end, 200, 201, 42]
    prefix = reasoning_prefix_ids(token_ids, think_end)
    assert prefix == [100, 101, 102]
    assert think_end not in prefix and 200 not in prefix and 42 not in prefix


def test_parse_trace_token_boundaries_and_eos_sanitization():
    think_end = 99
    token_ids = [100, 101, think_end, 200, 201, 42]
    text = "reasoning here</think>\nAnswer: 7<|im_end|>"
    parsed = parse_trace(text, token_ids, think_end_id=think_end, eos_ids={42})
    assert parsed.token_closed is True
    assert parsed.reasoning_token_ids == [100, 101]
    assert parsed.final_token_ids == [200, 201, 42]
    assert parsed.eos_observed is True
    # the decoded EOS marker must never leak into the semantic/scored answer
    assert "<|im_end|>" not in parsed.final_semantic
    assert parsed.answer_raw == "7"
    assert strip_special_tokens("<|im_end|>") == ""


def test_parse_trace_censored_has_no_final_channel():
    parsed = parse_trace("still thinking", [1, 2, 3], think_end_id=99, eos_ids={42})
    assert parsed.closed is False
    assert parsed.final_token_ids == []
    assert parsed.answer_raw is None


def test_run_prefix_probes_records_prefix_token_counts(fake_backend):
    prompt_ids = [10, 20, 30]
    prefix = [100 + i for i in range(30)]
    results = run_prefix_probes(
        fake_backend,
        family="arith",
        answer="3",
        prompt_token_ids=prompt_ids,
        prefix_token_ids=prefix,
        n_probes=3,
        probe_decoding=DecodingPolicy(mode="greedy", max_tokens=8),
        capture=CaptureSpec.minimal(),
        eos_token_ids={42},
    )
    assert len(results) >= 2
    for r in results:
        assert "reasoning_prefix_tokens" in r
        # the probe prefix never exceeds the reasoning length and never includes
        # the think-end token itself.
        assert 0 <= r["reasoning_prefix_tokens"] <= len(prefix)
        assert r["forced_close"] is True
        assert r["final_token_ids"]


class _UnsupportedBackend:
    name = "unsupported"

    def probe(self, *_args, **_kwargs):
        raise PrefixProbeUnsupported("no think-end token")


def test_run_prefix_probes_propagates_fail_closed():
    with pytest.raises(PrefixProbeUnsupported):
        run_prefix_probes(
            _UnsupportedBackend(),
            family="arith",
            answer="1",
            prompt_token_ids=[1, 2],
            prefix_token_ids=[1, 2, 3],
            n_probes=2,
            probe_decoding=DecodingPolicy(mode="greedy", max_tokens=4),
            capture=CaptureSpec.minimal(),
        )


def test_stable_sufficient_prefix_tokens():
    lengths = [0, 10, 20, 30]
    # correct at 10 and stays correct through an observed-correct natural final
    state = classify_transitions([False, True, True, True], "correct", prefix_token_lengths=lengths)
    assert state.first_correct_probe_index == 1
    assert state.first_correct_prefix_tokens == 10
    assert state.stable_sufficient_with_natural_final_tokens == 10
    # a late regression, with an observed wrong final, is not stable-sufficient
    state2 = classify_transitions(
        [False, True, True, False], "incorrect", prefix_token_lengths=lengths
    )
    assert state2.stable_sufficient_with_natural_final_tokens is None
    assert state2.harmful_overthinking_observed is True
    # the same probe pattern with a censored final makes no natural claim
    state3 = classify_transitions(
        [False, True, True, False], "censored", prefix_token_lengths=lengths
    )
    assert state3.stable_sufficient_with_natural_final_tokens is None
    assert state3.harmful_overthinking_observed is False
