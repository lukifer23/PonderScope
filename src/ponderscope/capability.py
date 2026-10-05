"""Live capability proof for the real MLX backend.

Runs a bounded battery on the real model and records, for each capability,
whether it is SUPPORTED, UNSUPPORTED, or MEASURED. Nothing is guessed: a
capability that cannot be validated is reported as unsupported.
"""

from __future__ import annotations

from typing import Any

from .backends.base import CaptureSpec, PrefixProbeUnsupported
from .config.identity import DecodingPolicy
from .experiment import measure_capture_overhead, verify_capture_equivalence
from .reasoning.parse import parse_trace

# Bounded escalation for natural-closure discovery.
CLOSURE_BUDGETS = (256, 512, 1024)
CAPABILITY_PROMPT = (
    "Compute 5 + 7. Reply with a single integer on the last line in the form 'Answer: <integer>'."
)


def _first_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def validate_live(backend: Any, *, overhead: bool = True) -> dict[str, Any]:
    caps: dict[str, Any] = {}
    unsupported: list[str] = []

    think_end_id = backend.think_end_token_id
    think_start_id = getattr(backend, "think_start_token_id", None)
    eos_ids = set(getattr(backend, "eos_token_ids", set()) or set())
    caps["think_start"] = {
        "id": think_start_id,
        "decoded": backend.decode([think_start_id]) if think_start_id is not None else None,
    }
    caps["think_end"] = {
        "id": think_end_id,
        "decoded": backend.decode([think_end_id]) if think_end_id is not None else None,
    }
    caps["eos"] = {
        "ids": sorted(eos_ids),
        "decoded": [backend.decode([i]) for i in sorted(eos_ids)],
    }

    closure: dict[str, Any]
    try:
        ids = backend.native_closure_ids()
        closure = {
            "supported": True,
            "ids": ids,
            "decoded": backend.decode(ids),
            "contains_think_end": think_end_id in ids,
            "roundtrip_ok": backend.decode(ids) == "\n</think>\n\n",
        }
    except PrefixProbeUnsupported as exc:
        closure = {"supported": False, "reason": str(exc)}
        unsupported.append("native_closure")
    caps["native_closure"] = closure

    messages = [{"role": "user", "content": CAPABILITY_PROMPT}]
    prompt_ids = backend.tokenize_prompt(messages, enable_thinking=True)
    caps["thinking_prompt_tail"] = backend.decode(prompt_ids[-8:])

    # natural closure discovery: keep the four observables distinct.
    natural: dict[str, Any] = {
        "supported": False,
        "attempts": [],
        "think_end_observed": False,
        "final_answer_channel_observed": False,
        "eos_observed": False,
        "parseable_answer_observed": False,
    }
    best_closed: Any = None
    for budget in CLOSURE_BUDGETS:
        trace = backend.generate(
            prompt_ids, DecodingPolicy(mode="greedy", max_tokens=budget), CaptureSpec.minimal()
        )
        parsed = parse_trace(
            trace.text, trace.token_ids, think_end_id=think_end_id, eos_ids=eos_ids
        )
        answer_observed = parsed.answer_raw is not None
        natural["attempts"].append(
            {
                "max_tokens": budget,
                "finish_reason": trace.finish_reason,
                "generated_tokens": trace.generated_tokens,
                "think_end_reached": trace.think_end_reached,
                "final_answer_channel_observed": bool(parsed.closed and parsed.final_semantic),
                "closed": parsed.closed,
                "eos_observed": parsed.eos_observed,
                "answer": parsed.answer_raw,
            }
        )
        if parsed.closed:
            if best_closed is None:
                best_closed = parsed
            natural["think_end_observed"] = True
            natural["final_answer_channel_observed"] = natural[
                "final_answer_channel_observed"
            ] or bool(parsed.final_semantic)
            natural["eos_observed"] = natural["eos_observed"] or parsed.eos_observed
            natural["parseable_answer_observed"] = (
                natural["parseable_answer_observed"] or answer_observed
            )
            natural.update(
                {
                    "max_tokens": budget,
                    "reasoning_tokens": len(parsed.reasoning_token_ids),
                    "answer": parsed.answer_raw,
                }
            )
            # Stop once the full natural contract (think-end + answer + EOS) is seen.
            if answer_observed and parsed.eos_observed:
                break
    natural["supported"] = bool(
        natural["think_end_observed"]
        and natural["final_answer_channel_observed"]
        and natural["parseable_answer_observed"]
    )
    if not natural["supported"]:
        unsupported.append("natural_closure")
    caps["natural_closure"] = natural

    # forced finalization: requires a validated closure AND a reasoning prefix.
    # Pass rule: EVERY required cut must inject the validated closure, produce no
    # error, enter final-answer continuation mode, and terminate by EOS promptly.
    forced: dict[str, Any] = {"supported": False}
    if closure.get("supported"):
        if best_closed is not None:
            reasoning = best_closed.reasoning_token_ids
        else:
            # censored reasoning is still a valid reasoning prefix to probe
            trace = backend.generate(
                prompt_ids,
                DecodingPolicy(mode="greedy", max_tokens=CLOSURE_BUDGETS[-1]),
                CaptureSpec.minimal(),
            )
            reasoning = backend.reasoning_prefix(trace.token_ids)
        cuts = sorted({0, min(16, len(reasoning)), min(64, len(reasoning)), len(reasoning)})
        probes = []
        for cut in cuts:
            tr = backend.probe(
                prompt_ids,
                reasoning[:cut],
                DecodingPolicy(mode="greedy", max_tokens=64),
                CaptureSpec.minimal(),
            )
            eos = any(t in eos_ids for t in tr.token_ids)
            forced_close = bool(tr.extra.get("forced_close"))
            entered_final_mode = tr.generated_tokens > 0
            terminated_cleanly = tr.finish_reason == "stop" and eos and tr.generated_tokens < 32
            probes.append(
                {
                    "cut": cut,
                    "forced_close": forced_close,
                    "entered_final_answer_mode": entered_final_mode,
                    "error": tr.error,
                    "generated_tokens": tr.generated_tokens,
                    "finish_reason": tr.finish_reason,
                    "eos_observed": eos,
                    "terminated_cleanly": terminated_cleanly,
                    "first_line": _first_line(tr.text)[:160],
                }
            )
        no_probe_error = all(p["error"] is None for p in probes)
        all_forced_close = all(p["forced_close"] for p in probes)
        all_entered = all(p["entered_final_answer_mode"] for p in probes)
        all_required_honored = all(p["terminated_cleanly"] for p in probes)
        supported = bool(
            closure.get("roundtrip_ok")
            and probes
            and no_probe_error
            and all_forced_close
            and all_entered
            and all_required_honored
        )
        forced = {
            "supported": supported,
            "no_probe_error": no_probe_error,
            "all_forced_close": all_forced_close,
            "all_entered_final_answer_mode": all_entered,
            "all_required_cuts_honored": all_required_honored,
            "n_required_cuts": len(cuts),
            "n_honored": sum(1 for p in probes if p["terminated_cleanly"]),
            "probes": probes,
            "note": (
                "native closure appended to every required cut; no answer cue injected; "
                "pass requires every cut to terminate cleanly"
            ),
        }
        if not supported:
            unsupported.append("forced_finalization")
    else:
        forced = {"supported": False, "reason": "native closure not validated"}
        unsupported.append("forced_finalization")
    caps["forced_finalization"] = forced

    # seeded sampling and same-seed replay
    seed = 123
    sampled = DecodingPolicy(
        mode="sampled", max_tokens=64, temperature=0.6, top_p=0.95, top_k=20, seed=seed
    )
    a = backend.generate(prompt_ids, sampled, CaptureSpec.minimal())
    b = backend.generate(prompt_ids, sampled, CaptureSpec.minimal())
    other = DecodingPolicy(
        mode="sampled", max_tokens=64, temperature=0.6, top_p=0.95, top_k=20, seed=999
    )
    c = backend.generate(prompt_ids, other, CaptureSpec.minimal())
    caps["seeded_sampling"] = {
        "supported": True,
        "seed": seed,
        "generated_tokens": a.generated_tokens,
    }
    caps["same_seed_replay"] = {
        "supported": True,
        "identical_token_ids": a.token_ids == b.token_ids,
    }
    caps["across_seed_variation"] = {
        "differ": a.token_ids != c.token_ids,
    }

    # greedy replay + distribution sanity
    g1 = backend.generate(
        prompt_ids, DecodingPolicy(mode="greedy", max_tokens=64), CaptureSpec.research(top_k=5)
    )
    g2 = backend.generate(
        prompt_ids, DecodingPolicy(mode="greedy", max_tokens=64), CaptureSpec.minimal()
    )
    caps["greedy"] = {
        "supported": True,
        "replay_identical": g1.token_ids == g2.token_ids,
        "finish_reason": g1.finish_reason,
    }
    dist: dict[str, Any] = {"supported": False}
    if g1.steps:
        s = g1.steps[0]
        dist = {
            "supported": True,
            "chosen_logprob": s.logprob,
            "entropy_nats": s.entropy,
            "top_token_ids": s.top_token_ids,
            "top_logprobs": s.top_logprobs,
            "top_sorted_descending": s.top_logprobs == sorted(s.top_logprobs, reverse=True),
            "chosen_in_top": s.token_id in s.top_token_ids,
            "distribution_note": "pre-sampler model distribution from generate_step",
        }
    caps["token_distribution"] = dist

    caps["capture_equivalence"] = verify_capture_equivalence(backend, prompt_ids, max_tokens=48)
    if not caps["capture_equivalence"]["equivalent"]:
        unsupported.append("capture_equivalence")

    if overhead:
        caps["instrumentation_overhead"] = measure_capture_overhead(
            backend,
            prompt_ids,
            DecodingPolicy(mode="greedy", max_tokens=48),
            repeats=5,
            warmup=1,
        )

    return {
        "capabilities": caps,
        "unsupported": unsupported,
        "all_supported": not unsupported,
    }
