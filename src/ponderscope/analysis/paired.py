"""Offline paired-outcome and stratification analysis.

Reads immutable evidence only (never re-runs a model) and produces reproducible,
task-clustered paired contingency tables that separate **closure** changes from
**correctness** changes, plus family/difficulty stratification and a repetition
diagnostic. Uses the same canonical draw identity as the comparison code.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from ..evidence.run import RunStore
from .analyze import natural_final_status_of
from .compare import _collapsed_draws, _key


def _outcome(record: dict[str, Any] | None) -> dict[str, Any]:
    if record is None:
        return {"available": False}
    term = record["termination"]
    return {
        "available": True,
        "closed": bool(term.get("think_end_reached")),
        "eos": bool(term.get("terminated_by_eos")),
        "capped": bool(term.get("capped")),
        "scorable": record["answer_normalized"] is not None,
        "correct": bool(record["correct"]),
        "natural_final_status": natural_final_status_of(record),
        "reasoning_tokens": record["reasoning_tokens"],
        "repeated_ngram_fraction_4": record["metrics"]["repeated_ngram_fraction_4"],
        "family": record.get("family"),
        "difficulty": (record.get("difficulty") or {}).get("level"),
    }


def _transition_label(a: dict[str, Any], b: dict[str, Any], key: str) -> str:
    """Classify a paired boolean outcome: both / only A / only B / neither."""
    av, bv = a.get(key), b.get(key)
    if av is None or bv is None:
        return "ambiguous"
    if av and bv:
        return "both"
    if av and not bv:
        return "only_a"
    if bv and not av:
        return "only_b"
    return "neither"


def paired_outcomes(
    run_a: RunStore,
    run_b: RunStore,
    *,
    mode: str = "sampled",
) -> dict[str, Any]:
    """Paired closure/correctness contingency tables over matched draws.

    ``run_a`` is the reference arm (A), ``run_b`` the comparison arm (B). All
    matched draws are retained; a draw that is ambiguous in either arm is
    labelled ``ambiguous`` rather than silently dropped, so the transition table
    is exhaustive over the declared matched population.
    """
    traces_a = [r for r in run_a.read_traces() if r["condition"]["mode"] == mode]
    traces_b = [r for r in run_b.read_traces() if r["condition"]["mode"] == mode]
    matched = {_key(r) for r in traces_a} & {_key(r) for r in traces_b}
    a = _collapsed_draws(run_a, matched, traces=traces_a)
    b = _collapsed_draws(run_b, matched, traces=traces_b)
    shared = sorted(set(a) & set(b), key=str)

    closure: Counter[str] = Counter()
    correctness: Counter[str] = Counter()
    scorable: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    per_family: dict[str, Counter] = defaultdict(Counter)
    per_difficulty: dict[str, Counter] = defaultdict(Counter)
    for draw in shared:
        oa = _outcome(a[draw]["record"])
        ob = _outcome(b[draw]["record"])
        cl = _transition_label(oa, ob, "closed")
        co = _transition_label(oa, ob, "correct")
        sc = _transition_label(oa, ob, "scorable")
        closure[cl] += 1
        correctness[co] += 1
        scorable[sc] += 1
        fam = (oa.get("family") if oa["available"] else ob.get("family")) or "?"
        diff = (oa.get("difficulty") if oa["available"] else ob.get("difficulty")) or "?"
        per_family[fam][cl] += 1
        per_family[fam][f"correct:{co}"] += 1
        per_difficulty[str(diff)][cl] += 1
        rows.append(
            {
                "draw_key": list(draw),
                "family": fam,
                "difficulty": diff,
                "closure_transition": cl,
                "correctness_transition": co,
                "scorable_transition": sc,
                "a": oa,
                "b": ob,
            }
        )
    return {
        "mode": mode,
        "unit": "stochastic draw (presentation x task x condition x seed)",
        "n_matched_draws": len(shared),
        "closure_transitions": dict(closure),
        "correctness_transitions": dict(correctness),
        "scorable_transitions": dict(scorable),
        "per_family": {f: dict(c) for f, c in sorted(per_family.items())},
        "per_difficulty": {d: dict(c) for d, c in sorted(per_difficulty.items())},
        "draws": rows,
        "note": (
            "Closure and correctness transitions are separate. A correct answer "
            "requires a scorable normalized answer; an unscorable string is never a "
            "wrong answer. Ambiguous draws are labelled, not dropped."
        ),
    }


def repetition_analysis(
    run_a: RunStore, run_b: RunStore, *, mode: str = "sampled"
) -> dict[str, Any]:
    """Exploratory stratification of the repeated-4gram difference.

    Reports the mean difference overall and within family, within closure state
    (closed vs censored), and split by reasoning-length quartile. Association
    only; not a validated loop detector.
    """
    traces_a = [r for r in run_a.read_traces() if r["condition"]["mode"] == mode]
    traces_b = [r for r in run_b.read_traces() if r["condition"]["mode"] == mode]
    matched = {_key(r) for r in traces_a} & {_key(r) for r in traces_b}
    a = _collapsed_draws(run_a, matched, traces=traces_a)
    b = _collapsed_draws(run_b, matched, traces=traces_b)
    shared = sorted(set(a) & set(b), key=str)

    def mean(xs: list[float]) -> float | None:
        return sum(xs) / len(xs) if xs else None

    overall: list[float] = []
    by_family: dict[str, list[float]] = defaultdict(list)
    by_closure: dict[str, list[float]] = defaultdict(list)
    by_length: dict[str, list[float]] = defaultdict(list)
    lengths = [
        (a[d]["record"]["reasoning_tokens"] + b[d]["record"]["reasoning_tokens"]) / 2
        for d in shared
        if a[d]["record"] and b[d]["record"]
    ]
    lengths_sorted = sorted(lengths)
    q1 = lengths_sorted[len(lengths_sorted) // 4] if lengths_sorted else 0
    q3 = lengths_sorted[(3 * len(lengths_sorted)) // 4] if lengths_sorted else 0
    for d in shared:
        ra, rb = a[d]["record"], b[d]["record"]
        if ra is None or rb is None:
            continue
        delta = (
            rb["metrics"]["repeated_ngram_fraction_4"] - ra["metrics"]["repeated_ngram_fraction_4"]
        )
        overall.append(delta)
        fam = ra.get("family") or "?"
        by_family[fam].append(delta)
        state = (
            "both_closed"
            if (ra["termination"]["think_end_reached"] and rb["termination"]["think_end_reached"])
            else "other"
        )
        by_closure[state].append(delta)
        avg_len = (ra["reasoning_tokens"] + rb["reasoning_tokens"]) / 2
        bucket = "short" if avg_len <= q1 else ("long" if avg_len >= q3 else "mid")
        by_length[bucket].append(delta)
    return {
        "mode": mode,
        "n_paired_draws": len(overall),
        "overall_mean_delta": mean(overall),
        "by_family_mean_delta": {f: mean(v) for f, v in sorted(by_family.items())},
        "by_closure_state_mean_delta": {k: mean(v) for k, v in sorted(by_closure.items())},
        "by_reasoning_length_quartile_mean_delta": {
            k: mean(v) for k, v in sorted(by_length.items())
        },
        "length_quartile_bounds": {"q1": q1, "q3": q3},
        "note": (
            "Exploratory association only; not a validated loop detector and not a "
            "confirmatory result."
        ),
    }
