"""Stochastic-draw collapse.

Statistical-unit hierarchy used by PonderScope:

    STRUCTURAL TASK       task_id                (wording-independent structure)
    PRESENTATION          presentation_id        (rendered stimulus)
    STOCHASTIC DRAW       presentation + condition + seed
    EXECUTION / REPLICATE stochastic draw + repeat

Same-seed technical repeats exist to measure replay determinism; they are **not**
independent samples and must not increase N. This module collapses each draw's
executions to a single primary observation iff every execution is
token-identical. If executions diverge, the draw is flagged ``ambiguous`` and
excluded from primary stochastic summaries, while all executions are retained
for reproducibility diagnostics.

For greedy conditions there is no sampling seed, so the draw is the
presentation/condition itself.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..config.identity import configuration_id


def draw_key(record: dict[str, Any]) -> tuple[str, str, int | None]:
    """Grouping key for a stochastic draw: presentation + condition + seed."""
    presentation = str(record.get("presentation_id") or record["task_id"])
    return (presentation, str(record["condition_id"]), record["condition"].get("seed"))


def _derived_draw_id(record: dict[str, Any]) -> str:
    presentation, condition_id, seed = draw_key(record)
    return configuration_id(
        {"presentation_id": presentation, "condition_id": condition_id, "seed": seed},
        prefix="draw",
    )


def _first_divergence(a: list[int], b: list[int]) -> int | None:
    for i in range(max(len(a), len(b))):
        av = a[i] if i < len(a) else None
        bv = b[i] if i < len(b) else None
        if av != bv:
            return i
    return None


def collapse_to_stochastic_draws(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse executions to one record per stochastic draw.

    Returns a list of draw dicts sorted by (family, task_id, presentation, seed).
    Each draw exposes ``n_executions``, ``token_identical``, ``ambiguous`` and a
    ``record`` (the primary representative) that is ``None`` for ambiguous draws.
    """
    groups: dict[tuple[str, str, int | None], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[draw_key(record)].append(record)

    draws: list[dict[str, Any]] = []
    for key, executions in groups.items():
        executions = sorted(executions, key=lambda r: r["condition"].get("repeat", 0))
        base_tokens = executions[0]["trace"]["token_ids"]
        identical = all(e["trace"]["token_ids"] == base_tokens for e in executions[1:])
        divergence = None
        if not identical:
            for e in executions[1:]:
                divergence = _first_divergence(base_tokens, e["trace"]["token_ids"])
                if divergence is not None:
                    break
        rep = executions[0]
        draws.append(
            {
                "stochastic_draw_id": rep.get("stochastic_draw_id") or _derived_draw_id(rep),
                "key": list(key),
                "task_id": rep["task_id"],
                "presentation_id": rep.get("presentation_id"),
                "family": rep.get("family"),
                "condition_id": rep["condition_id"],
                "mode": rep["condition"]["mode"],
                "seed": key[2],
                "n_executions": len(executions),
                "token_identical": identical,
                "ambiguous": not identical,
                "first_token_divergence": divergence,
                "execution_trial_ids": [e.get("trial_id") for e in executions],
                "record": rep if identical else None,
            }
        )
    draws.sort(
        key=lambda d: (
            str(d.get("family")),
            str(d["task_id"]),
            str(d.get("presentation_id")),
            -1 if d["seed"] is None else int(d["seed"]),
        )
    )
    return draws


def compact_draw(draw: dict[str, Any]) -> dict[str, Any]:
    """A publication-safe summary of one stochastic draw.

    Contains identity, flags, and per-draw scalar metrics only — never raw token
    ids, decoded text, prompts, or the full trace record. Raw evidence stays in
    the immutable run storage and optional verified bundles.
    """
    record = draw.get("record")
    entry: dict[str, Any] = {
        "stochastic_draw_id": draw["stochastic_draw_id"],
        "key": draw.get("key"),
        "task_id": draw["task_id"],
        "presentation_id": draw.get("presentation_id"),
        "family": draw.get("family"),
        "condition_id": draw["condition_id"],
        "mode": draw["mode"],
        "seed": draw["seed"],
        "n_executions": draw["n_executions"],
        "token_identical": draw["token_identical"],
        "ambiguous": draw["ambiguous"],
        "first_token_divergence": draw.get("first_token_divergence"),
        "execution_trial_ids": draw.get("execution_trial_ids", []),
        "scalars": None,
    }
    if record is not None:
        loop = record.get("loop") or {}
        entry["scalars"] = {
            "correct": record["correct"],
            "answer_normalized": record.get("answer_normalized"),
            "natural_final_status": record.get("natural_final_status"),
            "think_end_reached": record["termination"].get("think_end_reached"),
            "terminated_by_eos": record["termination"].get("terminated_by_eos"),
            "reasoning_tokens": record["reasoning_tokens"],
            "answer_tokens": record["answer_tokens"],
            "total_tokens": record["total_tokens"],
            "wall_ms": record["trace"].get("wall_ms"),
            "tokens_per_sec": record["trace"].get("tokens_per_sec"),
            "ttft_ms": record["trace"].get("ttft_ms"),
            "unique_token_ratio": record["metrics"]["unique_token_ratio"],
            "repeated_ngram_fraction_4": record["metrics"]["repeated_ngram_fraction_4"],
            "longest_run_length": loop.get("longest_run_length"),
        }
    return entry


def draw_summary(draws: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts that expose executions and unique draws separately.

    ``draws`` holds compact per-draw summaries (no raw evidence).
    """
    ambiguous = [d for d in draws if d["ambiguous"]]
    primary = [d for d in draws if not d["ambiguous"]]
    return {
        "n_executions": sum(d["n_executions"] for d in draws),
        "n_unique_draws": len(draws),
        "n_primary_draws": len(primary),
        "n_ambiguous_draws": len(ambiguous),
        "ambiguous_draw_ids": [d["stochastic_draw_id"] for d in ambiguous],
        "draws": [compact_draw(d) for d in draws],
    }


def draws_by_condition(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_condition[record["condition_id"]].append(record)
    return {cid: collapse_to_stochastic_draws(rs) for cid, rs in sorted(by_condition.items())}
