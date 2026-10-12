"""Offline paired prefix-recoverability analysis (Phase 1.8).

Reads two *sealed* forced-probe runs (BF16 and Q4) and their source runs, and
produces a paired, task-clustered description of whether a correct answer is
recoverable from identical reasoning-prefix checkpoints. It never runs a model.

Scientific scope (kept explicit and narrow):

- Forced finalization is an **oracle-style intervention** on a frozen historical
  trace. A correct forced answer does **not** establish natural closure, natural
  accuracy, or a deployable stopping policy.
- The continuation is a fresh deterministic greedy completion, **not** a
  continuation under the original sampled policy.
- Greedy forced-probe results are never pooled with naturally sampled
  final-answer accuracy in one metric.
- A checkpoint that is not eligible is missing, never a wrong answer. A
  ``censored``/``unparseable``/``error`` probe is indeterminate, never wrong.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from ..evidence.run import RunStore, verify_run
from ..reasoning.outcomes import FORCED_OUTCOMES
from .stats import paired_cluster_bootstrap_delta

ANALYSIS_VERSION = "phase1.8"

_CORRECT = "correct"
_OBSERVED_SCORABLE = ("correct", "scorable_incorrect")


def _load_probes(store: RunStore, *, verify: bool) -> list[dict[str, Any]]:
    report = verify_run(store.path)
    if verify and not report.get("pass"):
        raise RuntimeError(
            f"probe run {store.run_id} does not verify ({report.get('errors')}); "
            "refusing to analyze unverified evidence"
        )
    return store.read_probes()


def checkpoint_keys(store: RunStore) -> list[str]:
    """Ordered checkpoint labels: numeric ascending, then ``full_prefix``."""
    design = store.manifest.get("probe_design", {})
    numeric = [str(int(c)) for c in design.get("checkpoints", [])]
    keys = sorted(set(numeric), key=lambda x: int(x))
    if design.get("include_full_prefix"):
        keys.append("full_prefix")
    return keys


def _record_key(record: dict[str, Any]) -> tuple[str, Any]:
    return (record["source_task_id"], record.get("source_seed"))


def _select(record: dict[str, Any], key: str) -> bool:
    if key == "full_prefix":
        return bool(record.get("is_full_prefix"))
    return record.get("checkpoint_label") == key


def _index(records: list[dict[str, Any]], key: str) -> dict[tuple[str, Any], dict[str, Any]]:
    out: dict[tuple[str, Any], dict[str, Any]] = {}
    for record in records:
        if _select(record, key):
            out[_record_key(record)] = record
    return out


def _outcome_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = dict.fromkeys(FORCED_OUTCOMES, 0)
    for record in records:
        counts[record["outcome"]] = counts.get(record["outcome"], 0) + 1
    return counts


def _rate(numerator: int, denominator: int) -> float | None:
    return (numerator / denominator) if denominator else None


def _degenerate(deltas: list[float]) -> bool:
    return len(deltas) < 2 or len({round(d, 9) for d in deltas}) <= 1


def _paired_cluster_delta(
    a_by_cluster: dict[str, list[float]],
    b_by_cluster: dict[str, list[float]],
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Paired task-clustered bootstrap of ``mean_a - mean_b`` with degeneracy flag."""
    res = paired_cluster_bootstrap_delta(
        a_by_cluster, b_by_cluster, n_resamples=n_resamples, seed=seed
    )
    keys = sorted(set(a_by_cluster) & set(b_by_cluster))
    deltas = [
        float(_mean(a_by_cluster[k]) - _mean(b_by_cluster[k]))
        for k in keys
        if a_by_cluster[k] and b_by_cluster[k]
    ]
    degenerate = _degenerate(deltas)
    if degenerate or res.get("n_clusters", 0) < 2:
        res["lo"] = None
        res["hi"] = None
        res["excludes_zero"] = None
        res["ci_note"] = (
            "degenerate: fewer than two distinct paired deltas; no interval is manufactured"
        )
    res["degenerate"] = degenerate
    return res


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def single_arm_summary(records: list[dict[str, Any]], keys: list[str]) -> dict[str, Any]:
    """Per-checkpoint outcome distribution and eligibility for one arm."""
    per_key: dict[str, Any] = {}
    for key in keys:
        selection = [r for r in records if _select(r, key)]
        counts = _outcome_counts(selection)
        n = len(selection)
        per_key[key] = {
            "n_eligible": n,
            "outcomes": counts,
            "recoverable_correct_rate": _rate(counts[_CORRECT], n),
            "scorable_answer_rate": _rate(counts["correct"] + counts["scorable_incorrect"], n),
        }
    return {
        "n_probe_records": len(records),
        "n_source_draws": len({_record_key(r) for r in records}),
        "per_checkpoint": per_key,
    }


def _paired_status(oa: str | None, ob: str | None) -> dict[str, str]:
    """Pair a forced outcome from each arm into an explicit transition."""
    return {
        "correctness": _transition(oa, ob, _CORRECT),
        "scorable_incorrect": _transition(oa, ob, "scorable_incorrect"),
        "indeterminate": _indeterminate_transition(oa, ob),
    }


def _transition(oa: str | None, ob: str | None, value: str) -> str:
    if oa is None or ob is None:
        return "not_jointly_eligible"
    av, bv = oa == value, ob == value
    if av and bv:
        return "both"
    if av:
        return "only_bf16"
    if bv:
        return "only_q4"
    return "neither"


def _indeterminate_transition(oa: str | None, ob: str | None) -> str:
    if oa is None or ob is None:
        return "not_jointly_eligible"
    a_ind = oa not in _OBSERVED_SCORABLE
    b_ind = ob not in _OBSERVED_SCORABLE
    if a_ind and b_ind:
        return "both_indeterminate"
    if a_ind:
        return "only_bf16_indeterminate"
    if b_ind:
        return "only_q4_indeterminate"
    return "both_observed"


def _natural_status(record: dict[str, Any]) -> str:
    return "closed" if record.get("source_native_closed") else "censored"


def paired_recoverability(
    bf16_store: RunStore,
    q4_store: RunStore,
    *,
    verify: bool = True,
    n_resamples: int = 2000,
    seed: int = 0,
    max_examples: int = 6,
) -> dict[str, Any]:
    """Paired BF16/Q4 recoverability at matched, identical checkpoints.

    ``delta`` is always Q4 − BF16.
    """
    bf16_records = _load_probes(bf16_store, verify=verify)
    q4_records = _load_probes(q4_store, verify=verify)
    keys = checkpoint_keys(bf16_store)
    q4_keys = checkpoint_keys(q4_store)
    if keys != q4_keys:
        raise RuntimeError(f"probe runs declare different checkpoints: {keys} != {q4_keys}")

    per_checkpoint: dict[str, Any] = {}
    first_correct: dict[str, Counter] = {"bf16": Counter(), "q4": Counter()}
    stability = {
        "bf16": Counter(reverted=0, stable_from_first_correct=0, no_correct=0),
        "q4": Counter(reverted=0, stable_from_first_correct=0, no_correct=0),
    }
    for key in keys:
        a = _index(bf16_records, key)
        b = _index(q4_records, key)
        shared = sorted(set(a) & set(b), key=str)
        correctness: Counter[str] = Counter()
        scorable_incorrect: Counter[str] = Counter()
        indeterminate: Counter[str] = Counter()
        a_vals: dict[str, list[float]] = defaultdict(list)
        b_vals: dict[str, list[float]] = defaultdict(list)
        a_si: dict[str, list[float]] = defaultdict(list)
        b_si: dict[str, list[float]] = defaultdict(list)
        for pair in shared:
            ra, rb = a[pair], b[pair]
            oa, ob = ra["outcome"], rb["outcome"]
            trans = _paired_status(oa, ob)
            correctness[trans["correctness"]] += 1
            scorable_incorrect[trans["scorable_incorrect"]] += 1
            indeterminate[trans["indeterminate"]] += 1
            task = pair[0]
            a_vals[task].append(1.0 if oa == _CORRECT else 0.0)
            b_vals[task].append(1.0 if ob == _CORRECT else 0.0)
            a_si[task].append(1.0 if oa == "scorable_incorrect" else 0.0)
            b_si[task].append(1.0 if ob == "scorable_incorrect" else 0.0)

        a_counts = _outcome_counts([a[p] for p in shared])
        b_counts = _outcome_counts([b[p] for p in shared])
        n = len(shared)
        per_checkpoint[key] = {
            "n_jointly_eligible": n,
            "n_tasks": len({p[0] for p in shared}),
            "bf16": {
                "n_eligible": len(a),
                "outcomes": a_counts,
                "recoverable_correct_rate": _rate(a_counts[_CORRECT], n),
                "scorable_answer_rate": _rate(
                    a_counts["correct"] + a_counts["scorable_incorrect"], n
                ),
            },
            "q4": {
                "n_eligible": len(b),
                "outcomes": b_counts,
                "recoverable_correct_rate": _rate(b_counts[_CORRECT], n),
                "scorable_answer_rate": _rate(
                    b_counts["correct"] + b_counts["scorable_incorrect"], n
                ),
            },
            "paired_correctness_transitions": dict(correctness),
            "paired_scorable_incorrect_transitions": dict(scorable_incorrect),
            "paired_indeterminate_transitions": dict(indeterminate),
            "delta_recoverable_correct_q4_minus_bf16": _paired_cluster_delta(
                b_vals, a_vals, n_resamples=n_resamples, seed=seed
            ),
            "delta_scorable_incorrect_q4_minus_bf16": _paired_cluster_delta(
                b_si, a_si, n_resamples=n_resamples, seed=seed
            ),
        }

    # first tested recoverable-correct checkpoint, and stability, per arm
    for arm, records in (("bf16", bf16_records), ("q4", q4_records)):
        draws = sorted({_record_key(r) for r in records}, key=str)
        for draw in draws:
            seq = []
            for key in keys:
                rec = _index(records, key).get(draw)
                seq.append(None if rec is None else rec["outcome"])
            correct_positions = [i for i, o in enumerate(seq) if o == _CORRECT]
            if not correct_positions:
                stability[arm]["no_correct"] += 1
                continue
            first_correct[arm][keys[correct_positions[0]]] += 1
            first_idx = correct_positions[0]
            later = [o for o in seq[first_idx + 1 :] if o is not None]
            # a later *scorable_incorrect* is a reversion; later censored/unparseable
            # outcomes are indeterminate and are not a reversion.
            if any(o == "scorable_incorrect" for o in later):
                stability[arm]["reverted"] += 1
            else:
                stability[arm]["stable_from_first_correct"] += 1

    per_family = _per_family(bf16_records, q4_records, keys)
    strata = _censored_strata(bf16_records, q4_records, keys)
    examples = _representative_examples(bf16_records, q4_records, keys, max_examples=max_examples)

    return {
        "analysis_version": ANALYSIS_VERSION,
        "exploratory": True,
        "intervention": bf16_store.manifest.get("probe_design", {}).get(
            "intervention", "greedy_forced_finalization"
        ),
        "continuation_policy": bf16_store.manifest.get("probe_design", {}).get(
            "continuation_policy"
        ),
        "checkpoints": keys,
        "delta_orientation": "Q4 - BF16 (positive favours Q4 recoverability)",
        "probe_run_bf16": _run_ref(bf16_store),
        "probe_run_q4": _run_ref(q4_store),
        "bf16": single_arm_summary(bf16_records, keys),
        "q4": single_arm_summary(q4_records, keys),
        "per_checkpoint": per_checkpoint,
        "first_recoverable_correct_checkpoint": {
            "bf16": dict(first_correct["bf16"]),
            "q4": dict(first_correct["q4"]),
        },
        "probe_stability": {arm: dict(c) for arm, c in stability.items()},
        "per_family": per_family,
        "censored_source_strata": strata,
        "representative_examples": examples,
        "definitions": {
            "recoverable_correct": "forced-probe outcome == 'correct' at an eligible checkpoint",
            "jointly_eligible": "both arms produced a probe at the same checkpoint length",
            "indeterminate": "outcome in {censored, unparseable, error}: never an observed wrong answer",
            "delta": "task-clustered paired bootstrap of the per-task mean difference",
            "full_prefix": "the entire available reasoning prefix (separately labelled)",
        },
        "limitations": [
            "Exploratory, post-outcome: the source population was already measured and "
            "is not a fresh confirmatory sample.",
            "Forced finalization is an oracle-style intervention; a correct forced "
            "answer does not establish natural closure or a deployable stopping policy.",
            "The continuation is a fresh deterministic greedy completion, not a "
            "continuation under the original sampled policy.",
            "Higher checkpoints are jointly eligible only for longer trajectories, so "
            "the jointly-eligible set changes with checkpoint length.",
            "Greedy forced-probe recoverability is never pooled with naturally sampled "
            "final-answer accuracy.",
            "Degenerate paired samples produce no confidence interval, by design.",
        ],
    }


def _run_ref(store: RunStore) -> dict[str, Any]:
    source = store.manifest.get("prefix_probe_source", {})
    model = store.manifest.get("weight_variant", {})
    return {
        "probe_run_id": store.run_id,
        "probe_path": str(store.path),
        "source_run_id": source.get("source_run_id"),
        "source_evidence_seal_sha256": source.get("source_evidence_seal_sha256"),
        "source_manifest_sha256": source.get("source_manifest_sha256"),
        "source_artifact_id": store.manifest.get("source_artifact_id"),
        "weight_variant_id": store.manifest.get("weight_variant_id"),
        "deployment_id": store.manifest.get("deployment_id"),
        "representation": model.get("representation"),
        "precision": model.get("precision"),
        "quantization": model.get("quantization"),
        "quantization_bits": model.get("quantization_bits"),
        "quantization_group_size": model.get("quantization_group_size"),
    }


def _per_family(
    bf16_records: list[dict[str, Any]],
    q4_records: list[dict[str, Any]],
    keys: list[str],
) -> dict[str, Any]:
    families = sorted({r["source_family"] for r in bf16_records})
    out: dict[str, Any] = {}
    for family in families:
        a_f = [r for r in bf16_records if r["source_family"] == family]
        b_f = [r for r in q4_records if r["source_family"] == family]
        per_key: dict[str, Any] = {}
        for key in keys:
            a = _index(a_f, key)
            b = _index(b_f, key)
            shared = sorted(set(a) & set(b), key=str)
            a_correct = sum(1 for p in shared if a[p]["outcome"] == _CORRECT)
            b_correct = sum(1 for p in shared if b[p]["outcome"] == _CORRECT)
            per_key[key] = {
                "n_jointly_eligible": len(shared),
                "bf16_recoverable_correct": a_correct,
                "q4_recoverable_correct": b_correct,
            }
        out[family] = per_key
    return out


def _pair_draws(
    bf16_records: list[dict[str, Any]], q4_records: list[dict[str, Any]]
) -> list[tuple[str, Any]]:
    return sorted(
        {_record_key(r) for r in bf16_records} & {_record_key(r) for r in q4_records},
        key=str,
    )


def _censored_strata(
    bf16_records: list[dict[str, Any]],
    q4_records: list[dict[str, Any]],
    keys: list[str],
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for draw in _pair_draws(bf16_records, q4_records):
        # natural status from any record for the draw
        any_rec = next(
            (r for r in bf16_records + q4_records if _record_key(r) == draw),
            None,
        )
        if any_rec is None:
            continue
        a_status = "closed" if any_rec.get("source_native_closed") else "censored"
        # Q4 natural status may differ; find its own record
        q4_rec = next((r for r in q4_records if _record_key(r) == draw), None)
        b_status = (
            "closed" if q4_rec is not None and q4_rec.get("source_native_closed") else "censored"
        )
        if a_status == "closed" and b_status == "closed":
            group = "both_naturally_closed"
        elif a_status == "censored" and b_status == "censored":
            group = "both_naturally_censored"
        elif a_status == "closed":
            group = "only_bf16_closed"
        else:
            group = "only_q4_closed"
        groups[group].append(_draw_pattern(bf16_records, q4_records, keys, draw, group))

    out: dict[str, Any] = {}
    for group, rows in sorted(groups.items()):
        pattern = Counter(r["pattern"] for r in rows)
        fluctuate = sum(1 for r in rows if r["fluctuates"])
        too_noisy = sum(1 for r in rows if r["indeterminate_dominant"])
        out[group] = {
            "n_pairs": len(rows),
            "recoverability_pattern": dict(pattern),
            "n_fluctuating": fluctuate,
            "n_indeterminate_dominant": too_noisy,
            "draws": rows,
        }
    return out


def _draw_pattern(
    bf16_records: list[dict[str, Any]],
    q4_records: list[dict[str, Any]],
    keys: list[str],
    draw: tuple[str, Any],
    group: str,
) -> dict[str, Any]:
    seq: dict[str, list[str | None]] = {}
    for arm, records in (("bf16", bf16_records), ("q4", q4_records)):
        idx = {key: _index(records, key).get(draw) for key in keys}
        values: list[str | None] = []
        for key in keys:
            rec = idx[key]
            values.append(None if rec is None else rec["outcome"])
        seq[arm] = values

    def any_correct(values: list[str | None]) -> bool:
        return any(v == _CORRECT for v in values)

    def fluctuates(values: list[str | None]) -> bool:
        correct_positions = [i for i, v in enumerate(values) if v == _CORRECT]
        if not correct_positions:
            return False
        later = [v for v in values[correct_positions[0] + 1 :] if v is not None]
        return any(v == "scorable_incorrect" for v in later)

    bf16_any = any_correct(seq["bf16"])
    q4_any = any_correct(seq["q4"])
    if bf16_any and q4_any:
        pattern = "both_recover"
    elif bf16_any:
        pattern = "only_bf16_recovers"
    elif q4_any:
        pattern = "only_q4_recovers"
    else:
        pattern = "neither_recovers"
    observed_total = sum(1 for v in seq["bf16"] + seq["q4"] if v in _OBSERVED_SCORABLE)
    total = sum(1 for v in seq["bf16"] + seq["q4"] if v is not None)
    return {
        "task_id": draw[0],
        "seed": draw[1],
        "group": group,
        "pattern": pattern,
        "fluctuates": fluctuates(seq["bf16"]) or fluctuates(seq["q4"]),
        "indeterminate_dominant": total > 0 and observed_total / total < 0.5,
        "bf16_outcomes": seq["bf16"],
        "q4_outcomes": seq["q4"],
        "bf16_first_correct_checkpoint": _first_correct(keys, seq["bf16"]),
        "q4_first_correct_checkpoint": _first_correct(keys, seq["q4"]),
    }


def _first_correct(keys: list[str], values: list[str | None]) -> str | None:
    for key, value in zip(keys, values, strict=True):
        if value == _CORRECT:
            return key
    return None


def _representative_examples(
    bf16_records: list[dict[str, Any]],
    q4_records: list[dict[str, Any]],
    keys: list[str],
    *,
    max_examples: int,
) -> list[dict[str, Any]]:
    """Deterministically selected positive / negative / one-sided / unstable draws."""
    patterns: dict[str, dict[str, Any]] = {}
    for draw in _pair_draws(bf16_records, q4_records):
        row = _draw_pattern(bf16_records, q4_records, keys, draw, "example")
        row["source_seal"] = _draw_meta(bf16_records, q4_records, draw)
        if row["pattern"] == "both_recover":
            patterns.setdefault("positive_both_recover", row)
        elif row["pattern"] == "neither_recovers":
            patterns.setdefault("negative_neither_recovers", row)
        elif row["pattern"] in ("only_bf16_recovers", "only_q4_recovers"):
            patterns.setdefault("one_sided", row)
        if row["fluctuates"]:
            patterns.setdefault("unstable", row)
    return list(patterns.values())[:max_examples]


def _draw_meta(
    bf16_records: list[dict[str, Any]],
    q4_records: list[dict[str, Any]],
    draw: tuple[str, Any],
) -> dict[str, Any]:
    rec = next((r for r in bf16_records if _record_key(r) == draw), None) or next(
        (r for r in q4_records if _record_key(r) == draw), None
    )
    if rec is None:
        return {}
    return {
        "task_id": rec["source_task_id"],
        "family": rec["source_family"],
        "seed": rec.get("source_seed"),
        "gold_answer": rec.get("gold_answer"),
        "source_natural_status": ("closed" if rec.get("source_native_closed") else "censored"),
        "source_run_id": rec.get("source_run_id"),
        "source_trace_sha256": rec.get("source_trace_sha256"),
    }


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _delta_cell(delta: dict[str, Any]) -> str:
    mean = delta.get("mean")
    if mean is None:
        return "—"
    if delta.get("degenerate") or delta.get("lo") is None:
        return f"{_fmt(mean)} (degenerate)"
    return f"{_fmt(mean)} [{_fmt(delta['lo'])}, {_fmt(delta['hi'])}]"


def render_markdown(analysis: dict[str, Any]) -> str:
    a = analysis["probe_run_bf16"]
    b = analysis["probe_run_q4"]
    lines: list[str] = []
    lines.append("# Phase 1.8 — paired reasoning-prefix recoverability (EXPLORATORY)")
    lines.append("")
    lines.append(
        "_Generated from saved probe evidence; no numbers are hand-entered. This is "
        "an exploratory, post-outcome analysis, not a confirmatory result._"
    )
    lines.append("")
    lines.append("## Provenance")
    lines.append(f"- analysis version: `{analysis['analysis_version']}`")
    lines.append(f"- intervention: `{analysis['intervention']}`")
    lines.append(f"- continuation policy: {analysis.get('continuation_policy')}")
    lines.append(f"- checkpoints: {analysis['checkpoints']}")
    lines.append(f"- delta orientation: {analysis['delta_orientation']}")
    lines.append(
        f"- BF16 probe run `{a['probe_run_id']}` ← source `{a['source_run_id']}` "
        f"(seal `{str(a.get('source_evidence_seal_sha256'))[:16]}…`)"
    )
    lines.append(
        f"- Q4 probe run `{b['probe_run_id']}` ← source `{b['source_run_id']}` "
        f"(seal `{str(b.get('source_evidence_seal_sha256'))[:16]}…`)"
    )
    lines.append(
        f"- BF16 artifact `{a['source_artifact_id']}` / `{a['weight_variant_id']}` "
        f"quant={a.get('quantization') or 'none'}"
    )
    lines.append(
        f"- Q4 artifact `{b['source_artifact_id']}` / `{b['weight_variant_id']}` "
        f"quant={b.get('quantization')} bits={b.get('quantization_bits')} "
        f"group={b.get('quantization_group_size')}"
    )
    lines.append("")
    lines.append("## Paired recoverability at matched checkpoints (Q4 − BF16)")
    lines.append("")
    lines.append(
        "| checkpoint | jointly eligible | BF16 correct | Q4 correct | "
        "delta recoverable-correct | both | only BF16 | only Q4 | neither | "
        "indeterminate (both/onlyB/onlyQ) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for key in analysis["checkpoints"]:
        cp = analysis["per_checkpoint"][key]
        trans = cp["paired_correctness_transitions"]
        ind = cp["paired_indeterminate_transitions"]
        lines.append(
            f"| {key} | {cp['n_jointly_eligible']} | "
            f"{cp['bf16']['outcomes']['correct']} | {cp['q4']['outcomes']['correct']} | "
            f"{_delta_cell(cp['delta_recoverable_correct_q4_minus_bf16'])} | "
            f"{trans.get('both', 0)} | {trans.get('only_bf16', 0)} | "
            f"{trans.get('only_q4', 0)} | {trans.get('neither', 0)} | "
            f"{ind.get('both_indeterminate', 0)}/{ind.get('only_bf16_indeterminate', 0)}/"
            f"{ind.get('only_q4_indeterminate', 0)} |"
        )
    lines.append("")
    lines.append(
        "**'neither' means neither arm produced a correct forced answer; it is NOT "
        "evidence that both models answered incorrectly** (an indeterminate probe "
        "observed no answer at all)."
    )
    lines.append("")
    lines.append("## Per-arm outcome distributions (eligible counts)")
    lines.append("")
    lines.append(
        "| checkpoint | arm | eligible | correct | scorable_incorrect | unparseable | "
        "censored | error | recoverable-correct rate |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for key in analysis["checkpoints"]:
        cp = analysis["per_checkpoint"][key]
        for arm, label in (("bf16", "BF16"), ("q4", "Q4")):
            o = cp[arm]["outcomes"]
            lines.append(
                f"| {key} | {label} | {cp[arm]['n_eligible']} | {o['correct']} | "
                f"{o['scorable_incorrect']} | {o['unparseable']} | {o['censored']} | "
                f"{o['error']} | {_fmt(cp[arm]['recoverable_correct_rate'])} |"
            )
    lines.append("")
    lines.append("## Per-family recoverable-correct counts (jointly eligible)")
    lines.append("")
    header = "| family | " + " | ".join(f"{k}" for k in analysis["checkpoints"]) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(analysis["checkpoints"]) + 1))
    for family, per_key in analysis["per_family"].items():
        cells = []
        for key in analysis["checkpoints"]:
            row = per_key.get(key, {})
            cells.append(
                f"{row.get('bf16_recoverable_correct', 0)}/{row.get('n_jointly_eligible', 0)}"
                f" vs {row.get('q4_recoverable_correct', 0)}"
            )
        lines.append(f"| {family} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("_(cells: BF16 correct / jointly eligible vs Q4 correct)_")
    lines.append("")
    lines.append("## First recoverable-correct checkpoint and stability")
    lines.append("")
    first = analysis["first_recoverable_correct_checkpoint"]
    lines.append(f"- BF16 first-correct checkpoint counts: {first['bf16']}")
    lines.append(f"- Q4 first-correct checkpoint counts: {first['q4']}")
    lines.append(f"- BF16 probe stability: {analysis['probe_stability']['bf16']}")
    lines.append(f"- Q4 probe stability: {analysis['probe_stability']['q4']}")
    lines.append("")
    lines.append("## Censored-source strata (descriptive, post-hoc)")
    lines.append("")
    for group, data in analysis["censored_source_strata"].items():
        lines.append(
            f"- `{group}`: n={data['n_pairs']}, patterns={data['recoverability_pattern']}, "
            f"fluctuating={data['n_fluctuating']}, indeterminate-dominant="
            f"{data['n_indeterminate_dominant']}"
        )
    lines.append("")
    lines.append("## Representative examples (deterministic selection)")
    lines.append("")
    lines.append(
        "| case | family | task | seed | source natural | BF16 first | Q4 first | pattern |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for example in analysis["representative_examples"]:
        meta = example.get("source_seal", {})
        lines.append(
            f"| {meta.get('task_id', '?')} | {meta.get('family', '?')} | "
            f"{str(meta.get('task_id', ''))[:8]} | {meta.get('seed')} | "
            f"{meta.get('source_natural_status')} | {example['bf16_first_correct_checkpoint']} | "
            f"{example['q4_first_correct_checkpoint']} | {example['pattern']} |"
        )
    lines.append("")
    lines.append("## Definitions")
    lines.append("")
    for name, text in analysis["definitions"].items():
        lines.append(f"- **{name}**: {text}")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    for item in analysis["limitations"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def render_html(analysis: dict[str, Any]) -> str:
    from .report import _markdown_to_html

    md = render_markdown(analysis)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>PonderScope Phase 1.8 — prefix recoverability (exploratory)</title>
<style>
body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 1000px; margin: 2rem auto; padding: 0 1rem; color:#111; }}
h1,h2,h3 {{ font-weight: 600; }}
pre {{ background:#f6f8fa; padding: 0.75rem; overflow-x:auto; border-radius:6px; }}
code {{ background:#f0f0f0; padding:0 3px; border-radius:3px; }}
table {{ border-collapse: collapse; }} td, th {{ border:1px solid #ccc; padding:4px 8px; font-size: 0.85rem; }}
</style></head><body>
{_markdown_to_html(md)}
</body></html>
"""
