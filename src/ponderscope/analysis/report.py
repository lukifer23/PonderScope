"""Render Markdown and standalone HTML reports from saved evidence.

Every number in a report is derived from saved evidence. Nothing is transcribed
by hand.
"""

from __future__ import annotations

import html
import json
from typing import Any

import numpy as np

from ..evidence.run import RunStore


def _svg_histogram(
    values: list[float], width: int = 420, height: int = 140, bins: int = 20, color: str = "#3b6"
) -> str:
    if not values:
        return "<svg></svg>"
    arr = np.asarray(values, dtype=float)
    counts, edges = np.histogram(arr, bins=bins)
    max_count = counts.max() if counts.max() > 0 else 1
    bar_w = width / len(counts)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    ]
    for i, c in enumerate(counts):
        h = (c / max_count) * (height - 20)
        x = i * bar_w
        y = height - h
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - 1:.1f}" height="{h:.1f}" fill="{color}"/>'
        )
    parts.append(
        f'<text x="4" y="12" font-size="11" fill="#333">min={arr.min():.0f} max={arr.max():.0f} mean={arr.mean():.1f}</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _config_table(analysis: dict[str, Any]) -> list[str]:
    lines = [
        "| condition_id | mode | n | success_at_budget (95% CI) | completion_rate | censored_rate | conditional_acc_given_completed | reasoning tokens mean | tok/s |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for cid, c in analysis["configs"].items():
        acc = c["success_at_budget"]
        ci = c["success_at_budget_ci"]
        ci_txt = f"{_fmt(acc)} [{_fmt(ci[0])}, {_fmt(ci[1])}]" if ci else _fmt(acc)
        o = c["outcomes"]
        lines.append(
            f"| {cid} | {c['mode']} | {c['n']} | {ci_txt} | "
            f"{_fmt(o['completion_rate'])} | {_fmt(o['censored_rate'])} | "
            f"{_fmt(o['conditional_accuracy_given_completed'])} | "
            f"{_fmt(c['reasoning_tokens']['mean'])} | {_fmt(c['tokens_per_sec']['mean'])} |"
        )
    return lines


def _noise_section(analysis: dict[str, Any]) -> list[str]:
    nf = analysis["noise_floor"]
    greedy = nf["greedy_replay"]
    same_seed = nf["same_seed_replay"]
    across = nf["across_seed"]
    lines = ["## Repeatability / noise floor", ""]
    lines.append("### 1. Greedy replay (same deterministic condition, repeated)")
    if not greedy.get("available"):
        lines.append("- no repeated greedy conditions in this run")
    else:
        lines.append(
            f"- task/condition pairs with repeats: {greedy['n_task_conditions']}\n"
            f"- token-identical replays: {greedy['token_identical_count']} "
            f"({_fmt(greedy['token_identical_rate'])})\n"
            f"- answer-agreement rate: {_fmt(greedy['answer_agreement_rate'])}\n"
            f"- mean within-task reasoning-token std: {_fmt(greedy['mean_reasoning_token_std'])}\n"
            f"- mean within-task wall-ms std: {_fmt(greedy['mean_wall_ms_std'])}"
        )
    lines.append("")
    lines.append("### 2. Same-seed sampled replay (same seed + sampler, re-executed)")
    if not same_seed.get("available"):
        lines.append("- no repeated same-seed sampled conditions in this run")
    else:
        lines.append(
            f"- task/condition/seed groups: {same_seed['n_task_condition_seeds']}\n"
            f"- token-identical rate: {_fmt(same_seed['token_identical_rate'])}\n"
            f"- ambiguous (divergent) seed groups: {same_seed.get('ambiguous_seed_count', 0)}\n"
            f"- answer-agreement rate (observed answers only): {_fmt(same_seed['answer_agreement_rate'])}\n"
            f"- mean within-group reasoning-token std: {_fmt(same_seed['mean_reasoning_token_std'])}"
        )
    lines.append("")
    lines.append("### 3. Across-seed stochastic variation (fixed sampler policy, different seeds)")
    if not across.get("available"):
        lines.append("- fewer than two seeds per task/condition")
    else:
        lines.append(
            f"- task/condition groups: {across['n_task_conditions']}\n"
            f"- mean distinct observed answers across seeds: {_fmt(across['mean_distinct_answers'])}\n"
            f"- mean observed-answer accuracy std across seeds: {_fmt(across['mean_accuracy_std_across_seeds'])}\n"
            f"- any ambiguous seeds (divergent same-seed repeats): {across.get('any_ambiguous_seeds')}\n"
            "- note: final-answer diversity/accuracy is defined only over seeds with an "
            "observed answer; `—` means unobserved, not zero."
        )
    tokens = nf.get("cross_seed_tokens", {})
    lines.append("")
    lines.append("### 4. Across-seed token variation (valid even when all answers are censored)")
    if not tokens.get("available"):
        lines.append("- fewer than two seeds per task/condition")
    else:
        lines.append(
            f"- task/condition groups: {tokens['n_task_conditions']}\n"
            f"- mean first token divergence across seeds: {_fmt(tokens['mean_first_token_divergence'])}\n"
            f"- mean reasoning-token std across seeds: {_fmt(tokens['mean_reasoning_tokens_std_across_seeds'])}"
        )
    lines.append("")
    return lines


def _termination_section(analysis: dict[str, Any]) -> list[str]:
    lines = ["## Censoring / termination (do not read a capped run as a wrong answer)", ""]
    counter: dict[str, int] = {}
    for c in analysis["configs"].values():
        for key, val in c["termination"].items():
            counter[key] = counter.get(key, 0) + val
    if not counter:
        lines.append("- no termination records")
    for key in sorted(counter):
        lines.append(f"- {key}: {counter[key]}")
    lines.append("")
    lines.append("### Budget outcomes per condition")
    lines.append(
        "| condition_id | success_at_budget | completion_rate | censored_rate | error_rate | "
        "unparseable_rate | cond_acc|completed |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for cid, c in analysis["configs"].items():
        o = c["outcomes"]
        lines.append(
            f"| {cid} | {_fmt(o['success_at_budget'])} | {_fmt(o['completion_rate'])} | "
            f"{_fmt(o['censored_rate'])} | {_fmt(o['error_rate'])} | "
            f"{_fmt(o['unparseable_rate'])} | "
            f"{_fmt(o['conditional_accuracy_given_completed'])} |"
        )
    lines.append("")
    return lines


def _probe_section(analysis: dict[str, Any]) -> list[str]:
    probes = analysis["probes"]
    lines = ["## Prefix probes / trajectory states", ""]
    if probes["n"] == 0:
        lines.append("- no probes in this run")
        lines.append("")
        return lines
    lines.append(f"- n probes: {probes['n']}")
    lines.append(
        "- stable-sufficient WITH observed natural final: "
        f"{probes.get('n_stable_sufficient_with_natural_final', 0)}"
    )
    lines.append(
        "- observed-probe stable (natural final unobserved or uncounted): "
        f"{probes.get('n_observed_probe_stable', 0)}"
    )
    for state, count in sorted(probes["prefix_state_counts"].items()):
        lines.append(f"- {state}: {count}")
    lines.append("")
    return lines


def _survival_section(analysis: dict[str, Any]) -> list[str]:
    s = analysis.get("survival")
    if not s:
        return []
    lines = ["## Censor-aware time-to-closure (Kaplan-Meier / RMST)", ""]
    lines.append(
        f"- observation horizon tau: {s['tau']} tokens (an observation horizon, "
        "not a natural stopping threshold)"
    )
    lines.append(f"- event: {s['event_definition']}")
    lines.append("")
    lines.append("| condition | n | events | censored | median tokens | RMST(tau) |")
    lines.append("|---|---|---|---|---|---|")
    for cid, c in s["per_condition"].items():
        km = c["kaplan_meier"]
        lines.append(
            f"| {cid} | {km['n']} | {km['n_events']} | {km['n_censored']} | "
            f"{_fmt(c['median_tokens_to_closure'])} | {_fmt(c['rmst'])} |"
        )
    lines.append("")
    for cid, c in s["per_condition"].items():
        fam = c.get("by_family") or {}
        rows = [f for f in fam.values() if not f.get("insufficient_data")]
        if not rows:
            continue
        lines.append(f"### {cid} by family")
        lines.append("| family | n | events | censored | median | RMST |")
        lines.append("|---|---|---|---|---|---|")
        for f in rows:
            km = f["kaplan_meier"]
            lines.append(
                f"| {f['family']} | {f['n']} | {km['n_events']} | {km['n_censored']} | "
                f"{_fmt(f['median_tokens_to_closure'])} | {_fmt(f['rmst'])} |"
            )
        lines.append("")
    return lines


def _loop_section(analysis: dict[str, Any]) -> list[str]:
    lp = analysis.get("loop")
    if not lp:
        return []
    lines = ["## Loop-structure diagnostics (descriptive)", "", lp.get("note", ""), ""]
    lines.append(
        "| condition | n | longest_run median (max) | special longest runs | with degeneration onset |"
    )
    lines.append("|---|---|---|---|---|")
    for cid, c in lp["per_condition"].items():
        if not c.get("available"):
            continue
        lr = c["longest_run_length"]
        lines.append(
            f"| {cid} | {c['n']} | {_fmt(lr['median'])} ({_fmt(lr['max'])}) | "
            f"{c['special_longest_run_count']} | {c['n_with_degeneration_onset']} |"
        )
    lines.append("")
    return lines


def _comparison_section(comparison: dict[str, Any]) -> list[str]:
    lines = ["## Deployment comparison", ""]
    lines.append(f"- A: {comparison['description_a']}")
    lines.append(f"- B: {comparison['description_b']}")
    lines.append(f"- matched observations ({comparison['mode']}): {comparison['n_matched_trials']}")
    validity = comparison.get("validity", {})
    if comparison.get("refused"):
        lines.append(
            "- **REFUSED: requested contrast is confounded or under-specified. "
            f"Reasons: {validity.get('reasons')}**"
        )
        lines.append("")
        return lines
    if comparison.get("exploratory"):
        lines.append(
            f"- **EXPLORATORY / CONFOUNDED (override applied). Reasons: {validity.get('reasons')}**"
        )
    lines.append("")
    lines.append(
        "| metric | delta (A−B) | 95% CI | noise A | noise B | combined | classification |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for metric, m in comparison["metrics"].items():
        d = m["delta"]
        if d.get("mean") is None:
            lines.append(
                f"| {metric} | — | — | {_fmt(m['noise_scale_a'])} | {_fmt(m['noise_scale_b'])} "
                f"| {_fmt(m['noise_scale'])} | insufficient_data |"
            )
            continue
        lines.append(
            f"| {metric} | {_fmt(d['mean'])} | [{_fmt(d['lo'])}, {_fmt(d['hi'])}] | "
            f"{_fmt(m['noise_scale_a'])} | {_fmt(m['noise_scale_b'])} | {_fmt(m['noise_scale'])} | "
            f"{m['classification']} |"
        )
    if "trajectory_states" in comparison:
        lines.append("")
        lines.append(f"- trajectory states A: {comparison['trajectory_states']['a']}")
        lines.append(f"- trajectory states B: {comparison['trajectory_states']['b']}")
    lines.append("")
    return lines


def render_markdown(store: RunStore, analysis: dict[str, Any]) -> str:
    manifest = store.manifest
    env = json.loads((store.path / "environment.json").read_text())
    traces = store.read_traces()
    model = manifest.get("weight_variant", manifest.get("artifact", {}))
    runtime = manifest.get("runtime", {})

    lines: list[str] = []
    lines.append(f"# PonderScope report — {manifest['run_id']}")
    lines.append("")
    lines.append("_Generated from saved evidence; no numbers are hand-entered._")
    lines.append("")
    lines.append("## Deployment identity")
    lines.append(f"- description: `{manifest.get('deployment_description')}`")
    lines.append(
        f"- source artifact id: `{manifest.get('source_artifact_id', manifest.get('artifact_id'))}`"
    )
    lines.append(
        f"- weight variant id: `{manifest.get('weight_variant_id', manifest.get('artifact_id'))}`"
    )
    lines.append(f"- deployment id: `{manifest.get('deployment_id')}`")
    if manifest.get("prompt_policy"):
        lines.append(f"- prompt policy: `{manifest.get('prompt_policy')}`")
    policy = manifest.get("model_policy")
    if policy:
        lines.append(
            f"- model policy: `{policy.get('profile_id')}` "
            f"(label `{policy.get('condition_label')}`) recommended=`{policy.get('recommended')}`"
        )
    conditions = manifest.get("conditions", [])
    lines.append("- condition ids: `" + ", ".join(c["condition_id"] for c in conditions) + "`")
    source = model.get("source", model)
    lines.append(f"- model repo: `{source.get('repo_id')}`")
    lines.append(f"- revision: `{source.get('revision')}`")
    lines.append(f"- representation: `{model.get('representation')}`")
    lines.append(f"- precision: `{model.get('precision')}`")
    lines.append(
        f"- quantization: `{model.get('quantization')}` bits={model.get('quantization_bits')} group={model.get('quantization_group_size')}"
    )
    lines.append(
        f"- runtime: `{runtime['runtime']} {runtime['runtime_version']}` backend=`{runtime['backend']}`"
    )
    lines.append(f"- hardware: `{runtime['hardware']}` os=`{runtime['os']}`")
    lines.append(f"- created: `{manifest['created_utc']}`")
    lines.append("")
    lines.append("## Machine / runtime environment")
    lines.append(f"- python: `{env.get('python_version')}`")
    lines.append(f"- platform: `{env.get('platform')}`")
    lines.append(f"- mlx device: `{env.get('mlx_default_device')}`")
    lines.append(f"- memory bytes: `{env.get('physical_memory_bytes')}`")
    lines.append(f"- packages: `{env.get('packages')}`")
    lines.append("")
    lines.append("## Task population")
    lines.append(f"- n tasks: {analysis['n_tasks']}  n generations: {analysis['n_generations']}")
    lines.append(f"- spec: `{manifest['spec']}`")
    lines.append("")

    lines.append("## Configurations")
    lines.extend(_config_table(analysis))
    lines.append("")

    # distributions
    lines.append("## Reasoning-length distribution")
    lines.append("")
    for cid, c in analysis["configs"].items():
        vals = [r["reasoning_tokens"] for r in traces if r["condition_id"] == cid]
        lines.append(f"### {cid} ({c['mode']})")
        lines.append("")
        lines.append("```")
        lines.append(_ascii_hist(vals))
        lines.append("```")
        lines.append(f"- {json.dumps(c['reasoning_tokens'])}")
        lines.append("")

    lines.append("## Latency / throughput")
    for cid, c in analysis["configs"].items():
        lines.append(
            f"- {cid}: wall_ms {json.dumps(c['wall_ms'])}; tok/s {json.dumps(c['tokens_per_sec'])}; ttft_ms {json.dumps(c['ttft_ms'])}"
        )
    lines.append("")

    lines.append("## Repetition")
    for cid, c in analysis["configs"].items():
        lines.append(
            f"- {cid}: unique_token_ratio {json.dumps(c['unique_token_ratio'])}; "
            f"repeated_ngram_fraction_4 {json.dumps(c['repeated_ngram_fraction_4'])}; "
            f"text_repeat_ratio {json.dumps(c['text_repeat_ratio'])}"
        )
    lines.append("")

    lines.extend(_noise_section(analysis))
    lines.extend(_termination_section(analysis))
    lines.extend(_survival_section(analysis))
    lines.extend(_loop_section(analysis))
    lines.extend(_probe_section(analysis))

    comp_path = store.path / "comparison.json"
    if comp_path.exists():
        lines.extend(_comparison_section(json.loads(comp_path.read_text())))

    lines.append("## Limitations")
    lines.append(
        "- Single model revision on a single machine; results do not generalize to other weights or hardware."
    )
    lines.append("- Greedy determinism is a same-process, same-machine replay test.")
    lines.append(
        "- Where the reasoning channel was never closed, the trajectory is censored, not a natural stop."
    )
    lines.append("- No causal claims: deployment differences are associations under this protocol.")
    lines.append("")
    return "\n".join(lines)


def _ascii_hist(values: list[float], bins: int = 20, width: int = 40) -> str:
    if not values:
        return "(no data)"
    arr = np.asarray(values, dtype=float)
    counts, edges = np.histogram(arr, bins=bins)
    max_c = counts.max() if counts.max() > 0 else 1
    out = []
    for i, c in enumerate(counts):
        bar = "#" * int(round((c / max_c) * width))
        out.append(f"{edges[i]:8.0f}-{edges[i + 1]:<8.0f} {c:4d} {bar}")
    return "\n".join(out)


def render_html(store: RunStore, analysis: dict[str, Any]) -> str:
    md = render_markdown(store, analysis)
    traces = store.read_traces()
    charts = []
    for cid in analysis["configs"]:
        vals = [r["reasoning_tokens"] for r in traces if r["condition_id"] == cid]
        charts.append(f"<h3>{html.escape(cid)} — reasoning tokens</h3>")
        charts.append(_svg_histogram(vals))
    body = _markdown_to_html(md)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>PonderScope report {html.escape(store.run_id)}</title>
<style>
body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; color:#111; }}
h1,h2,h3 {{ font-weight: 600; }}
pre {{ background:#f6f8fa; padding: 0.75rem; overflow-x:auto; border-radius:6px; }}
code {{ background:#f0f0f0; padding:0 3px; border-radius:3px; }}
table {{ border-collapse: collapse; }} td, th {{ border:1px solid #ccc; padding:4px 8px; font-size: 0.9rem; }}
svg {{ display:block; margin: 0.5rem 0 1rem; background:#fafafa; }}
</style></head><body>
{body}
<hr>
<h2>Plots derived from saved evidence</h2>
{"".join(charts)}
</body></html>
"""


def _markdown_to_html(md: str) -> str:
    """Very small Markdown-to-HTML for our own controlled output."""
    out: list[str] = []
    in_code = False
    in_table = False
    for line in md.splitlines():
        if line.startswith("```"):
            if in_code:
                out.append("</pre>")
            else:
                out.append("<pre>")
            in_code = not in_code
            continue
        if in_code:
            out.append(html.escape(line))
            continue
        if line.startswith("|") and line.endswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells):
                continue
            if not in_table:
                out.append("<table>")
                in_table = True
            tag = "th" if out[-1].startswith("<table>") else "td"
            out.append(
                "<tr>" + "".join(f"<{tag}>{html.escape(c)}</{tag}>" for c in cells) + "</tr>"
            )
            continue
        elif in_table:
            out.append("</table>")
            in_table = False
        if line.startswith("### "):
            out.append(f"<h3>{html.escape(line[4:])}</h3>")
        elif line.startswith("## "):
            out.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("# "):
            out.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("- "):
            out.append(f"<li>{html.escape(line[2:])}</li>")
        elif line.strip() == "":
            out.append("")
        else:
            out.append(f"<p>{html.escape(line)}</p>")
    if in_table:
        out.append("</table>")
    return "\n".join(out)


def generate_report(store: RunStore, analysis: dict[str, Any] | None = None) -> tuple[str, str]:
    if analysis is None:
        analysis = store.read_analysis()
    md = render_markdown(store, analysis)
    html_text = render_html(store, analysis)
    store.write_summary(md, html_text)
    return md, html_text
