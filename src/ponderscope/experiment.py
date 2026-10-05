"""Experiment orchestration: run declared deployments over procedural tasks.

This module only records evidence. It never analyses or interprets during a run.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .backends import CaptureSpec, get_backend
from .backends.base import PrefixProbeUnsupported
from .config.conditions import condition_specs, decoding_for
from .config.identity import (
    DecodingPolicy,
    Deployment,
    SourceArtifactIdentity,
    TrialIdentity,
)
from .config.schema import ExperimentSpec
from .evidence.environment import capture_code_state
from .evidence.run import RunStore
from .reasoning.metrics import compute_repetition
from .reasoning.parse import parse_trace, reasoning_prefix_ids
from .reasoning.probes import run_prefix_probes
from .reasoning.transitions import (
    classify_transitions,
    natural_final_status_for_record,
    natural_final_status_from_termination,
)
from .tasks import generate_pack, generate_pack_metadata, normalize, score


def build_capture(spec: ExperimentSpec, *, digest: bool | None = None) -> CaptureSpec:
    return CaptureSpec(
        entropy=spec.capture_entropy,
        top_k=spec.capture_top_k,
        logprob_digest=spec.capture_logprob_digest if digest is None else digest,
        per_token_timing=True,
        chosen_logprob=True,
    )


def verify_capture_equivalence(
    backend: Any,
    prompt_token_ids: list[int],
    *,
    max_tokens: int = 64,
) -> dict[str, Any]:
    """Check that capture level does not change the decoded token sequence.

    Greedy and same-seed sampled decoding are deterministic; if the minimal and
    research lanes ever diverge, that is an instrumentation effect and must be
    reported, not silently averaged into performance.
    """

    def _first_divergence(a: list[int], b: list[int]) -> int | None:
        for i in range(max(len(a), len(b))):
            av = a[i] if i < len(a) else None
            bv = b[i] if i < len(b) else None
            if av != bv:
                return i
        return None

    greedy = DecodingPolicy(mode="greedy", max_tokens=max_tokens)
    sampled = DecodingPolicy(
        mode="sampled",
        max_tokens=max_tokens,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        seed=7,
    )
    result: dict[str, Any] = {}
    for label, decoding in (("greedy", greedy), ("sampled_same_seed", sampled)):
        minimal = backend.generate(prompt_token_ids, decoding, CaptureSpec.minimal())
        research = backend.generate(
            prompt_token_ids, decoding, CaptureSpec.research(top_k=5, digest=False)
        )
        result[label] = {
            "identical": minimal.token_ids == research.token_ids,
            "first_divergence": _first_divergence(minimal.token_ids, research.token_ids),
            "minimal_tokens": len(minimal.token_ids),
            "research_tokens": len(research.token_ids),
        }
    result["equivalent"] = all(v["identical"] for v in result.values() if isinstance(v, dict))
    result["note"] = (
        "Deterministic decoding must be token-identical across capture lanes; a "
        "difference is an instrumentation effect."
    )
    return result


def measure_capture_overhead(
    backend: Any,
    prompt_token_ids: list[int],
    decoding: DecodingPolicy,
    *,
    repeats: int = 5,
    warmup: int = 1,
) -> dict[str, Any]:
    """Bounded qualification of instrumentation overhead.

    Same prompt, same greedy condition, separate warmup, then counterbalanced
    repeated runs of three capture levels (order rotated each repetition so a
    lane is not systematically penalised by later runs). Reports median, mean,
    std, and IQR of TTFT, wall time, end-to-end tokens/sec, and decode
    tokens/sec. The ``minimal`` lane is the primary performance measurement;
    research/digest throughput is instrumentation-affected and never presented
    as native deployment throughput.
    """
    import statistics

    modes = {
        "minimal": CaptureSpec.minimal(),
        "research": CaptureSpec.research(top_k=5, digest=False),
        "digest": CaptureSpec.research(top_k=5, digest=True),
    }
    order = list(modes)
    samples: dict[str, list[dict[str, float | None]]] = {name: [] for name in modes}
    run_order: list[str] = []

    def _one(capture: CaptureSpec) -> dict[str, float | None]:
        trace = backend.generate(prompt_token_ids, decoding, capture)
        return {
            "ttft_ms": trace.ttft_ms,
            "wall_ms": trace.wall_ms,
            "output_tokens_per_sec": trace.tokens_per_sec,
            "decode_tokens_per_sec": trace.decode_tokens_per_sec,
            "generated_tokens": float(trace.generated_tokens),
        }

    for _ in range(max(0, warmup)):
        for name in order:
            _one(modes[name])
    for r in range(max(1, repeats)):
        rotated = order[r % len(order) :] + order[: r % len(order)]
        for name in rotated:
            run_order.append(name)
            samples[name].append(_one(modes[name]))

    def _stat(values: list[float | None], fn: Any) -> float | None:
        usable = [v for v in values if v is not None]
        return float(fn(usable)) if usable else None

    def _iqr(values: list[float | None]) -> float | None:
        usable = [v for v in values if v is not None]
        if len(usable) < 2:
            return None
        q1, q3 = np.percentile(usable, [25, 75])
        return float(q3 - q1)

    summary: dict[str, Any] = {}
    for name, rows in samples.items():
        summary[name] = {
            "ttft_ms_median": _stat([r["ttft_ms"] for r in rows], statistics.median),
            "wall_ms_median": _stat([r["wall_ms"] for r in rows], statistics.median),
            "output_tokens_per_sec_median": _stat(
                [r["output_tokens_per_sec"] for r in rows], statistics.median
            ),
            "output_tokens_per_sec_std": _stat(
                [r["output_tokens_per_sec"] for r in rows], statistics.pstdev
            ),
            "decode_tokens_per_sec_median": _stat(
                [r["decode_tokens_per_sec"] for r in rows], statistics.median
            ),
            "decode_tokens_per_sec_iqr": _iqr([r["decode_tokens_per_sec"] for r in rows]),
            "n": len(rows),
        }
    base_decode = summary["minimal"]["decode_tokens_per_sec_median"]
    base_wall = summary["minimal"]["wall_ms_median"]
    for name in modes:
        d = summary[name]["decode_tokens_per_sec_median"]
        w = summary[name]["wall_ms_median"]
        summary[name]["decode_overhead_fraction"] = (
            (base_decode - d) / base_decode if (base_decode and d) else None
        )
        summary[name]["wall_overhead_fraction"] = (
            (w - base_wall) / base_wall if (base_wall and w) else None
        )
    return {
        "modes": summary,
        "repeats": repeats,
        "warmup": warmup,
        "run_order": run_order,
        "samples": samples,
        "primary_performance_lane": "minimal",
        "note": (
            "Counterbalanced (rotated-order) repeated runs; separate warmup. 'minimal' "
            "captures token ids + termination only; 'research' adds entropy/top-k/chosen "
            "logprob/timing; 'digest' adds full-distribution digests. Use 'minimal' for "
            "primary performance; other lanes are instrumentation-affected."
        ),
    }


def run_experiment(
    spec: ExperimentSpec,
    *,
    model_repo: str,
    model_revision: str,
    backend_name: str = "mlx",
    runs_dir: str | Path = "runs",
    run_suffix: str | None = None,
    progress: Callable[[str], None] | None = None,
    allow_dirty: bool = False,
    code_state: dict[str, Any] | None = None,
) -> RunResult:
    def _progress(message: str) -> None:
        if progress is not None:
            progress(message)

    code_state = code_state if code_state is not None else capture_code_state()
    tracked_dirty = bool(code_state.get("tracked_dirty", code_state.get("git_dirty", False)))
    if tracked_dirty and not allow_dirty:
        raise RuntimeError(
            "refusing to run an official experiment from a dirty tracked worktree; "
            "commit your changes or pass allow_dirty=True (records an exploratory run)"
        )
    code_state = {
        **code_state,
        "exploratory": tracked_dirty,
        "publication_grade": not tracked_dirty,
    }

    backend = get_backend(backend_name)
    request = SourceArtifactIdentity(repo_id=model_repo, revision=model_revision)
    loaded = backend.load(request)
    runtime = backend.runtime_identity()

    base_decoding = DecodingPolicy(
        mode="greedy",
        max_tokens=spec.max_tokens,
        stop_on_eos=True,
        context={"enable_thinking": True},
    )
    deployment0 = Deployment(model=loaded, runtime=runtime, decoding=base_decoding, label=spec.name)

    store = RunStore.create(
        deployment0,
        spec,
        runs_dir=runs_dir,
        run_suffix=run_suffix,
        code_state=code_state,
        prompt_policy=spec.prompt_policy,
    )
    store.add_observation(
        f"model loaded; precision={loaded.precision}; quantization={loaded.quantization}; "
        f"load_audit_ok={loaded.load_audit.get('ok')}"
    )
    store.update_status("RUNNING")

    tasks = generate_pack(
        families=spec.families or None,
        n_per_family=spec.n_per_family,
        pack=spec.task_pack,
        split=spec.split,
        seed=spec.task_seed,
        prompt_policy=spec.prompt_policy,
    )
    pack_meta = generate_pack_metadata(
        tasks,
        families=spec.families or None,
        n_per_family=spec.n_per_family,
        split=spec.split,
        seed=spec.task_seed,
        pack=spec.task_pack,
        prompt_policy=spec.prompt_policy,
    )
    store.manifest["task_pack"] = pack_meta
    store.write_tasks([t.to_dict() for t in tasks])

    capture = build_capture(spec)
    conditions = condition_specs(spec)
    eos_ids = set(getattr(backend, "eos_token_ids", set()) or set())
    think_end_id = getattr(backend, "think_end_token_id", None)
    n_gen = 0
    n_probe = 0
    probe_supported = True

    traces_writer = store.open_traces()
    probes_writer = store.open_probes()
    try:
        _progress(f"run {store.run_id}: {len(tasks)} tasks, {len(conditions)} conditions each")
        for task in tasks:
            _progress(f"task {task.family}/{task.task_id[:8]} starting")
            prompt_ids = backend.tokenize_prompt(
                [{"role": "user", "content": task.prompt}], enable_thinking=True
            )
            greedy_reasoning_prefix: list[int] | None = None
            greedy_record: dict[str, Any] | None = None
            for condition in conditions:
                decoding = decoding_for(spec, condition)
                trace = backend.generate(prompt_ids, decoding, capture)
                trial = TrialIdentity(
                    task_id=task.task_id,
                    condition=decoding,
                    seed=condition.get("seed"),
                    repeat=condition["repeat"],
                )
                record = _make_record(
                    backend=backend,
                    store=store,
                    artifact=loaded,
                    runtime=runtime,
                    task=task,
                    condition=condition,
                    decoding=decoding,
                    trial=trial,
                    trace=trace,
                    eos_ids=eos_ids,
                    think_end_id=think_end_id,
                )
                traces_writer.append(record)
                n_gen += 1
                if condition["mode"] == "greedy" and condition["repeat"] == 0:
                    greedy_reasoning_prefix = reasoning_prefix_ids(trace.token_ids, think_end_id)
                    greedy_record = record
            if spec.probe and greedy_reasoning_prefix is not None:
                probe_decoding = DecodingPolicy(
                    mode="greedy", max_tokens=spec.probe_max_tokens, stop_on_eos=True
                )
                try:
                    results = run_prefix_probes(
                        backend,
                        family=task.family,
                        answer=task.answer,
                        prompt_token_ids=prompt_ids,
                        prefix_token_ids=greedy_reasoning_prefix,
                        n_probes=spec.n_probes,
                        probe_decoding=probe_decoding,
                        capture=CaptureSpec(entropy=False, top_k=0, logprob_digest=False),
                        eos_token_ids=eos_ids,
                    )
                except PrefixProbeUnsupported as exc:
                    probe_supported = False
                    store.add_observation(f"probe unsupported: {exc}")
                    results = []
                if results:
                    natural_status = (
                        natural_final_status_for_record(greedy_record)
                        if greedy_record is not None
                        else "censored"
                    )
                    state = classify_transitions(
                        [r["correct"] for r in results],
                        natural_status,
                        prefix_token_lengths=[r["reasoning_prefix_tokens"] for r in results],
                    )
                    for result in results:
                        probes_writer.append(
                            {
                                "run_id": store.run_id,
                                "task_id": task.task_id,
                                "family": task.family,
                                "condition_id": probe_decoding.condition_id,
                                "probe_capability": "prefix_probe",
                                **result,
                            }
                        )
                        n_probe += 1
                    store.add_observation(
                        f"probe {task.task_id}: prefix_state={state.prefix_state} "
                        f"prefix_flips={state.prefix_flips} "
                        f"natural_final_status={state.natural_final_status} "
                        f"observed_probe_stable_from_tokens="
                        f"{state.observed_probe_stable_from_tokens} "
                        f"stable_sufficient_with_natural_final_tokens="
                        f"{state.stable_sufficient_with_natural_final_tokens} "
                        f"harmful_overthinking_observed={state.harmful_overthinking_observed}"
                    )
            _progress(
                f"task {task.family}/{task.task_id[:8]} done "
                f"({n_gen} generations, {n_probe} probes)"
            )
        store.manifest["probe_supported"] = probe_supported
    except BaseException as exc:
        store.mark_failed(type(exc).__name__, str(exc), partial=True)
        raise
    finally:
        traces_writer.close()
        probes_writer.close()

    seal = store.seal()
    return RunResult(store=store, n_generations=n_gen, n_probes=n_probe, seal=seal)


def _make_record(
    *,
    backend: Any,
    store: RunStore,
    artifact: Any,
    runtime: Any,
    task: Any,
    condition: dict[str, Any],
    decoding: DecodingPolicy,
    trial: TrialIdentity,
    trace: Any,
    eos_ids: set[int],
    think_end_id: int | None,
) -> dict[str, Any]:
    deployment = Deployment(
        model=artifact, runtime=runtime, decoding=decoding, label=store.manifest["name"]
    )
    parsed = parse_trace(trace.text, trace.token_ids, think_end_id=think_end_id, eos_ids=eos_ids)
    answer_norm = (
        normalize(task.family, parsed.answer_raw) if parsed.answer_raw is not None else None
    )
    correct = bool(
        answer_norm is not None and score(task.family, parsed.answer_raw or "", task.answer)
    )
    rep = compute_repetition(trace.token_ids, trace.text)
    error_type = None
    if trace.error:
        error_type = trace.error.split(":", 1)[0]
    natural_final_status = natural_final_status_from_termination(
        correct=correct,
        answer_observed=parsed.answer_raw is not None,
        think_end_reached=trace.think_end_reached,
        capped=trace.capped,
        finish_reason=trace.finish_reason,
        error_type=error_type,
    )
    return {
        "record_type": "generation",
        "run_id": store.run_id,
        "artifact_id": deployment.artifact_id,
        "source_artifact_id": deployment.source_artifact_id,
        "weight_variant_id": deployment.weight_variant_id,
        "deployment_id": deployment.deployment_id,
        "condition_id": decoding.condition_id,
        "trial_id": trial.trial_id,
        "deployment": deployment.to_dict(),
        "task_id": task.task_id,
        "family": task.family,
        "split": task.split,
        "difficulty": task.difficulty,
        "condition": condition,
        "prompt": task.prompt,
        "prompt_tokens": trace.prompt_tokens,
        "gold_answer": task.answer,
        "trace": trace.to_dict(),
        "parse": parsed.to_dict(),
        "answer_normalized": answer_norm,
        "correct": correct,
        "natural_final_status": natural_final_status,
        "reasoning_tokens": len(parsed.reasoning_token_ids),
        "answer_tokens": len(parsed.final_token_ids),
        "total_tokens": len(trace.token_ids),
        "metrics": rep.to_dict(),
        "termination": {
            "finish_reason": trace.finish_reason,
            "terminated_by_eos": trace.terminated_by_eos,
            "capped": trace.capped,
            "think_end_reached": trace.think_end_reached,
            "eos_observed": parsed.eos_observed,
            "missing_answer": parsed.answer_raw is None,
            "error_type": error_type,
        },
    }


@dataclass
class RunResult:
    store: RunStore
    n_generations: int
    n_probes: int
    seal: dict[str, Any] | None = None
