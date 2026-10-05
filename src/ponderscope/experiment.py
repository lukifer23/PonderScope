"""Experiment orchestration: run declared deployments over procedural tasks.

This module only records evidence. It never analyses or interprets during a run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .backends import CaptureSpec, get_backend
from .backends.base import PrefixProbeUnsupported
from .config.conditions import condition_specs, decoding_for
from .config.identity import DecodingPolicy, Deployment, ModelIdentity, TrialIdentity
from .config.schema import ExperimentSpec
from .evidence.run import RunStore
from .reasoning.metrics import compute_repetition
from .reasoning.parse import parse_trace, reasoning_prefix_ids
from .reasoning.probes import run_prefix_probes
from .reasoning.transitions import classify_transitions
from .tasks import generate_pack, generate_pack_metadata, normalize, score

BARE_BASELINE_NOTE = "bare baseline captures no per-token metrics"


def build_capture(spec: ExperimentSpec, *, digest: bool | None = None) -> CaptureSpec:
    return CaptureSpec(
        entropy=spec.capture_entropy,
        top_k=spec.capture_top_k,
        logprob_digest=spec.capture_logprob_digest if digest is None else digest,
        per_token_timing=True,
        chosen_logprob=True,
    )


def measure_capture_overhead(
    backend: Any,
    prompt_token_ids: list[int],
    decoding: DecodingPolicy,
    *,
    repeats: int = 5,
    warmup: int = 1,
) -> dict[str, Any]:
    """Bounded qualification of instrumentation overhead.

    Same prompt, same greedy condition, warmup, then alternating repeated runs of
    three capture levels. Reports medians (and full samples) of TTFT, total wall
    time, end-to-end output tokens/sec, and decode tokens/sec. Overhead is only
    claimed as such where measured.
    """
    import statistics

    modes = {
        "minimal": CaptureSpec.minimal(),
        "research": CaptureSpec.research(top_k=5, digest=False),
        "digest": CaptureSpec.research(top_k=5, digest=True),
    }
    samples: dict[str, list[dict[str, float | None]]] = {name: [] for name in modes}

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
        for capture in modes.values():
            _one(capture)
    for _ in range(max(1, repeats)):
        for name, capture in modes.items():
            samples[name].append(_one(capture))

    def _median(values: list[float | None]) -> float | None:
        usable = [v for v in values if v is not None]
        return float(statistics.median(usable)) if usable else None

    summary: dict[str, Any] = {}
    for name, rows in samples.items():
        summary[name] = {
            "ttft_ms_median": _median([r["ttft_ms"] for r in rows]),
            "wall_ms_median": _median([r["wall_ms"] for r in rows]),
            "output_tokens_per_sec_median": _median([r["output_tokens_per_sec"] for r in rows]),
            "decode_tokens_per_sec_median": _median([r["decode_tokens_per_sec"] for r in rows]),
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
        "samples": samples,
        "note": (
            "Same prompt, same greedy condition, alternating runs. 'minimal' captures "
            "token ids + termination only; 'research' adds entropy/top-k/chosen "
            "logprob/timing; 'digest' adds full-distribution digests."
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
) -> RunResult:
    backend = get_backend(backend_name)
    declared = ModelIdentity(repo_id=model_repo, revision=model_revision, precision="unknown")
    loaded = backend.load(declared)
    runtime = backend.runtime_identity()

    base_decoding = DecodingPolicy(
        mode="greedy",
        max_tokens=spec.max_tokens,
        stop_on_eos=True,
        context={"enable_thinking": True},
    )
    deployment0 = Deployment(model=loaded, runtime=runtime, decoding=base_decoding, label=spec.name)

    store = RunStore.create(deployment0, spec, runs_dir=runs_dir, run_suffix=run_suffix)
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
    )
    pack_meta = generate_pack_metadata(
        tasks,
        families=spec.families or None,
        n_per_family=spec.n_per_family,
        split=spec.split,
        seed=spec.task_seed,
        pack=spec.task_pack,
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
        for task in tasks:
            prompt_ids = backend.tokenize_prompt(
                [{"role": "user", "content": task.prompt}], enable_thinking=True
            )
            greedy_reasoning_prefix: list[int] | None = None
            greedy_correct = False
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
                    greedy_correct = bool(record["correct"])
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
                    state = classify_transitions(
                        final_correct=greedy_correct,
                        prefix_correct=[r["correct"] for r in results],
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
                        f"probe {task.task_id}: primary={state.primary} flips={state.flips} "
                        f"stable_sufficient_prefix_tokens={state.stable_sufficient_prefix_tokens}"
                    )
        store.manifest["probe_supported"] = probe_supported
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
    return {
        "record_type": "generation",
        "run_id": store.run_id,
        "artifact_id": deployment.artifact_id,
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
