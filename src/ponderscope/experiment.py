"""Experiment orchestration: run declared deployments over procedural tasks.

This module only records evidence. It never analyses or interprets during a run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .backends import CaptureSpec, get_backend
from .config.identity import DecodingPolicy, Deployment, ModelIdentity
from .config.schema import ExperimentSpec
from .evidence.run import RunStore
from .reasoning.metrics import compute_repetition
from .reasoning.parse import parse_reasoning
from .reasoning.probes import run_prefix_probes
from .reasoning.transitions import classify_transitions
from .tasks import generate_pack, normalize, score

BARE_BASELINE_NOTE = "bare baseline captures no per-token metrics"


def measure_capture_overhead(
    backend: Any,
    prompt_token_ids: list[int],
    decoding: DecodingPolicy,
    *,
    repeats: int = 3,
) -> dict[str, Any]:
    """Approximate instrumentation overhead: capture on vs capture off.

    The observer should not meaningfully distort the observed. This measures the
    cost of per-token entropy/top-k/digest capture by running the same prompt with
    and without it.
    """
    on = CaptureSpec(entropy=True, top_k=5, logprob_digest=False, per_token_timing=True)
    off = CaptureSpec(entropy=False, top_k=0, logprob_digest=False, per_token_timing=False)
    on_tps: list[float] = []
    off_tps: list[float] = []
    for _ in range(repeats):
        on_tps.append(backend.generate(prompt_token_ids, decoding, on).tokens_per_sec)
        off_tps.append(backend.generate(prompt_token_ids, decoding, off).tokens_per_sec)
    on_mean = sum(on_tps) / len(on_tps)
    off_mean = sum(off_tps) / len(off_tps)
    return {
        "capture_on_tps": on_mean,
        "capture_off_tps": off_mean,
        "overhead_fraction": (off_mean - on_mean) / off_mean if off_mean else None,
        "repeats": repeats,
        "note": "Same prompt, same decoding; difference is per-token capture cost.",
    }


@dataclass
class RunResult:
    store: RunStore
    n_generations: int
    n_probes: int


def build_capture(spec: ExperimentSpec, *, digest: bool | None = None) -> CaptureSpec:
    return CaptureSpec(
        entropy=spec.capture_entropy,
        top_k=spec.capture_top_k,
        logprob_digest=spec.capture_logprob_digest if digest is None else digest,
        per_token_timing=True,
    )


def _token_split(ids: list[int], think_end_id: int | None) -> tuple[int, int]:
    if think_end_id is not None and think_end_id in ids:
        idx = ids.index(think_end_id)
        return idx, len(ids) - idx - 1
    return len(ids), 0


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
        f"model loaded strict=False; precision={loaded.precision}; "
        f"quantization={loaded.quantization}"
    )
    store.update_status("RUNNING")

    tasks = generate_pack(
        families=spec.families or None,
        n_per_family=spec.n_per_family,
        split=spec.split,
        seed=spec.task_seed,
    )
    store.write_tasks([t.to_dict() for t in tasks])

    capture = build_capture(spec)
    n_gen = 0
    n_probe = 0

    traces_writer = store.open_traces()
    probes_writer = store.open_probes()
    try:
        for task in tasks:
            prompt_ids = backend.tokenize_prompt(
                [{"role": "user", "content": task.prompt}], enable_thinking=True
            )
            conditions = _conditions(spec)
            greedy_prefix_ids: list[int] | None = None
            greedy_correct = False
            for condition in conditions:
                decoding = _decoding_for(spec, condition)
                trace = backend.generate(prompt_ids, decoding, capture)
                record = _make_record(
                    backend=backend,
                    store=store,
                    deployment_model=loaded,
                    runtime=runtime,
                    task=task,
                    prompt_ids=prompt_ids,
                    condition=condition,
                    decoding=decoding,
                    trace=trace,
                )
                traces_writer.append(record)
                n_gen += 1
                if condition["mode"] == "greedy" and condition["repeat"] == 0:
                    greedy_prefix_ids = trace.token_ids
                    greedy_correct = bool(record["correct"])
            if spec.probe and greedy_prefix_ids:
                probe_decoding = DecodingPolicy(
                    mode="greedy", max_tokens=spec.probe_max_tokens, stop_on_eos=True
                )
                results = run_prefix_probes(
                    backend,
                    family=task.family,
                    answer=task.answer,
                    prompt_token_ids=prompt_ids,
                    prefix_token_ids=greedy_prefix_ids,
                    n_probes=spec.n_probes,
                    probe_decoding=probe_decoding,
                    capture=CaptureSpec(entropy=False, top_k=0, logprob_digest=False),
                )
                state = classify_transitions(
                    final_correct=greedy_correct,
                    prefix_correct=[r["correct"] for r in results],
                )
                for result in results:
                    probes_writer.append(
                        {
                            "run_id": store.run_id,
                            "task_id": task.task_id,
                            "family": task.family,
                            "config_id": store.config_id,
                            **result,
                        }
                    )
                    n_probe += 1
                store.add_observation(
                    f"probe {task.task_id}: primary={state.primary} flips={state.flips}"
                )
    finally:
        traces_writer.close()
        probes_writer.close()

    store.update_status("EVIDENCE_COMPLETE")
    return RunResult(store=store, n_generations=n_gen, n_probes=n_probe)


def _conditions(spec: ExperimentSpec) -> list[dict[str, Any]]:
    conditions: list[dict[str, Any]] = []
    for r in range(spec.greedy_repeats):
        conditions.append({"mode": "greedy", "repeat": r})
    for seed in spec.sampled_seeds:
        for r in range(spec.sampled_repeats_per_seed):
            conditions.append({"mode": "sampled", "seed": seed, "repeat": r})
    return conditions


def _decoding_for(spec: ExperimentSpec, condition: dict[str, Any]) -> DecodingPolicy:
    if condition["mode"] == "greedy":
        return DecodingPolicy(
            mode="greedy",
            max_tokens=spec.max_tokens,
            stop_on_eos=True,
            context={"enable_thinking": True},
        )
    return DecodingPolicy(
        mode="sampled",
        max_tokens=spec.max_tokens,
        temperature=spec.sampled_temperature,
        top_p=spec.sampled_top_p,
        top_k=spec.sampled_top_k,
        seed=int(condition["seed"]),
        stop_on_eos=True,
        context={"enable_thinking": True},
    )


def _make_record(
    *,
    backend: Any,
    store: RunStore,
    deployment_model: ModelIdentity,
    runtime: Any,
    task: Any,
    prompt_ids: list[int],
    condition: dict[str, Any],
    decoding: DecodingPolicy,
    trace: Any,
) -> dict[str, Any]:
    deployment = Deployment(
        model=deployment_model, runtime=runtime, decoding=decoding, label=store.manifest["name"]
    )
    parsed = parse_reasoning(trace.text, trace.think_end_reached)
    answer_norm = (
        normalize(task.family, parsed.answer_raw) if parsed.answer_raw is not None else None
    )
    correct = bool(
        answer_norm is not None and score(task.family, parsed.answer_raw or "", task.answer)
    )
    reasoning_tokens, answer_tokens = _token_split(trace.token_ids, backend.think_end_token_id)
    rep = compute_repetition(trace.token_ids, trace.text)
    return {
        "record_type": "generation",
        "run_id": store.run_id,
        "config_id": deployment.config_id,
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
        "reasoning_tokens": reasoning_tokens,
        "answer_tokens": answer_tokens,
        "total_tokens": len(trace.token_ids),
        "metrics": rep.to_dict(),
        "termination": {
            "finish_reason": trace.finish_reason,
            "terminated_by_eos": trace.terminated_by_eos,
            "capped": trace.capped,
            "think_end_reached": trace.think_end_reached,
            "missing_answer": parsed.answer_raw is None,
        },
    }
