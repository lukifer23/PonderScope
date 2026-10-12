"""Replay forced-finalization probes on saved historical reasoning prefixes.

This module implements the ``probe-existing`` workflow: it reads a *sealed*,
already-measured generation run, verifies its integrity, recovers the exact
saved task presentation and prompt tokenization, extracts the reasoning-only
token prefix of each saved trajectory, and runs **new** forced-finalization
interventions on those immutable prefixes.

Design constraints (Phase 1.8):

- The source run is **never** mutated or appended to. New probes live in their
  own sealed evidence directory.
- The source deployment / condition / stochastic draw / seed / trial identity is
  preserved on every probe record, paired with the original trace hash and the
  source evidence-seal hash.
- A saved reasoning prefix was previously *generated output*, not ordinary user
  prompt text. The replayed prefix is therefore declared as generated history to
  the logits processors (:meth:`MlxBackend.probe`), and the continuation is a
  fresh deterministic greedy completion — an intervention on a frozen trace, not
  a continuation of the original sampled policy.
- Any failure (unverifiable source, missing raw traces, prompt re-tokenization
  mismatch, artifact identity mismatch, unvalidatable native closure) fails
  closed: nothing is fabricated and no probe is run.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..backends.base import Backend, CaptureSpec
from ..config.identity import (
    DecodingPolicy,
    Deployment,
    SourceArtifactIdentity,
    sha256_file,
)
from ..config.schema import ExperimentSpec
from ..evidence.probe_run import PROBE_RUN_KIND, create_probe_run
from ..evidence.run import RunStore, verify_run
from .probes import run_prefix_probes_at
from .transitions import natural_final_status_for_record


class ReplayError(RuntimeError):
    """Raised when a source run cannot be replayed faithfully."""


def _sha256_ids(token_ids: list[int]) -> str:
    payload = json.dumps(list(token_ids), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_record(record: dict[str, Any]) -> str:
    from ..config.identity import canonical_json

    return hashlib.sha256(canonical_json(record).encode("utf-8")).hexdigest()


def reasoning_tokens_only(
    token_ids: list[int],
    think_end_id: int | None,
    eos_ids: set[int] | frozenset[int] = frozenset(),
) -> list[int]:
    """Reasoning-only token prefix: never a final-answer or EOS token.

    When the native close is present, everything from it onward (final answer and
    EOS) is excluded. When it is absent (a censored / no-closure trajectory),
    any trailing EOS tokens are stripped defensively so they can never enter a
    replayed prefix.
    """
    if think_end_id is not None and think_end_id in token_ids:
        return list(token_ids[: token_ids.index(think_end_id)])
    seq = list(token_ids)
    while seq and seq[-1] in eos_ids:
        seq.pop()
    return seq


@dataclass
class SourceDraw:
    """One saved source trajectory to be probed."""

    task_id: str
    presentation_id: str | None
    family: str
    difficulty: Any
    prompt: str
    prompt_policy: str | None
    split: str | None
    seed: int | None
    mode: str
    condition_id: str
    deployment_id: str
    stochastic_draw_id: str
    trial_id: str
    trace_sha256: str
    reasoning_token_ids: list[int] = field(default_factory=list)
    native_closed: bool = False
    natural_final_status: str = "unparseable"
    finish_reason: str | None = None
    prompt_token_ids: list[int] = field(default_factory=list)
    source_prompt_tokens: int = 0

    def identity(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "presentation_id": self.presentation_id,
            "family": self.family,
            "seed": self.seed,
            "mode": self.mode,
            "condition_id": self.condition_id,
            "deployment_id": self.deployment_id,
            "stochastic_draw_id": self.stochastic_draw_id,
            "trial_id": self.trial_id,
        }


def verify_source_run(source_path: str | Path) -> dict[str, Any]:
    """Verify a source run's seal and require its raw traces to be present."""
    path = Path(source_path)
    report = verify_run(path)
    if not report.get("pass"):
        raise ReplayError(
            f"source run {path} does not verify ({report.get('errors')}); refusing to probe it"
        )
    if not (path / "traces.jsonl").exists():
        raise ReplayError(
            f"source run {path} has no raw traces.jsonl; refusing to fabricate or "
            "regenerate the original population"
        )
    return report


def load_source_draws(
    source_store: RunStore,
    backend: Backend,
    *,
    seeds: list[int] | None = None,
    families: list[str] | None = None,
) -> list[SourceDraw]:
    """Recover the exact saved draws, asserting prompt re-tokenization identity."""
    think_end_id = getattr(backend, "think_end_token_id", None)
    eos_ids = set(getattr(backend, "eos_token_ids", set()) or set())
    draws: list[SourceDraw] = []
    for record in source_store.read_traces():
        condition = record.get("condition", {})
        if seeds is not None and condition.get("seed") not in seeds:
            continue
        if families is not None and record.get("family") not in families:
            continue
        prompt = record["prompt"]
        prompt_policy = record.get("prompt_policy")
        prompt_ids = backend.tokenize_prompt(
            [{"role": "user", "content": prompt}], enable_thinking=True
        )
        saved_prompt_tokens = record["prompt_tokens"]
        trace_prompt_tokens = record["trace"]["prompt_tokens"]
        if not (len(prompt_ids) == saved_prompt_tokens == trace_prompt_tokens):
            raise ReplayError(
                f"prompt re-tokenization mismatch for {record['task_id']}: "
                f"re-tokenized={len(prompt_ids)} record={saved_prompt_tokens} "
                f"trace={trace_prompt_tokens}; refusing to replay a prefix under a "
                "different prompt tokenization"
            )
        expected_presentation = source_store.manifest.get("spec", {}).get("prompt_policy")
        if (
            record.get("presentation_id") is not None
            and expected_presentation is not None
            and prompt_policy != expected_presentation
        ):
            raise ReplayError(
                f"presentation prompt policy {prompt_policy!r} != run {expected_presentation!r}"
            )
        reasoning = reasoning_tokens_only(record["trace"]["token_ids"], think_end_id, eos_ids)
        draws.append(
            SourceDraw(
                task_id=record["task_id"],
                presentation_id=record.get("presentation_id"),
                family=record.get("family", "?"),
                difficulty=record.get("difficulty"),
                prompt=prompt,
                prompt_policy=prompt_policy,
                split=record.get("split"),
                seed=condition.get("seed"),
                mode=condition.get("mode", "?"),
                condition_id=record["condition_id"],
                deployment_id=record.get("deployment_id", ""),
                stochastic_draw_id=record.get("stochastic_draw_id", ""),
                trial_id=record["trial_id"],
                trace_sha256=_sha256_record(record),
                reasoning_token_ids=reasoning,
                native_closed=bool(record["termination"].get("think_end_reached")),
                natural_final_status=natural_final_status_for_record(record),
                finish_reason=record["termination"].get("finish_reason"),
                prompt_token_ids=prompt_ids,
                source_prompt_tokens=saved_prompt_tokens,
            )
        )
    draws.sort(key=lambda d: (d.family, d.task_id, str(d.seed)))
    return draws


def numeric_checkpoints(checkpoints: list[int], *, include_zero: bool) -> list[int]:
    values = sorted({int(c) for c in checkpoints})
    if include_zero:
        values = sorted(set(values) | {0})
    return values


def plan_draw(
    draw: SourceDraw,
    checkpoints: list[int],
    *,
    include_full: bool,
) -> dict[str, Any]:
    """Per-draw checkpoint eligibility plan (no substitution, explicit NOT ELIGIBLE)."""
    rlen = len(draw.reasoning_token_ids)
    eligible = {str(c): (c == 0 or rlen >= c) for c in checkpoints}
    lengths = sorted({c for c in checkpoints if eligible[str(c)]})
    if include_full:
        lengths = sorted(set(lengths) | {rlen})
    entries: list[dict[str, Any]] = []
    for index, length in enumerate(lengths):
        label = str(length) if length in checkpoints else "full_prefix"
        entries.append(
            {
                "checkpoint": length,
                "checkpoint_label": label,
                "checkpoint_index": index,
                "is_full_prefix": length == rlen,
            }
        )
    required = ["full_prefix"] if include_full else []
    ineligible = [str(c) for c in checkpoints if not eligible[str(c)]]
    return {
        **draw.identity(),
        "source_reasoning_tokens": rlen,
        "source_native_closed": draw.native_closed,
        "source_natural_final_status": draw.natural_final_status,
        "requested_numeric_checkpoints": checkpoints,
        "requested_labels": [str(c) for c in checkpoints] + required,
        "checkpoint_eligible": {**eligible, "full_prefix": include_full},
        "ineligible_checkpoints": ineligible,
        "planned": entries,
    }


@dataclass
class ProbeRunResult:
    store: RunStore
    n_probes: int
    n_source_draws: int
    plan: list[dict[str, Any]] = field(default_factory=list)
    seal: dict[str, Any] | None = None


def run_probe_existing(
    source_run_path: str | Path,
    *,
    backend: Backend,
    model_repo: str,
    model_revision: str,
    artifact_path: str | None = None,
    run_label: str,
    checkpoints: list[int],
    include_zero: bool = True,
    include_full: bool = True,
    probe_max_tokens: int = 256,
    seeds: list[int] | None = None,
    families: list[str] | None = None,
    capture: CaptureSpec | None = None,
    runs_dir: str | Path = "runs",
    run_suffix: str | None = None,
    allow_dirty: bool = False,
    code_state: dict[str, Any] | None = None,
    progress: Callable[[str], None] | None = None,
) -> ProbeRunResult:
    """Run forced-finalization probes over the saved prefixes of a source run."""

    def _progress(message: str) -> None:
        if progress is not None:
            progress(message)

    from ..evidence.environment import capture_code_state

    code_state = code_state if code_state is not None else capture_code_state()
    tracked_dirty = bool(code_state.get("tracked_dirty", code_state.get("git_dirty", False)))
    if tracked_dirty and not allow_dirty:
        raise ReplayError(
            "refusing to run an official probe experiment from a dirty tracked worktree; "
            "commit your changes or pass allow_dirty=True (records an exploratory run)"
        )
    code_state = {
        **code_state,
        "exploratory": tracked_dirty,
        "publication_grade": not tracked_dirty,
    }

    source_path = Path(source_run_path)
    source_store = RunStore.load(source_path)
    verify_source_run(source_path)
    source_manifest = source_store.manifest
    source_spec = source_manifest.get("spec", {})

    # Validate the native close *before* creating an evidence directory.
    native_close: list[int] | None = None
    if hasattr(backend, "native_closure_ids"):
        native_close = list(backend.native_closure_ids())

    loaded = backend.load(
        SourceArtifactIdentity(repo_id=model_repo, revision=model_revision),
        artifact_path=artifact_path,
    )
    if loaded.source.source_artifact_id != source_manifest.get("source_artifact_id"):
        raise ReplayError(
            "loaded source artifact does not match the source run "
            f"({loaded.source.source_artifact_id} != "
            f"{source_manifest.get('source_artifact_id')}); refusing to misattribute the probe"
        )
    if loaded.weight_variant_id != source_manifest.get("weight_variant_id"):
        raise ReplayError(
            "loaded weight variant does not match the source run "
            f"({loaded.weight_variant_id} != {source_manifest.get('weight_variant_id')}); "
            "refusing to probe a different deployment"
        )
    runtime = backend.runtime_identity()

    draws = load_source_draws(source_store, backend, seeds=seeds, families=families)
    if not draws:
        raise ReplayError(
            f"no source draws matched seeds={seeds} families={families} in {source_path}"
        )

    checkpoints = numeric_checkpoints(checkpoints, include_zero=include_zero)
    probe_decoding = DecodingPolicy(
        mode="greedy",
        max_tokens=probe_max_tokens,
        stop_on_eos=True,
        context={"enable_thinking": True},
    )
    capture = capture or CaptureSpec.minimal()

    number_of_probes = 0
    plan: list[dict[str, Any]] = []
    for draw in draws:
        entry = plan_draw(draw, checkpoints, include_full=include_full)
        plan.append(entry)
        number_of_probes += len(entry["planned"])

    spec = ExperimentSpec(
        name=run_label,
        task_pack=source_spec.get("task_pack", "tasks-v1"),
        prompt_policy=source_spec.get("prompt_policy", "pp-v1"),
        model_policy=source_spec.get("model_policy", ""),
        families=families or source_spec.get("families", []),
        n_per_family=source_spec.get("n_per_family", 6),
        task_seed=source_spec.get("task_seed", 1729),
        split=source_spec.get("split", "test"),
        greedy_repeats=1,
        sampled_seeds=[],
        max_tokens=probe_max_tokens,
        probe=True,
        probe_max_tokens=probe_max_tokens,
        probe_checkpoints=list(checkpoints),
        capture_level="minimal",
        notes=(
            "Phase 1.8 EXPLORATORY forced-finalization replay over saved source "
            "trajectories. Not a fresh population; not a continuation under the "
            "source sampled policy."
        ),
    )
    deployment = Deployment(model=loaded, runtime=runtime, decoding=probe_decoding, label=run_label)

    source_info = {
        "kind": PROBE_RUN_KIND,
        "source_run_id": source_store.run_id,
        "source_path": str(source_path),
        "source_evidence_seal_sha256": sha256_file(str(source_path / "evidence.json")),
        "source_manifest_sha256": sha256_file(str(source_path / "manifest.json")),
        "source_source_artifact_id": source_manifest.get("source_artifact_id"),
        "source_weight_variant_id": source_manifest.get("weight_variant_id"),
        "source_deployment_id": source_manifest.get("deployment_id"),
        "source_conditions": source_manifest.get("conditions", []),
        "source_population_lock": source_manifest.get("population_lock"),
        "source_seeds_requested": seeds,
        "source_families_requested": families,
        "seal_verified": True,
    }
    design = {
        "intervention": "greedy_forced_finalization",
        "continuation_policy": (
            "fresh deterministic greedy completion from the saved prefix; NOT a "
            "continuation under the original sampled decoding policy"
        ),
        "source_reasoning_prefix_is_generated_history": True,
        "checkpoints": checkpoints,
        "include_zero": include_zero,
        "include_full_prefix": include_full,
        "probe_max_tokens": probe_max_tokens,
        "decoding": probe_decoding.to_dict(),
        "capture": capture.to_dict(),
        "native_close_sequence": native_close,
        "n_source_draws": len(draws),
        "n_planned_probes": number_of_probes,
        "draws": plan,
    }

    store = create_probe_run(
        spec=spec,
        deployment=deployment,
        conditions=[probe_decoding],
        runs_dir=runs_dir,
        source_info=source_info,
        probe_design=design,
        code_state=code_state,
        prompt_policy=spec.prompt_policy,
        run_suffix=run_suffix,
    )
    store.update_status("RUNNING")

    n_probes = 0
    probes_writer = store.open_probes()
    try:
        _progress(
            f"probe run {store.run_id}: {len(draws)} source draws, "
            f"{number_of_probes} planned probes over checkpoints {checkpoints}"
        )
        for draw, entry in zip(draws, plan, strict=True):
            lengths = [p["checkpoint"] for p in entry["planned"]]
            if not lengths:
                continue
            gold_answer = source_gold_answer(source_store, draw.task_id)
            records = run_prefix_probes_at(
                backend,
                family=draw.family,
                answer=gold_answer,
                prompt_token_ids=draw.prompt_token_ids,
                prefix_token_ids=draw.reasoning_token_ids,
                lengths=lengths,
                probe_decoding=probe_decoding,
                capture=capture,
                eos_token_ids=set(getattr(backend, "eos_token_ids", set()) or set()),
            )
            by_length = {p["checkpoint"]: p for p in entry["planned"]}
            for record in records:
                meta = by_length[record["reasoning_prefix_tokens"]]
                probes_writer.append(
                    {
                        "record_type": "forced_prefix_probe",
                        "run_id": store.run_id,
                        "run_kind": PROBE_RUN_KIND,
                        "arm_label": run_label,
                        "source_run_id": source_store.run_id,
                        "source_evidence_seal_sha256": source_info["source_evidence_seal_sha256"],
                        "source_manifest_sha256": source_info["source_manifest_sha256"],
                        "source_trace_sha256": draw.trace_sha256,
                        "source_task_id": draw.task_id,
                        "source_presentation_id": draw.presentation_id,
                        "source_family": draw.family,
                        "source_seed": draw.seed,
                        "source_mode": draw.mode,
                        "source_condition_id": draw.condition_id,
                        "source_deployment_id": draw.deployment_id,
                        "source_stochastic_draw_id": draw.stochastic_draw_id,
                        "source_trial_id": draw.trial_id,
                        "source_prompt_tokens": draw.source_prompt_tokens,
                        "source_reasoning_tokens": len(draw.reasoning_token_ids),
                        "source_native_closed": draw.native_closed,
                        "source_natural_final_status": draw.natural_final_status,
                        "source_finish_reason": draw.finish_reason,
                        "source_difficulty": draw.difficulty,
                        "gold_answer": gold_answer,
                        "checkpoint": meta["checkpoint"],
                        "checkpoint_label": meta["checkpoint_label"],
                        "checkpoint_index": meta["checkpoint_index"],
                        "checkpoint_eligible": True,
                        "is_full_prefix": meta["is_full_prefix"],
                        "prefix_token_ids_sha256": _sha256_ids(
                            draw.reasoning_token_ids[: meta["checkpoint"]]
                        ),
                        "artifact_id": deployment.artifact_id,
                        "source_artifact_id": deployment.source_artifact_id,
                        "weight_variant_id": deployment.weight_variant_id,
                        "deployment_id": deployment.deployment_id,
                        "condition_id": probe_decoding.condition_id,
                        "probe_decoding": probe_decoding.to_dict(),
                        "probe_capture": capture.to_dict(),
                        "probe_max_tokens": probe_max_tokens,
                        **record,
                    }
                )
                n_probes += 1
            _progress(
                f"  {draw.family}/{draw.task_id[:8]} seed={draw.seed}: "
                f"{len(records)} probes ({n_probes} total)"
            )
    except BaseException as exc:
        store.mark_failed(type(exc).__name__, str(exc), partial=True)
        raise
    finally:
        probes_writer.close()

    store.manifest["probe_design"]["n_completed_probes"] = n_probes
    seal = store.seal()
    return ProbeRunResult(
        store=store, n_probes=n_probes, n_source_draws=len(draws), plan=plan, seal=seal
    )


def source_gold_answer(source_store: RunStore, task_id: str) -> str:
    """Recover the exact gold answer saved with the source task."""
    for task in source_store.read_tasks():
        if task["task_id"] == task_id:
            return str(task["answer"])
    raise ReplayError(f"gold answer for source task {task_id} is not present in tasks.jsonl")


@dataclass
class BudgetDiagnosticResult:
    source_run_id: str
    arm_label: str
    task_id: str
    family: str
    seed: int | None
    checkpoint: str
    prefix_length: int
    per_budget: list[dict[str, Any]]
    source_evidence_seal_sha256: str
    source_trace_sha256: str


def run_budget_diagnostic(
    source_run_path: str | Path,
    *,
    backend: Backend,
    model_repo: str,
    model_revision: str,
    artifact_path: str | None = None,
    arm_label: str,
    task_id: str,
    seed: int,
    checkpoint: str,
    budgets: list[int],
    capture: CaptureSpec | None = None,
) -> BudgetDiagnosticResult:
    """Measure EOS completion / scorable-answer / censoring at several probe budgets.

    This is the sentinel diagnostic used to *choose and justify* the primary probe
    horizon. It runs on a single predeclared source trajectory per arm and never
    retries a failed probe until it becomes correct.
    """
    source_path = Path(source_run_path)
    source_store = RunStore.load(source_path)
    verify_source_run(source_path)
    source_manifest = source_store.manifest

    if hasattr(backend, "native_closure_ids"):
        backend.native_closure_ids()
    loaded = backend.load(
        SourceArtifactIdentity(repo_id=model_repo, revision=model_revision),
        artifact_path=artifact_path,
    )
    if loaded.source.source_artifact_id != source_manifest.get("source_artifact_id"):
        raise ReplayError("loaded source artifact does not match the source run")
    if loaded.weight_variant_id != source_manifest.get("weight_variant_id"):
        raise ReplayError("loaded weight variant does not match the source run")

    draws = [
        d for d in load_source_draws(source_store, backend, seeds=[seed]) if d.task_id == task_id
    ]
    if not draws:
        raise ReplayError(f"task {task_id} (seed {seed}) not found in {source_path}")
    draw = draws[0]
    rlen = len(draw.reasoning_token_ids)
    prefix_length = rlen if checkpoint == "full_prefix" else int(checkpoint)
    if prefix_length > rlen:
        raise ReplayError(
            f"checkpoint {checkpoint} ({prefix_length}) exceeds reasoning length {rlen}"
        )
    gold = source_gold_answer(source_store, task_id)
    capture = capture or CaptureSpec.minimal()

    per_budget: list[dict[str, Any]] = []
    for budget in budgets:
        decoding = DecodingPolicy(
            mode="greedy",
            max_tokens=int(budget),
            stop_on_eos=True,
            context={"enable_thinking": True},
        )
        records = run_prefix_probes_at(
            backend,
            family=draw.family,
            answer=gold,
            prompt_token_ids=draw.prompt_token_ids,
            prefix_token_ids=draw.reasoning_token_ids,
            lengths=[prefix_length],
            probe_decoding=decoding,
            capture=capture,
            eos_token_ids=set(getattr(backend, "eos_token_ids", set()) or set()),
        )
        record = records[0]
        per_budget.append(
            {
                "probe_max_tokens": int(budget),
                "outcome": record["outcome"],
                "answer_scorable": record["answer_scorable"],
                "answer_observed": record["answer_observed"],
                "eos_observed": record["eos_observed"],
                "finish_reason": record["finish_reason"],
                "generated_answer_tokens": len(record["generated_answer_token_ids"]),
                "correct": record["correct"],
            }
        )
    return BudgetDiagnosticResult(
        source_run_id=source_store.run_id,
        arm_label=arm_label,
        task_id=task_id,
        family=draw.family,
        seed=seed,
        checkpoint=checkpoint,
        prefix_length=prefix_length,
        per_budget=per_budget,
        source_evidence_seal_sha256=sha256_file(str(source_path / "evidence.json")),
        source_trace_sha256=draw.trace_sha256,
    )
