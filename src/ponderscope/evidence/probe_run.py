"""Probe-run evidence storage (Phase 1.8).

A prefix-probe replay run is a first-class, *create-once*, sealed evidence
directory built with the same :class:`~ponderscope.evidence.run.RunStore`
machinery as a generation run: atomic writes, a final-manifest seal, structural
verification, bundling and offline reanalysis. It never appends to, or mutates,
the historical source run it reads.

The only new manifest sections are ``run_kind``, ``prefix_probe_source`` and
``probe_design``. No new raw-file type is introduced, so historical seals (which
cover ``environment.json``/``tasks.jsonl``/``traces.jsonl``/``probes.jsonl``)
continue to verify unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config.identity import DecodingPolicy, Deployment
from ..config.schema import ExperimentSpec
from .run import RunStore

PROBE_RUN_KIND = "prefix_probe_replay"


def create_probe_run(
    *,
    spec: ExperimentSpec,
    deployment: Deployment,
    conditions: list[DecodingPolicy],
    runs_dir: str | Path,
    source_info: dict[str, Any],
    probe_design: dict[str, Any],
    code_state: dict[str, Any] | None,
    prompt_policy: str | None,
    run_suffix: str | None = None,
) -> RunStore:
    """Create a sealed-ready probe evidence directory (not yet sealed)."""
    store = RunStore.create(
        deployment,
        spec,
        runs_dir=runs_dir,
        run_suffix=run_suffix,
        conditions=conditions,
        code_state=code_state,
        prompt_policy=prompt_policy,
    )
    store.manifest["run_kind"] = PROBE_RUN_KIND
    store.manifest["prefix_probe_source"] = source_info
    store.manifest["probe_design"] = probe_design
    store.add_observation(
        f"prefix-probe replay run over source {source_info.get('source_run_id')}; "
        f"intervention=greedy forced finalization; this run never mutates its source"
    )
    return store
