"""Run directories: create-once, immutable, atomic evidence storage."""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config.identity import Deployment
from ..config.schema import ExperimentSpec
from .environment import capture_code_state, capture_environment
from .store import (
    JsonlWriter,
    atomic_write_json,
    atomic_write_text,
    read_json,
    read_jsonl,
)

RUNS_DIRNAME = "runs"
MANIFEST = "manifest.json"
ENVIRONMENT = "environment.json"
TASKS = "tasks.jsonl"
TRACES = "traces.jsonl"
PROBES = "probes.jsonl"
ANALYSIS = "analysis.json"
SUMMARY_MD = "summary.md"
SUMMARY_HTML = "summary.html"


def _utc_now() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _slug(text: str) -> str:
    keep = [c.lower() if c.isalnum() else "-" for c in text]
    slug = "".join(keep).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "run"


@dataclass
class RunStore:
    """Handle to a single immutable run directory."""

    path: Path
    manifest: dict[str, Any]

    # -- ids -----------------------------------------------------------------
    @property
    def run_id(self) -> str:
        return self.manifest["run_id"]

    @property
    def config_id(self) -> str:
        return self.manifest["config_id"]

    # -- creation ------------------------------------------------------------
    @classmethod
    def create(
        cls,
        deployment: Deployment,
        spec: ExperimentSpec,
        runs_dir: str | Path = RUNS_DIRNAME,
        created_utc: str | None = None,
        run_suffix: str | None = None,
    ) -> RunStore:
        runs_dir = Path(runs_dir)
        created_utc = created_utc or _utc_now()
        suffix = f"-{_slug(run_suffix)}" if run_suffix else ""
        run_id = f"{created_utc}-{_slug(spec.name)}{suffix}-{deployment.config_id}"
        path = runs_dir / run_id
        if path.exists():
            raise FileExistsError(f"run already exists and will not be overwritten: {path}")
        path.mkdir(parents=True)
        environment = capture_environment()
        manifest = {
            "run_id": run_id,
            "created_utc": created_utc,
            "name": spec.name,
            "config_id": deployment.config_id,
            "deployment": deployment.to_dict(),
            "deployment_description": deployment.describe(),
            "spec": spec.to_dict(),
            "code": capture_code_state(),
            "status": {
                "run": "IN_PROGRESS",
                "observations": [],
            },
        }
        atomic_write_json(path / MANIFEST, manifest)
        atomic_write_json(path / ENVIRONMENT, environment)
        return cls(path=path, manifest=manifest)

    # -- loading -------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> RunStore:
        path = Path(path)
        manifest = read_json(path / MANIFEST)
        return cls(path=path, manifest=manifest)

    @classmethod
    def latest(cls, runs_dir: str | Path = RUNS_DIRNAME) -> RunStore:
        runs_dir = Path(runs_dir)
        candidates = [p for p in runs_dir.iterdir() if (p / MANIFEST).exists()]
        if not candidates:
            raise FileNotFoundError(f"no runs found in {runs_dir}")
        candidates.sort(key=lambda p: (p.stat().st_mtime_ns, p.name))
        return cls.load(candidates[-1])

    # -- evidence ------------------------------------------------------------
    def update_status(self, status: str, observations: list[str] | None = None) -> None:
        self.manifest["status"]["run"] = status
        if observations:
            self.manifest["status"]["observations"].extend(observations)
        atomic_write_json(self.path / MANIFEST, self.manifest)

    def add_observation(self, text: str) -> None:
        self.manifest["status"]["observations"].append(text)
        atomic_write_json(self.path / MANIFEST, self.manifest)

    def write_tasks(self, tasks: list[dict[str, Any]]) -> None:
        from .store import write_jsonl

        write_jsonl(self.path / TASKS, tasks)

    def read_tasks(self) -> list[dict[str, Any]]:
        return list(read_jsonl(self.path / TASKS))

    def open_traces(self) -> JsonlWriter:
        return JsonlWriter(self.path / TRACES).open()

    def read_traces(self) -> list[dict[str, Any]]:
        return list(read_jsonl(self.path / TRACES))

    def open_probes(self) -> JsonlWriter:
        return JsonlWriter(self.path / PROBES).open()

    def read_probes(self) -> list[dict[str, Any]]:
        path = self.path / PROBES
        return list(read_jsonl(path)) if path.exists() else []

    def write_analysis(self, analysis: dict[str, Any]) -> None:
        atomic_write_json(self.path / ANALYSIS, analysis)

    def read_analysis(self) -> dict[str, Any]:
        return read_json(self.path / ANALYSIS)

    def write_summary(self, markdown: str, html: str) -> None:
        atomic_write_text(self.path / SUMMARY_MD, markdown)
        atomic_write_text(self.path / SUMMARY_HTML, html)

    def has_analysis(self) -> bool:
        return (self.path / ANALYSIS).exists()

    def to_dict(self) -> dict[str, Any]:
        return {"path": str(self.path), "run_id": self.run_id, "config_id": self.config_id}


def load_manifest_text(path: str | Path) -> str:
    return json.dumps(read_json(Path(path) / MANIFEST), indent=2)
