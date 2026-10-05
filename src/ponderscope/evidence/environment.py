"""Environment capture for evidence manifests."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import subprocess
import sys
from typing import Any


def _pkg_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _sysctl(key: str) -> str | None:
    try:
        out = subprocess.run(
            ["sysctl", "-n", key], capture_output=True, text=True, timeout=5, check=False
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return None


def capture_environment() -> dict[str, Any]:
    """Best-effort, read-only snapshot of the execution environment."""
    env: dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "os": f"{platform.system()} {platform.release()}",
        "cpu_brand": _sysctl("machdep.cpu.brand_string"),
        "physical_memory_bytes": _int_or_none(_sysctl("hw.memsize")),
        "packages": {
            name: _pkg_version(name)
            for name in (
                "mlx",
                "mlx-lm",
                "mlx-metal",
                "numpy",
                "transformers",
                "tokenizers",
                "huggingface-hub",
                "safetensors",
                "ponderscope",
            )
        },
    }
    # MLX device, if importable.
    try:
        import mlx.core as mx

        env["mlx_default_device"] = str(mx.default_device())
        env["mlx_version"] = getattr(mx, "__version__", None)
    except Exception as exc:  # pragma: no cover - depends on host
        env["mlx_error"] = f"{type(exc).__name__}: {exc}"
    env["cwd"] = os.getcwd()
    return env


def _git(repo: str, *args: str) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            ["git", "-C", repo, *args],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def capture_code_state(include_diff_hash: bool = True) -> dict[str, Any]:
    """Version and cleanliness of the measurement code itself, for provenance.

    ``tracked_dirty`` is the official-run criterion: it ignores untracked
    evidence output under ``runs/``/``scratch/`` (which any run necessarily
    creates) and reports only tracked source modifications. ``git_dirty`` retains
    the stricter "any porcelain output" meaning for transparency. When tracked
    files are modified, a deterministic SHA-256 of ``git diff HEAD`` is recorded.
    """
    from .. import __version__

    state: dict[str, Any] = {"version": __version__}
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    sha = _git(repo, "rev-parse", "HEAD")
    if sha is not None and sha.returncode == 0:
        state["git_sha"] = sha.stdout.strip()

    porcelain = _git(repo, "status", "--porcelain")
    if porcelain is not None and porcelain.returncode == 0:
        entries = [line for line in porcelain.stdout.splitlines() if line.strip()]
        state["git_dirty"] = bool(entries)
        state["untracked_evidence_count"] = sum(
            1
            for line in entries
            if line.startswith("?? ") and line[3:].split("/")[0] in {"runs", "scratch"}
        )

    tracked = _git(repo, "status", "--porcelain", "--untracked-files=no")
    if tracked is not None and tracked.returncode == 0:
        tracked_entries = [line for line in tracked.stdout.splitlines() if line.strip()]
        state["tracked_dirty"] = bool(tracked_entries)
        state["tracked_changes"] = tracked_entries[:50]

    if include_diff_hash and state.get("tracked_dirty"):
        diff = _git(repo, "diff", "HEAD")
        if diff is not None and diff.returncode == 0:
            state["working_tree_diff_sha256"] = hashlib.sha256(
                diff.stdout.encode("utf-8")
            ).hexdigest()
    return state


def _int_or_none(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None
