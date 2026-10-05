"""Environment capture for evidence manifests."""

from __future__ import annotations

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


def capture_code_state() -> dict[str, Any]:
    """Version of the measurement code itself (PonderScope), for provenance."""
    from .. import __version__

    state: dict[str, Any] = {"version": __version__}
    try:
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        sha = subprocess.run(
            ["git", "-C", repo, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if sha.returncode == 0:
            state["git_sha"] = sha.stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", repo, "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if dirty.returncode == 0:
            state["git_dirty"] = bool(dirty.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return state


def _int_or_none(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None
