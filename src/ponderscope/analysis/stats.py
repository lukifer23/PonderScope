"""Bootstrap and summary statistics. Pure local computation, no external judges."""

from __future__ import annotations

from typing import Any

import numpy as np


def summarize(values: list[float] | np.ndarray) -> dict[str, float | int | None]:
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "p10": None,
            "p90": None,
        }
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


def bootstrap_ci(
    values: list[float] | np.ndarray,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float] | None:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    means = arr[idx].mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


def paired_bootstrap_delta(
    a: list[float] | np.ndarray,
    b: list[float] | np.ndarray,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    """Paired delta = a - b over matched observations."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.size != y.size:
        raise ValueError(f"paired arrays differ in length: {x.size} != {y.size}")
    if x.size == 0:
        return {"n": 0, "mean": None, "lo": None, "hi": None, "excludes_zero": None}
    d = x - y
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_resamples, d.size))
    means = d[idx].mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return {
        "n": int(d.size),
        "mean": float(d.mean()),
        "lo": lo,
        "hi": hi,
        "excludes_zero": bool(lo > 0 or hi < 0),
    }


def classify_effect(delta: dict[str, Any], noise_scale: float) -> str:
    """Classify a paired delta against a within-configuration noise scale.

    - below_noise: CI includes zero, or point effect is within one noise scale.
    - comparable: effect exceeds noise but is not clearly larger (<= 2x noise).
    - clearly_larger: CI excludes zero and |mean| > 2x noise scale.
    """
    mean = delta.get("mean")
    if mean is None:
        return "insufficient_data"
    if not delta.get("excludes_zero"):
        return "below_noise"
    if noise_scale is None or noise_scale <= 0:
        return "comparable"
    if abs(mean) > 2 * noise_scale:
        return "clearly_larger"
    return "comparable"
