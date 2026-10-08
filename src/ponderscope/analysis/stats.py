"""Bootstrap and summary statistics. Pure local computation, no external judges.

Statistical unit is the **task**, not the generation. Repeated seeds/repeats from
the same task are not independent samples, so all interval estimates resample at
the task level ("cluster bootstrap"). The estimand is stated explicitly for each
helper.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
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


def classify_effect(delta: dict[str, Any], noise_scale: float | None) -> str:
    """Classify a paired delta against a within-configuration noise scale.

    - insufficient_data: no usable point estimate, or no noise estimate *and* the
      cluster-bootstrap CI includes zero.
    - ci_only: no within-deployment noise estimate is available (e.g. one
      execution per stochastic draw), but the task-clustered bootstrap CI
      excludes zero. This is explicitly NOT a noise-floor-calibrated claim; the
      uncertainty is the cluster bootstrap alone.
    - below_noise: CI includes zero, or point effect is within one noise scale.
    - comparable: effect exceeds noise but is not clearly larger (<= 2x noise).
    - clearly_larger: CI excludes zero and |mean| > 2x noise scale.
    """
    mean = delta.get("mean")
    if mean is None:
        return "insufficient_data"
    if noise_scale is None:
        return "ci_only" if delta.get("excludes_zero") else "insufficient_data"
    if not delta.get("excludes_zero"):
        return "below_noise"
    if noise_scale <= 0:
        return "comparable"
    if abs(mean) > 2 * noise_scale:
        return "clearly_larger"
    return "comparable"


def combine_noise_scales(*scales: float | None) -> float | None:
    """Combine within-deployment noise from both compared conditions.

    Uses the larger of the available scales so an effect is never called
    significant against only the quieter deployment. Returns None if no scale is
    available, which callers must treat as insufficient evidence.
    """
    usable = [s for s in scales if s is not None]
    if not usable:
        return None
    return float(max(usable))


def cluster_means(values_by_cluster: dict[str, list[float]]) -> dict[str, float]:
    """Per-cluster mean, dropping empty clusters."""
    return {k: float(np.mean(v)) for k, v in values_by_cluster.items() if v}


def cluster_bootstrap_ci(
    values_by_cluster: dict[str, list[float]],
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    """Cluster (task) bootstrap CI of the macro mean.

    Estimand: the mean over clusters of the per-cluster mean. Resampling is at
    the cluster level, preserving within-cluster seed/repeat structure.
    """
    clusters = cluster_means(values_by_cluster)
    keys = sorted(clusters)
    if not keys:
        return {"n_clusters": 0, "n_obs": 0, "mean": None, "lo": None, "hi": None}
    vals = np.array([clusters[k] for k in keys], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, vals.size, size=(n_resamples, vals.size))
    means = vals[idx].mean(axis=1)
    return {
        "n_clusters": int(vals.size),
        "n_obs": int(sum(len(v) for v in values_by_cluster.values())),
        "mean": float(vals.mean()),
        "lo": float(np.percentile(means, 100 * alpha / 2)),
        "hi": float(np.percentile(means, 100 * (1 - alpha / 2))),
    }


def paired_cluster_bootstrap_delta(
    a_by_cluster: dict[str, list[float]],
    b_by_cluster: dict[str, list[float]],
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    """Paired, task-clustered bootstrap of the macro mean difference A - B.

    Estimand: mean over matched clusters of (mean_A(cluster) - mean_B(cluster)).
    """
    a_means = cluster_means(a_by_cluster)
    b_means = cluster_means(b_by_cluster)
    keys = sorted(set(a_means) & set(b_means))
    if not keys:
        return {
            "n_clusters": 0,
            "n_obs": 0,
            "mean": None,
            "lo": None,
            "hi": None,
            "excludes_zero": None,
        }
    d = np.array([a_means[k] - b_means[k] for k in keys], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_resamples, d.size))
    means = d[idx].mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return {
        "n_clusters": int(d.size),
        "n_obs": int(sum(len(v) for v in a_by_cluster.values()))
        + int(sum(len(v) for v in b_by_cluster.values())),
        "mean": float(d.mean()),
        "lo": lo,
        "hi": hi,
        "excludes_zero": bool(lo > 0 or hi < 0),
    }


def within_cluster_std(values_by_cluster: dict[str, list[float]]) -> float | None:
    """Mean within-cluster standard deviation (noise floor for a metric)."""
    stds = [float(np.std(v, ddof=1)) for v in values_by_cluster.values() if len(v) > 1]
    if not stds:
        return None
    return float(np.mean(stds))


def group_values_by_task(
    records: list[dict[str, Any]],
    key: str,
    extract: Callable[[dict[str, Any]], float | None],
) -> dict[str, list[float]]:
    """Group extracted metric values by ``key`` (usually 'task_id').

    ``None`` extractions are dropped, not coerced to zero: a metric that is not
    estimable (e.g. conditional accuracy with no completed answer) must not be
    silently treated as an observed value.
    """
    out: dict[str, list[float]] = defaultdict(list)
    for r in records:
        value = extract(r)
        if value is None:
            continue
        out[r[key]].append(float(value))
    return dict(out)
