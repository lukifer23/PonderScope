"""Design-planning precision simulation.

A deterministic, seeded Monte Carlo used to estimate the achievable precision of
a paired, task-clustered comparison *before* any held-out model outcome is
observed. It uses no model data: assumptions are stated explicitly and the whole
computation is reproducible from the recorded seed.

This is planning information, not a measurement of model performance.
"""

from __future__ import annotations

import numpy as np


def simulate_closure_precision(
    *,
    n_tasks: int = 30,
    n_seeds: int = 2,
    p0: float = 0.5,
    delta: float = 0.0,
    between_task_sd: float = 0.15,
    n_monte_carlo: int = 300,
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Mean 95% task-clustered bootstrap CI half-width for a closure-rate delta.

    Model: each task has a baseline closure probability drawn from
    ``Normal(p0, between_task_sd)`` (clipped to [0,1]); each seed yields a
    Bernoulli draw in each arm, with the treatment arm shifted by ``delta``.
    The interval is a percentile bootstrap that resamples task clusters.
    """
    rng = np.random.default_rng(seed)
    half_widths: list[float] = []
    for _ in range(n_monte_carlo):
        p_task = np.clip(rng.normal(p0, between_task_sd, size=n_tasks), 0.0, 1.0)
        a = rng.random((n_tasks, n_seeds)) < p_task[:, None]
        b = rng.random((n_tasks, n_seeds)) < np.clip(p_task + delta, 0.0, 1.0)[:, None]
        d = a.mean(axis=1) - b.mean(axis=1)
        idx = rng.integers(0, n_tasks, size=(n_resamples, n_tasks))
        means = d[idx].mean(axis=1)
        lo, hi = np.percentile(means, [2.5, 97.5])
        half_widths.append((hi - lo) / 2.0)
    arr = np.asarray(half_widths)
    return {
        "mean_ci_half_width": float(arr.mean()),
        "median_ci_half_width": float(np.median(arr)),
        "p90_ci_half_width": float(np.percentile(arr, 90)),
    }


def simulate_rmst_precision(
    *,
    n_tasks: int = 30,
    per_task_sd_tokens: float = 250.0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Analytic 95% CI half-width for a paired per-task RMST difference.

    SE = per_task_sd / sqrt(n_tasks); half-width = z * SE. This is the standard
    normal-approximation precision for a paired cluster mean difference.
    """
    from statistics import NormalDist

    z = NormalDist().inv_cdf(1 - alpha / 2)
    se = per_task_sd_tokens / np.sqrt(n_tasks)
    return {
        "se_tokens": float(se),
        "ci_half_width_tokens": float(z * se),
    }


def design_plan(
    *,
    seed: int = 0,
    p0: float = 0.5,
    delta: float = 0.0,
    between_task_sd: float = 0.15,
    n_monte_carlo: int = 300,
    n_resamples: int = 2000,
    per_task_sd_tokens: float = 250.0,
    task_sizes: tuple[int, ...] = (30, 50, 100),
    seeds_per_task: int = 2,
) -> dict:
    """Full design plan across a sensitivity grid of task counts."""
    sensitivity = []
    for n_tasks in task_sizes:
        closure = simulate_closure_precision(
            n_tasks=n_tasks,
            n_seeds=seeds_per_task,
            p0=p0,
            delta=delta,
            between_task_sd=between_task_sd,
            n_monte_carlo=n_monte_carlo,
            n_resamples=n_resamples,
            seed=seed,
        )
        rmst = simulate_rmst_precision(n_tasks=n_tasks, per_task_sd_tokens=per_task_sd_tokens)
        sensitivity.append(
            {
                "n_tasks": n_tasks,
                "n_draws_per_arm": n_tasks * seeds_per_task,
                "closure_ci_half_width": closure["mean_ci_half_width"],
                "rmst_ci_half_width_tokens": rmst["ci_half_width_tokens"],
            }
        )
    return {
        "kind": "design-planning (not model data)",
        "assumptions": {
            "closure_baseline_p0": p0,
            "true_closure_delta": delta,
            "between_task_sd": between_task_sd,
            "within_task": "Bernoulli per seed",
            "seeds_per_task": seeds_per_task,
            "monte_carlo_replicates": n_monte_carlo,
            "bootstrap_resamples": n_resamples,
            "interval_method": "task-clustered percentile bootstrap",
            "rmst_per_task_sd_tokens": per_task_sd_tokens,
            "simulation_seed": seed,
        },
        "sensitivity": sensitivity,
        "note": (
            "Planning estimates under stated assumptions; not a measurement of model "
            "performance. A null result is not evidence of equivalence."
        ),
    }
