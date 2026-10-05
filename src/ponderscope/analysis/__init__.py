"""Analysis of saved evidence."""

from .analyze import (
    across_seed_variation,
    analyze_run,
    cross_seed_token_variation,
    greedy_replay_variation,
    noise_floor,
    prefix_invariance,
    same_seed_replay_variation,
)
from .compare import compare_configs
from .report import generate_report
from .stats import (
    bootstrap_ci,
    classify_effect,
    cluster_bootstrap_ci,
    combine_noise_scales,
    paired_bootstrap_delta,
    paired_cluster_bootstrap_delta,
    summarize,
)

__all__ = [
    "across_seed_variation",
    "analyze_run",
    "bootstrap_ci",
    "classify_effect",
    "cluster_bootstrap_ci",
    "combine_noise_scales",
    "compare_configs",
    "cross_seed_token_variation",
    "generate_report",
    "greedy_replay_variation",
    "noise_floor",
    "paired_bootstrap_delta",
    "paired_cluster_bootstrap_delta",
    "prefix_invariance",
    "same_seed_replay_variation",
    "summarize",
]
