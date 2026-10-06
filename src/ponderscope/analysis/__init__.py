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
from .draws import (
    collapse_to_stochastic_draws,
    draw_summary,
    draws_by_condition,
)
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
from .survival import (
    kaplan_meier,
    observations_from_records,
    rmst,
    rmst_delta_clustered,
    survival_by_family,
    survival_summary,
)

__all__ = [
    "across_seed_variation",
    "analyze_run",
    "bootstrap_ci",
    "classify_effect",
    "cluster_bootstrap_ci",
    "collapse_to_stochastic_draws",
    "combine_noise_scales",
    "compare_configs",
    "cross_seed_token_variation",
    "draw_summary",
    "draws_by_condition",
    "generate_report",
    "greedy_replay_variation",
    "kaplan_meier",
    "noise_floor",
    "observations_from_records",
    "paired_bootstrap_delta",
    "paired_cluster_bootstrap_delta",
    "prefix_invariance",
    "rmst",
    "rmst_delta_clustered",
    "same_seed_replay_variation",
    "summarize",
    "survival_by_family",
    "survival_summary",
]
