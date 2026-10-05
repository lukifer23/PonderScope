"""Analysis of saved evidence."""

from .analyze import analyze_run, greedy_determinism, sampled_noise
from .compare import compare_configs
from .report import generate_report
from .stats import (
    bootstrap_ci,
    classify_effect,
    paired_bootstrap_delta,
    summarize,
)

__all__ = [
    "analyze_run",
    "bootstrap_ci",
    "classify_effect",
    "compare_configs",
    "generate_report",
    "greedy_determinism",
    "paired_bootstrap_delta",
    "sampled_noise",
    "summarize",
]
