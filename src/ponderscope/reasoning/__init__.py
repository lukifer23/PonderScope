"""Reasoning trajectory parsing and measurement."""

from .metrics import RepetitionMetrics, TraceMetrics, compute_repetition
from .parse import (
    ParsedReasoning,
    ParsedTrace,
    extract_answer,
    parse_reasoning,
    parse_trace,
    reasoning_prefix_ids,
    split_channels,
    strip_special_tokens,
)
from .probes import run_prefix_probes
from .transitions import (
    NATURAL_FINAL_STATUSES,
    PREFIX_STATES,
    STATES,
    TrajectoryState,
    classify_transitions,
    natural_final_correctness,
    natural_final_status_for_record,
    natural_final_status_from_termination,
    prefix_lengths,
)

__all__ = [
    "NATURAL_FINAL_STATUSES",
    "PREFIX_STATES",
    "STATES",
    "ParsedReasoning",
    "ParsedTrace",
    "RepetitionMetrics",
    "TraceMetrics",
    "TrajectoryState",
    "classify_transitions",
    "compute_repetition",
    "extract_answer",
    "natural_final_correctness",
    "natural_final_status_for_record",
    "natural_final_status_from_termination",
    "parse_reasoning",
    "parse_trace",
    "prefix_lengths",
    "reasoning_prefix_ids",
    "run_prefix_probes",
    "split_channels",
    "strip_special_tokens",
]
