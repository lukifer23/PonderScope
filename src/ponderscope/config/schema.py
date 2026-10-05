"""Configuration schemas for PonderScope."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .identity import DecodingPolicy, Deployment, ModelIdentity, RuntimeIdentity

__all__ = [
    "DecodingPolicy",
    "Deployment",
    "ModelIdentity",
    "RuntimeIdentity",
    "ExperimentSpec",
]


@dataclass(frozen=True)
class ExperimentSpec:
    """A declared experiment: which deployment, which tasks, how many repeats."""

    name: str
    task_pack: str
    task_versions: dict[str, str] = field(default_factory=dict)
    families: list[str] = field(default_factory=list)
    n_per_family: int = 4
    task_seed: int = 0
    split: str = "dev"
    greedy_repeats: int = 1
    sampled_seeds: list[int] = field(default_factory=list)
    sampled_repeats_per_seed: int = 1
    max_tokens: int = 512
    sampled_temperature: float = 0.6
    sampled_top_p: float = 0.95
    sampled_top_k: int = 20
    probe: bool = False
    n_probes: int = 4
    probe_max_tokens: int = 64
    capture_entropy: bool = True
    capture_top_k: int = 5
    capture_logprob_digest: bool = False
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "task_pack": self.task_pack,
            "task_versions": self.task_versions,
            "families": self.families,
            "n_per_family": self.n_per_family,
            "task_seed": self.task_seed,
            "split": self.split,
            "greedy_repeats": self.greedy_repeats,
            "sampled_seeds": self.sampled_seeds,
            "sampled_repeats_per_seed": self.sampled_repeats_per_seed,
            "max_tokens": self.max_tokens,
            "sampled_temperature": self.sampled_temperature,
            "sampled_top_p": self.sampled_top_p,
            "sampled_top_k": self.sampled_top_k,
            "probe": self.probe,
            "n_probes": self.n_probes,
            "probe_max_tokens": self.probe_max_tokens,
            "capture_entropy": self.capture_entropy,
            "capture_top_k": self.capture_top_k,
            "capture_logprob_digest": self.capture_logprob_digest,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentSpec:
        import dataclasses

        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})
