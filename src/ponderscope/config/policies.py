"""Versioned model-policy profiles.

A model-policy profile records an upstream-declared generation recipe for an
exact model revision, structurally and with provenance, so that a run can state
*which declared policy* it claims to reproduce instead of silently promoting
upstream values into generic PonderScope defaults.

The profile is data, not code: the JSON files under ``model_policies/`` are the
authoritative record. This module only loads and validates them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_POLICY_DIR = Path("model_policies")


class ModelPolicyError(ValueError):
    """Raised when a model-policy profile is missing or malformed."""


@dataclass(frozen=True)
class ModelPolicyProfile:
    """A loaded, versioned model-policy profile with provenance."""

    profile_id: str
    repo_id: str
    revision: str
    generation_mode: str
    recommended: dict[str, Any]
    provenance: dict[str, Any]
    mlx_mapping: dict[str, Any] = field(default_factory=dict)
    quotes: dict[str, str] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelPolicyProfile:
        for key in ("profile_id", "model", "generation_mode", "recommended", "provenance"):
            if key not in data:
                raise ModelPolicyError(f"model-policy profile missing required key: {key!r}")
        model = data["model"]
        return cls(
            profile_id=str(data["profile_id"]),
            repo_id=str(model["repo_id"]),
            revision=str(model["revision"]),
            generation_mode=str(data["generation_mode"]),
            recommended=dict(data["recommended"]),
            provenance=dict(data["provenance"]),
            mlx_mapping=dict(data.get("mlx_mapping", {})),
            quotes=dict(data.get("quotes", {})),
            raw=dict(data),
        )

    @classmethod
    def load(cls, path: str | Path) -> ModelPolicyProfile:
        path = Path(path)
        if not path.exists():
            raise ModelPolicyError(f"no model-policy profile at {path}")
        try:
            data = json.loads(path.read_text())
        except ValueError as exc:  # pragma: no cover - malformed file
            raise ModelPolicyError(f"invalid JSON in {path}: {exc}") from exc
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return self.raw

    def sampling_kwargs(self) -> dict[str, Any]:
        """The upstream-recommended sampling values, structurally."""
        return dict(self.recommended)

    @property
    def condition_label(self) -> str:
        """Honest label for the MLX mapping of this upstream profile."""
        return str(self.mlx_mapping.get("equivalence_label", "upstream-profile"))

    def provenance_record(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "generation_mode": self.generation_mode,
            "recommended": self.recommended,
            "provenance": self.provenance,
            "mlx_mapping": self.mlx_mapping,
            "condition_label": self.condition_label,
        }


def resolve_policy_path(
    profile_id_or_path: str, policy_dir: str | Path = DEFAULT_POLICY_DIR
) -> Path:
    """Resolve a profile id (or explicit path) to a JSON file, failing closed."""
    candidate = Path(profile_id_or_path)
    if candidate.suffix == ".json" and candidate.exists():
        return candidate
    path = Path(policy_dir) / f"{profile_id_or_path}.json"
    if path.exists():
        return path
    # An explicit path that does not exist is still reported as missing.
    if candidate.suffix == ".json":
        return candidate
    return path


def load_model_policy(
    profile_id_or_path: str, policy_dir: str | Path = DEFAULT_POLICY_DIR
) -> ModelPolicyProfile:
    return ModelPolicyProfile.load(resolve_policy_path(profile_id_or_path, policy_dir))
