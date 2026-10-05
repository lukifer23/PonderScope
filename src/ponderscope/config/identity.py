"""Layered deployment identity.

Scientific identity is deliberately layered so that unrelated provenance
(human labels, local filesystem paths, sampling seeds, repeat indices) can never
silently change what is being measured.

    ArtifactIdentity   weights + tokenizer + quantization         -> artifact_id
    DeploymentIdentity artifact + runtime + hardware + OS         -> deployment_id
    ConditionIdentity  decoding policy (NO seed)                  -> condition_id
    TrialIdentity      task + condition + seed + repeat           -> trial_id

A random seed is trial state, not a deployment property. A run may contain
several decoding conditions, so a run is never described by a single
decoding-specific id; it records the artifact/deployment ids plus one
condition id per condition.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

# Bump when the identity schema changes in a way that should invalidate
# cross-version configuration-id comparisons.
IDENTITY_SCHEMA_VERSION = "ponderscope-identity/2"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _canonicalize(value: Any) -> Any:
    """Canonicalize a metadata tree so hashing is stable.

    - mappings are sorted by key
    - None and empty containers are dropped
    - floats are rounded to a fixed decimal count to avoid repr wobble
    - bytes are represented by their sha256
    """
    if isinstance(value, dict):
        out = {}
        for key in sorted(value):
            canon = _canonicalize(value[key])
            if canon is None or canon == {} or canon == []:
                continue
            out[str(key)] = canon
        return out
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [_canonicalize(v) for v in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, bytes):
        return {"sha256": sha256_bytes(value)}
    if isinstance(value, (int, str)):
        return value
    return str(value)


def canonical_json(metadata: dict[str, Any]) -> str:
    return json.dumps(_canonicalize(metadata), sort_keys=True, separators=(",", ":"))


def configuration_id(metadata: dict[str, Any], length: int = 12, prefix: str = "dep") -> str:
    """Stable short id derived from canonicalized metadata.

    The identity-schema version is part of the hash, so ids are never compared
    across schema changes.
    """
    digest = _sha256_text(IDENTITY_SCHEMA_VERSION + "\n" + canonical_json(metadata))
    return f"{prefix}-{digest[:length]}"


@dataclass(frozen=True)
class ArtifactIdentity:
    """Identity of the weights and tokenizer, independent of runtime and path.

    ``local_path`` is retained for provenance but is deliberately excluded from
    :attr:`artifact_id`: a cache path must never change artifact identity.
    """

    repo_id: str
    revision: str
    weight_files: dict[str, str] = field(default_factory=dict)  # filename -> sha256
    tokenizer_files: dict[str, str] = field(default_factory=dict)
    chat_template_sha256: str | None = None
    precision: str = "unknown"  # e.g. bfloat16, float16, float32
    quantization: str | None = None  # e.g. mlx-4bit
    quantization_bits: int | None = None
    quantization_group_size: int | None = None
    quantization_params: dict[str, Any] = field(default_factory=dict)
    local_path: str | None = None  # provenance only; excluded from the hash
    load_audit: dict[str, Any] = field(default_factory=dict)

    def identity_dict(self) -> dict[str, Any]:
        """Fields that define artifact identity (excludes path/label)."""
        d = asdict(self)
        d.pop("local_path", None)
        return d

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def artifact_id(self) -> str:
        return configuration_id(self.identity_dict(), prefix="art")


# Backwards-compatible name: the backend loads an artifact.
ModelIdentity = ArtifactIdentity


@dataclass(frozen=True)
class RuntimeIdentity:
    """Identity of the software/hardware stack executing the weights."""

    runtime: str  # e.g. mlx-lm
    runtime_version: str
    backend: str  # e.g. mlx-metal
    hardware: str
    os: str
    python_version: str
    device: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DecodingPolicy:
    """Decoding/sampling/context configuration.

    ``seed`` is trial state, not condition identity: it is excluded from
    :attr:`condition_id`. Changing temperature/top-p/top-k/min-p/budget does
    change condition identity.
    """

    mode: str  # "greedy" | "sampled"
    max_tokens: int
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    seed: int | None = None
    stop_on_eos: bool = True
    context: dict[str, Any] = field(default_factory=dict)

    def identity_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("seed", None)
        return d

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def condition_id(self) -> str:
        return configuration_id(self.identity_dict(), prefix="cond")


@dataclass(frozen=True)
class TrialIdentity:
    """Identity of a single generation: task + condition + seed + repeat."""

    task_id: str
    condition: DecodingPolicy
    seed: int | None
    repeat: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "condition": self.condition.to_dict(),
            "seed": self.seed,
            "repeat": self.repeat,
        }

    @property
    def trial_id(self) -> str:
        return configuration_id(self.to_dict(), prefix="trial")


@dataclass(frozen=True)
class DeploymentIdentity:
    """artifact + runtime/hardware. Decoding is deliberately excluded."""

    artifact: ArtifactIdentity
    runtime: RuntimeIdentity

    def to_dict(self) -> dict[str, Any]:
        return {"artifact": self.artifact.to_dict(), "runtime": self.runtime.to_dict()}

    @property
    def deployment_id(self) -> str:
        return configuration_id(
            {"artifact": self.artifact.identity_dict(), "runtime": self.runtime.to_dict()},
            prefix="dep",
        )


@dataclass(frozen=True)
class Deployment:
    """Convenience composite: artifact + runtime + a decoding condition.

    Retained so callers can describe a concrete execution. It exposes the
    layered ids explicitly; it no longer fabricates a single overloaded
    ``config_id``.
    """

    model: ArtifactIdentity
    runtime: RuntimeIdentity
    decoding: DecodingPolicy
    label: str | None = None
    identity_schema: str = IDENTITY_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity_schema": self.identity_schema,
            "label": self.label,
            "model": self.model.to_dict(),
            "runtime": self.runtime.to_dict(),
            "decoding": self.decoding.to_dict(),
        }

    @property
    def artifact_id(self) -> str:
        return self.model.artifact_id

    @property
    def deployment_id(self) -> str:
        return self.identity().deployment_id

    @property
    def condition_id(self) -> str:
        return self.decoding.condition_id

    def identity(self) -> DeploymentIdentity:
        return DeploymentIdentity(artifact=self.model, runtime=self.runtime)

    def describe(self) -> str:
        m = self.model
        quant = ""
        if m.quantization:
            quant = f" quant={m.quantization}"
            if m.quantization_bits:
                quant += f"{m.quantization_bits}bit"
            if m.quantization_group_size:
                quant += f" g{m.quantization_group_size}"
        return (
            f"{m.repo_id}@{m.revision[:12]} ({m.precision}{quant}) "
            f"under {self.runtime.runtime} {self.runtime.runtime_version} "
            f"[{self.runtime.backend}, {self.runtime.hardware}] "
            f"{self.decoding.mode} art={self.artifact_id} dep={self.deployment_id} "
            f"cond={self.condition_id}"
        )


def host_runtime_identity(
    runtime: str, runtime_version: str, backend: str, device: str | None = None
) -> RuntimeIdentity:
    """Capture the current host identity. Runtime fields are passed by the backend."""
    return RuntimeIdentity(
        runtime=runtime,
        runtime_version=runtime_version,
        backend=backend,
        hardware=platform.processor() or platform.machine(),
        os=f"{platform.system()} {platform.release()}",
        python_version=sys.version.split()[0],
        device=device,
    )
