"""Layered deployment identity.

Scientific identity is deliberately layered so that unrelated provenance
(human labels, local filesystem paths, sampling seeds, repeat indices, and
runtime load audits) can never silently change what is being measured:

    SourceArtifactIdentity  repo + immutable revision + source weight/tokenizer/
                            template hashes                          -> src id
    WeightVariantIdentity   an actual executable representation of a source
                            artifact: original OR a derived quantized/converted
                            variant, with explicit lineage            -> wvar id
    DeploymentIdentity      weight variant + runtime + hardware + OS  -> dep id
    ConditionIdentity       decoding policy (NO seed)                 -> cond id
    TrialIdentity           task + condition + seed + repeat          -> trial id

A backend-specific *load audit* is evidence about how a runtime interpreted an
artifact; it is deliberately excluded from both the source and weight-variant
identity, so the same weights can never receive two different artifact ids
because a different backend audited them.
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
IDENTITY_SCHEMA_VERSION = "ponderscope-identity/3"


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
class SourceArtifactIdentity:
    """Immutable upstream source: repo, revision, and source file hashes.

    Independent of runtime, path, quantization, and load audit.
    """

    repo_id: str
    revision: str
    weight_files: dict[str, str] = field(default_factory=dict)
    tokenizer_files: dict[str, str] = field(default_factory=dict)
    chat_template_sha256: str | None = None

    def identity_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def source_artifact_id(self) -> str:
        return configuration_id(self.identity_dict(), prefix="src")


@dataclass(frozen=True)
class WeightVariantIdentity:
    """An actual executable weight representation.

    For a native checkpoint ``representation == "original"`` and the variant
    weight hashes equal the source hashes. For a converted/quantized variant
    ``representation == "derived"``, ``variant_weight_files`` are the actual
    files that load, and ``derived_from_source_artifact_id`` records lineage.
    """

    source: SourceArtifactIdentity
    representation: str = "original"  # "original" | "derived"
    variant_weight_files: dict[str, str] = field(default_factory=dict)
    precision: str = "unknown"
    quantization: str | None = None
    quantization_bits: int | None = None
    quantization_group_size: int | None = None
    quantization_params: dict[str, Any] = field(default_factory=dict)
    conversion: dict[str, Any] = field(default_factory=dict)
    local_path: str | None = None  # provenance only; excluded from the hash
    load_audit: dict[str, Any] = field(default_factory=dict)  # excluded from the hash

    @property
    def repo_id(self) -> str:
        return self.source.repo_id

    @property
    def revision(self) -> str:
        return self.source.revision

    @property
    def derived_from_source_artifact_id(self) -> str | None:
        return None if self.representation == "original" else self.source.source_artifact_id

    def identity_dict(self) -> dict[str, Any]:
        """Fields that define weight-variant identity (excludes path/audit)."""
        d = asdict(self)
        d.pop("local_path", None)
        d.pop("load_audit", None)
        d["derived_from_source_artifact_id"] = self.derived_from_source_artifact_id
        return d

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def weight_variant_id(self) -> str:
        return configuration_id(self.identity_dict(), prefix="wvar")


# Backwards-compatible aliases: the loaded artifact is now a weight variant.
ModelIdentity = WeightVariantIdentity
ArtifactIdentity = WeightVariantIdentity


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
    """weight variant + runtime/hardware. Decoding is deliberately excluded."""

    variant: WeightVariantIdentity
    runtime: RuntimeIdentity

    def to_dict(self) -> dict[str, Any]:
        return {"weight_variant": self.variant.to_dict(), "runtime": self.runtime.to_dict()}

    @property
    def deployment_id(self) -> str:
        return configuration_id(
            {"weight_variant": self.variant.identity_dict(), "runtime": self.runtime.to_dict()},
            prefix="dep",
        )


@dataclass(frozen=True)
class Deployment:
    """Convenience composite: weight variant + runtime + a decoding condition."""

    model: WeightVariantIdentity
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
    def source_artifact_id(self) -> str:
        return self.model.source.source_artifact_id

    @property
    def weight_variant_id(self) -> str:
        return self.model.weight_variant_id

    @property
    def artifact_id(self) -> str:
        # Backwards-compatible alias for the executable weight variant.
        return self.weight_variant_id

    @property
    def deployment_id(self) -> str:
        return self.identity().deployment_id

    @property
    def condition_id(self) -> str:
        return self.decoding.condition_id

    def identity(self) -> DeploymentIdentity:
        return DeploymentIdentity(variant=self.model, runtime=self.runtime)

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
            f"{m.repo_id}@{m.revision[:12]} ({m.representation}, {m.precision}{quant}) "
            f"under {self.runtime.runtime} {self.runtime.runtime_version} "
            f"[{self.runtime.backend}, {self.runtime.hardware}] "
            f"{self.decoding.mode} src={self.source_artifact_id} "
            f"wvar={self.weight_variant_id} dep={self.deployment_id} cond={self.condition_id}"
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
