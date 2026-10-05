"""Canonical deployment identity.

A deployment is never identified by a model name alone. Every execution
condition is reduced to a canonical metadata mapping, hashed to a stable
configuration id. Reports compare deployment configurations, not vague model
names.
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
IDENTITY_SCHEMA_VERSION = "ponderscope-identity/1"


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


def configuration_id(metadata: dict[str, Any], length: int = 12) -> str:
    """Stable short id derived from canonicalized deployment metadata."""
    digest = _sha256_text(IDENTITY_SCHEMA_VERSION + "\n" + canonical_json(metadata))
    return f"dep-{digest[:length]}"


@dataclass(frozen=True)
class ModelIdentity:
    """Identity of the weights and tokenizer, independent of the runtime."""

    repo_id: str
    revision: str
    local_path: str | None = None
    weight_files: dict[str, str] = field(default_factory=dict)  # filename -> sha256
    tokenizer_files: dict[str, str] = field(default_factory=dict)
    chat_template_sha256: str | None = None
    precision: str = "unknown"  # e.g. bfloat16, float16, float32
    quantization: str | None = None  # e.g. mlx-4bit
    quantization_bits: int | None = None
    quantization_group_size: int | None = None
    quantization_params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    """Decoding/sampling/context configuration."""

    mode: str  # "greedy" | "sampled"
    max_tokens: int
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    seed: int | None = None
    stop_on_eos: bool = True
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Deployment:
    """A complete deployment identity: model + runtime + decoding policy."""

    model: ModelIdentity
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
    def config_id(self) -> str:
        return configuration_id(self.to_dict())

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
            f"{self.decoding.mode} cfg={self.config_id}"
        )


def host_runtime_identity(
    runtime: str, runtime_version: str, backend: str, device: str | None = None
):
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
