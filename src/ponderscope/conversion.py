"""Reproducible MLX weight-variant conversion.

This module is the only first-party path that produces a *derived* executable
weight representation (currently an affine Q4 MLX artifact) from an immutable
upstream source snapshot. It records full provenance so that the derived
artifact's scientific identity is reproducible and its lineage points at the
exact source artifact.

Design rules:

- The upstream source revision is resolved to an immutable local snapshot; the
  source weight/tokenizer/chat-template hashes are recorded.
- The output directory is create-once: a different existing artifact is never
  overwritten.
- Available storage is checked before a multi-gigabyte conversion.
- The conversion command/API, tool version, requested parameters, and the
  *actual* detected quantization scheme (which may include excluded or
  higher-precision modules) are recorded. A nominal "Q4" artifact is never
  claimed to be uniformly four-bit.
- The derived artifact is validated by a real load, not asserted.

Nothing here runs a model unless explicitly invoked.
"""

from __future__ import annotations

import datetime as _dt
import json
import shutil
from pathlib import Path
from typing import Any

from .backends.mlx_backend import (
    detect_precision,
    detect_quantization,
    hash_snapshot_files,
    resolve_local_snapshot,
)
from .config.identity import SourceArtifactIdentity, WeightVariantIdentity

VARIANT_SCHEMA = "ponderscope-weight-variant/1"
VARIANT_PROVENANCE = "ponderscope_variant.json"


class ConversionError(RuntimeError):
    """Raised when a conversion cannot proceed safely."""


def _utc_now() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def source_identity_from_snapshot(
    repo_id: str, revision: str, snapshot: Path
) -> SourceArtifactIdentity:
    """Build the immutable source identity by hashing the local snapshot."""
    weights, tokenizers = hash_snapshot_files(snapshot)
    chat = snapshot / "chat_template.jinja"
    from .config.identity import sha256_file

    chat_sha = sha256_file(str(chat)) if chat.exists() else None
    return SourceArtifactIdentity(
        repo_id=repo_id,
        revision=revision,
        weight_files=weights,
        tokenizer_files=tokenizers,
        chat_template_sha256=chat_sha,
    )


def _dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def check_storage(snapshot: Path, output_dir: Path, *, factor: float = 1.0) -> dict[str, Any]:
    """Require enough free space for the derived artifact before converting."""
    output_dir = Path(output_dir)
    probe = output_dir
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    if not probe.exists():
        raise ConversionError(f"no existing ancestor directory for output {output_dir}")
    usage = shutil.disk_usage(str(probe))
    source_bytes = _dir_size(snapshot)
    required = int(source_bytes * factor)
    if usage.free < required:
        raise ConversionError(
            f"insufficient free space at {probe}: free={usage.free} required>={required} "
            f"(source={source_bytes} bytes, factor={factor})"
        )
    return {
        "free_bytes": usage.free,
        "source_bytes": source_bytes,
        "required_bytes": required,
    }


def _mlx_lm_version() -> str:
    import mlx_lm

    return str(getattr(mlx_lm, "__version__", "unknown"))


def convert_q4_variant(
    repo_id: str,
    revision: str,
    output_dir: str | Path,
    *,
    bits: int = 4,
    group_size: int = 64,
    mode: str = "affine",
    dtype: str = "bfloat16",
    storage_factor: float = 1.0,
) -> dict[str, Any]:
    """Convert the pinned source snapshot to an MLX affine quantized variant.

    Returns the provenance record (also written to ``output_dir``). Raises
    :class:`ConversionError` on any unsafe precondition.
    """
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ConversionError(
            f"refusing to overwrite an existing artifact at {output_dir}; "
            "choose a new output directory or remove the old artifact explicitly"
        )
    snapshot = resolve_local_snapshot(repo_id, revision)
    storage = check_storage(snapshot, output_dir, factor=storage_factor)
    source = source_identity_from_snapshot(repo_id, revision, snapshot)

    from mlx_lm.convert import convert

    params = {
        "quantize": True,
        "q_bits": bits,
        "q_group_size": group_size,
        "q_mode": mode,
        "dtype": dtype,
    }
    command = (
        f"mlx_lm convert --hf-path {snapshot} --mlx-path {output_dir} "
        f"-q --q-bits {bits} --q-group-size {group_size} --q-mode {mode} --dtype {dtype}"
    )
    convert(
        hf_path=str(snapshot),
        mlx_path=str(output_dir),
        quantize=True,
        q_bits=bits,
        q_group_size=group_size,
        q_mode=mode,
        dtype=dtype,
    )

    # Validate and record the *actual* derived artifact.
    weights, tokenizers = hash_snapshot_files(output_dir)
    if not weights:
        raise ConversionError(f"conversion produced no model*.safetensors in {output_dir}")
    from .config.identity import sha256_file

    chat = output_dir / "chat_template.jinja"
    chat_sha = sha256_file(str(chat)) if chat.exists() else None
    if source.chat_template_sha256 is not None and chat_sha != source.chat_template_sha256:
        raise ConversionError(
            "conversion did not preserve the source chat template "
            f"(source={source.chat_template_sha256}, derived={chat_sha})"
        )

    tokenizer_byte_identity = {
        name: tokenizers.get(name) == digest
        for name, digest in source.tokenizer_files.items()
        if name != "chat_template.jinja"
    }

    precision, quantization, think_end_id, eos_ids, bits_per_weight = _inspect_artifact(output_dir)
    quant_mode = quantization.get("mode") if quantization.get("quantized") else None

    provenance: dict[str, Any] = {
        "schema": VARIANT_SCHEMA,
        "created_utc": _utc_now(),
        "source_artifact": source.to_dict(),
        "representation": "derived",
        "precision": precision,
        "quantization": f"mlx-{quant_mode}" if quant_mode else None,
        "quantization_bits": quantization.get("bits"),
        "quantization_group_size": quantization.get("group_size"),
        "quantization_params": quantization,
        "bits_per_weight": bits_per_weight,
        "conversion": {
            "tool": "mlx-lm",
            "tool_version": _mlx_lm_version(),
            "function": "mlx_lm.convert.convert",
            "params": params,
        },
        "variant_weight_files": weights,
        "variant_tokenizer_files": tokenizers,
        "chat_template_sha256": chat_sha,
        # Tokenizer files are re-serialized by save_pretrained, so byte identity
        # is not expected; semantic identity is verified by `variant-audit`.
        "tokenizer_byte_identity_vs_source": tokenizer_byte_identity,
        "think_end_token_id": think_end_id,
        "eos_token_ids": sorted(eos_ids),
        "local": {"source_path": str(snapshot), "output_path": str(output_dir)},
        "storage": storage,
        "command": command,
    }
    (output_dir / VARIANT_PROVENANCE).write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def _inspect_artifact(
    artifact_dir: Path,
) -> tuple[str, dict[str, Any], int | None, set[int], float | None]:
    """Load the derived artifact and return actual precision/quant/token info."""
    from mlx_lm.utils import compute_bits_per_weight, load_model, load_tokenizer

    model, config = load_model(artifact_dir, lazy=False, strict=False)
    tokenizer = load_tokenizer(artifact_dir, eos_token_ids=config.get("eos_token_id"))
    precision = detect_precision(model)["dominant"]
    quantization = detect_quantization(model)
    think_end_id = getattr(tokenizer, "think_end_id", None)
    eos_ids = set(tokenizer.eos_token_ids)
    try:
        bits_per_weight = float(compute_bits_per_weight(model))
    except Exception:  # pragma: no cover - best-effort reporting
        bits_per_weight = None
    return precision, quantization, think_end_id, eos_ids, bits_per_weight


def load_variant_provenance(artifact_dir: str | Path) -> dict[str, Any]:
    """Read and validate a derived-variant provenance record (fail closed)."""
    path = Path(artifact_dir) / VARIANT_PROVENANCE
    if not path.exists():
        raise ConversionError(
            f"no derived-variant provenance at {path}; refusing to treat an "
            "unlabeled directory as a scientific weight variant"
        )
    try:
        data = json.loads(path.read_text())
    except ValueError as exc:
        raise ConversionError(f"invalid provenance JSON in {path}: {exc}") from exc
    if data.get("schema") != VARIANT_SCHEMA:
        raise ConversionError(
            f"unsupported variant provenance schema {data.get('schema')!r} in {path}"
        )
    for key in ("source_artifact", "variant_weight_files", "conversion"):
        if key not in data:
            raise ConversionError(f"variant provenance missing required key {key!r}")
    return data


def verify_variant_artifact(artifact_dir: str | Path, provenance: dict[str, Any]) -> dict[str, Any]:
    """Recompute derived hashes and confirm they match the recorded provenance."""
    artifact_dir = Path(artifact_dir)
    weights, tokenizers = hash_snapshot_files(artifact_dir)
    errors: list[str] = []
    if weights != provenance.get("variant_weight_files", {}):
        errors.append("derived weight-file hashes do not match provenance")
    recorded_tok = provenance.get("variant_tokenizer_files", {})
    for name, digest in recorded_tok.items():
        if tokenizers.get(name) != digest:
            errors.append(f"derived tokenizer file {name} does not match provenance")
    from .config.identity import sha256_file

    chat = artifact_dir / "chat_template.jinja"
    chat_sha = sha256_file(str(chat)) if chat.exists() else None
    if chat_sha != provenance.get("chat_template_sha256"):
        errors.append("derived chat template does not match provenance")
    return {"ok": not errors, "errors": errors, "weights": weights, "tokenizers": tokenizers}


def variant_identity(
    provenance: dict[str, Any],
    *,
    local_path: str | None,
    load_audit: dict[str, Any] | None = None,
) -> WeightVariantIdentity:
    """Construct the executable weight-variant identity from provenance.

    Path-independent fields only participate in identity: the local path and the
    load audit are excluded by :class:`WeightVariantIdentity` itself.
    """
    source = SourceArtifactIdentity(**provenance["source_artifact"])
    conversion = {
        "tool": provenance["conversion"].get("tool"),
        "tool_version": provenance["conversion"].get("tool_version"),
        "function": provenance["conversion"].get("function"),
        "params": provenance["conversion"].get("params", {}),
    }
    return WeightVariantIdentity(
        source=source,
        representation="derived",
        variant_weight_files=provenance["variant_weight_files"],
        precision=provenance.get("precision", "unknown"),
        quantization=provenance.get("quantization"),
        quantization_bits=provenance.get("quantization_bits"),
        quantization_group_size=provenance.get("quantization_group_size"),
        quantization_params=provenance.get("quantization_params", {}),
        conversion=conversion,
        local_path=local_path,
        load_audit=load_audit or {},
    )


def _tokenizer_semantics(tokenizer: Any) -> dict[str, Any]:
    probe = "The quick brown fox; 2+2=4."
    return {
        "vocab_size": int(getattr(tokenizer, "vocab_size", -1)),
        "think_start_id": getattr(tokenizer, "think_start_id", None),
        "think_end_id": getattr(tokenizer, "think_end_id", None),
        "eos_token_ids": sorted(set(tokenizer.eos_token_ids)),
        "probe_encode": [int(t) for t in tokenizer.encode(probe, add_special_tokens=False)],
        "probe_decode": tokenizer.decode(
            [int(t) for t in tokenizer.encode(probe, add_special_tokens=False)]
        ),
    }


def audit_variant_load(
    repo_id: str,
    revision: str,
    artifact_path: str | Path,
    *,
    prompt: str = "What is 17 + 25? Answer with the integer only.",
    max_tokens: int = 32,
) -> dict[str, Any]:
    """Real load + short-generation audit of a derived variant.

    Verifies semantic tokenizer identity against the pinned source snapshot
    (vocab size, think/EOS ids, a probe encode/decode round-trip), confirms the
    native reasoning channel is detectable, and records measured load time and
    MLX peak memory. This performs real inference; it never fabricates a result.
    """
    import time

    import mlx.core as mx

    from .backends.base import CaptureSpec
    from .backends.mlx_backend import MlxBackend
    from .config.identity import DecodingPolicy

    backend = MlxBackend()
    t0 = time.perf_counter()
    loaded = backend.load(
        SourceArtifactIdentity(repo_id=repo_id, revision=revision),
        artifact_path=str(artifact_path),
    )
    load_seconds = time.perf_counter() - t0
    peak_after_load = int(mx.get_peak_memory())

    tokenizer = backend._tokenizer
    derived_semantics = _tokenizer_semantics(tokenizer)

    source_snapshot = resolve_local_snapshot(repo_id, revision)
    from mlx_lm.utils import load_tokenizer

    source_tokenizer = load_tokenizer(source_snapshot)
    source_semantics = _tokenizer_semantics(source_tokenizer)
    tokenizer_matches = derived_semantics == source_semantics

    prompt_ids = backend.tokenize_prompt(
        [{"role": "user", "content": prompt}], enable_thinking=True
    )
    trace = backend.generate(
        prompt_ids,
        DecodingPolicy(mode="greedy", max_tokens=max_tokens),
        CaptureSpec.minimal(),
    )
    peak_after_generate = int(mx.get_peak_memory())

    bits_per_weight: float | None = None
    try:
        from mlx_lm.utils import compute_bits_per_weight

        bits_per_weight = float(compute_bits_per_weight(backend._model))
    except Exception:  # pragma: no cover - best-effort reporting
        bits_per_weight = None

    reasoning_channel_detected = (
        derived_semantics["think_start_id"] is not None
        and derived_semantics["think_end_id"] is not None
    )

    return {
        "artifact_path": str(artifact_path),
        "source_repo": repo_id,
        "revision": revision,
        "representation": loaded.representation,
        "precision": loaded.precision,
        "quantization": loaded.quantization,
        "quantization_bits": loaded.quantization_bits,
        "quantization_group_size": loaded.quantization_group_size,
        "quantization_params": loaded.quantization_params,
        "bits_per_weight": bits_per_weight,
        "load_audit": loaded.load_audit,
        "source_artifact_id": loaded.source.source_artifact_id,
        "weight_variant_id": loaded.weight_variant_id,
        "derived_from_source_artifact_id": loaded.derived_from_source_artifact_id,
        "load_seconds": load_seconds,
        "mlx_peak_after_load_bytes": peak_after_load,
        "mlx_peak_after_generate_bytes": peak_after_generate,
        "physical_memory_bytes": _physical_memory(),
        "tokenizer": {
            "derived": derived_semantics,
            "source": source_semantics,
            "semantic_match": tokenizer_matches,
            "reasoning_channel_detected": reasoning_channel_detected,
        },
        "generation": {
            "prompt_tokens": trace.prompt_tokens,
            "generated_tokens": trace.generated_tokens,
            "finish_reason": trace.finish_reason,
            "think_end_reached": trace.think_end_reached,
            "eos_observed": trace.terminated_by_eos,
            "error": trace.error,
            "wall_ms": trace.wall_ms,
            "tokens_per_sec": trace.tokens_per_sec,
        },
        "ok": bool(
            loaded.representation == "derived"
            and loaded.load_audit.get("ok")
            and loaded.load_audit.get("n_quantized_modules", 0) > 0
            and tokenizer_matches
            and reasoning_channel_detected
            and trace.error is None
        ),
    }


def _physical_memory() -> int | None:
    import subprocess

    try:
        out = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5, check=False
        )
        return int(out.stdout.strip()) if out.returncode == 0 else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
