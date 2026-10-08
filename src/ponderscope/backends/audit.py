"""Model-load audit: prove which source weights load, rename, or are ignored.

PonderScope must never *assert* that ``strict=False`` only drops non-text
weights. This audit reconstructs the sanitize step and compares it against the
instantiated model's parameter keys, using an explicit allow-list of the known
non-text Qwen3.5 components (vision tower and MTP). Every dropped key is
classified; any unexpected dropped key or any lost text weight is a hard
failure.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, cast

# Explicit allow-list of source components that the text architecture drops.
NON_TEXT_PREFIXES = ("model.visual.", "model.visual", "vision_tower", "mtp.")


def _category(key: str) -> str:
    parts = key.split(".")
    return parts[0] if parts else key


def audit_derived_load(model: Any, expected_quantization: dict[str, Any] | None) -> dict[str, Any]:
    """Validate a *derived* (quantized) artifact load.

    For a derived artifact the source-key rename accounting used for native
    checkpoints does not apply. Instead this proves the artifact really is a
    quantized representation and that the actual per-module scheme matches the
    recorded conversion provenance. A nominal "Q4" claim is never accepted unless
    the loaded modules confirm it.
    """
    import mlx.nn as nn

    scheme_counts: dict[tuple[int, int, str], int] = {}
    n_quantized = 0
    for module in model.modules():
        if isinstance(module, nn.QuantizedLinear):
            n_quantized += 1
            key = (int(module.bits), int(module.group_size), str(module.mode))
            scheme_counts[key] = scheme_counts.get(key, 0) + 1

    actual = sorted(
        (
            {"bits": b, "group_size": g, "mode": m, "layers": n}
            for (b, g, m), n in scheme_counts.items()
        ),
        key=lambda d: (d["bits"], d["group_size"], d["mode"]),
    )
    expected = None
    if expected_quantization and expected_quantization.get("quantized"):
        expected = sorted(
            (
                {
                    "bits": int(s["bits"]),
                    "group_size": int(s["group_size"]),
                    "mode": str(s["mode"]),
                    "layers": int(s["layers"]),
                }
                for s in expected_quantization.get("layer_schemes", [])
            ),
            key=lambda d: (d["bits"], d["group_size"], d["mode"]),
        )

    mismatches: list[str] = []
    if n_quantized == 0:
        mismatches.append("no QuantizedLinear modules found in a derived artifact")
    if expected is not None and expected != actual:
        mismatches.append(f"actual quant schemes {actual} != recorded {expected}")

    return {
        "ok": not mismatches,
        "kind": "derived",
        "n_quantized_modules": n_quantized,
        "actual_schemes": actual,
        "expected_schemes": expected,
        "mismatches": mismatches,
    }


def audit_model_load(model: Any, snapshot: Path) -> dict[str, Any]:
    import glob

    import mlx.core as mx
    from mlx.utils import tree_flatten

    raw: dict[str, Any] = {}
    for wf in sorted(glob.glob(str(snapshot / "model*.safetensors"))):
        shard = cast("dict[str, Any]", mx.load(wf))
        raw.update(shard)

    def is_non_text(key: str) -> bool:
        return key.startswith(NON_TEXT_PREFIXES)

    raw_keys = set(raw)
    non_text_source = {k for k in raw_keys if is_non_text(k)}
    text_source = raw_keys - non_text_source

    sanitized = model.sanitize(dict(raw)) if hasattr(model, "sanitize") else dict(raw)
    sanitized_keys = set(sanitized)
    flat = cast("list[tuple[str, Any]]", tree_flatten(model.parameters()))
    param_keys = {name for name, _ in flat}

    # Keys the sanitizer did not reproduce verbatim. These are a mix of (a)
    # allow-listed non-text components that are genuinely dropped and (b) text
    # keys the sanitizer *renames* (e.g. `model.language_model.*` -> instantiated
    # parameter names). Renames are not loss; loss is detected structurally.
    dropped_by_sanitize = raw_keys - sanitized_keys
    dropped_allowed_non_text = {k for k in dropped_by_sanitize if is_non_text(k)}
    renamed_or_dropped_text = sorted(dropped_by_sanitize - dropped_allowed_non_text)
    dropped_unknown_non_text = sorted(
        k for k in dropped_by_sanitize if not is_non_text(k) and k not in text_source
    )

    # Parameters with no sanitized source, and sanitized weights never used.
    missing_from_sanitized = sorted(param_keys - sanitized_keys)
    unused_sanitized = sorted(sanitized_keys - param_keys)

    dropped_categories = Counter(_category(k) for k in dropped_by_sanitize)
    text_key_count_consistent = len(text_source) == len(param_keys)
    # A rename is accepted only when the sanitized output is exactly the used
    # parameter set and text source/param counts match 1:1. Otherwise every
    # non-reproduced text key is treated as loss.
    renames_accounted_by_count = text_key_count_consistent and not (
        missing_from_sanitized or unused_sanitized
    )
    text_source_lost = [] if renames_accounted_by_count else renamed_or_dropped_text
    dropped_unexpected = [] if renames_accounted_by_count else renamed_or_dropped_text

    ok = not (
        missing_from_sanitized or unused_sanitized or dropped_unknown_non_text or text_source_lost
    )
    return {
        "ok": ok,
        "allowlist": list(NON_TEXT_PREFIXES),
        "n_source_keys": len(raw_keys),
        "n_text_source_keys": len(text_source),
        "n_non_text_source_keys": len(non_text_source),
        "n_sanitized_keys": len(sanitized_keys),
        "n_loaded_params": len(param_keys),
        "n_dropped_by_sanitize": len(dropped_by_sanitize),
        "n_dropped_allowed_non_text": len(dropped_allowed_non_text),
        "n_renamed_or_dropped_text": len(renamed_or_dropped_text),
        "n_dropped_unexpected": len(dropped_unexpected),
        "n_text_source_lost": len(text_source_lost),
        "dropped_key_categories": dict(sorted(dropped_categories.items())),
        "missing_from_sanitized_params": missing_from_sanitized,
        "unused_sanitized_params": unused_sanitized,
        "dropped_unexpected_keys": dropped_unexpected,
        "text_source_keys_lost": text_source_lost,
        "renames_or_drops_accounted_by_count": renames_accounted_by_count,
        "text_key_count_consistent": text_key_count_consistent,
        "load_strict": False,
    }
