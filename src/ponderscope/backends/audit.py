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

    # Keys actually removed by the sanitize step, split into allowed/unexpected.
    dropped_by_sanitize = raw_keys - sanitized_keys
    dropped_allowed_non_text = {k for k in dropped_by_sanitize if is_non_text(k)}
    dropped_unexpected = sorted(dropped_by_sanitize - dropped_allowed_non_text)
    # Text weights that the sanitizer dropped are a hard failure.
    text_source_lost = sorted(text_source & dropped_by_sanitize)

    # Parameters with no sanitized source, and sanitized weights never used.
    missing_from_sanitized = sorted(param_keys - sanitized_keys)
    unused_sanitized = sorted(sanitized_keys - param_keys)

    dropped_categories = Counter(_category(k) for k in dropped_by_sanitize)
    text_key_count_consistent = len(text_source) == len(param_keys)

    ok = not (missing_from_sanitized or unused_sanitized or dropped_unexpected or text_source_lost)
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
        "n_dropped_unexpected": len(dropped_unexpected),
        "n_text_source_lost": len(text_source_lost),
        "dropped_key_categories": dict(sorted(dropped_categories.items())),
        "missing_from_sanitized_params": missing_from_sanitized,
        "unused_sanitized_params": unused_sanitized,
        "dropped_unexpected_keys": dropped_unexpected,
        "text_source_keys_lost": text_source_lost,
        "text_key_count_consistent": text_key_count_consistent,
        "load_strict": False,
    }
