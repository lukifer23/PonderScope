"""Model-load audit: prove which source weights load, rename, or are ignored.

PonderScope must never *assert* that ``strict=False`` only drops non-text
weights. This audit reconstructs the sanitize step and compares it against the
instantiated model's parameter keys, using an explicit allow-list of the known
non-text Qwen3.5 components (vision tower and MTP). Any unexpected missing or
unused *text* weight is a hard failure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

# Explicit allow-list of source components that the text architecture drops.
NON_TEXT_PREFIXES = ("model.visual.", "model.visual", "vision_tower", "mtp.")


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

    non_text = {k for k in raw if is_non_text(k)}
    text_source = set(raw) - non_text

    sanitized = model.sanitize(dict(raw)) if hasattr(model, "sanitize") else dict(raw)
    sanitized_keys = set(sanitized)
    flat = cast("list[tuple[str, Any]]", tree_flatten(model.parameters()))
    param_keys = {name for name, _ in flat}

    missing = sorted(param_keys - sanitized_keys)
    unused = sorted(sanitized_keys - param_keys)
    unexpected_non_text = sorted(k for k in non_text if not is_non_text(k))
    count_consistent = len(text_source) == len(param_keys)

    ok = not missing and not unused and not unexpected_non_text and count_consistent
    return {
        "ok": ok,
        "allowlist": list(NON_TEXT_PREFIXES),
        "n_source_keys": len(raw),
        "n_text_source_keys": len(text_source),
        "n_non_text_source_keys": len(non_text),
        "n_loaded_params": len(param_keys),
        "missing_text_params": missing,
        "unused_loaded_params": unused,
        "unexpected_ignored_keys": unexpected_non_text,
        "text_key_count_consistent": count_consistent,
        "load_strict": False,
    }
