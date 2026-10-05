"""MLX / MLX-LM reasoning measurement backend.

This is the only real backend in PonderScope. It runs on Apple Silicon through
MLX-LM, records the exact model revision and file hashes, and captures only
evidence MLX-LM genuinely provides (per-token chosen logprob, full-distribution
entropy, top-k, optional distribution digest, per-token timing).
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, cast

import mlx.core as mx

from ..config.identity import DecodingPolicy, ModelIdentity, RuntimeIdentity, sha256_file
from .base import BackendCapabilities, CaptureSpec, TokenStep, Trace

_WEIGHT_GLOBS = ("model*.safetensors",)
_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "special_tokens_map.json",
    "added_tokens.json",
)
_HASH_CACHE = Path.home() / ".cache" / "ponderscope" / "weight_hashes.json"


def _model_dir_name(repo_id: str) -> str:
    return "models--" + repo_id.replace("/", "--")


def resolve_local_snapshot(repo_id: str, revision: str) -> Path:
    """Locate an immutable snapshot in the Hugging Face cache without network.

    We do not use ``snapshot_download`` because it validates metadata files
    (README/LICENSE) that are irrelevant to execution and whose absence would
    otherwise abort a run on a perfectly usable cached checkpoint.
    """
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    snap = cache_root / _model_dir_name(repo_id) / "snapshots" / revision
    if not snap.exists():
        raise FileNotFoundError(
            f"no local snapshot for {repo_id}@{revision} at {snap}; "
            "PonderScope does not silently download a different revision"
        )
    if not (snap / "config.json").exists():
        raise FileNotFoundError(f"snapshot {snap} has no config.json")
    if not any(p.exists() for p in snap.glob("model*.safetensors")):
        raise FileNotFoundError(f"snapshot {snap} has no model*.safetensors weights")
    return snap


def _load_hash_cache() -> dict[str, str]:
    import json

    if _HASH_CACHE.exists():
        try:
            return json.loads(_HASH_CACHE.read_text())
        except (ValueError, OSError):
            return {}
    return {}


def _save_hash_cache(cache: dict[str, str]) -> None:
    import json

    _HASH_CACHE.parent.mkdir(parents=True, exist_ok=True)
    tmp = _HASH_CACHE.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, indent=2))
    tmp.replace(_HASH_CACHE)


def hash_snapshot_files(snapshot: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Return (weight_file_hashes, tokenizer_file_hashes).

    Hashing is cached by (path, size, mtime) so repeated runs do not re-hash
    gigabytes of weights. Cached hashes are keyed by content identity anyway.
    """
    cache = _load_hash_cache()
    weights: dict[str, str] = {}
    tokenizers: dict[str, str] = {}
    changed = False

    def _hash(rel_path: Path, target: dict[str, str]) -> None:
        nonlocal changed
        st = rel_path.stat()
        key = f"{rel_path.resolve()}|{st.st_size}|{int(st.st_mtime)}"
        digest = cache.get(key)
        if digest is None:
            digest = sha256_file(str(rel_path))
            cache[key] = digest
            changed = True
        target[rel_path.name] = digest

    for pattern in _WEIGHT_GLOBS:
        for path in sorted(snapshot.glob(pattern)):
            if path.is_file():
                _hash(path, weights)
    for name in _TOKENIZER_FILES:
        path = snapshot / name
        if path.exists() and path.is_file():
            _hash(path, tokenizers)
    if changed:
        _save_hash_cache(cache)
    return weights, tokenizers


def detect_precision(model: Any) -> dict[str, Any]:
    """Inspect the actual in-memory weight dtypes."""
    from mlx.utils import tree_flatten

    flat = cast("list[tuple[str, Any]]", tree_flatten(model.parameters()))
    counts: dict[str, int] = {}
    for _name, arr in flat:
        key = str(arr.dtype).split(".")[-1]
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return {"dominant": "unknown", "dtype_counts": {}}
    dominant = max(counts.items(), key=lambda kv: kv[1])[0]
    return {"dominant": dominant, "dtype_counts": counts}


def detect_quantization(model: Any) -> dict[str, Any]:
    """Detect MLX quantization scheme from module structure."""
    import mlx.nn as nn

    seen: dict[tuple[int, int, str], int] = {}
    for module in model.modules():
        if isinstance(module, nn.QuantizedLinear):
            key = (int(module.bits), int(module.group_size), str(module.mode))
            seen[key] = seen.get(key, 0) + 1
    if not seen:
        return {"quantized": False}
    # majority scheme
    (bits, group, mode), _ = max(seen.items(), key=lambda kv: kv[1])
    return {
        "quantized": True,
        "bits": bits,
        "group_size": group,
        "mode": mode,
        "layer_schemes": [
            {"bits": b, "group_size": g, "mode": m, "layers": n} for (b, g, m), n in seen.items()
        ],
    }


class MlxBackend:
    name = "mlx"

    def __init__(self) -> None:
        self._model: Any = None
        self._tokenizer: Any = None
        self._snapshot: Path | None = None
        self._eos_ids: set[int] = set()
        self._think_end_id: int | None = None
        self._think_start_id: int | None = None
        self._model_identity: ModelIdentity | None = None

    # -- capabilities --------------------------------------------------------
    def capabilities(self) -> BackendCapabilities:
        import mlx_lm

        runtime_version = getattr(mlx_lm, "__version__", "unknown")
        return BackendCapabilities(
            name="mlx",
            runtime="mlx-lm",
            runtime_version=runtime_version,
            greedy=True,
            seeded_sampling=True,
            logprobs=True,
            entropy=True,
            full_logprob_digest=True,
            per_token_timing=True,
            ttft=True,
            prefix_probe=True,
            notes=[
                "Per-token metrics (entropy, top-k, digest) are computed from the "
                "full normalized logprob vector returned by mlx_lm.generate_step.",
                "Per-token timing includes measurement overhead; a bare baseline is "
                "measured separately to quantify instrumentation cost.",
            ],
        )

    # -- loading -------------------------------------------------------------
    def load(self, model_identity: ModelIdentity) -> ModelIdentity:
        from mlx_lm.utils import load_model, load_tokenizer

        snapshot = resolve_local_snapshot(model_identity.repo_id, model_identity.revision)

        # strict=False because multimodal checkpoints carry vision-tower and MTP
        # weights that the text architecture intentionally drops. This is
        # recorded as a provenance caveat, not hidden.
        model, config = load_model(snapshot, lazy=False, strict=False)
        tokenizer = load_tokenizer(snapshot, eos_token_ids=config.get("eos_token_id"))

        weights, tokenizer_files = hash_snapshot_files(snapshot)
        precision = detect_precision(model)
        quantization = detect_quantization(model)
        chat_template = snapshot / "chat_template.jinja"
        chat_sha = sha256_file(str(chat_template)) if chat_template.exists() else None

        actual = ModelIdentity(
            repo_id=model_identity.repo_id,
            revision=model_identity.revision,
            local_path=str(snapshot),
            weight_files=weights,
            tokenizer_files=tokenizer_files,
            chat_template_sha256=chat_sha,
            precision=precision["dominant"],
            quantization=(f"mlx-{quantization['mode']}") if quantization.get("quantized") else None,
            quantization_bits=quantization.get("bits"),
            quantization_group_size=quantization.get("group_size"),
            quantization_params={
                "detected": quantization,
                "dtype_counts": precision["dtype_counts"],
                "load_strict": False,
                "dropped_multimodal_and_mtp": True,
            },
        )

        if model_identity.precision not in ("unknown", actual.precision):
            raise ValueError(
                f"declared precision {model_identity.precision!r} != detected "
                f"{actual.precision!r}; refusing to mislabel the deployment"
            )

        self._model = model
        self._tokenizer = tokenizer
        self._snapshot = snapshot
        self._eos_ids = set(tokenizer.eos_token_ids)
        self._think_end_id = getattr(tokenizer, "think_end_id", None)
        self._think_start_id = getattr(tokenizer, "think_start_id", None)
        self._model_identity = actual
        return actual

    @property
    def model_identity(self) -> ModelIdentity:
        if self._model_identity is None:
            raise RuntimeError("backend not loaded")
        return self._model_identity

    @property
    def think_end_token_id(self) -> int | None:
        return self._think_end_id

    def runtime_identity(self) -> RuntimeIdentity:
        import platform as _platform
        import sys as _sys

        import mlx_lm

        from ..evidence.environment import capture_environment

        env = capture_environment()
        return RuntimeIdentity(
            runtime="mlx-lm",
            runtime_version=getattr(mlx_lm, "__version__", "unknown"),
            backend="mlx-metal",
            hardware=env.get("cpu_brand") or _platform.processor() or _platform.machine(),
            os=f"{_platform.system()} {_platform.release()}",
            python_version=_sys.version.split()[0],
            device=str(mx.default_device()),
        )

    # -- tokenization --------------------------------------------------------
    def tokenize_prompt(self, messages: list[dict[str, str]], enable_thinking: bool) -> list[int]:
        if self._tokenizer is None:
            raise RuntimeError("backend not loaded")
        template_kwargs: dict[str, Any] = {}
        if getattr(self._tokenizer, "has_thinking", False):
            template_kwargs["enable_thinking"] = enable_thinking
        ids = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, **template_kwargs
        )
        return [int(t) for t in ids]

    def decode(self, token_ids: list[int]) -> str:
        if self._tokenizer is None:
            raise RuntimeError("backend not loaded")
        return self._tokenizer.decode(token_ids)

    def encode_text(self, text: str) -> list[int]:
        if self._tokenizer is None:
            raise RuntimeError("backend not loaded")
        return [int(t) for t in self._tokenizer.encode(text, add_special_tokens=False)]

    # -- generation ----------------------------------------------------------
    def _prepare_sampler(self, decoding: DecodingPolicy) -> tuple[Any, int | None]:
        from mlx_lm.sample_utils import make_sampler

        if decoding.mode == "greedy":
            return None, None
        if decoding.mode != "sampled":
            raise ValueError(f"unknown decoding mode: {decoding.mode!r}")
        if decoding.seed is None:
            raise ValueError("sampled decoding requires an explicit seed")
        sampler = make_sampler(
            temp=decoding.temperature if decoding.temperature is not None else 0.6,
            top_p=decoding.top_p if decoding.top_p is not None else 1.0,
            top_k=decoding.top_k if decoding.top_k is not None else 0,
            min_p=decoding.min_p if decoding.min_p is not None else 0.0,
        )
        return sampler, decoding.seed

    def _run(
        self,
        prompt_ids: list[int],
        decoding: DecodingPolicy,
        capture: CaptureSpec,
        extra_tokens: list[int] | None = None,
    ) -> Trace:
        from mlx_lm.generate import generate_step

        if self._model is None:
            raise RuntimeError("backend not loaded")

        sampler, seed = self._prepare_sampler(decoding)
        if seed is not None:
            mx.random.seed(seed)

        base = list(prompt_ids)
        if extra_tokens:
            base = base + list(extra_tokens)

        prompt = mx.array(base)
        max_tokens = decoding.max_tokens

        token_ids: list[int] = []
        steps: list[TokenStep] = []
        ttft_ms: float | None = None
        t_start = time.perf_counter()
        t_prev = t_start
        finish_reason = "error"
        error: str | None = None
        think_end_reached = False

        try:
            gen = generate_step(prompt, self._model, max_tokens=max_tokens, sampler=sampler)
            for token, logprobs in gen:
                now = time.perf_counter()
                tid = int(token)  # type: ignore[arg-type]
                if not token_ids:
                    ttft_ms = (now - t_start) * 1000.0
                step = self._capture_step(len(token_ids), tid, logprobs, capture, now - t_prev)
                steps.append(step)
                token_ids.append(tid)
                t_prev = now
                if self._think_end_id is not None and tid == self._think_end_id:
                    think_end_reached = True
                if tid in self._eos_ids:
                    finish_reason = "stop"
                    break
                if len(token_ids) >= max_tokens:
                    finish_reason = "length"
                    break
            else:
                finish_reason = "length"
        except Exception as exc:  # pragma: no cover - hardware/runtime failures
            finish_reason = "error"
            error = f"{type(exc).__name__}: {exc}"

        wall_ms = (time.perf_counter() - t_start) * 1000.0
        text = self.decode(token_ids)
        terminated_by_eos = finish_reason == "stop"
        capped = finish_reason == "length"
        tps = (len(token_ids) / (wall_ms / 1000.0)) if wall_ms > 0 else 0.0

        return Trace(
            token_ids=token_ids,
            text=text,
            prompt_tokens=len(base),
            generated_tokens=len(token_ids),
            ttft_ms=ttft_ms,
            wall_ms=wall_ms,
            tokens_per_sec=tps,
            finish_reason=finish_reason,
            terminated_by_eos=terminated_by_eos,
            capped=capped,
            steps=steps,
            error=error,
            extra={
                "think_end_reached": think_end_reached,
                "think_end_token_id": self._think_end_id,
                "base_prompt_tokens": len(prompt_ids),
                "extra_tokens": len(extra_tokens) if extra_tokens else 0,
                "capture": capture.to_dict(),
                "seed": seed,
            },
        )

    def _capture_step(
        self,
        index: int,
        token_id: int,
        logprobs: Any,
        capture: CaptureSpec,
        dt: float,
    ) -> TokenStep:
        step = TokenStep(index=index, token_id=token_id)
        if capture.per_token_timing:
            step.t_ms = dt * 1000.0

        lp32 = logprobs.astype(mx.float32)
        mx.eval(lp32)
        step.logprob = float(lp32[token_id])

        if capture.entropy:
            ent = -(mx.exp(lp32) * lp32).sum()
            mx.eval(ent)
            step.entropy = float(ent)

        if capture.top_k and capture.top_k > 0:
            k = min(capture.top_k, lp32.shape[-1])
            top = mx.argpartition(-lp32, kth=k - 1)[-k:]
            vals = lp32[top]
            order = mx.argsort(-vals)
            top = top[order]
            vals = vals[order]
            mx.eval(top, vals)
            step.top_token_ids = [int(t) for t in top.tolist()]  # type: ignore[arg-type, union-attr]
            step.top_logprobs = [float(v) for v in vals.tolist()]  # type: ignore[arg-type, union-attr]

        if capture.logprob_digest:
            import numpy as np

            raw = np.asarray(lp32, dtype=np.float32).tobytes()
            step.logprob_digest = hashlib.sha256(raw).hexdigest()[:16]

        return step

    def generate(
        self,
        prompt_token_ids: list[int],
        decoding: DecodingPolicy,
        capture: CaptureSpec,
    ) -> Trace:
        return self._run(prompt_token_ids, decoding, capture)

    def probe(
        self,
        prompt_token_ids: list[int],
        prefix_token_ids: list[int],
        decoding: DecodingPolicy,
        capture: CaptureSpec,
    ) -> Trace:
        """Forced finalization from a saved reasoning prefix.

        The prefix is replayed, then the model's natively learned think-end token
        is appended followed by a fixed answer cue, forcing it out of the
        reasoning channel and into an answer. Empirically (MEASURED) the raw
        think-end token alone is not always honoured by this model, so the cue is
        part of the intervention. This is a measurement primitive, not a
        deployable stopping method.
        """
        if self._think_end_id is None:
            raise RuntimeError("model has no think-end token; prefix probing unsupported")
        cue = self.encode_text("\n\nAnswer:")
        extra = list(prefix_token_ids) + [self._think_end_id] + cue
        return self._run(prompt_token_ids, decoding, capture, extra_tokens=extra)
