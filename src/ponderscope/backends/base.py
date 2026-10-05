"""Backend contract for PonderScope.

This is a narrow, real interface. Only genuinely implemented backends may
satisfy it. There are deliberately no placeholder backends.

A backend executes a :class:`~ponderscope.config.identity.Deployment` and returns
a :class:`Trace` of what actually happened. Everything a backend cannot honestly
provide must be reported as unavailable rather than approximated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..config.identity import DecodingPolicy, ModelIdentity


@dataclass(frozen=True)
class CaptureSpec:
    """What per-token evidence to capture. Expensive fields are opt-in."""

    entropy: bool = True
    top_k: int = 5
    logprob_digest: bool = False  # digest of the full distribution; expensive
    per_token_timing: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TokenStep:
    index: int
    token_id: int
    logprob: float | None = None  # logprob of the sampled/greedy token
    entropy: float | None = None  # Shannon entropy (nats) of the full distribution
    top_token_ids: list[int] = field(default_factory=list)
    top_logprobs: list[float] = field(default_factory=list)
    logprob_digest: str | None = None
    t_ms: float | None = None  # time since previous token, including capture overhead

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None and v != []}


@dataclass
class Trace:
    """A complete record of one generation."""

    token_ids: list[int]
    text: str
    prompt_tokens: int
    generated_tokens: int
    ttft_ms: float | None
    wall_ms: float
    tokens_per_sec: float
    finish_reason: str  # "stop" | "length" | "error"
    terminated_by_eos: bool
    capped: bool
    steps: list[TokenStep] = field(default_factory=list)
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def think_end_reached(self) -> bool:
        return bool(self.extra.get("think_end_reached", False))

    def to_dict(self) -> dict[str, Any]:
        return {
            "token_ids": self.token_ids,
            "text": self.text,
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "ttft_ms": self.ttft_ms,
            "wall_ms": self.wall_ms,
            "tokens_per_sec": self.tokens_per_sec,
            "finish_reason": self.finish_reason,
            "terminated_by_eos": self.terminated_by_eos,
            "capped": self.capped,
            "steps": [s.to_dict() for s in self.steps],
            "error": self.error,
            "extra": self.extra,
        }


@dataclass(frozen=True)
class BackendCapabilities:
    name: str
    runtime: str
    runtime_version: str
    greedy: bool
    seeded_sampling: bool
    logprobs: bool
    entropy: bool
    full_logprob_digest: bool
    per_token_timing: bool
    ttft: bool
    prefix_probe: bool
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@runtime_checkable
class Backend(Protocol):
    name: str

    def capabilities(self) -> BackendCapabilities: ...

    def load(self, model_identity: ModelIdentity) -> ModelIdentity: ...

    def runtime_identity(self) -> Any: ...

    def tokenize_prompt(
        self, messages: list[dict[str, str]], enable_thinking: bool
    ) -> list[int]: ...

    def generate(
        self,
        prompt_token_ids: list[int],
        decoding: DecodingPolicy,
        capture: CaptureSpec,
    ) -> Trace: ...

    def probe(
        self,
        prompt_token_ids: list[int],
        prefix_token_ids: list[int],
        decoding: DecodingPolicy,
        capture: CaptureSpec,
    ) -> Trace: ...

    def decode(self, token_ids: list[int]) -> str: ...


def get_backend(name: str) -> Backend:
    """Return the real backend registered under ``name``."""
    if name == "mlx":
        from .mlx_backend import MlxBackend

        return MlxBackend()
    raise ValueError(f"unknown backend: {name!r} (available: ['mlx'])")
