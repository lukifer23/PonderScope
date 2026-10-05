"""Test fixtures. The fake backend exists ONLY here; there is no fake production backend."""

from __future__ import annotations

from typing import Any

import pytest

from ponderscope.backends.base import BackendCapabilities, CaptureSpec, TokenStep, Trace
from ponderscope.config.identity import DecodingPolicy, ModelIdentity, RuntimeIdentity


class FakeBackend:
    """Deterministic synthetic backend for software tests only."""

    name = "fake"

    def __init__(self) -> None:
        self._loaded = False
        self._think_end_id = 99

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            name="fake",
            runtime="fake-runtime",
            runtime_version="0.0.1",
            greedy=True,
            seeded_sampling=True,
            logprobs=True,
            entropy=True,
            full_logprob_digest=True,
            per_token_timing=True,
            ttft=True,
            prefix_probe=True,
        )

    def load(self, model_identity: ModelIdentity) -> ModelIdentity:
        self._loaded = True
        return ModelIdentity(
            repo_id=model_identity.repo_id,
            revision=model_identity.revision,
            local_path="/fake",
            weight_files={"fake.safetensors": "0" * 64},
            tokenizer_files={"tokenizer.json": "1" * 64},
            precision="float32",
            quantization=None,
        )

    def runtime_identity(self) -> RuntimeIdentity:
        return RuntimeIdentity(
            runtime="fake-runtime",
            runtime_version="0.0.1",
            backend="fake",
            hardware="test-host",
            os="test-os",
            python_version="3.12",
        )

    @property
    def think_end_token_id(self) -> int:
        return self._think_end_id

    def tokenize_prompt(self, messages: list[dict[str, str]], enable_thinking: bool) -> list[int]:
        content = messages[-1]["content"]
        return [10, 20, 30, len(content) % 50]

    def _answer_for(self, prompt_ids: list[int], seedless: int) -> str:
        return str((sum(prompt_ids) + seedless) % 7)

    def generate(
        self, prompt_token_ids: list[int], decoding: DecodingPolicy, capture: CaptureSpec
    ) -> Trace:
        offset = decoding.seed if decoding.seed is not None else 0
        n = 20 + (offset % 3)
        token_ids = [100 + i for i in range(n)]
        answer = self._answer_for(prompt_token_ids, offset)
        token_ids = token_ids + [self._think_end_id] + [200, 201]
        text = "reasoning " * n + "</think>\nAnswer: " + answer
        steps = [
            TokenStep(index=i, token_id=t, logprob=-0.5, entropy=0.3, t_ms=1.0)
            for i, t in enumerate(token_ids)
        ]
        return Trace(
            token_ids=token_ids,
            text=text,
            prompt_tokens=len(prompt_token_ids),
            generated_tokens=len(token_ids),
            ttft_ms=5.0,
            wall_ms=10.0 + offset,
            tokens_per_sec=50.0,
            finish_reason="stop",
            terminated_by_eos=True,
            capped=False,
            steps=steps,
            extra={"think_end_reached": True, "seed": decoding.seed},
        )

    def probe(
        self,
        prompt_token_ids: list[int],
        prefix_token_ids: list[int],
        decoding: DecodingPolicy,
        capture: CaptureSpec,
    ) -> Trace:
        # Answer correctness alternates with prefix length so transitions occur.
        answer = str((len(prefix_token_ids) + sum(prompt_token_ids)) % 2 == 0).lower()
        answer = (len(prefix_token_ids) + sum(prompt_token_ids)) % 2
        token_ids = [self._think_end_id, 300, 301]
        text = "</think>\nAnswer: " + str(answer)
        return Trace(
            token_ids=token_ids,
            text=text,
            prompt_tokens=len(prompt_token_ids) + len(prefix_token_ids),
            generated_tokens=len(token_ids),
            ttft_ms=2.0,
            wall_ms=3.0,
            tokens_per_sec=100.0,
            finish_reason="stop",
            terminated_by_eos=True,
            capped=False,
            extra={"think_end_reached": True, "seed": decoding.seed},
        )

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(str(t) for t in token_ids)


@pytest.fixture
def fake_backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture
def model_identity() -> ModelIdentity:
    return ModelIdentity(repo_id="fake/model", revision="deadbeef", precision="unknown")


@pytest.fixture
def deployment_kwargs() -> dict[str, Any]:
    return {"model_repo": "fake/model", "model_revision": "deadbeef"}
