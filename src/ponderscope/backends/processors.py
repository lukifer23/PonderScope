"""PonderScope's narrow logits processors.

These exist because MLX-LM's built-in presence penalty is *not* the same
semantics as the OpenAI-compatible presence penalty that upstream Qwen serving
examples assume. MLX-LM subtracts the penalty for tokens seen in the last
``presence_context_size`` positions of the accumulated token history, and that
history includes the prompt. OpenAI/vLLM-style presence penalty is defined over
the *generated* text so far, with no rolling window and without counting prompt
tokens.

This module implements only the generated-history presence penalty. It does not
patch the installed MLX-LM package. Repetition and frequency penalties continue
to use MLX-LM's built-in processors; upstream ``repetition_penalty=1.0`` is a
no-op and is never manufactured into an effect here.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx


def make_generated_history_presence_penalty(penalty: float, prompt_len: int) -> Any:
    """Presence penalty over the generated token history.

    A token receives the penalty once if it has appeared anywhere in the
    generated output so far. Prompt tokens (the first ``prompt_len`` tokens of
    the accumulated history) never count. ``penalty == 0`` is a no-op.

    ``generate_step`` passes each processor the full, monotonically growing
    ``tokens`` array (prompt + generated). This processor keeps a Python set of
    generated token ids and only scans the newly appended suffix on each call, so
    it is O(new tokens) per step and fully deterministic. A processor instance is
    created fresh for every generation, so state never leaks across generations.
    """
    if penalty < 0:
        raise ValueError(f"presence penalty must be non-negative, got {penalty}")
    seen: set[int] = set()
    consumed = 0

    def generated_history_presence_penalty(tokens: Any, logits: Any) -> Any:
        nonlocal consumed
        if penalty == 0:
            return logits
        token_list = tokens.tolist() if hasattr(tokens, "tolist") else list(tokens)
        # Skip the prompt prefix. On the first call ``tokens`` is exactly the
        # prompt, so nothing is added to ``seen``.
        if consumed < prompt_len:
            consumed = min(prompt_len, len(token_list))
        if len(token_list) > consumed:
            for token in token_list[consumed:]:
                seen.add(int(token))
            consumed = len(token_list)
        if seen:
            indices = mx.array(sorted(seen))
            logits[:, indices] -= penalty
        return logits

    return generated_history_presence_penalty
