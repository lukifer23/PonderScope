"""Model-distribution helpers (no MLX dependency, unit-testable).

``mlx_lm.generate_step`` yields the *pre-sampler* normalized log-probability
vector (logits minus log-sum-exp, after any logits processors). It is the raw
model distribution, NOT the post-temperature/top-p/top-k sampling distribution.
Top-k extraction here therefore reports the model's own most likely tokens.
"""

from __future__ import annotations

import numpy as np


def top_k_from_logprobs(logprobs: np.ndarray, k: int) -> tuple[list[int], list[float]]:
    """Return the actual top-k token ids and logprobs, descending by logprob.

    Uses ``argpartition(-lp, k-1)[:k]`` (the first k smallest of ``-lp`` equal the
    k largest of ``lp``); taking ``[-k:]`` would instead return the *least* likely
    tokens.
    """
    arr = np.asarray(logprobs, dtype=np.float32).reshape(-1)
    if k <= 0 or arr.size == 0:
        return [], []
    kk = min(k, arr.size)
    idx = np.argpartition(-arr, kk - 1)[:kk]
    vals = arr[idx]
    order = np.argsort(-vals, kind="stable")
    idx = idx[order]
    vals = vals[order]
    return [int(i) for i in idx.tolist()], [float(v) for v in vals.tolist()]
