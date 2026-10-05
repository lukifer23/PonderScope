# Methodology

PonderScope is measurement software. This document defines what it measures and
how. Terminology is used strictly: **IMPLEMENTED**, **LIVE VALIDATED**,
**MEASURED**, **HYPOTHESIS**, **UNSUPPORTED**, **NOT YET TESTED**.

## The central distinction: model behavior vs deployment behavior

A result observed under one runtime is never described as a property of the
model. Reports identify the deployment as:

> `<repo>@<immutable-revision> (precision, quantization) under <runtime>
> <version> [backend, hardware], <decoding mode>, cfg=<config-id>`

## Deployment identity

Every execution condition is reduced to a canonical metadata mapping and hashed
to a configuration id (`dep-<12hex>`). The mapping includes, where applicable:

- upstream model repository and immutable revision
- weight-file sha256 hashes and tokenizer/chat-template hashes
- numerical precision and quantization method/parameters (bits, group size, mode)
- inference runtime and version; backend; hardware; OS; Python version
- decoding mode (greedy/sampled), sampler parameters, seed
- context configuration and output/reasoning cap

Canonicalization sorts keys, drops nulls/empties, rounds floats, and hashes the
result. Reports compare configuration ids, never vague model names.

## Backend contract

A backend executes a `ModelIdentity` and returns a `Trace`. The only real
backend is **MLX / MLX-LM** on Apple Silicon (`backends/mlx_backend.py`). There
are deliberately no placeholder CUDA/llama.cpp/Ollama/vLLM/PyTorch backends.
Future real backends satisfy the same narrow `Backend` protocol.

Capabilities are reported honestly. MLX-LM provides, and PonderScope captures:

- **per-token chosen logprob** (from the normalized logprob vector returned by
  `mlx_lm.generate_step`)
- **Shannon entropy** of the full normalized distribution (nats)
- **top-k** token ids and logprobs
- optional **digest** of the full distribution (for determinism fingerprinting)
- **per-token timing** and **TTFT**
- `finish_reason` (`stop`/`length`/`error`) and EOS/think-end status

Anything MLX-LM does not provide is marked UNSUPPORTED rather than approximated.

### Provenance caveat (recorded, not hidden)

`Qwen/Qwen3.5-0.8B` is a multimodal checkpoint. The text architecture drops the
vision-tower and MTP weights during load (`strict=False`). This is recorded in
the run manifest (`dropped_multimodal_and_mtp: true`). It does not change the
text weights, but it is stated because provenance matters more than convenience.

## Reasoning channels

The chat template opens the thinking block in the prompt (`<think>\n`). The
generated text is therefore reasoning followed by the model's learned closing
delimiter `</think>` (token 248069), then the final answer. PonderScope parses:

- **natural termination** — EOS (`<|im_end|>`, token 248046) observed
- **forced-finalization intervention** — prefix probe appending `</think>`
- **capped / censored** — `max_tokens` reached before EOS (no final channel)

Trajectory states: `initially_correct`, `wrong_to_correct`,
`correct_to_wrong`, `multiple_flips`, `stable_correct`, `never_correct`. Both a
primary label and independent flags are stored; nothing is collapsed into one
composite score.

## Metrics

Token-level, deterministic, no LLM judge:

- reasoning / answer / total token counts (split at the think-end token)
- unique-token ratio; distinct-1/2/4; repeated 4-/8-gram fractions
- longest identical token run; most-common token count
- line-level text repetition ratio
- answer extraction success / missing-answer rate
- TTFT, wall time, tokens/sec

## Prefix probes

Sparse forced-finalization probes replay a saved reasoning prefix, append the
model's own think-end token, and read the forced answer. **This concept already
exists in prior literature; PonderScope uses it as a measurement primitive, not
as its claimed novelty.** An oracle prefix is never presented as a deployable
stopping method.

## The within-configuration noise floor

A defining feature. Before comparing deployments, identical conditions are
repeated:

- **Greedy diagnostic:** same condition repeated; compare exact token ids and
  (optionally) per-token distribution digests. If they diverge, record the first
  divergence index.
- **Seeded sampled condition:** a frozen sampling profile with several seeds;
  repeat the same seed where supported. This separates deterministic-replay
  failure, seed-driven variation, and task-driven variation.

Distributions are computed for accuracy, reasoning tokens, latency, repetition,
answer flips, and termination failures. Deployment effects are only interpreted
relative to this floor.

## Comparison and statistics

Paired bootstrap confidence intervals over matched `(task, condition)` pairs;
each metric's delta is classified as `below_noise`, `comparable`, or
`clearly_larger` relative to the measured within-condition scale. No causal
language is used.

## Policy-transfer contract (future scope, documented now)

Evidence is saved so a later phase can: calibrate a stopping rule on deployment
A; freeze the rule and threshold; evaluate it unchanged on deployment B; measure
accuracy/risk/compute degradation; and compare against recalibration on B.
Candidate baselines: fixed token budget, natural termination, answer stability,
confidence threshold, temporal confidence/stability, a small learned controller.
This is **NOT YET TESTED**.
