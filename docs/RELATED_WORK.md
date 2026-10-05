# Related work

Up-to-date literature audit for PonderScope, performed **2026-10-05**, focused on
the intended research gap: *reasoning reproducibility and adaptive-policy
portability across exact deployment configurations*.

This audit actively tries to falsify PonderScope's novelty. Where prior work
already answers a question, that is stated plainly. PonderScope must not claim
novelty for measuring overthinking, answer stability, quantization effects, or
early stopping; those have significant prior work.

Legend for the table: **Y** = studied, **N** = not studied, **~** = partially /
incidentally, **?** = not determinable from the abstract/available text.

## Structured comparison

| paper / system | date | central question | models | deployment configs considered | reasoning metrics | quant? | multi-runtime? | multi-hardware? | within-condition variance? | stopping-controller transfer across deployments? | direct overlap with PonderScope | remaining gap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Thinking Past the Answer: Harmful Overthinking (2606.02835) | 2026-06 | Does post-correct reasoning refine or harm? | multimodal + language LRMs | single inference path | prefix sufficiency, correct→wrong, verbose vs harmful | N | N | N | N | N | prefix probes, answer-state transitions, non-monotonicity | no deployment variation, no noise floor |
| Large Language Models Decide Early and Explain Later (2604.22266) | 2026-04 | When is the final answer determined? | Qwen3-4B, others | single runtime | forced-answer prefix probing, answer stability, early stopping | N | N | N | N | N | prefix probes, answer transitions, stop-when-stable | no deployment variation |
| Dynamic Early Exit in Reasoning Models (DEER, 2504.15895) | 2025-04 | Adaptive early exit via trial-answer confidence | LRMs | single runtime | confidence at reasoning transitions | N | N | N | N | N | early-exit motivation | no deployment variation |
| Conformal Thinking: Risk Control on a Compute Budget (2602.03814) | 2026-02 | Calibrate stopping to a risk target | several | single runtime | dual-threshold conformal stopping, efficiency loss | N | N | N | N | N (calibrates on one validation set) | risk-controlled stopping, calibration | no transfer of a frozen policy to a different deployment |
| Answer Convergence / Learn-to-Stop (EMNLP 2025, 2025.emnlp-main.904) | 2025 | Answer stabilization and learned stopping | 5 open-weight LLMs | single runtime | answer convergence, switch probabilities | N | N | N | ~ (repeat sampling) | N | stop-when-stable, learned controller | no deployment variation |
| ES-CoT: Early Stopping CoT (2509.14004) | 2025-09 | Stop when answers stabilize (run-jump test) | Qwen, QwQ, DeepSeek | single runtime | step-answer run lengths | N | N | N | N | N | answer-stability stopping | no deployment variation |
| OS-Pruner (2607.11089) | 2026-07 | Optimal stopping on CoT prefixes | several | single runtime | learned stop policy, accuracy–length utility | N | N | N | N | N | learned controller baseline | no deployment variation |
| MUTO / Token-Level Marginal Utility (ACL 2026) | 2026-07 | Per-token utility for efficient reasoning | DS-R1-Distill-Qwen 1.5B/7B | single runtime | token-level marginal utility | N | N | N | N | N | training-time, not measurement | orthogonal training method |
| ReEfBench (2601.03550, ACL 2026) | 2026-01 | Quantify reasoning efficiency | many LLMs | single runtime | logical depth, exploration, redundancy | N | N | N | N | N | reasoning-efficiency metrics | no deployment variation |
| ReasonBENCH (2512.07795) | 2025-12 | Instability of reasoning across repeated executions | 12 models, 10 strategies | single execution config, repeated runs | quality/cost distributions, greedy T=0 nondeterminism | N | N | N | **Y** | N | **closest to Gate 4**: within-condition variance as a first-class object | varies execution, **not deployment**; no precision/quant/runtime/hardware shift; no controller transfer |
| Quantization Hurts Reasoning? (2504.04823) | 2025-04 | Does quantization degrade reasoning models? | R1-distills, QwQ, Qwen3 | weight/KV/activation quant, single runtime | accuracy vs bit-width, output length | **Y** | N | N | ~ (reported per-config variance) | N | quantization vs reasoning accuracy/length | single runtime family; no reasoning-trajectory reproducibility; no controller transfer |
| Quantization Inflates Reasoning (CTIR, 2606.25519) | 2026-06 | Hidden token-inflation cost of low-bit reasoning | Qwen3 sizes | INT3/INT4 PTQ, single runtime (vLLM) | CoT token inflation, semantic repetition | **Y** | N | N | N | N | **quantization changes reasoning length/behavior** | single runtime; no noise floor; no cross-runtime/hardware; no controller transfer |
| Extreme Low-Bit Inference: Failure Modes (2606.02011) | 2026 | How do LRM traces fail at 2-bit? | Qwen3-32B/8B | FP16 vs 2-bit, single runtime | loops, budget exhaustion, think-closed rate, commit gap | **Y** | N | N | N | N | trace-level failure modes under quantization | single runtime; no noise floor; no transfer |
| Quantized Reasoning Models Think They Need to Think Longer (2606.00206) | 2026 | Overthinking under aggressive quantization | LRMs | PTQ, single runtime | intermediate-answer commitment | **Y** | N | N | N | N | quantization → overthinking | single runtime |
| Quantization Meets Reasoning (2501.03035 / 2505.11574) | 2025 | Error taxonomy of low-bit math reasoning | several | AWQ/GPTQ/SmoothQuant, single runtime | step-aligned error taxonomy, first-step flips | **Y** | N | N | N | N | reasoning-step error analysis | single runtime |
| BitCal-TTS (2605.05561) | 2026-05 | Can bit-aware calibration fix halting under 4-bit? | Qwen2.5 3B/7B/14B | BF16 vs 4-bit; single runtime (HF/bitsandbytes) | entropy, trace/hidden stability, premature stops | **Y** | N | N | N | **N** (it re-calibrates per bit-width; it does not transfer a frozen policy) | **closest to H3**: stopping calibration under quantization | no frozen-policy transfer test; single runtime; greedy only; tiny N; no noise floor |
| MARS (2606.12935) | 2026 | Risk-controlled stopping for parallel test-time scaling | R1-8B, Qwen3 | single runtime | vote margin, switch probabilities | N | N | N | ~ (parallel sampling) | N | adaptive stopping | parallel-decoding setting; no deployment variation |
| ORCA: Online Reasoning Calibration (2604.01170) | 2026 | Test-time training for generalizable conformal reasoning | several | single runtime | conformal risk | N | N | N | N | ~ (generalization across tasks, not deployments) | calibration generalization | task generalization, not deployment transfer |
| Understanding/Mitigating Numerical Nondeterminism (NeurIPS 2025, 2506.09501) | 2025-06 | Precision vs inference reproducibility | LLMs | FP32/FP16/BF16, 12 runtime configs, GPU | none (token identity) | ~ | ~ (kernel configs) | ~ (GPU) | **Y** | N | **precision → nondeterminism** | no reasoning dynamics; no cross-runtime/hardware reasoning; no controller transfer |
| Beyond Reproducibility: Token Probabilities Expose Nondeterminism (2601.06118) | 2026-01 | Nondeterminism at the token-probability level | LLMs incl. Ascend | multiple accelerators/hardware | none (token probabilities) | N | ~ | **Y** | **Y** | N | hardware → probabilistic variation | not reasoning; no controller transfer |
| Non-Determinism of "Deterministic" LLM Settings (2408.04667) | 2024-08 | Are T=0 outputs deterministic? | several | repeated calls | none | N | ~ | ~ | **Y** | N | greedy nondeterminism | not reasoning |
| Defeating Nondeterminism in LLM Inference (Thinking Machines) | 2025-09 | Batch-invariant kernels | — | vLLM | none | N | N | N | **Y** | N | determinism engineering | not reasoning |
| LLM-42 (2601.17768) | 2026 | Deterministic inference via verified speculation | — | SGLang | none | N | N | N | **Y** | N | determinism engineering | not reasoning |
| Production-Grade Local LLM Inference on Apple Silicon (2511.05502) | 2025-10 | Compare MLX/MLC/Ollama/llama.cpp/PyTorch MPS | Qwen-2.5 | 5 runtimes, M2 Ultra | none (perf/latency) | ~ | **Y** | **Y** | N | N | **multi-runtime on Apple Silicon** | performance only; no reasoning behavior |
| Native LLM/MLLM Inference on Apple Silicon (vllm-mlx, 2601.19139) | 2026-01 | MLX throughput vs llama.cpp | Qwen3 etc. | MLX vs llama.cpp, M4 Max | none (throughput) | ~ | **Y** | N | N | N | multi-runtime | throughput only |
| The Illusion of Equivalency (2607.08734) | 2026 | Do quantized models preserve behavior? | Llama/Mistral/Vicuna | llama.cpp legacy/K-quants | correctness agreement, behavioral drift | **Y** | N | N | ~ | N | quantization → behavioral drift | single runtime; not reasoning trajectories |
| Which Quantization Should I Use? (2601.14277) | 2026 | Unified llama.cpp quant evaluation | Llama-3.1-8B | llama.cpp quants | accuracy, PPL | **Y** | N | N | N | N | quant eval | no reasoning-trajectory reproducibility |
| The Illusion of Thinking (NeurIPS 2025) | 2025 | Reasoning vs problem complexity | o1/R1-class + others | single runtime | accuracy vs thinking tokens | N | N | N | N | N | overthinking vs complexity | no deployment variation |
| Reasoning on a Budget (survey, 2507.02076) | 2025-07 | Survey of adaptive/controllable test-time compute | many | — | survey | ~ | ~ | ~ | N | N | landscape | none directly |

## Closest competing work

1. **ReasonBENCH (2512.07795)** is the closest to PonderScope's noise-floor gate.
   It establishes within-condition instability as a first-class object and shows
   that even greedy (T=0) decoding varies across repeated executions. However it
   varies the *execution*, not the *deployment configuration*: same runtime,
   same precision, same hardware. It does not measure cross-deployment
   reasoning-trajectory differences and does not test stopping-policy transfer.

2. **BitCal-TTS (2605.05561)** is the closest to the policy-transfer hypothesis
   (H3). It explicitly recognises that quantization distorts the signals a
   controller uses, and it *recalibrates* confidence per bit-width. Crucially it
   does not freeze a controller calibrated on deployment A and evaluate it
   unchanged on deployment B, and it is single-runtime, greedy-only, with small
   evaluation shards.

3. **Quantization Inflates Reasoning / CTIR (2606.25519)** and **Extreme Low-Bit
   Inference (2606.02011)** establish that quantization changes reasoning-trace
   behaviour (length, repetition, loops, budget exhaustion) even when accuracy is
   preserved. Both are single-runtime and neither establishes a within-condition
   noise floor nor tests controller transfer.

4. **Numerical nondeterminism work (2506.09501, 2601.06118, 2408.04667)** shows
   that precision and hardware affect reproducibility at the token/probability
   level, but does not connect this to reasoning dynamics or adaptive compute.

5. **Apple-Silicon runtime comparisons (2511.05502, 2601.19139)** compare MLX,
   llama.cpp, and others, but purely on latency/throughput, never on reasoning
   trajectories.

## Gap assessment

No located work simultaneously:

1. holds the weights and immutable revision **exactly fixed** while varying the
   *deployment configuration* (precision, quantization, runtime, hardware);
2. first establishes a **within-configuration noise floor** on *reasoning
   dynamics* (length, repetition, transitions, termination), separating
   deterministic-replay failure from seed-driven and task-driven variation;
3. measures **cross-deployment reasoning-trajectory reproducibility**; and
4. tests whether an adaptive-compute / stopping policy **calibrated on
   deployment A transfers unchanged to deployment B**, versus recalibration.

The closest neighbours each cover one axis (ReasonBENCH: execution variance;
BitCal-TTS: quantization-aware recalibration; CTIR: quantization trace
inflation) but none covers the combination. **The proposed gap remains
defensible**, conditional on PonderScope's measured gates actually working (see
`RESULTS.md` and the GO/NO-GO decision there).

PonderScope explicitly does **not** claim novelty for: overthinking
measurement, answer-stability stopping, prefix probing / forced answer
elicitation, confidence-based early exit, quantization effects on reasoning, or
deterministic-inference engineering. It reuses the prefix probe strictly as a
measurement primitive.
