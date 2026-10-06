# PonderScope report — 20261006T021446Z-calibration-08b-generated-history-benchmark-dep-d7a370157187

_Generated from saved evidence; no numbers are hand-entered._

## Deployment identity
- description: `Qwen/Qwen3.5-0.8B@2fc06364715b (original, bfloat16) under mlx-lm 0.32.0 [mlx-metal, Apple M3 Pro] src=src-52161236e42a wvar=wvar-46d3e044d08b dep=dep-d7a370157187`
- source artifact id: `src-52161236e42a`
- weight variant id: `wvar-46d3e044d08b`
- deployment id: `dep-d7a370157187`
- prompt policy: `pp-v1`
- model policy: `qwen35-08b-thinking-benchmark-v1` (label `qwen-generated-history-presence-v1`) recommended=`{'temperature': 0.6, 'top_p': 0.95, 'top_k': 20, 'min_p': 0.0, 'presence_penalty': 1.5, 'repetition_penalty': 1.0, 'enable_thinking': True}`
- condition ids: `cond-4555bfefa490`
- model repo: `Qwen/Qwen3.5-0.8B`
- revision: `2fc06364715b967f1860aea9cf38778875588b17`
- representation: `original`
- precision: `bfloat16`
- quantization: `None` bits=None group=None
- runtime: `mlx-lm 0.32.0` backend=`mlx-metal`
- hardware: `Apple M3 Pro` os=`Darwin 27.2.0`
- created: `20261006T021446Z`

## Machine / runtime environment
- python: `3.12.12`
- platform: `macOS-27.2-arm64-arm-64bit`
- mlx device: `Device(gpu, 0)`
- memory bytes: `19327352832`
- packages: `{'mlx': '0.32.3', 'mlx-lm': '0.32.0', 'mlx-metal': '0.32.3', 'numpy': '2.5.3', 'transformers': '5.18.0', 'tokenizers': '0.23.2', 'huggingface-hub': '1.33.0', 'safetensors': '0.8.0', 'ponderscope': '0.1.0'}`

## Task population
- n tasks: 10  n generations: 10
- spec: `{'name': 'calibration-08b-generated-history-benchmark', 'task_pack': 'tasks-v1', 'prompt_policy': 'pp-v1', 'model_policy': 'qwen35-08b-thinking-benchmark-v1', 'families': ['arith', 'path', 'order', 'logic', 'sm'], 'n_per_family': 2, 'task_seed': 0, 'split': 'calibration', 'greedy_repeats': 0, 'sampled_seeds': [0], 'sampled_repeats_per_seed': 1, 'max_tokens': 2048, 'sampled_temperature': 0.6, 'sampled_top_p': 0.95, 'sampled_top_k': 20, 'sampled_min_p': 0.0, 'sampled_presence_penalty': 1.5, 'sampled_presence_context_size': 20, 'sampled_presence_scope': 'generated_history', 'sampled_repetition_penalty': 1.0, 'sampled_repetition_context_size': 20, 'sampled_frequency_penalty': 0.0, 'sampled_frequency_context_size': 20, 'probe': False, 'n_probes': 4, 'probe_max_tokens': 64, 'capture_entropy': True, 'capture_top_k': 5, 'capture_logprob_digest': False, 'capture_level': 'minimal', 'notes': 'Small T=0.6 policy diagnostic: same 0.8B revision, tasks-v1 calibration, pp-v1, published non-video benchmark THINKING configuration mapped with explicit generated-history presence semantics. 2048-token observation horizon, minimal capture, one predeclared seed (0). NOT expanded into a full sweep.'}`

## Configurations
| condition_id | mode | exec | draws | success_at_budget (95% CI) | completion_rate | censored_rate | conditional_acc_given_completed | reasoning tokens mean | tok/s |
|---|---|---|---|---|---|---|---|---|---|
| cond-4555bfefa490 | sampled | 10 | 10 | 0 [0, 0] | 0 | 1 | — | 2048 | 62.98 |

## Reasoning-length distribution

### cond-4555bfefa490 (sampled)

```
    2048-2048        0 
    2048-2048        0 
    2048-2048        0 
    2048-2048        0 
    2048-2048        0 
    2048-2048        0 
    2048-2048        0 
    2048-2048        0 
    2048-2048        0 
    2048-2048        0 
    2048-2048       10 ########################################
    2048-2048        0 
    2048-2048        0 
    2048-2048        0 
    2048-2048        0 
    2048-2048        0 
    2048-2048        0 
    2048-2048        0 
    2048-2048        0 
    2048-2048        0 
```
- {"n": 10, "mean": 2048.0, "median": 2048.0, "std": 0.0, "min": 2048.0, "max": 2048.0, "p10": 2048.0, "p90": 2048.0}

## Latency / throughput
- cond-4555bfefa490: wall_ms {"n": 10, "mean": 32546.616470700246, "median": 32136.22739550192, "std": 1024.571344915889, "min": 31168.29504200723, "max": 34169.58245799469, "p10": 31729.76731670351, "p90": 33986.513270801515}; tok/s {"n": 10, "mean": 62.98064800969537, "median": 63.72959911734564, "std": 1.960170179955616, "min": 59.93634843263434, "max": 65.70779688910791, "p10": 60.2593903847898, "p90": 64.54734996877667}; ttft_ms {"n": 10, "mean": 90.46703729982255, "median": 73.6596874994575, "std": 56.02648641862, "min": 49.050208006519824, "max": 240.27341599867214, "p10": 57.161196097149514, "p90": 131.07911599799985}

## Repetition
- cond-4555bfefa490: unique_token_ratio {"n": 10, "mean": 0.16328125, "median": 0.172607421875, "std": 0.03233996645081976, "min": 0.107421875, "max": 0.20068359375, "p10": 0.126318359375, "p90": 0.198046875}; repeated_ngram_fraction_4 {"n": 10, "mean": 0.503960880195599, "median": 0.471638141809291, "std": 0.11882231404843065, "min": 0.32567237163814183, "max": 0.6777506112469438, "p10": 0.3960880195599022, "p90": 0.6680684596577018}; text_repeat_ratio {"n": 10, "mean": 0.0762008214682613, "median": 0.1001393625994792, "std": 0.06662112567711673, "min": 0.0, "max": 0.16874999999999996, "p10": 0.0, "p90": 0.13652817919075141}

## Repeatability / noise floor

### 1. Greedy replay (same deterministic condition, repeated)
- no repeated greedy conditions in this run

### 2. Same-seed sampled replay (same seed + sampler, re-executed)
- no repeated same-seed sampled conditions in this run

### 3. Across-seed stochastic variation (fixed sampler policy, different seeds)
- fewer than two seeds per task/condition

### 4. Across-seed token variation (valid even when all answers are censored)
- fewer than two seeds per task/condition

## Censoring / termination (do not read a capped run as a wrong answer)

- capped_length: 10
- missing_answer: 10
- status:censored: 10

### Budget outcomes per condition
| condition_id | success_at_budget | completion_rate | censored_rate | error_rate | unparseable_rate | cond_acc|completed |
|---|---|---|---|---|---|---|
| cond-4555bfefa490 | 0 | 0 | 1 | 0 | 0 | — |

## Censor-aware time-to-closure (Kaplan-Meier / RMST)

- observation horizon tau: 2048.0 tokens (an observation horizon, not a natural stopping threshold)
- event: PRIMARY: native think-end reasoning closure observed
- secondary event: SECONDARY: EOS observed (generation termination)
- analysis unit: unique stochastic draws (same-seed technical repeats collapsed)

| condition | exec | draws (used) | events | censored | median tokens | RMST(tau) |
|---|---|---|---|---|---|---|
| cond-4555bfefa490 | 10 | 10 (10) | 0 | 10 | — | 2048 |

Secondary endpoint (generation termination via EOS):

| condition | events | censored | median tokens | RMST(tau) |
|---|---|---|---|---|
| cond-4555bfefa490 | 0 | 10 | — | 2048 |

### cond-4555bfefa490 by family
| family | n | events | censored | median | RMST |
|---|---|---|---|---|---|
| arith | 2 | 0 | 2 | — | 2048 |
| logic | 2 | 0 | 2 | — | 2048 |
| order | 2 | 0 | 2 | — | 2048 |
| path | 2 | 0 | 2 | — | 2048 |
| sm | 2 | 0 | 2 | — | 2048 |

## Loop-structure diagnostics (descriptive)

Descriptive loop structure with an explicitly documented onset rule; not a validated universal loop detector.

| condition | n | longest_run median (max) | special longest runs | with degeneration onset |
|---|---|---|---|---|
| cond-4555bfefa490 | 10 | 1 (3) | 0 | 0 |

## Prefix probes / trajectory states

- no probes in this run

## Limitations
- Single model revision on a single machine; results do not generalize to other weights or hardware.
- Greedy determinism is a same-process, same-machine replay test.
- Where the reasoning channel was never closed, the trajectory is censored, not a natural stop.
- No causal claims: deployment differences are associations under this protocol.
