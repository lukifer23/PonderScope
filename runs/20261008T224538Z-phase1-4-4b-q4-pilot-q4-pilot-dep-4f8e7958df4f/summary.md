# PonderScope report — 20261008T224538Z-phase1-4-4b-q4-pilot-q4-pilot-dep-4f8e7958df4f

_Generated from saved evidence; no numbers are hand-entered._

## Deployment identity
- description: `Qwen/Qwen3.5-4B@851bf6e806ef (derived, bfloat16 quant=mlx-affine4bit g64) under mlx-lm 0.32.0 [mlx-metal, Apple M3 Pro] src=src-6209ef70e318 wvar=wvar-b7a30758909a dep=dep-4f8e7958df4f`
- source artifact id: `src-6209ef70e318`
- weight variant id: `wvar-b7a30758909a`
- deployment id: `dep-4f8e7958df4f`
- prompt policy: `pp-v1`
- model policy: `qwen35-4b-thinking-api-v1` (label `qwen-generated-history-presence-v1`) recommended=`{'temperature': 1.0, 'top_p': 0.95, 'top_k': 20, 'min_p': 0.0, 'presence_penalty': 1.5, 'repetition_penalty': 1.0, 'enable_thinking': True}`
- condition ids: `cond-f5e762d3078f`
- model repo: `Qwen/Qwen3.5-4B`
- revision: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- representation: `derived`
- precision: `bfloat16`
- quantization: `mlx-affine` bits=4 group=64
- runtime: `mlx-lm 0.32.0` backend=`mlx-metal`
- hardware: `Apple M3 Pro` os=`Darwin 27.2.0`
- created: `20261008T224538Z`

## Machine / runtime environment
- python: `3.12.12`
- platform: `macOS-27.2-arm64-arm-64bit`
- mlx device: `Device(gpu, 0)`
- memory bytes: `19327352832`
- packages: `{'mlx': '0.32.3', 'mlx-lm': '0.32.0', 'mlx-metal': '0.32.3', 'numpy': '2.5.3', 'transformers': '5.18.0', 'tokenizers': '0.23.2', 'huggingface-hub': '1.33.0', 'safetensors': '0.8.0', 'ponderscope': '0.1.0'}`

## Task population
- n tasks: 10  n generations: 10
- spec: `{'name': 'phase1-4-4b-q4-pilot', 'task_pack': 'tasks-v1', 'prompt_policy': 'pp-v1', 'model_policy': 'qwen35-4b-thinking-api-v1', 'families': ['arith', 'path', 'order', 'logic', 'sm'], 'n_per_family': 2, 'task_seed': 0, 'split': 'calibration', 'greedy_repeats': 0, 'sampled_seeds': [0], 'sampled_repeats_per_seed': 1, 'max_tokens': 2048, 'sampled_temperature': 1.0, 'sampled_top_p': 0.95, 'sampled_top_k': 20, 'sampled_min_p': 0.0, 'sampled_presence_penalty': 1.5, 'sampled_presence_context_size': 20, 'sampled_presence_scope': 'generated_history', 'sampled_repetition_penalty': 1.0, 'sampled_repetition_context_size': 20, 'sampled_frequency_penalty': 0.0, 'sampled_frequency_context_size': 20, 'probe': False, 'n_probes': 4, 'probe_max_tokens': 64, 'capture_entropy': True, 'capture_top_k': 5, 'capture_logprob_digest': False, 'capture_level': 'minimal', 'notes': 'Phase 1.4 pilot: Qwen3.5-4B controlled MLX affine Q4 derived from the pinned revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a, supplied via --artifact-path. Identical tasks-v1 calibration presentations, pp-v1, generated-history thinking policy, and 2048 horizon as the BF16 baseline; one predeclared seed (0) = 10 stochastic draws. This is a compatibility/operational validation step, NOT a statistically powered conclusion.'}`

## Configurations
| condition_id | mode | exec | draws | success_at_budget (95% CI) | completion_rate | censored_rate | conditional_acc_given_completed | reasoning tokens mean | tok/s |
|---|---|---|---|---|---|---|---|---|---|
| cond-f5e762d3078f | sampled | 10 | 10 | 0.6 [0.3, 0.9] | 0.6 | 0.3 | 1 | 1194 | 40.71 |

## Reasoning-length distribution

### cond-f5e762d3078f (sampled)

```
     296-384         1 #############
     384-471         1 #############
     471-559         0 
     559-646         0 
     646-734         1 #############
     734-822         1 #############
     822-909         1 #############
     909-997         1 #############
     997-1084        0 
    1084-1172        0 
    1172-1260        0 
    1260-1347        0 
    1347-1435        0 
    1435-1522        0 
    1522-1610        0 
    1610-1698        0 
    1698-1785        0 
    1785-1873        1 #############
    1873-1960        0 
    1960-2048        3 ########################################
```
- {"n": 10, "mean": 1194.5, "median": 875.5, "std": 722.1503767683478, "min": 296.0, "max": 2048.0, "p10": 378.8, "p90": 2048.0}

## Latency / throughput
- cond-f5e762d3078f: wall_ms {"n": 10, "mean": 32121.418633299618, "median": 27575.397312499263, "std": 15941.502405334479, "min": 8979.17283300194, "max": 50958.58687500004, "p10": 12605.322820795118, "p90": 49956.88526219965}; tok/s {"n": 10, "mean": 40.70869382795842, "median": 40.92876630173228, "std": 2.3260974630970512, "min": 35.43910721049283, "max": 43.84487331978003, "p10": 39.317061837747666, "p90": 43.648400581222624}; ttft_ms {"n": 10, "mean": 453.403128998616, "median": 295.66220800188603, "std": 495.19407277565466, "min": 173.1075419957051, "max": 1842.2660829965025, "p10": 220.47637889481848, "p90": 614.8443576938002}

## Repetition
- cond-f5e762d3078f: unique_token_ratio {"n": 10, "mean": 0.22575438662582714, "median": 0.188232421875, "std": 0.0887746629844338, "min": 0.13604060913705585, "max": 0.3644251626898048, "p10": 0.1375965632567234, "p90": 0.36171425539941526}; repeated_ngram_fraction_4 {"n": 10, "mean": 0.3884109240323173, "median": 0.3535452322738386, "std": 0.18685455526129477, "min": 0.17234262125902994, "max": 0.6733102253032929, "p10": 0.20463152239987561, "p90": 0.6577445563649157}; text_repeat_ratio {"n": 10, "mean": 0.04741509279229287, "median": 0.03868630201028517, "std": 0.05620220058962184, "min": 0.0, "max": 0.18181818181818177, "p10": 0.0, "p90": 0.0907624633431085}

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

- capped_length: 4
- eos_observed: 6
- missing_answer: 3
- natural_eos: 6
- status:censored: 3
- status:correct: 6
- status:incorrect: 1
- think_end_reached: 7

### Budget outcomes per condition
| condition_id | success_at_budget | completion_rate | censored_rate | error_rate | unparseable_rate | cond_acc|completed |
|---|---|---|---|---|---|---|
| cond-f5e762d3078f | 0.6 | 0.6 | 0.3 | 0 | 0 | 1 |

## Censor-aware time-to-closure (Kaplan-Meier / RMST)

- observation horizon tau: 2048.0 tokens (an observation horizon, not a natural stopping threshold)
- event: PRIMARY: native think-end reasoning closure observed
- secondary event: SECONDARY: EOS observed (generation termination)
- analysis unit: unique stochastic draws (same-seed technical repeats collapsed)

| condition | exec | draws (used) | events | censored | median tokens | RMST(tau) |
|---|---|---|---|---|---|---|
| cond-f5e762d3078f | 10 | 10 (10) | 7 | 3 | 833 | 1195 |

Secondary endpoint (generation termination via EOS):

| condition | events | censored | median tokens | RMST(tau) |
|---|---|---|---|---|
| cond-f5e762d3078f | 6 | 4 | 833 | 1213 |

### cond-f5e762d3078f by family
| family | n | events | censored | median | RMST |
|---|---|---|---|---|---|
| arith | 2 | 2 | 0 | 389 | 557 |
| logic | 2 | 0 | 2 | — | 2048 |
| order | 2 | 2 | 0 | 297 | 1081 |
| path | 2 | 1 | 1 | 779 | 1414 |
| sm | 2 | 2 | 0 | 833 | 876.5 |

## Loop-structure diagnostics (descriptive)

Descriptive loop structure with an explicitly documented onset rule; not a validated universal loop detector.

| condition | n | longest_run median (max) | special longest runs | with degeneration onset |
|---|---|---|---|---|
| cond-f5e762d3078f | 10 | 1 (3) | 0 | 0 |

## Prefix probes / trajectory states

- no probes in this run

## Limitations
- Single model revision on a single machine; results do not generalize to other weights or hardware.
- Greedy determinism is a same-process, same-machine replay test.
- Where the reasoning channel was never closed, the trajectory is censored, not a natural stop.
- No causal claims: deployment differences are associations under this protocol.
