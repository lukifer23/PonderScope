# PonderScope report — 20261006T022442Z-calibration-4b-generated-history-api-seed0-dep-e1c20569173a

_Generated from saved evidence; no numbers are hand-entered._

## Deployment identity
- description: `Qwen/Qwen3.5-4B@851bf6e806ef (original, bfloat16) under mlx-lm 0.32.0 [mlx-metal, Apple M3 Pro] src=src-6209ef70e318 wvar=wvar-9b6f9b40b789 dep=dep-e1c20569173a`
- source artifact id: `src-6209ef70e318`
- weight variant id: `wvar-9b6f9b40b789`
- deployment id: `dep-e1c20569173a`
- prompt policy: `pp-v1`
- model policy: `qwen35-4b-thinking-api-v1` (label `qwen-generated-history-presence-v1`) recommended=`{'temperature': 1.0, 'top_p': 0.95, 'top_k': 20, 'min_p': 0.0, 'presence_penalty': 1.5, 'repetition_penalty': 1.0, 'enable_thinking': True}`
- condition ids: `cond-f5e762d3078f`
- model repo: `Qwen/Qwen3.5-4B`
- revision: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- representation: `original`
- precision: `bfloat16`
- quantization: `None` bits=None group=None
- runtime: `mlx-lm 0.32.0` backend=`mlx-metal`
- hardware: `Apple M3 Pro` os=`Darwin 27.2.0`
- created: `20261006T022442Z`

## Machine / runtime environment
- python: `3.12.12`
- platform: `macOS-27.2-arm64-arm-64bit`
- mlx device: `Device(gpu, 0)`
- memory bytes: `19327352832`
- packages: `{'mlx': '0.32.3', 'mlx-lm': '0.32.0', 'mlx-metal': '0.32.3', 'numpy': '2.5.3', 'transformers': '5.18.0', 'tokenizers': '0.23.2', 'huggingface-hub': '1.33.0', 'safetensors': '0.8.0', 'ponderscope': '0.1.0'}`

## Task population
- n tasks: 10  n generations: 10
- spec: `{'name': 'calibration-4b-generated-history-api-seed0', 'task_pack': 'tasks-v1', 'prompt_policy': 'pp-v1', 'model_policy': 'qwen35-4b-thinking-api-v1', 'families': ['arith', 'path', 'order', 'logic', 'sm'], 'n_per_family': 2, 'task_seed': 0, 'split': 'calibration', 'greedy_repeats': 0, 'sampled_seeds': [0], 'sampled_repeats_per_seed': 1, 'max_tokens': 2048, 'sampled_temperature': 1.0, 'sampled_top_p': 0.95, 'sampled_top_k': 20, 'sampled_min_p': 0.0, 'sampled_presence_penalty': 1.5, 'sampled_presence_context_size': 20, 'sampled_presence_scope': 'generated_history', 'sampled_repetition_penalty': 1.0, 'sampled_repetition_context_size': 20, 'sampled_frequency_penalty': 0.0, 'sampled_frequency_context_size': 20, 'probe': False, 'n_probes': 4, 'probe_max_tokens': 64, 'capture_entropy': True, 'capture_top_k': 5, 'capture_logprob_digest': False, 'capture_level': 'minimal', 'notes': 'Stage B pilot: Qwen3.5-4B native BF16, same tasks-v1 calibration presentations, pp-v1, same generated-history thinking semantics as Stage A2, 2048 horizon, one predeclared seed (0) = 10 draws. Extend to seeds {0,1,2} only if execution is clean and runtime is reasonable.'}`

## Configurations
| condition_id | mode | exec | draws | success_at_budget (95% CI) | completion_rate | censored_rate | conditional_acc_given_completed | reasoning tokens mean | tok/s |
|---|---|---|---|---|---|---|---|---|---|
| cond-f5e762d3078f | sampled | 10 | 10 | 0.7 [0.4, 1] | 0.7 | 0.3 | 1 | 1394 | 13.17 |

## Reasoning-length distribution

### cond-f5e762d3078f (sampled)

```
     522-598         1 #############
     598-675         1 #############
     675-751         0 
     751-827         0 
     827-904         0 
     904-980         1 #############
     980-1056        1 #############
    1056-1132        0 
    1132-1209        0 
    1209-1285        0 
    1285-1361        0 
    1361-1438        0 
    1438-1514        1 #############
    1514-1590        1 #############
    1590-1666        1 #############
    1666-1743        0 
    1743-1819        0 
    1819-1895        0 
    1895-1972        0 
    1972-2048        3 ########################################
```
- {"n": 10, "mean": 1394.3, "median": 1513.5, "std": 578.3128814827567, "min": 522.0, "max": 2048.0, "p10": 654.3, "p90": 2048.0}

## Latency / throughput
- cond-f5e762d3078f: wall_ms {"n": 10, "mean": 116908.08349180008, "median": 134575.88731249416, "std": 40572.63887005343, "min": 46543.10904099839, "max": 160132.80870800372, "p10": 62350.222441591904, "p90": 151984.6911089102}; tok/s {"n": 10, "mean": 13.167748150304215, "median": 13.412175047496657, "std": 0.4302252463742947, "min": 12.463621439323532, "max": 13.576803985502808, "p10": 12.63580326470173, "p90": 13.557892171709137}; ttft_ms {"n": 10, "mean": 523.7379165992024, "median": 399.7419999941485, "std": 380.76060143716217, "min": 265.52545800223015, "max": 1580.7626659952803, "p10": 348.51914609462256, "p90": 679.0941038037996}

## Repetition
- cond-f5e762d3078f: unique_token_ratio {"n": 10, "mean": 0.20756923489191056, "median": 0.21724451351498836, "std": 0.05413649030635501, "min": 0.11813904861474124, "max": 0.266553480475382, "p10": 0.13151833835901108, "p90": 0.2617627699225382}; repeated_ngram_fraction_4 {"n": 10, "mean": 0.3976140724823456, "median": 0.35378545410479945, "std": 0.16663707385191556, "min": 0.2181622783449758, "max": 0.6848167539267016, "p10": 0.23306317159977877, "p90": 0.6207376228906193}; text_repeat_ratio {"n": 10, "mean": 0.05657732223187887, "median": 0.04691762138570649, "std": 0.046202494532634274, "min": 0.0, "max": 0.16417910447761197, "p10": 0.01888111888111884, "p90": 0.10271928031077492}

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

- capped_length: 3
- eos_observed: 7
- missing_answer: 3
- natural_eos: 7
- status:censored: 3
- status:correct: 7
- think_end_reached: 7

### Budget outcomes per condition
| condition_id | success_at_budget | completion_rate | censored_rate | error_rate | unparseable_rate | cond_acc|completed |
|---|---|---|---|---|---|---|
| cond-f5e762d3078f | 0.7 | 0.7 | 0.3 | 0 | 0 | 1 |

## Censor-aware time-to-closure (Kaplan-Meier / RMST)

- observation horizon tau: 2048.0 tokens (an observation horizon, not a natural stopping threshold)
- event: PRIMARY: native think-end reasoning closure observed
- secondary event: SECONDARY: EOS observed (generation termination)
- analysis unit: unique stochastic draws (same-seed technical repeats collapsed)

| condition | exec | draws (used) | events | censored | median tokens | RMST(tau) |
|---|---|---|---|---|---|---|
| cond-f5e762d3078f | 10 | 10 (10) | 7 | 3 | 1718 | 1547 |

Secondary endpoint (generation termination via EOS):

| condition | events | censored | median tokens | RMST(tau) |
|---|---|---|---|---|
| cond-f5e762d3078f | 7 | 3 | 1718 | 1547 |

### cond-f5e762d3078f by family
| family | n | events | censored | median | RMST |
|---|---|---|---|---|---|
| arith | 2 | 2 | 0 | 589 | 903.5 |
| logic | 2 | 0 | 2 | — | 2048 |
| order | 2 | 2 | 0 | 799 | 1258 |
| path | 2 | 1 | 1 | 1864 | 1956 |
| sm | 2 | 2 | 0 | 1222 | 1568 |

## Loop-structure diagnostics (descriptive)

Descriptive loop structure with an explicitly documented onset rule; not a validated universal loop detector.

| condition | n | longest_run median (max) | special longest runs | with degeneration onset |
|---|---|---|---|---|
| cond-f5e762d3078f | 10 | 1.5 (3) | 0 | 0 |

## Prefix probes / trajectory states

- no probes in this run

## Limitations
- Single model revision on a single machine; results do not generalize to other weights or hardware.
- Greedy determinism is a same-process, same-machine replay test.
- Where the reasoning channel was never closed, the trajectory is censored, not a natural stop.
- No causal claims: deployment differences are associations under this protocol.
