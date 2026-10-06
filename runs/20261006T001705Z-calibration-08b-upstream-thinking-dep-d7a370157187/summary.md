# PonderScope report — 20261006T001705Z-calibration-08b-upstream-thinking-dep-d7a370157187

_Generated from saved evidence; no numbers are hand-entered._

## Deployment identity
- description: `Qwen/Qwen3.5-0.8B@2fc06364715b (original, bfloat16) under mlx-lm 0.32.0 [mlx-metal, Apple M3 Pro] greedy src=src-52161236e42a wvar=wvar-46d3e044d08b dep=dep-d7a370157187 cond=cond-2f3369427321`
- source artifact id: `src-52161236e42a`
- weight variant id: `wvar-46d3e044d08b`
- deployment id: `dep-d7a370157187`
- prompt policy: `pp-v1`
- model policy: `qwen35-08b-thinking-upstream-v1` (label `qwen-upstream-profile-on-mlx`) recommended=`{'temperature': 1.0, 'top_p': 0.95, 'top_k': 20, 'min_p': 0.0, 'presence_penalty': 1.5, 'repetition_penalty': 1.0, 'enable_thinking': True}`
- condition ids: `cond-74a29a115886`
- model repo: `Qwen/Qwen3.5-0.8B`
- revision: `2fc06364715b967f1860aea9cf38778875588b17`
- representation: `original`
- precision: `bfloat16`
- quantization: `None` bits=None group=None
- runtime: `mlx-lm 0.32.0` backend=`mlx-metal`
- hardware: `Apple M3 Pro` os=`Darwin 27.2.0`
- created: `20261006T001705Z`

## Machine / runtime environment
- python: `3.12.12`
- platform: `macOS-27.2-arm64-arm-64bit`
- mlx device: `Device(gpu, 0)`
- memory bytes: `19327352832`
- packages: `{'mlx': '0.32.3', 'mlx-lm': '0.32.0', 'mlx-metal': '0.32.3', 'numpy': '2.5.3', 'transformers': '5.18.0', 'tokenizers': '0.23.2', 'huggingface-hub': '1.33.0', 'safetensors': '0.8.0', 'ponderscope': '0.1.0'}`

## Task population
- n tasks: 10  n generations: 60
- spec: `{'name': 'calibration-08b-upstream-thinking', 'task_pack': 'tasks-v1', 'prompt_policy': 'pp-v1', 'model_policy': 'qwen35-08b-thinking-upstream-v1', 'families': ['arith', 'path', 'order', 'logic', 'sm'], 'n_per_family': 2, 'task_seed': 0, 'split': 'calibration', 'greedy_repeats': 0, 'sampled_seeds': [0, 1, 2], 'sampled_repeats_per_seed': 2, 'max_tokens': 2048, 'sampled_temperature': 1.0, 'sampled_top_p': 0.95, 'sampled_top_k': 20, 'sampled_min_p': 0.0, 'sampled_presence_penalty': 1.5, 'sampled_presence_context_size': 20, 'sampled_repetition_penalty': 1.0, 'sampled_repetition_context_size': 20, 'sampled_frequency_penalty': 0.0, 'sampled_frequency_context_size': 20, 'probe': False, 'n_probes': 4, 'probe_max_tokens': 64, 'capture_entropy': True, 'capture_top_k': 5, 'capture_logprob_digest': False, 'capture_level': 'minimal', 'notes': 'Stage A: same 0.8B revision, tasks-v1 calibration, pp-v1, upstream-recommended TEXT THINKING sampling mapped faithfully onto MLX-LM 0.32.0 (qwen-upstream-profile-on-mlx). 2048-token observation horizon, minimal capture, seeds {0,1,2} x 2 repeats to preserve same-seed replay evidence. Greedy is NOT the primary new condition.'}`

## Configurations
| condition_id | mode | n | success_at_budget (95% CI) | completion_rate | censored_rate | conditional_acc_given_completed | reasoning tokens mean | tok/s |
|---|---|---|---|---|---|---|---|---|
| cond-74a29a115886 | sampled | 60 | 0.06667 [0, 0.1667] | 0.06667 | 0.9333 | 1 | 1951 | 67.65 |

## Reasoning-length distribution

### cond-74a29a115886 (sampled)

```
     280-368         2 #
     368-457         0 
     457-545         0 
     545-634         0 
     634-722         0 
     722-810         0 
     810-899         0 
     899-987         2 #
     987-1076        0 
    1076-1164        0 
    1164-1252        0 
    1252-1341        0 
    1341-1429        0 
    1429-1518        0 
    1518-1606        0 
    1606-1694        0 
    1694-1783        0 
    1783-1871        0 
    1871-1960        0 
    1960-2048       56 ########################################
```
- {"n": 60, "mean": 1951.3, "median": 2048.0, "std": 374.119816754119, "min": 280.0, "max": 2048.0, "p10": 2048.0, "p90": 2048.0}

## Latency / throughput
- cond-74a29a115886: wall_ms {"n": 60, "mean": 28958.154518732776, "median": 29782.572562995483, "std": 5399.600286323016, "min": 5483.594165998511, "max": 35168.62354100158, "p10": 29132.98725840723, "p90": 31532.50069579226}; tok/s {"n": 60, "mean": 67.65058402294525, "median": 68.6954439684518, "std": 2.7243348091926127, "min": 58.233726367263856, "max": 70.73992837816972, "p10": 64.77693131962593, "p90": 70.21969279422397}; ttft_ms {"n": 60, "mean": 77.82349369954318, "median": 63.81941700237803, "std": 83.54479160120535, "min": 43.989541998598725, "max": 703.6982499994338, "p10": 47.337546189373825, "p90": 94.06250000465661}

## Repetition
- cond-74a29a115886: unique_token_ratio {"n": 60, "mean": 0.14072883164787797, "median": 0.130615234375, "std": 0.03581237676906193, "min": 0.08544921875, "max": 0.27851458885941643, "p10": 0.10874023437500001, "p90": 0.17182617187500002}; repeated_ngram_fraction_4 {"n": 60, "mean": 0.4678636707006822, "median": 0.45843520782396086, "std": 0.1334839552894706, "min": 0.25623471882640586, "max": 0.7286063569682152, "p10": 0.3308557457212714, "p90": 0.6425916870415648}; text_repeat_ratio {"n": 60, "mean": 0.08898819668575096, "median": 0.07111251580278127, "std": 0.07798574857969134, "min": 0.0, "max": 0.33526011560693647, "p10": 0.008338219008533621, "p90": 0.1738542297597416}

## Repeatability / noise floor

### 1. Greedy replay (same deterministic condition, repeated)
- no repeated greedy conditions in this run

### 2. Same-seed sampled replay (same seed + sampler, re-executed)
- task/condition/seed groups: 30
- token-identical rate: 1
- ambiguous (divergent) seed groups: 0
- answer-agreement rate (observed answers only): 1
- mean within-group reasoning-token std: 0

### 3. Across-seed stochastic variation (fixed sampler policy, different seeds)
- task/condition groups: 10
- mean distinct observed answers across seeds: 1
- mean observed-answer accuracy std across seeds: —
- any ambiguous seeds (divergent same-seed repeats): False
- note: final-answer diversity/accuracy is defined only over seeds with an observed answer; `—` means unobserved, not zero.

### 4. Across-seed token variation (valid even when all answers are censored)
- task/condition groups: 10
- mean first token divergence across seeds: 6.2
- mean reasoning-token std across seeds: 136.8

## Censoring / termination (do not read a capped run as a wrong answer)

- capped_length: 56
- eos_observed: 4
- missing_answer: 56
- natural_eos: 4
- status:censored: 56
- status:correct: 4
- think_end_reached: 4

### Budget outcomes per condition
| condition_id | success_at_budget | completion_rate | censored_rate | error_rate | unparseable_rate | cond_acc|completed |
|---|---|---|---|---|---|---|
| cond-74a29a115886 | 0.06667 | 0.06667 | 0.9333 | 0 | 0 | 1 |

## Censor-aware time-to-closure (Kaplan-Meier / RMST)

- observation horizon tau: 2048.0 tokens (an observation horizon, not a natural stopping threshold)
- event: natural native think-end or EOS observed

| condition | n | events | censored | median tokens | RMST(tau) |
|---|---|---|---|---|---|
| cond-74a29a115886 | 60 | 4 | 56 | — | 1957 |

### cond-74a29a115886 by family
| family | n | events | censored | median | RMST |
|---|---|---|---|---|---|
| arith | 12 | 4 | 8 | — | 1591 |
| logic | 12 | 0 | 12 | — | 2048 |
| order | 12 | 0 | 12 | — | 2048 |
| path | 12 | 0 | 12 | — | 2048 |
| sm | 12 | 0 | 12 | — | 2048 |

## Loop-structure diagnostics (descriptive)

Descriptive loop structure with an explicitly documented onset rule; not a validated universal loop detector.

| condition | n | longest_run median (max) | special longest runs | with degeneration onset |
|---|---|---|---|---|
| cond-74a29a115886 | 60 | 1 (3) | 0 | 4 |

## Prefix probes / trajectory states

- no probes in this run

## Limitations
- Single model revision on a single machine; results do not generalize to other weights or hardware.
- Greedy determinism is a same-process, same-machine replay test.
- Where the reasoning channel was never closed, the trajectory is censored, not a natural stop.
- No causal claims: deployment differences are associations under this protocol.
