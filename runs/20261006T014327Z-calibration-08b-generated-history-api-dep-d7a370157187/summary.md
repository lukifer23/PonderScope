# PonderScope report — 20261006T014327Z-calibration-08b-generated-history-api-dep-d7a370157187

_Generated from saved evidence; no numbers are hand-entered._

## Deployment identity
- description: `Qwen/Qwen3.5-0.8B@2fc06364715b (original, bfloat16) under mlx-lm 0.32.0 [mlx-metal, Apple M3 Pro] src=src-52161236e42a wvar=wvar-46d3e044d08b dep=dep-d7a370157187`
- source artifact id: `src-52161236e42a`
- weight variant id: `wvar-46d3e044d08b`
- deployment id: `dep-d7a370157187`
- prompt policy: `pp-v1`
- model policy: `qwen35-08b-thinking-api-v2` (label `qwen-generated-history-presence-v1`) recommended=`{'temperature': 1.0, 'top_p': 0.95, 'top_k': 20, 'min_p': 0.0, 'presence_penalty': 1.5, 'repetition_penalty': 1.0, 'enable_thinking': True}`
- condition ids: `cond-f5e762d3078f`
- model repo: `Qwen/Qwen3.5-0.8B`
- revision: `2fc06364715b967f1860aea9cf38778875588b17`
- representation: `original`
- precision: `bfloat16`
- quantization: `None` bits=None group=None
- runtime: `mlx-lm 0.32.0` backend=`mlx-metal`
- hardware: `Apple M3 Pro` os=`Darwin 27.2.0`
- created: `20261006T014327Z`

## Machine / runtime environment
- python: `3.12.12`
- platform: `macOS-27.2-arm64-arm-64bit`
- mlx device: `Device(gpu, 0)`
- memory bytes: `19327352832`
- packages: `{'mlx': '0.32.3', 'mlx-lm': '0.32.0', 'mlx-metal': '0.32.3', 'numpy': '2.5.3', 'transformers': '5.18.0', 'tokenizers': '0.23.2', 'huggingface-hub': '1.33.0', 'safetensors': '0.8.0', 'ponderscope': '0.1.0'}`

## Task population
- n tasks: 10  n generations: 30
- spec: `{'name': 'calibration-08b-generated-history-api', 'task_pack': 'tasks-v1', 'prompt_policy': 'pp-v1', 'model_policy': 'qwen35-08b-thinking-api-v2', 'families': ['arith', 'path', 'order', 'logic', 'sm'], 'n_per_family': 2, 'task_seed': 0, 'split': 'calibration', 'greedy_repeats': 0, 'sampled_seeds': [0, 1, 2], 'sampled_repeats_per_seed': 1, 'max_tokens': 2048, 'sampled_temperature': 1.0, 'sampled_top_p': 0.95, 'sampled_top_k': 20, 'sampled_min_p': 0.0, 'sampled_presence_penalty': 1.5, 'sampled_presence_context_size': 20, 'sampled_presence_scope': 'generated_history', 'sampled_repetition_penalty': 1.0, 'sampled_repetition_context_size': 20, 'sampled_frequency_penalty': 0.0, 'sampled_frequency_context_size': 20, 'probe': False, 'n_probes': 4, 'probe_max_tokens': 64, 'capture_entropy': True, 'capture_top_k': 5, 'capture_logprob_digest': False, 'capture_level': 'minimal', 'notes': 'Stage A2: same 0.8B revision, tasks-v1 calibration, pp-v1, upstream API/best-practice TEXT THINKING numeric values mapped with explicit generated-history presence semantics (qwen-generated-history-presence-v1). 2048-token observation horizon, minimal capture, seeds {0,1,2} x ONE primary execution = 30 stochastic draws. Same-seed technical replay is measured separately, not duplicated into N.'}`

## Configurations
| condition_id | mode | exec | draws | success_at_budget (95% CI) | completion_rate | censored_rate | conditional_acc_given_completed | reasoning tokens mean | tok/s |
|---|---|---|---|---|---|---|---|---|---|
| cond-f5e762d3078f | sampled | 30 | 30 | 0.03333 [0, 0.1] | 0.03333 | 0.9667 | 1 | 1986 | 62.33 |

## Reasoning-length distribution

### cond-f5e762d3078f (sampled)

```
     188-281         1 #
     281-374         0 
     374-467         0 
     467-560         0 
     560-653         0 
     653-746         0 
     746-839         0 
     839-932         0 
     932-1025        0 
    1025-1118        0 
    1118-1211        0 
    1211-1304        0 
    1304-1397        0 
    1397-1490        0 
    1490-1583        0 
    1583-1676        0 
    1676-1769        0 
    1769-1862        0 
    1862-1955        0 
    1955-2048       29 ########################################
```
- {"n": 30, "mean": 1986.0, "median": 2048.0, "std": 339.58798565320296, "min": 188.0, "max": 2048.0, "p10": 2048.0, "p90": 2048.0}

## Latency / throughput
- cond-f5e762d3078f: wall_ms {"n": 30, "mean": 31959.847894334234, "median": 32492.364186997293, "std": 5510.82241287111, "min": 3815.1255830016453, "max": 36796.06108299049, "p10": 31425.93062439264, "p90": 35453.28289619647}; tok/s {"n": 30, "mean": 62.327844477368664, "median": 62.89521038454738, "std": 2.633829024497424, "min": 55.65813132500526, "max": 66.62556707419373, "p10": 57.7668338852074, "p90": 64.97986073460291}; ttft_ms {"n": 30, "mean": 71.41295960124505, "median": 68.59439550316893, "std": 16.447657193369043, "min": 46.440249992883764, "max": 121.81458300619852, "p10": 50.99599620007211, "p90": 92.94540830305779}

## Repetition
- cond-f5e762d3078f: unique_token_ratio {"n": 30, "mean": 0.20820529513888889, "median": 0.209228515625, "std": 0.05705910476039936, "min": 0.11767578125, "max": 0.45416666666666666, "p10": 0.15302734375000002, "p90": 0.238525390625}; repeated_ngram_fraction_4 {"n": 30, "mean": 0.3591958019112858, "median": 0.3254278728606357, "std": 0.13371153067181135, "min": 0.17603911980440098, "max": 0.6210268948655256, "p10": 0.18668276025708477, "p90": 0.5530073349633252}; text_repeat_ratio {"n": 30, "mean": 0.05709009126308695, "median": 0.041960093896713624, "std": 0.06063156857861727, "min": 0.0, "max": 0.2616279069767442, "p10": 0.006081081081081121, "p90": 0.15506161971830987}

## Repeatability / noise floor

### 1. Greedy replay (same deterministic condition, repeated)
- no repeated greedy conditions in this run

### 2. Same-seed sampled replay (same seed + sampler, re-executed)
- no repeated same-seed sampled conditions in this run

### 3. Across-seed stochastic variation (fixed sampler policy, different seeds)
- task/condition groups: 10
- estimable answer diversity: 1 / 10 (proportion 0.1)
- mean distinct observed answers across seeds: 1 (support: 1 tasks)
- mean observed-answer accuracy std across seeds: —
- any ambiguous seeds (divergent same-seed repeats): False
- note: final-answer diversity/accuracy is defined only over seeds with an observed answer; `—` means unobserved, not zero. No mean is shown without its support count.

### 4. Across-seed token variation (valid even when all answers are censored)
- task/condition groups: 10
- mean first token divergence across seeds: 6.3
- mean reasoning-token std across seeds: 87.68

## Censoring / termination (do not read a capped run as a wrong answer)

- capped_length: 29
- eos_observed: 1
- missing_answer: 29
- natural_eos: 1
- status:censored: 29
- status:correct: 1
- think_end_reached: 1

### Budget outcomes per condition
| condition_id | success_at_budget | completion_rate | censored_rate | error_rate | unparseable_rate | cond_acc|completed |
|---|---|---|---|---|---|---|
| cond-f5e762d3078f | 0.03333 | 0.03333 | 0.9667 | 0 | 0 | 1 |

## Censor-aware time-to-closure (Kaplan-Meier / RMST)

- observation horizon tau: 2048.0 tokens (an observation horizon, not a natural stopping threshold)
- event: PRIMARY: native think-end reasoning closure observed
- secondary event: SECONDARY: EOS observed (generation termination)
- analysis unit: unique stochastic draws (same-seed technical repeats collapsed)

| condition | exec | draws (used) | events | censored | median tokens | RMST(tau) |
|---|---|---|---|---|---|---|
| cond-f5e762d3078f | 30 | 30 (30) | 1 | 29 | — | 1988 |

Secondary endpoint (generation termination via EOS):

| condition | events | censored | median tokens | RMST(tau) |
|---|---|---|---|---|
| cond-f5e762d3078f | 1 | 29 | — | 1988 |

### cond-f5e762d3078f by family
| family | n | events | censored | median | RMST |
|---|---|---|---|---|---|
| arith | 6 | 1 | 5 | — | 1747 |
| logic | 6 | 0 | 6 | — | 2048 |
| order | 6 | 0 | 6 | — | 2048 |
| path | 6 | 0 | 6 | — | 2048 |
| sm | 6 | 0 | 6 | — | 2048 |

## Loop-structure diagnostics (descriptive)

Descriptive loop structure with an explicitly documented onset rule; not a validated universal loop detector.

| condition | n | longest_run median (max) | special longest runs | with degeneration onset |
|---|---|---|---|---|
| cond-f5e762d3078f | 30 | 1.5 (4) | 0 | 1 |

## Prefix probes / trajectory states

- no probes in this run

## Limitations
- Single model revision on a single machine; results do not generalize to other weights or hardware.
- Greedy determinism is a same-process, same-machine replay test.
- Where the reasoning channel was never closed, the trajectory is censored, not a natural stop.
- No causal claims: deployment differences are associations under this protocol.
