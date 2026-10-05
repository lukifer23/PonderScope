# PonderScope report — 20261005T231850Z-calibration-pp2-dep-d7a370157187

_Generated from saved evidence; no numbers are hand-entered._

## Deployment identity
- description: `Qwen/Qwen3.5-0.8B@2fc06364715b (original, bfloat16) under mlx-lm 0.32.0 [mlx-metal, Apple M3 Pro] greedy src=src-52161236e42a wvar=wvar-46d3e044d08b dep=dep-d7a370157187 cond=cond-2f3369427321`
- source artifact id: `src-52161236e42a`
- weight variant id: `wvar-46d3e044d08b`
- deployment id: `dep-d7a370157187`
- prompt policy: `pp-v2`
- condition ids: `cond-2f3369427321`
- model repo: `Qwen/Qwen3.5-0.8B`
- revision: `2fc06364715b967f1860aea9cf38778875588b17`
- representation: `original`
- precision: `bfloat16`
- quantization: `None` bits=None group=None
- runtime: `mlx-lm 0.32.0` backend=`mlx-metal`
- hardware: `Apple M3 Pro` os=`Darwin 27.2.0`
- created: `20261005T231850Z`

## Machine / runtime environment
- python: `3.12.12`
- platform: `macOS-27.2-arm64-arm-64bit`
- mlx device: `Device(gpu, 0)`
- memory bytes: `19327352832`
- packages: `{'mlx': '0.32.3', 'mlx-lm': '0.32.0', 'mlx-metal': '0.32.3', 'numpy': '2.5.3', 'transformers': '5.18.0', 'tokenizers': '0.23.2', 'huggingface-hub': '1.33.0', 'safetensors': '0.8.0', 'ponderscope': '0.1.0'}`

## Task population
- n tasks: 10  n generations: 10
- spec: `{'name': 'calibration-pp2', 'task_pack': 'tasks-v1', 'prompt_policy': 'pp-v2', 'families': ['arith', 'path', 'order', 'logic', 'sm'], 'n_per_family': 2, 'task_seed': 0, 'split': 'calibration', 'greedy_repeats': 1, 'sampled_seeds': [], 'sampled_repeats_per_seed': 1, 'max_tokens': 2048, 'sampled_temperature': 0.6, 'sampled_top_p': 0.95, 'sampled_top_k': 20, 'sampled_min_p': 0.0, 'probe': True, 'n_probes': 4, 'probe_max_tokens': 64, 'capture_entropy': True, 'capture_top_k': 5, 'capture_logprob_digest': False, 'capture_level': 'minimal', 'notes': 'Termination calibration, prompt policy pp-v2 (minimal neutral instruction). Matched structures to pp-v1; generous 2048 horizon; greedy + minimal capture.'}`

## Configurations
| condition_id | mode | n | success_at_budget (95% CI) | completion_rate | censored_rate | conditional_acc_given_completed | reasoning tokens mean | tok/s |
|---|---|---|---|---|---|---|---|---|
| cond-2f3369427321 | greedy | 10 | 0.1 [0, 0.3] | 0.1 | 0.9 | 1 | 1882 | 62.49 |

## Reasoning-length distribution

### cond-2f3369427321 (greedy)

```
     385-468         1 ####
     468-551         0 
     551-634         0 
     634-718         0 
     718-801         0 
     801-884         0 
     884-967         0 
     967-1050        0 
    1050-1133        0 
    1133-1216        0 
    1216-1300        0 
    1300-1383        0 
    1383-1466        0 
    1466-1549        0 
    1549-1632        0 
    1632-1715        0 
    1715-1799        0 
    1799-1882        0 
    1882-1965        0 
    1965-2048        9 ########################################
```
- {"n": 10, "mean": 1881.7, "median": 2048.0, "std": 525.8867748860015, "min": 385.0, "max": 2048.0, "p10": 1881.7, "p90": 2048.0}

## Latency / throughput
- cond-2f3369427321: wall_ms {"n": 10, "mean": 30518.04660420312, "median": 32083.52577050391, "std": 9029.947310313608, "min": 5715.543167010765, "max": 37079.88070900319, "p10": 27831.27852890175, "p90": 36216.94863369921}; tok/s {"n": 10, "mean": 62.49369885738355, "median": 63.84693157556724, "std": 4.758663776633255, "min": 55.23210864868653, "max": 68.23498460321672, "p10": 56.55160338438565, "p90": 67.6781126554841}; ttft_ms {"n": 10, "mean": 67.4229958007345, "median": 62.877791504433844, "std": 16.96577822427294, "min": 43.67508299765177, "max": 99.58991699386388, "p10": 51.25803389819339, "p90": 89.07649139291607}

## Repetition
- cond-2f3369427321: unique_token_ratio {"n": 10, "mean": 0.0896073717948718, "median": 0.065673828125, "std": 0.08093647969234956, "min": 0.04443359375, "max": 0.31794871794871793, "p10": 0.053662109375, "p90": 0.10606244991987171}; repeated_ngram_fraction_4 {"n": 10, "mean": 0.7929555290208044, "median": 0.8398533007334963, "std": 0.15830041578265, "min": 0.35400516795865633, "max": 0.8997555012224939, "p10": 0.7307550400232495, "p90": 0.8865525672371638}; text_repeat_ratio {"n": 10, "mean": 0.4243672825524641, "median": 0.4697535052012664, "std": 0.2365128648334008, "min": 0.0, "max": 0.6844919786096257, "p10": 0.01636363636363637, "p90": 0.6299170877692195}

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

- capped_length: 9
- eos_observed: 1
- missing_answer: 9
- natural_eos: 1
- status:censored: 9
- status:correct: 1
- think_end_reached: 1

### Budget outcomes per condition
| condition_id | success_at_budget | completion_rate | censored_rate | error_rate | unparseable_rate | cond_acc|completed |
|---|---|---|---|---|---|---|
| cond-2f3369427321 | 0.1 | 0.1 | 0.9 | 0 | 0 | 1 |

## Prefix probes / trajectory states

- n probes: 50
- stable-sufficient WITH observed natural final: 1
- observed-probe stable (natural final unobserved or uncounted): 3
- multiple_flips: 1
- never_correct: 6
- wrong_to_correct: 3

## Limitations
- Single model revision on a single machine; results do not generalize to other weights or hardware.
- Greedy determinism is a same-process, same-machine replay test.
- Where the reasoning channel was never closed, the trajectory is censored, not a natural stop.
- No causal claims: deployment differences are associations under this protocol.
