# PonderScope report — 20261005T231234Z-calibration-pp1-dep-d7a370157187

_Generated from saved evidence; no numbers are hand-entered._

## Deployment identity
- description: `Qwen/Qwen3.5-0.8B@2fc06364715b (original, bfloat16) under mlx-lm 0.32.0 [mlx-metal, Apple M3 Pro] greedy src=src-52161236e42a wvar=wvar-46d3e044d08b dep=dep-d7a370157187 cond=cond-2f3369427321`
- source artifact id: `src-52161236e42a`
- weight variant id: `wvar-46d3e044d08b`
- deployment id: `dep-d7a370157187`
- prompt policy: `pp-v1`
- condition ids: `cond-2f3369427321`
- model repo: `Qwen/Qwen3.5-0.8B`
- revision: `2fc06364715b967f1860aea9cf38778875588b17`
- representation: `original`
- precision: `bfloat16`
- quantization: `None` bits=None group=None
- runtime: `mlx-lm 0.32.0` backend=`mlx-metal`
- hardware: `Apple M3 Pro` os=`Darwin 27.2.0`
- created: `20261005T231234Z`

## Machine / runtime environment
- python: `3.12.12`
- platform: `macOS-27.2-arm64-arm-64bit`
- mlx device: `Device(gpu, 0)`
- memory bytes: `19327352832`
- packages: `{'mlx': '0.32.3', 'mlx-lm': '0.32.0', 'mlx-metal': '0.32.3', 'numpy': '2.5.3', 'transformers': '5.18.0', 'tokenizers': '0.23.2', 'huggingface-hub': '1.33.0', 'safetensors': '0.8.0', 'ponderscope': '0.1.0'}`

## Task population
- n tasks: 10  n generations: 10
- spec: `{'name': 'calibration-pp1', 'task_pack': 'tasks-v1', 'prompt_policy': 'pp-v1', 'families': ['arith', 'path', 'order', 'logic', 'sm'], 'n_per_family': 2, 'task_seed': 0, 'split': 'calibration', 'greedy_repeats': 1, 'sampled_seeds': [], 'sampled_repeats_per_seed': 1, 'max_tokens': 2048, 'sampled_temperature': 0.6, 'sampled_top_p': 0.95, 'sampled_top_k': 20, 'sampled_min_p': 0.0, 'probe': True, 'n_probes': 4, 'probe_max_tokens': 64, 'capture_entropy': True, 'capture_top_k': 5, 'capture_logprob_digest': False, 'capture_level': 'minimal', 'notes': 'Termination calibration, prompt policy pp-v1 (explicit Answer: cue). Generous 2048 horizon; greedy + minimal capture. Calibration split only.'}`

## Configurations
| condition_id | mode | n | success_at_budget (95% CI) | completion_rate | censored_rate | conditional_acc_given_completed | reasoning tokens mean | tok/s |
|---|---|---|---|---|---|---|---|---|
| cond-2f3369427321 | greedy | 10 | 0.1 [0, 0.3] | 0.1 | 0.9 | 1 | 1873 | 62.37 |

## Reasoning-length distribution

### cond-2f3369427321 (greedy)

```
     295-383         1 ####
     383-470         0 
     470-558         0 
     558-646         0 
     646-733         0 
     733-821         0 
     821-909         0 
     909-996         0 
     996-1084        0 
    1084-1172        0 
    1172-1259        0 
    1259-1347        0 
    1347-1434        0 
    1434-1522        0 
    1522-1610        0 
    1610-1697        0 
    1697-1785        0 
    1785-1873        0 
    1873-1960        0 
    1960-2048        9 ########################################
```
- {"n": 10, "mean": 1872.7, "median": 2048.0, "std": 554.3472738275169, "min": 295.0, "max": 2048.0, "p10": 1872.7, "p90": 2048.0}

## Latency / throughput
- cond-2f3369427321: wall_ms {"n": 10, "mean": 30029.265054098505, "median": 31896.260229499603, "std": 8960.98377409812, "min": 4941.920750003192, "max": 35880.3760830051, "p10": 28474.013149694656, "p90": 34862.20555829059}; tok/s {"n": 10, "mean": 62.36816167011009, "median": 63.64367095999401, "std": 3.0263241893271284, "min": 57.078554451664296, "max": 65.87604670806981, "p10": 58.75099324031683, "p90": 65.38568023009796}; ttft_ms {"n": 10, "mean": 88.07096679811366, "median": 65.96958399313735, "std": 65.27022526447207, "min": 47.596500007784925, "max": 270.14108300500084, "p10": 58.67295030329842, "p90": 113.70210829772981}

## Repetition
- cond-2f3369427321: unique_token_ratio {"n": 10, "mean": 0.09514322916666666, "median": 0.068115234375, "std": 0.08065288086818474, "min": 0.04345703125, "max": 0.31666666666666665, "p10": 0.051806640625, "p90": 0.1362565104166666}; repeated_ngram_fraction_4 {"n": 10, "mean": 0.7679375663727741, "median": 0.8466992665036674, "std": 0.17318025620412994, "min": 0.3434343434343434, "max": 0.8953545232273838, "p10": 0.5796246079375664, "p90": 0.8812713936430318}; text_repeat_ratio {"n": 10, "mean": 0.4369290857294761, "median": 0.4238953488372093, "std": 0.2441783660315285, "min": 0.04347826086956519, "max": 0.764367816091954, "p10": 0.05050167224080271, "p90": 0.6559822361546499}

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
- never_correct: 7
- wrong_to_correct: 3

## Limitations
- Single model revision on a single machine; results do not generalize to other weights or hardware.
- Greedy determinism is a same-process, same-machine replay test.
- Where the reasoning channel was never closed, the trajectory is censored, not a natural stop.
- No causal claims: deployment differences are associations under this protocol.
