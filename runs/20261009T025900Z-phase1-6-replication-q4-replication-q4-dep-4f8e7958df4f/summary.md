# PonderScope report — 20261009T025900Z-phase1-6-replication-q4-replication-q4-dep-4f8e7958df4f

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
- created: `20261009T025900Z`

## Machine / runtime environment
- python: `3.12.12`
- platform: `macOS-27.2-arm64-arm-64bit`
- mlx device: `Device(gpu, 0)`
- memory bytes: `19327352832`
- packages: `{'mlx': '0.32.3', 'mlx-lm': '0.32.0', 'mlx-metal': '0.32.3', 'numpy': '2.5.3', 'transformers': '5.18.0', 'tokenizers': '0.23.2', 'huggingface-hub': '1.33.0', 'safetensors': '0.8.0', 'ponderscope': '0.1.0'}`

## Task population
- n tasks: 30  n generations: 60
- spec: `{'name': 'phase1-6-replication-q4', 'task_pack': 'tasks-v1', 'prompt_policy': 'pp-v1', 'model_policy': 'qwen35-4b-thinking-api-v1', 'families': ['arith', 'path', 'order', 'logic', 'sm'], 'n_per_family': 6, 'task_seed': 1729, 'split': 'test', 'greedy_repeats': 0, 'sampled_seeds': [0, 1], 'sampled_repeats_per_seed': 1, 'max_tokens': 2048, 'sampled_temperature': 1.0, 'sampled_top_p': 0.95, 'sampled_top_k': 20, 'sampled_min_p': 0.0, 'sampled_presence_penalty': 1.5, 'sampled_presence_context_size': 20, 'sampled_presence_scope': 'generated_history', 'sampled_repetition_penalty': 1.0, 'sampled_repetition_context_size': 20, 'sampled_frequency_penalty': 0.0, 'sampled_frequency_context_size': 20, 'probe': False, 'n_probes': 4, 'probe_max_tokens': 64, 'capture_entropy': True, 'capture_top_k': 5, 'capture_logprob_digest': False, 'capture_level': 'minimal', 'notes': 'Phase 1.6 independent replication (Q4 arm). Controlled MLX affine Q4/group-64 derived from the identical pinned source revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a, loaded via --artifact-path models/qwen3.5-4b-mlx-q4-affine-g64, on the held-out test split (task_seed=1729, 30 previously unseen tasks, seeds {0,1} = 60 stochastic draws, 2048 horizon). Identical controlled dimensions to the BF16 arm; only the weight representation differs. Preregistered before inspecting outcomes.'}`

## Configurations
| condition_id | mode | exec | draws | success_at_budget (95% CI) | completion_rate | censored_rate | conditional_acc_given_completed | reasoning tokens mean | tok/s |
|---|---|---|---|---|---|---|---|---|---|
| cond-f5e762d3078f | sampled | 60 | 60 | 0.4833 [0.3167, 0.65] | 0.4833 | 0.4833 | 1 | 1418 | 37.04 |

## Reasoning-length distribution

### cond-f5e762d3078f (sampled)

```
     215-307         2 ###
     307-398         1 #
     398-490         1 #
     490-582         6 ########
     582-673         2 ###
     673-765         2 ###
     765-857         4 ######
     857-948         3 ####
     948-1040        3 ####
    1040-1132        2 ###
    1132-1223        1 #
    1223-1315        1 #
    1315-1406        0 
    1406-1498        1 #
    1498-1590        0 
    1590-1681        1 #
    1681-1773        0 
    1773-1865        0 
    1865-1956        1 #
    1956-2048       29 ########################################
```
- {"n": 60, "mean": 1417.7666666666667, "median": 1784.5, "std": 673.6758101454595, "min": 215.0, "max": 2048.0, "p10": 526.1, "p90": 2048.0}

## Latency / throughput
- cond-f5e762d3078f: wall_ms {"n": 60, "mean": 41925.35953328382, "median": 50494.582312501734, "std": 16610.71920126924, "min": 8434.352708994993, "max": 66230.65445800603, "p10": 18164.46680840745, "p90": 57102.67524940282}; tok/s {"n": 60, "mean": 37.03794223022219, "median": 36.90324891341227, "std": 2.6683360233323525, "min": 30.463075332399114, "max": 41.94502930844606, "p10": 33.571741459037455, "p90": 39.995206410497914}; ttft_ms {"n": 60, "mean": 314.47379224967636, "median": 318.5559994963114, "std": 89.49406968990502, "min": 162.6592919928953, "max": 524.4285420048982, "p10": 171.41734189644922, "p90": 416.2315534951631}

## Repetition
- cond-f5e762d3078f: unique_token_ratio {"n": 60, "mean": 0.2012899768169968, "median": 0.1971847739888977, "std": 0.05929213042828427, "min": 0.11130434782608696, "max": 0.3342245989304813, "p10": 0.13037109375, "p90": 0.29486486486486485}; repeated_ngram_fraction_4 {"n": 60, "mean": 0.40048720834583534, "median": 0.33667481662591686, "std": 0.15688160344498975, "min": 0.1189873417721519, "max": 0.7271142109851787, "p10": 0.23597150434443268, "p90": 0.6250779396196108}; text_repeat_ratio {"n": 60, "mean": 0.03890841714451586, "median": 0.01807166528583265, "std": 0.04760884998379281, "min": 0.0, "max": 0.19767441860465118, "p10": 0.0, "p90": 0.10709474629546475}

## Repeatability / noise floor

### 1. Greedy replay (same deterministic condition, repeated)
- no repeated greedy conditions in this run

### 2. Same-seed sampled replay (same seed + sampler, re-executed)
- no repeated same-seed sampled conditions in this run

### 3. Across-seed stochastic variation (fixed sampler policy, different seeds)
- task/condition groups: 30
- estimable answer diversity: 18 / 30 (proportion 0.6)
- mean distinct observed answers across seeds: 1 (support: 18 tasks)
- mean observed-answer accuracy std across seeds: 0
- any ambiguous seeds (divergent same-seed repeats): False
- note: final-answer diversity/accuracy is defined only over seeds with an observed answer; `—` means unobserved, not zero. No mean is shown without its support count.

### 4. Across-seed token variation (valid even when all answers are censored)
- task/condition groups: 30
- mean first token divergence across seeds: 7.833
- mean reasoning-token std across seeds: 143.1

## Censoring / termination (do not read a capped run as a wrong answer)

- capped_length: 31
- eos_observed: 29
- missing_answer: 29
- natural_eos: 29
- status:censored: 29
- status:correct: 29
- status:incorrect: 2
- think_end_reached: 31

### Budget outcomes per condition
| condition_id | success_at_budget | completion_rate | censored_rate | error_rate | unparseable_rate | cond_acc|completed |
|---|---|---|---|---|---|---|
| cond-f5e762d3078f | 0.4833 | 0.4833 | 0.4833 | 0 | 0 | 1 |

## Family reconciliation (closures vs EOS are distinct)

| condition | family | n | native closures | EOS | censored | no-closure | unparseable | errors | observed answers | correct answers |
|---|---|---|---|---|---|---|---|---|---|---|
| cond-f5e762d3078f | **ALL** | 60 | 31 | 29 | 29 | 0 | 0 | 0 | 30 | 29 |
| cond-f5e762d3078f | arith | 12 | 11 | 11 | 1 | 0 | 0 | 0 | 11 | 11 |
| cond-f5e762d3078f | logic | 12 | 0 | 0 | 12 | 0 | 0 | 0 | 0 | 0 |
| cond-f5e762d3078f | order | 12 | 11 | 10 | 1 | 0 | 0 | 0 | 11 | 10 |
| cond-f5e762d3078f | path | 12 | 2 | 1 | 10 | 0 | 0 | 0 | 1 | 1 |
| cond-f5e762d3078f | sm | 12 | 7 | 7 | 5 | 0 | 0 | 0 | 7 | 7 |

## Censor-aware time-to-closure (Kaplan-Meier / RMST)

- observation horizon tau: 2048.0 tokens (an observation horizon, not a natural stopping threshold)
- event: PRIMARY: native think-end reasoning closure observed
- secondary event: SECONDARY: EOS observed (generation termination), timed at the terminal token, independent of the reasoning-closure time
- analysis unit: unique stochastic draws (same-seed technical repeats collapsed)

| condition | exec | draws (used) | events | censored | median tokens | RMST(tau) |
|---|---|---|---|---|---|---|
| cond-f5e762d3078f | 60 | 60 (60) | 31 | 29 | 1662 | 1418 |

Secondary endpoint (generation termination via EOS):

| condition | events | censored | median tokens | RMST(tau) |
|---|---|---|---|---|
| cond-f5e762d3078f | 29 | 31 | — | 1540 |

### cond-f5e762d3078f by family
| family | n | events | censored | median | RMST |
|---|---|---|---|---|---|
| arith | 12 | 11 | 1 | 683 | 771.2 |
| logic | 12 | 0 | 12 | — | 2048 |
| order | 12 | 11 | 1 | 616 | 886.9 |
| path | 12 | 2 | 10 | — | 1910 |
| sm | 12 | 7 | 5 | 1255 | 1475 |

## Loop-structure diagnostics (descriptive)

Descriptive loop structure with an explicitly documented onset rule; not a validated universal loop detector.

| condition | n | longest_run median (max) | special longest runs | with degeneration onset |
|---|---|---|---|---|
| cond-f5e762d3078f | 60 | 1 (4) | 0 | 4 |

## Prefix probes / trajectory states

- no probes in this run

## Limitations
- Single model revision on a single machine; results do not generalize to other weights or hardware.
- Greedy determinism is a same-process, same-machine replay test.
- Where the reasoning channel was never closed, the trajectory is censored, not a natural stop.
- No causal claims: deployment differences are associations under this protocol.
