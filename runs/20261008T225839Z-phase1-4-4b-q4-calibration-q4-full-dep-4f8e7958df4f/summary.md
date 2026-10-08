# PonderScope report — 20261008T225839Z-phase1-4-4b-q4-calibration-q4-full-dep-4f8e7958df4f

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
- created: `20261008T225839Z`

## Machine / runtime environment
- python: `3.12.12`
- platform: `macOS-27.2-arm64-arm-64bit`
- mlx device: `Device(gpu, 0)`
- memory bytes: `19327352832`
- packages: `{'mlx': '0.32.3', 'mlx-lm': '0.32.0', 'mlx-metal': '0.32.3', 'numpy': '2.5.3', 'transformers': '5.18.0', 'tokenizers': '0.23.2', 'huggingface-hub': '1.33.0', 'safetensors': '0.8.0', 'ponderscope': '0.1.0'}`

## Task population
- n tasks: 10  n generations: 30
- spec: `{'name': 'phase1-4-4b-q4-calibration', 'task_pack': 'tasks-v1', 'prompt_policy': 'pp-v1', 'model_policy': 'qwen35-4b-thinking-api-v1', 'families': ['arith', 'path', 'order', 'logic', 'sm'], 'n_per_family': 2, 'task_seed': 0, 'split': 'calibration', 'greedy_repeats': 0, 'sampled_seeds': [0, 1, 2], 'sampled_repeats_per_seed': 1, 'max_tokens': 2048, 'sampled_temperature': 1.0, 'sampled_top_p': 0.95, 'sampled_top_k': 20, 'sampled_min_p': 0.0, 'sampled_presence_penalty': 1.5, 'sampled_presence_context_size': 20, 'sampled_presence_scope': 'generated_history', 'sampled_repetition_penalty': 1.0, 'sampled_repetition_context_size': 20, 'sampled_frequency_penalty': 0.0, 'sampled_frequency_context_size': 20, 'probe': False, 'n_probes': 4, 'probe_max_tokens': 64, 'capture_entropy': True, 'capture_top_k': 5, 'capture_logprob_digest': False, 'capture_level': 'minimal', 'notes': 'Phase 1.4 primary contrast: Qwen3.5-4B controlled MLX affine Q4 (--artifact-path models/qwen3.5-4b-mlx-q4-affine-g64) derived from the identical pinned source revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a. Matches the frozen BF16 baseline run 20261006T024435Z-calibration-4b-generated-history-api on every controlled dimension (tasks-v1 calibration, pp-v1, generated-history thinking policy, 2048 horizon, seeds {0,1,2} x ONE execution = 30 stochastic draws); only the weight representation differs.'}`

## Configurations
| condition_id | mode | exec | draws | success_at_budget (95% CI) | completion_rate | censored_rate | conditional_acc_given_completed | reasoning tokens mean | tok/s |
|---|---|---|---|---|---|---|---|---|---|
| cond-f5e762d3078f | sampled | 30 | 30 | 0.6333 [0.3667, 0.8667] | 0.6333 | 0.3 | 1 | 1251 | 41.74 |

## Reasoning-length distribution

### cond-f5e762d3078f (sampled)

```
     296-384         1 ####
     384-471         1 ####
     471-559         1 ####
     559-646         2 #########
     646-734         2 #########
     734-822         4 ##################
     822-909         4 ##################
     909-997         1 ####
     997-1084        1 ####
    1084-1172        0 
    1172-1260        0 
    1260-1347        0 
    1347-1435        0 
    1435-1522        2 #########
    1522-1610        0 
    1610-1698        0 
    1698-1785        0 
    1785-1873        1 ####
    1873-1960        1 ####
    1960-2048        9 ########################################
```
- {"n": 30, "mean": 1250.6, "median": 897.5, "std": 639.1388926813413, "min": 296.0, "max": 2048.0, "p10": 611.0, "p90": 2048.0}

## Latency / throughput
- cond-f5e762d3078f: wall_ms {"n": 30, "mean": 33001.33202223394, "median": 27332.353958001477, "std": 14141.900183335814, "min": 9081.215166006587, "max": 51014.1759170001, "p10": 16476.398246096505, "p90": 49986.215570603235}; tok/s {"n": 30, "mean": 41.741017329533314, "median": 41.82751363661471, "std": 0.9208759140448929, "min": 40.14570387909606, "max": 43.7645818994075, "p10": 40.59530787438371, "p90": 43.00136647268554}; ttft_ms {"n": 30, "mean": 285.9092236336437, "median": 287.0356669991452, "std": 85.65459488256637, "min": 161.5855419950094, "max": 462.6224160019774, "p10": 167.44943779995083, "p90": 367.06127939760347}

## Repetition
- cond-f5e762d3078f: unique_token_ratio {"n": 30, "mean": 0.21360396549347876, "median": 0.201171875, "std": 0.06748702303968696, "min": 0.1257995735607676, "max": 0.3644251626898048, "p10": 0.1375965632567234, "p90": 0.3021556755321848}; repeated_ngram_fraction_4 {"n": 30, "mean": 0.4040113540947527, "median": 0.3535452322738386, "std": 0.16861071304160843, "min": 0.17234262125902994, "max": 0.720855614973262, "p10": 0.23696406972510592, "p90": 0.6577445563649158}; text_repeat_ratio {"n": 30, "mean": 0.04306955718143014, "median": 0.029644268774703553, "std": 0.05174049913456842, "min": 0.0, "max": 0.18181818181818177, "p10": 0.0, "p90": 0.09334375000000011}

## Repeatability / noise floor

### 1. Greedy replay (same deterministic condition, repeated)
- no repeated greedy conditions in this run

### 2. Same-seed sampled replay (same seed + sampler, re-executed)
- no repeated same-seed sampled conditions in this run

### 3. Across-seed stochastic variation (fixed sampler policy, different seeds)
- task/condition groups: 10
- estimable answer diversity: 7 / 10 (proportion 0.7)
- mean distinct observed answers across seeds: 1.286 (support: 7 tasks)
- mean observed-answer accuracy std across seeds: 0.1347
- any ambiguous seeds (divergent same-seed repeats): False
- note: final-answer diversity/accuracy is defined only over seeds with an observed answer; `—` means unobserved, not zero. No mean is shown without its support count.

### 4. Across-seed token variation (valid even when all answers are censored)
- task/condition groups: 10
- mean first token divergence across seeds: 4.4
- mean reasoning-token std across seeds: 163.1

## Censoring / termination (do not read a capped run as a wrong answer)

- capped_length: 11
- eos_observed: 19
- missing_answer: 9
- natural_eos: 19
- status:censored: 9
- status:correct: 19
- status:incorrect: 2
- think_end_reached: 21

### Budget outcomes per condition
| condition_id | success_at_budget | completion_rate | censored_rate | error_rate | unparseable_rate | cond_acc|completed |
|---|---|---|---|---|---|---|
| cond-f5e762d3078f | 0.6333 | 0.6333 | 0.3 | 0 | 0 | 1 |

## Censor-aware time-to-closure (Kaplan-Meier / RMST)

- observation horizon tau: 2048.0 tokens (an observation horizon, not a natural stopping threshold)
- event: PRIMARY: native think-end reasoning closure observed
- secondary event: SECONDARY: EOS observed (generation termination)
- analysis unit: unique stochastic draws (same-seed technical repeats collapsed)

| condition | exec | draws (used) | events | censored | median tokens | RMST(tau) |
|---|---|---|---|---|---|---|
| cond-f5e762d3078f | 30 | 30 (30) | 21 | 9 | 920 | 1251 |

Secondary endpoint (generation termination via EOS):

| condition | events | censored | median tokens | RMST(tau) |
|---|---|---|---|---|
| cond-f5e762d3078f | 19 | 11 | 920 | 1262 |

### cond-f5e762d3078f by family
| family | n | events | censored | median | RMST |
|---|---|---|---|---|---|
| arith | 6 | 6 | 0 | 725 | 844.8 |
| logic | 6 | 0 | 6 | — | 2048 |
| order | 6 | 6 | 0 | 621 | 983.5 |
| path | 6 | 3 | 3 | 1485 | 1538 |
| sm | 6 | 6 | 0 | 851 | 842 |

## Loop-structure diagnostics (descriptive)

Descriptive loop structure with an explicitly documented onset rule; not a validated universal loop detector.

| condition | n | longest_run median (max) | special longest runs | with degeneration onset |
|---|---|---|---|---|
| cond-f5e762d3078f | 30 | 1 (3) | 0 | 0 |

## Prefix probes / trajectory states

- no probes in this run

## Limitations
- Single model revision on a single machine; results do not generalize to other weights or hardware.
- Greedy determinism is a same-process, same-machine replay test.
- Where the reasoning channel was never closed, the trajectory is censored, not a natural stop.
- No causal claims: deployment differences are associations under this protocol.
