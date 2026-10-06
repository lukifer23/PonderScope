# PonderScope report — 20261006T004654Z-loop-diagnostic-dep-d7a370157187

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
- created: `20261006T004654Z`

## Machine / runtime environment
- python: `3.12.12`
- platform: `macOS-27.2-arm64-arm-64bit`
- mlx device: `Device(gpu, 0)`
- memory bytes: `19327352832`
- packages: `{'mlx': '0.32.3', 'mlx-lm': '0.32.0', 'mlx-metal': '0.32.3', 'numpy': '2.5.3', 'transformers': '5.18.0', 'tokenizers': '0.23.2', 'huggingface-hub': '1.33.0', 'safetensors': '0.8.0', 'ponderscope': '0.1.0'}`

## Task population
- n tasks: 3  n generations: 3
- spec: `{'name': 'loop-diagnostic', 'task_pack': 'tasks-v1', 'prompt_policy': 'pp-v1', 'model_policy': 'qwen35-08b-thinking-upstream-v1', 'families': ['arith', 'order', 'sm'], 'n_per_family': 1, 'task_seed': 0, 'split': 'calibration', 'greedy_repeats': 0, 'sampled_seeds': [0], 'sampled_repeats_per_seed': 1, 'max_tokens': 2048, 'sampled_temperature': 1.0, 'sampled_top_p': 0.95, 'sampled_top_k': 20, 'sampled_min_p': 0.0, 'sampled_presence_penalty': 1.5, 'sampled_presence_context_size': 20, 'sampled_repetition_penalty': 1.0, 'sampled_repetition_context_size': 20, 'sampled_frequency_penalty': 0.0, 'sampled_frequency_context_size': 20, 'probe': False, 'n_probes': 4, 'probe_max_tokens': 64, 'capture_entropy': True, 'capture_top_k': 5, 'capture_logprob_digest': False, 'capture_level': 'research', 'notes': 'Predeclared small research-capture subset: arith index 0 (historically terminating), order index 0 and sm index 0 (historically looping). Fixed seed 0, upstream thinking policy, 2048 horizon. Used to inspect entropy / chosen logprob / top-k / repetition onset at loop onset, and (separately) to confirm minimal-vs-research token equivalence under the penalty processors.'}`

## Configurations
| condition_id | mode | n | success_at_budget (95% CI) | completion_rate | censored_rate | conditional_acc_given_completed | reasoning tokens mean | tok/s |
|---|---|---|---|---|---|---|---|---|
| cond-74a29a115886 | sampled | 3 | 0.3333 [0, 1] | 0.3333 | 0.6667 | 1 | 1459 | 44.53 |

## Reasoning-length distribution

### cond-74a29a115886 (sampled)

```
     280-368         1 ####################
     368-457         0 
     457-545         0 
     545-634         0 
     634-722         0 
     722-810         0 
     810-899         0 
     899-987         0 
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
    1960-2048        2 ########################################
```
- {"n": 3, "mean": 1458.6666666666667, "median": 2048.0, "std": 1020.7552759272584, "min": 280.0, "max": 2048.0, "p10": 633.6, "p90": 2048.0}

## Latency / throughput
- cond-74a29a115886: wall_ms {"n": 3, "mean": 35949.83406933898, "median": 42222.906208000495, "std": 25822.265714863715, "min": 7568.975250003859, "max": 58057.620750012575, "p10": 14499.761441603187, "p90": 54890.67784161016}; tok/s {"n": 3, "mean": 44.52945652522667, "median": 48.50447740169861, "std": 8.040816866904656, "min": 35.27530018528285, "max": 49.80859198869857, "p10": 37.921135628566006, "p90": 49.54776907129858}; ttft_ms {"n": 3, "mean": 176.95813867127677, "median": 214.47916601027828, "std": 102.92860839830864, "min": 60.5327500088606, "max": 255.86249999469146, "p10": 91.32203320914414, "p90": 247.58583319780882}

## Repetition
- cond-74a29a115886: unique_token_ratio {"n": 3, "mean": 0.1830074671198055, "median": 0.1474609375, "std": 0.08360753159361378, "min": 0.123046875, "max": 0.27851458885941643, "p10": 0.1279296875, "p90": 0.2523038585875331}; repeated_ngram_fraction_4 {"n": 3, "mean": 0.4914490801877542, "median": 0.45525672371638143, "std": 0.14140501714362202, "min": 0.3716577540106952, "max": 0.6474327628361858, "p10": 0.38837754795183244, "p90": 0.608997555012225}; text_repeat_ratio {"n": 3, "mean": 0.08057357459883918, "median": 0.06930693069306926, "std": 0.08675731647391034, "min": 0.0, "max": 0.1724137931034483, "p10": 0.013861386138613853, "p90": 0.1517924206213725}

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

- capped_length: 2
- eos_observed: 1
- missing_answer: 2
- natural_eos: 1
- status:censored: 2
- status:correct: 1
- think_end_reached: 1

### Budget outcomes per condition
| condition_id | success_at_budget | completion_rate | censored_rate | error_rate | unparseable_rate | cond_acc|completed |
|---|---|---|---|---|---|---|
| cond-74a29a115886 | 0.3333 | 0.3333 | 0.6667 | 0 | 0 | 1 |

## Censor-aware time-to-closure (Kaplan-Meier / RMST)

- observation horizon tau: 2048.0 tokens (an observation horizon, not a natural stopping threshold)
- event: natural native think-end or EOS observed

| condition | n | events | censored | median tokens | RMST(tau) |
|---|---|---|---|---|---|
| cond-74a29a115886 | 3 | 1 | 2 | — | 1491 |

## Loop-structure diagnostics (descriptive)

Descriptive loop structure with an explicitly documented onset rule; not a validated universal loop detector.

| condition | n | longest_run median (max) | special longest runs | with degeneration onset |
|---|---|---|---|---|
| cond-74a29a115886 | 3 | 1 (3) | 0 | 0 |

## Prefix probes / trajectory states

- no probes in this run

## Limitations
- Single model revision on a single machine; results do not generalize to other weights or hardware.
- Greedy determinism is a same-process, same-machine replay test.
- Where the reasoning channel was never closed, the trajectory is censored, not a natural stop.
- No causal claims: deployment differences are associations under this protocol.
