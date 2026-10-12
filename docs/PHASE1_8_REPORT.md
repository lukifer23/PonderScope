# Phase 1.8 report — paired reasoning-prefix recoverability (EXPLORATORY, INCOMPLETE)

> **STATUS: BF16 ARM COMPLETE · Q4 ARM NOT RUN · PAIRED CONCLUSION BLOCKED.**
> This is an exploratory, post-outcome analysis. The Q4 arm has **not** been run,
> so **no BF16-vs-Q4 difference may be inferred from this document**. Every
> BF16-only number below describes one arm; it is not a deployment contrast.
> Machine-readable status and the exact required Q4 command:
> `runs/phase1_8-status.json`.

This document is human-curated. The generated paired tables (produced only after
both arms exist) will be written to `docs/PHASE1_8_RECOVERABILITY.md`, with the
raw analysis at `runs/phase1_8-recoverability.json`.

## 1. Question

Do the controlled BF16 and Q4 deployments of the same `Qwen/Qwen3.5-4B` revision
differ in **answer recoverability along matched reasoning trajectories** —
especially where natural generation failed to terminate within 2048 tokens?
Forced finalization is an oracle-style intervention on a frozen trace, not a
stopping policy (see `docs/PHASE1_8_PROTOCOL.md`).

## 2. Design (frozen before execution)

- Replays the **existing** saved sampled trajectories of the Phase 1.6
  `test/1729` population: 30 structural tasks × seed 0 × {BF16, Q4}.
- Checkpoints `0 / 256 / 512 / 1024 / 1536` + the full available prefix;
  horizon **512** continuation tokens (chosen by a predeclared sentinel
  diagnostic; see `docs/PHASE1_8_PROTOCOL.md` §"Horizon decision").
- Greedy deterministic forced finalization; the replayed prefix is declared as
  generated history. A budget-exhausted probe is `censored`, never a wrong
  answer. No source run is modified.

## 3. Execution status

| arm | run id | probes | verify | status |
|---|---|---|---|---|
| BF16 | `20261012T162909Z-phase1-8-probe-bf16-recoverability-dep-e1c20569173a` | 152 | PASS | **COMPLETE** |
| Q4 | — | — | — | **NOT RUN** |

- BF16 wall time: `20261012T162909Z` → `20261012T173713Z` (≈68 min; ≈27 s/probe).
- BF16 `publication_grade=true`, `code git_sha 71f6534`; `src/` is byte-identical
  at current `main`, so a later Q4 arm runs the same intervention code.
- BF16 bundle: `models/bundles/phase1_8-probe-bf16.tar.gz`
  (sha256 `2829a27584186dedcd411642ad1b5bdbe68e5b3fc106e60091aeaa4733ed6085`).
  Raw `probes.jsonl` is preserved locally and in the bundle (gitignored).

## 4. BF16 single-arm results (seed 0, horizon 512) — NOT a contrast

| checkpoint | n eligible | correct | censored | scorable-incorrect | unparseable | recoverable-correct |
|---|---|---|---|---|---|---|
| 0 | 30 | 23 | 7 | 0 | 0 | 0.77 |
| 256 | 30 | 25 | 5 | 0 | 0 | 0.83 |
| 512 | 27 | 23 | 4 | 0 | 0 | 0.85 |
| 1024 | 18 | 13 | 5 | 0 | 0 | 0.72 |
| 1536 | 17 | 9 | 8 | 0 | 0 | 0.53 |
| full prefix | 30 | 24 | 6 | 0 | 0 | 0.80 |

- **27/30 draws are recoverable-correct at some eligible checkpoint**, while
  **17/30 source trajectories were naturally censored**. For BF16 alone this is
  consistent with censoring hiding recoverable correct answers — but it is one
  arm and cannot answer whether Q4 differs.
- Every probe is either `correct` or `censored`: **zero scorable-incorrect and
  zero unparseable**. When BF16 is forced to answer it answers correctly; when it
  cannot, it keeps reasoning and is censored.
- Recoverability is lower at 1024/1536, whose denominators are small and enriched
  for loop-prone trajectories.
- 3/30 draws never recover at any checkpoint: `logic/716b73bf`,
  `path/148e3892`, `path/7ad1e635` (all naturally censored).

## 5. What this does NOT show

- **No BF16/Q4 comparison** — the Q4 arm does not exist yet.
- No natural closure, natural accuracy, stable recoverability, or policy-transfer
  claim: forced finalization is oracle-style and uses a fresh greedy completion,
  not the sampled policy that produced the traces.
- No confirmatory claim: the population was already measured and inspected.

## 6. Next required pass (blocking)

Before any Phase 1.8 conclusion, the **Q4 arm must be run** at the identical
design (seed 0, checkpoints `0/256/512/1024/1536` + full prefix, horizon 512),
then the offline paired analysis. Exact commands are recorded in
`runs/phase1_8-status.json` (`arms.q4.required_command`, `next_required_step`) and
in `docs/ROADMAP.md`. Only after both arms are sealed, verified, and joined may
the six Phase 1.8 research questions be answered.

## 7. Limitations

- Exploratory and post-outcome; `test/1729` is not a fresh confirmatory sample.
- One arm only so far; single seed (seed 1 is an optional later extension).
- Forced finalization is oracle-style; a correct forced answer is not a stopping
  policy. Greedy forced-probe recoverability is never pooled with naturally
  sampled accuracy.
- Residual censoring at horizon 512 is a reported outcome, not a defect; failed
  probes are never retried until correct.
- Single machine/runtime; no cross-hardware claims.
