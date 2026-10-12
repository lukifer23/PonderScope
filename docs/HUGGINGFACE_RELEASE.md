# PonderScope Hugging Face release — 2026-10-12

## Public repositories

- Account/namespace: **Dontbeafed69**, verified from the authenticated Chrome UI.
- Model: [Dontbeafed69/Qwen3.5-4B-MLX-4bit-PonderScope](https://huggingface.co/Dontbeafed69/Qwen3.5-4B-MLX-4bit-PonderScope).
- Dataset: [Dontbeafed69/PonderScope-Qwen3.5-4B-Quantization-Reasoning](https://huggingface.co/datasets/Dontbeafed69/PonderScope-Qwen3.5-4B-Quantization-Reasoning).
- Both repositories are **public and ungated**. Browser uploads used the existing
  authenticated Chrome session; no browser credentials were extracted. The CLI
  was unauthenticated; fresh downloads and API checks therefore proved public access.

Reviewed research/export-analysis source and initial local/remote `main`:
`c83dd2d49f0acfdfae78f355dc94a37eeede0d81`.
Concurrent research work advanced `main` to
`36fbf917a6dba9075f5d8449b5a558a2f60b0382` before publication documentation.
That work was preserved; its new Phase 1.8 evidence is not part of this release.
Only this release document, README/results/reproducibility links, and the export
utility belong to this publication change. No model or raw traces are committed
to GitHub. No training, reconversion, or BF16/Q4 experiment was performed for the
release. Bounded usability tests are separate from historical study executions.

## Model artifact and source provenance

Base developer: Qwen / Alibaba Cloud. Source model `Qwen/Qwen3.5-4B` at immutable
revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
The exact upstream revision's Apache-2.0 license, including Copyright 2026 Alibaba
Cloud, was downloaded and preserved with attribution and conversion changes.
The GitHub code's MIT license is separate.

MLX-LM 0.32.0 / MLX 0.32.3, affine 4-bit/group-64, BF16 conversion dtype;
248 actual quantized linear modules and **4.502633743002342 effective bits per
weight**. Text-only artifact: no vision-tower or MTP weights; upstream visual-token
configuration fields do not confer multimodal support. Nonquantized parameters
remain in their supported precision. No training/fine-tuning by PonderScope.

Weight file `model.safetensors`: **2,367,238,661 bytes**, SHA-256
`d58b87def677c7b3235e21cdca85289a162a5ba8949687b9573592e3c74cd477`.
Complete remote repository: **2,387,351,669 bytes** in
**15 files**, including Hub-created `.gitattributes`.
Model Hub revision: `444756bfcd873ed5df40c526bbaa81217457efab`.

## Evidence coverage and dataset structure

Calibration: 10 tasks, seeds 0/1/2, 30 draws/arm, **60 per-draw rows / 30 pairs**.
Independent replication: 30 unseen frozen tasks, seeds 0/1, 60 draws/arm,
**120 per-draw rows / 60 pairs**. Total: **180 executions, 40 distinct structural
tasks**, two deployments. No population overlap, technical repeats, pilot draws,
failed-seal smoke, or subsequent research is included in these totals.

Parquet data tables expose task/presentation identities, source/variant/deployment
and condition IDs, seeds, ground truth and observed answers, historical/corrected
outcomes, native closure/EOS/censoring flags, tokens, wall time, both throughput
definitions, repetition/loop diagnostics and exact raw archive/member/line references.
The 90-row paired convenience table is also split into 30/60-row viewer files.
The card explicitly keeps calibration and replication distinct.

All four main source seals verified with the existing PonderScope verifier before
export. All 180 executions have public raw trace coverage. Four archives contain
traces, tasks, environments, exact specs, historical analyses, source seals,
sanitation ledgers and new public integrity seals. The frozen population lock,
corrected comparisons, schema/data dictionary and load audits are published.
Complete remote dataset: **3,868,170 bytes** in
**26 files**, including Hub-created `.gitattributes`.
Dataset Hub revision: `2252f23505ec546bfbcd7ee5eee1b1fa3edf08c7`.
Dataset original research material is **MIT** licensed, based on the original MIT
procedural-task/scoring/instrumentation material, with a separate attribution
notice. Dataset licensing does not relicense upstream weights or independently
protected third-party content. No third-party benchmark corpus was imported.

## Privacy and hash boundaries

The originals remain immutable and private. Public copies replace local home
prefixes with `/REDACTED_HOME` in metadata/deployment paths; conversion provenance
also removes local paths and disk-space measurements. Scientific prompt/output,
token arrays, scoring inputs, termination, timing and metrics remain unchanged.
No access tokens, passwords, browser session details, sensitive environment
variables, private repository files, serial numbers or unrelated private files
were selected. Hardware/runtime conditions remain visible.

Per-file original and public SHA-256 values plus exact JSON locations are in
`redactions.json` inside each archive and in `metadata/evidence-coverage.json`.
Source seals remain byte-identical as `source-evidence-seal.json`; they do not
cover the sanitized bytes. The clearly labeled new `evidence.json` covers the
public derivative. The existing verifier passes both original sources locally
and fresh downloaded public derivatives separately. Integrity hashes are not
cryptographic signatures or independent source attestation.

| Study / arm | Source seal SHA-256 | Public archive SHA-256 |
|---|---|---|
| `calibration-bf16.tar.gz` | `31fd272b2e2836fc58fa6104785db795afb4717c9fc6d69c3d13d4400b690431` | `c7949d638496b9c1b8d01687f26ce0c8adb373d57eacfa5adec6b1f812eac097` |
| `calibration-q4.tar.gz` | `5547e02e23eb698a43208b712a1ea9331ab4e6c9cfcc94b07fa346563ff4c126` | `91095409f23fc7da0fa8cdb83a0d92ccd20eb2b3b77640b345883d91e900d1a1` |
| `replication-bf16.tar.gz` | `0c65e5edfe0f70796ed6a3ee24034bb96c8441f6603754fefa561f186d576a51` | `97cd9db65c92d7f458c833de6f99089d3f747bf61873805d135b47d792f206b9` |
| `replication-q4.tar.gz` | `2b6b87e9db55dc8f978d70881e19148993aa3f3763f5523f710e9660e8f873e2` | `d3b8f6038bba07dd2a353f6693c4c6cf6b3f5a542f108a676f1ea2be39ba5b01` |

## Verification results

- Standard `mlx_lm.load` with default strict loading passed on the local artifact.
- An isolated `uv run --no-project` runtime passed chat-template, native thinking
  closure, final answer, EOS-token and actual-quantization checks.
- A fresh **anonymous Hub download** of all model files passed the same bounded
  generation check. Output contained native `</think>` and final answer `4`.
- Hub `hf cache verify --fail-on-missing-files` verified **15 model and 26 dataset
  files** against remote checksums. The CLI's own local `.cache` metadata is not a
  repository file; an independent inventory check excluded it and found an exact
  match to every remote file. Every release-file `SHA256SUMS` entry also matched.
- Fresh downloaded/extracted public bundles: **4/4 existing-verifier PASS**.
- Public `datasets.load_dataset` succeeded: `draws` 60/120 rows and
  `paired_outcomes` 30/60 rows, with the intended calibration/replication splits.
- Both cards render with correct YAML metadata, licenses, cross-links, source
  attribution, source revision, model base relation, and no README placeholders.
- All card links were tested: GitHub/upstream/cross-repository links return HTTP
  200; the uv documentation link, which rejected a terminal HTTP client, loaded
  correctly in Chrome.
- The final export utility recreated all five Parquet files and four evidence
  archives **byte-for-byte** using the reviewed immutable analysis source.
- Local project gates passed: frozen dev sync, Ruff check/format, mypy, pytest,
  and `git diff --check`. The baseline GitHub CI at `36fbf91` passed in run
  [38277867762](https://github.com/lukifer23/PonderScope/actions/runs/38277867762).
  Publication-commit CI is available in the repository's Actions history and is
  checked after pushing this change.

At the last pre-commit viewer check, Hugging Face's viewer was still processing:
“the server is busier than usual and the response is not ready yet.” This is
separate from the successful public dataset loader and checksum verification.
No viewer success is claimed until the rendered rows are observed.

## Tested standalone use

Apple Silicon Mac, MLX-LM 0.32.0, MLX 0.32.3, Python 3.12,
Transformers 5.18.0. The release's downloaded `smoke_test.py` imports no PonderScope
code, uses `mlx_lm.load`, preserves the chat template, and checks 248 actual
affine-4bit/group-64 modules with a 256-token greedy usability generation.

```bash
hf download Dontbeafed69/Qwen3.5-4B-MLX-4bit-PonderScope --local-dir ponder-model
uv run --no-project --python 3.12 \
  --with mlx-lm==0.32.0 --with mlx==0.32.3 --with transformers==5.18.0 \
  python ponder-model/smoke_test.py --model ponder-model

hf download Dontbeafed69/PonderScope-Qwen3.5-4B-Quantization-Reasoning \
  --repo-type dataset --local-dir ponder-evidence
cd ponder-evidence
shasum -a 256 -c SHA256SUMS
```

Offline reanalysis: extract each `evidence/*.tar.gz` into a separate directory;
use the reviewed GitHub source at `c83dd2d` and
`uv run ponderscope verify --run <unpacked-run> --json`, followed by `analyze` or
`compare --a <bf16> --b <q4> --mode sampled --require-complete`. Preserve historical
analyses before writing new reports. This checks the **new public derivative
seal**, not the separately archived original-byte source seal. It requires no
model inference. Comparison JSON uses BF16-minus-Q4; primary-results JSON and
both cards explicitly use Q4-minus-BF16.

For release maintainers with the retained verified original run directories,
`scripts/export_huggingface.py` creates a new output and verifies imported
analysis/evidence module bytes against an explicit Git revision. It refuses
source mismatches, invalid seals, incomplete pairs, unexpected repeats, task
population overlap and redactions that would touch scientific generation fields.
The optional Parquet exporter dependency does not change the project's lockfile.

```bash
mkdir /tmp/ponderscope-reviewed-source
git archive c83dd2d49f0acfdfae78f355dc94a37eeede0d81 | \
  tar -x -C /tmp/ponderscope-reviewed-source
PYTHONPATH=/tmp/ponderscope-reviewed-source/src \
  uv run --with pyarrow==23.0.1 python scripts/export_huggingface.py \
  --project . --analysis-source-sha c83dd2d49f0acfdfae78f355dc94a37eeede0d81 \
  --output /tmp/ponderscope-evidence-export
```

## Scientific presentation and caveats

Replication: native reasoning closures BF16 27/60 vs Q4 31/60; closure-rate
Q4-minus-BF16 +0.067, 95% CI [-0.033, +0.183]; restricted-mean closure difference
bootstrap mean -48.1, CI [-171.9, +62.3], empirical point -48.8; success-at-budget
+0.050, CI [-0.017, +0.133]. All primary intervals include zero. Calibration also
failed to resolve a primary difference. No pooling of these populations.

The historical 12.8/37.0 tok/s headline measures **end-to-end** generation,
including TTFT; actual decode-only arithmetic means are 12.9/37.3. Separate short
load audits recorded approximately 8.53/2.56 GB MLX peaks, not matched full-study
peak memory. These distinctions are explicit in the cards and tables.

Supported: a large speed/memory advantage on the measured Mac/MLX setup with no
statistically distinguishable change in the primary reasoning outcomes at this
precision. Excluded: accuracy equivalence, improved reasoning, a novel
quantization method, superiority to other community conversions, peer review,
and generalization to other Qwen models/runtimes/devices. Phase 1.7 unparseable
corrections are applied without changing raw historical classifications.
Repeated-4gram findings remain exploratory. Right censoring is not a wrong
answer; 2048 tokens is a measurement horizon, not the model's natural stopping
point. One model, machine/runtime and small procedural populations limit reuse.

## Exact published inventory and SHA-256

The Hub's automatically created `.gitattributes` is listed here as well as the
release-controlled files. Each repository's `SHA256SUMS` covers all manually
published release files except itself; the manifest hash and `.gitattributes`
hash are recorded below. Large artifacts and raw traces live only on Hugging Face.

### Model

| File | Bytes | SHA-256 |
|---|---:|---|
| `.gitattributes` | 1570 | `34448b82c17d60fec9b65b1f093c115ddbaadc04beb1b0140b6bfed2e012a930` |
| `LICENSE` | 11544 | `bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a` |
| `NOTICE` | 544 | `9daf031eb7bf47d243e046970f6ddf44fccfff980729a3a2aafd9f3e5941efa6` |
| `README.md` | 9046 | `d0eddfed4e1ffd099fdd2819f5215cb327d8e14afd07cfa35cd0aaa0baf675e6` |
| `SHA256SUMS` | 1089 | `9247c9c484ff7bd787ff920fe917512cc45575d9e56c016fe9a7d36e3b1d5d6d` |
| `chat_template.jinja` | 7756 | `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715` |
| `config.json` | 2878 | `8591d683d8d132234766399897415b9427967c05e77c1c169a40de3e81177e62` |
| `model.safetensors` | 2367238661 | `d58b87def677c7b3235e21cdca85289a162a5ba8949687b9573592e3c74cd477` |
| `model.safetensors.index.json` | 81008 | `2e97890b24a47b9c290489efeb2b68b7e4dd4101360de80122165ebae4e0d6cf` |
| `ponderscope_variant.json` | 3154 | `787732e108f66413847640b6931976a644b96a983aeec5154d493a1183ca5b44` |
| `smoke_test.py` | 1361 | `477aebcb2d2530633bd8c0040212bb6373e1b0c73237ca3b2bb5e711756c1c6e` |
| `standalone-isolated-test.json` | 960 | `eb2fa1c8fc8d3a829213b24c150ae7bb87b171495613df8125030220235d2303` |
| `standalone-test.json` | 1257 | `23f6aa041178e71e499fc9317d1f22643a74ae55b20b4d5e92d619301b5a2553` |
| `tokenizer.json` | 19989339 | `a5cd9732badce41de57e6efce8302930ded1c1188c5f81feb2bd6c24c4a1941f` |
| `tokenizer_config.json` | 1502 | `4d2fc7899270dc76b57a7aa1d5ec3234e6a25934c6e2aea6166f989f1d453892` |

### Dataset

| File | Bytes | SHA-256 |
|---|---:|---|
| `.gitattributes` | 2504 | `9e75dd981de037ec3769f24f790e126bc5a160b6871f510214e68dc70649aeeb` |
| `LICENSE` | 1066 | `625ce3a533245f5f5bc3a65450e468a1fb5bdef4d38c7af04ce59caa67edce7a` |
| `NOTICE` | 655 | `82ba48c732c2fd7e4bcf522933408e7864b2354b079ff9ed5f23d050b79ca4af` |
| `README.md` | 17149 | `f0bef85b1ddb23eb0ccc832a28c871b23ac5d97d86ef22e846b2d14a2987e08f` |
| `SHA256SUMS` | 2317 | `29332abf6219e6329a9dd3ca2e17a2012ffdbc4742cc836f0214646c1c73a018` |
| `data/calibration.parquet` | 33352 | `cdf93105423e395d6feecb2e38e0e4a33c563da3b7d5c270532911b0d564a9f6` |
| `data/paired-calibration.parquet` | 7339 | `a11f8784c4efb5064ea572cc9233cb9b1b633d745e8f955792f0bd095f65ac01` |
| `data/paired-outcomes.parquet` | 9521 | `6e2c186177db11c670d902ff5e6ef10f2ad7e885f9d017bb1ce398bbc875dcb9` |
| `data/paired-replication.parquet` | 8607 | `15b041841e47bb9fb828ea1c9cacfeeeb4972d1be590c73e7e27672994b00bcb` |
| `data/replication.parquet` | 43365 | `9ccf7dc7b35e2c987bcfc5b604ccd0deb55b079feb0cb70518ca12593eeb6bfc` |
| `evidence/calibration-bf16.tar.gz` | 1066021 | `c7949d638496b9c1b8d01687f26ce0c8adb373d57eacfa5adec6b1f812eac097` |
| `evidence/calibration-q4.tar.gz` | 460630 | `91095409f23fc7da0fa8cdb83a0d92ccd20eb2b3b77640b345883d91e900d1a1` |
| `evidence/replication-bf16.tar.gz` | 995653 | `97cd9db65c92d7f458c833de6f99089d3f747bf61873805d135b47d792f206b9` |
| `evidence/replication-q4.tar.gz` | 981539 | `d3b8f6038bba07dd2a353f6693c4c6cf6b3f5a542f108a676f1ea2be39ba5b01` |
| `metadata/DATA_DICTIONARY.md` | 4619 | `e9af19796be7ea5d45f2469a00243dc4db345ed6f6fa3d5c74063f4b60fe7db8` |
| `metadata/calibration-comparison.json` | 33866 | `1920a0f09ac1db42967cce81d27e9d7a3671fcc0832f17a92d8a088d60503da1` |
| `metadata/calibration-paired-outcomes.json` | 32158 | `c93202718718543aadeff6f0de821727c19215ad9bd25f6d897698da2832ab99` |
| `metadata/evidence-coverage.json` | 55670 | `d8798f591d60546aa79eb6ad7d5447791d8ad7290a476d50b66475c0dfc6b9ad` |
| `metadata/export-provenance.json` | 1345 | `7e498e32d3c22704058366bb99a26e210c6fcbd6ced0a3faf4320917e57b76ef` |
| `metadata/model-load-audit-4b-q4.phase1_4.json` | 2468 | `3c4843e272d48a0591ba1217764be0c8388f72f7c8eb7fe34be3732fb2cababe` |
| `metadata/model-load-audit-4b.phase1_3b.json` | 872 | `7d471f760762da69763ae61660b9a789e019d39ff8bc6c94e4c9295081a210b4` |
| `metadata/phase1_6-population-lock.json` | 4448 | `f5ef21c17433ebfb87e8a9ca09073f7ebe9c0dde62bee4aeb0b139ed644d2eed` |
| `metadata/primary-results.json` | 1553 | `d1c90f0b12c102deed53c24ed8b0d31e24d745158aac22f69dd4cecf2d54d0f1` |
| `metadata/replication-comparison.json` | 34037 | `1f714dd64b386d1474973cb4f62c6b7fcaab42042581160be87c6a212719fc81` |
| `metadata/replication-paired-outcomes.json` | 63570 | `87509aa028d8622c9518b22b3e5f8782f69e9f56efdc6c52524314e51059c8d4` |
| `metadata/table-schemas.json` | 3846 | `5f72caa2098a53fb5783db7f5dd4f74fe7b637d11a7cdbb6123333f3a21f4506` |
