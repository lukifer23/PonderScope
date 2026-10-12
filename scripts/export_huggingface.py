"""Export verified historical studies; never runs or converts a model.

Run with the reviewed PonderScope source on PYTHONPATH and pyarrow installed.
Creates a new output directory, leaving original immutable evidence untouched.
Public path redactions are logged by JSON location and old-value hash. The new
seals explicitly cover sanitized derivatives, not original historical bytes.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import inspect
import io
import json
import re
import subprocess
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any

from ponderscope.analysis.analyze import natural_final_status_of
from ponderscope.analysis.compare import compare_configs
from ponderscope.analysis.paired import paired_outcomes
from ponderscope.evidence.run import RAW_FILES, RunStore, verify_run

STUDIES = {
    "calibration": (
        "20261006T024435Z-calibration-4b-generated-history-api-dep-e1c20569173a",
        "20261008T225839Z-phase1-4-4b-q4-calibration-q4-full-dep-4f8e7958df4f",
        30,
        10,
    ),
    "replication": (
        "20261009T005354Z-phase1-6-replication-bf16-replication-bf16-dep-e1c20569173a",
        "20261009T025900Z-phase1-6-replication-q4-replication-q4-dep-4f8e7958df4f",
        60,
        30,
    ),
}
HOME_PATH = re.compile(r"/Users/[^/\s\"']+")
SECRET = re.compile(r"hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|-----BEGIN .*PRIVATE KEY-----")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sanitize(value: Any, changes: list[dict[str, Any]], location: str = "$") -> Any:
    if isinstance(value, dict):
        return {k: sanitize(v, changes, f"{location}.{k}") for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v, changes, f"{location}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, str):
        clean = HOME_PATH.sub("/REDACTED_HOME", value)
        if clean != value:
            changes.append(
                {
                    "location": location,
                    "old_value_sha256": digest(value.encode()),
                    "replacement": "home-directory prefix -> /REDACTED_HOME",
                }
            )
        if SECRET.search(clean):
            raise ValueError(f"credential-like content at {location}")
        return clean
    return value


def public_bundle(source: Path, out: Path) -> dict[str, Any]:
    original = verify_run(source)
    if not original["pass"]:
        raise ValueError(f"unverified source: {source.name}: {original['errors']}")
    members: dict[str, bytes] = {}
    redactions: dict[str, Any] = {}
    names = ["manifest.json", *RAW_FILES, "analysis.json", "analysis.phase1_6.json"]
    for name in names:
        p = source / name
        if not p.exists():
            continue
        changes: list[dict[str, Any]] = []
        raw = p.read_bytes()
        if name.endswith(".jsonl"):
            # Unchanged files remain byte-identical. Only path-bearing records
            # are reserialized; prompt/output/token arrays are never redacted.
            lines = []
            for line_no, line in enumerate(raw.splitlines(keepends=True), 1):
                record = json.loads(line)
                clean = sanitize(record, changes, f"line[{line_no}]")
                for key in (
                    "prompt",
                    "gold_answer",
                    "trace",
                    "parse",
                    "termination",
                    "metrics",
                    "loop",
                ):
                    if key in record and clean[key] != record[key]:
                        raise ValueError(
                            f"redaction would alter scientific content: {name}:{line_no}:{key}"
                        )
                lines.append(
                    line
                    if clean == record
                    else (json.dumps(clean, ensure_ascii=False) + "\n").encode()
                )
            data = b"".join(lines)
        else:
            record = json.loads(raw)
            clean = sanitize(record, changes)
            data = raw if clean == record else (json.dumps(clean, indent=2) + "\n").encode()
        members[name] = data
        redactions[name] = {
            "original_sha256": digest(raw),
            "published_sha256": digest(data),
            "changes": changes,
        }
    manifest = json.loads(members["manifest.json"])
    manifest["publication"] = {
        "kind": "sanitized derivative of verified historical evidence",
        "release_date": "2026-10-12",
        "source_manifest_sha256": redactions["manifest.json"]["original_sha256"],
        "source_seal_file": "source-evidence-seal.json",
        "note": "Source identity fields and all generation content retained. Local home paths redacted. New evidence.json covers published bytes only.",
    }
    members["manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    redactions["manifest.json"]["published_sha256"] = digest(members["manifest.json"])
    source_seal = (source / "evidence.json").read_bytes()
    members["source-evidence-seal.json"] = source_seal
    seal = json.loads(source_seal)
    seal["files"] = {n: digest(members[n]) for n in RAW_FILES if n in members}
    seal["manifest_sha256"] = digest(members["manifest.json"])
    seal["sealed_utc"] = "20261012T000000Z"
    seal["publication"] = {
        "kind": "sanitized derivative integrity seal",
        "historical_sealed_utc": json.loads(source_seal)["sealed_utc"],
        "note": "Release-day timestamp; not a historical experiment seal or a cryptographic signature.",
    }
    members["evidence.json"] = (json.dumps(seal, indent=2) + "\n").encode()
    members["spec.json"] = (json.dumps(manifest["spec"], indent=2) + "\n").encode()
    members["redactions.json"] = (json.dumps(redactions, indent=2) + "\n").encode()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, data in sorted(members.items()):
            info = tarfile.TarInfo(f"{source.name}/{name}")
            info.size, info.mtime, info.mode = len(data), 0, 0o644
            tar.addfile(info, io.BytesIO(data))
    archive = gzip.compress(buf.getvalue(), mtime=0)
    out.write_bytes(archive)
    # Verify the derivative with the existing verifier, separately from source.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
            tar.extractall(tmp, filter="data")
        published = verify_run(Path(tmp) / source.name)
    if not published["pass"]:
        raise ValueError(published["errors"])
    original.pop("path", None)
    published.pop("path", None)
    return {
        "run_id": source.name,
        "archive": out.name,
        "bytes": len(archive),
        "sha256": digest(archive),
        "source_verification": original,
        "derivative_verification": published,
        "source_seal_sha256": digest(source_seal),
        "redactions": redactions,
    }


def draw_row(r: dict[str, Any], study: str, arm: str, line: int) -> dict[str, Any]:
    term, trace = r["termination"], r["trace"]
    outcome = natural_final_status_of(r)
    return {
        "study_id": "phase1.4" if study == "calibration" else "phase1.6",
        "study_population": study,
        "population_split": r["split"],
        "structural_task_id": r["task_id"],
        "presentation_id": r["presentation_id"],
        "task_family": r["family"],
        "task_difficulty": r["difficulty"]["level"],
        "source_model_id": r["deployment"]["model"]["source"]["repo_id"],
        "source_revision": r["deployment"]["model"]["source"]["revision"],
        "source_artifact_id": r["source_artifact_id"],
        "weight_variant_id": r["weight_variant_id"],
        "deployment_id": r["deployment_id"],
        "deployment_arm": arm,
        "decoding_condition_id": r["condition_id"],
        "decoding_mode": r["condition"]["mode"],
        "sample_seed": r["condition"]["seed"],
        "technical_repeat": r["condition"]["repeat"],
        "stochastic_draw_id": r["stochastic_draw_id"],
        "trial_id": r["trial_id"],
        "ground_truth": r["gold_answer"],
        "generated_answer": r["parse"].get("answer_raw"),
        "normalized_answer": r["answer_normalized"],
        "generated_final_channel": r["parse"].get("final_raw"),
        "outcome_classification": outcome,
        "historical_outcome_classification": r["natural_final_status"],
        "outcome_analysis_version": "phase1.7",
        "success_at_budget": bool(r["correct"]),
        "native_reasoning_closure": bool(term["think_end_reached"]),
        "generation_terminated_by_eos": bool(term["terminated_by_eos"]),
        "capped": bool(term["capped"]),
        "censored": outcome == "censored",
        "budget_exhausted_after_closure": bool(term["capped"] and term["think_end_reached"]),
        "unparseable": outcome == "unparseable",
        "error": outcome == "error",
        "finish_reason": term["finish_reason"],
        "reasoning_tokens": r["reasoning_tokens"],
        "answer_tokens": r["answer_tokens"],
        "total_tokens": r["total_tokens"],
        "generation_wall_ms": trace["wall_ms"],
        "decode_tokens_per_second": trace["decode_tokens_per_sec"],
        "generation_tokens_per_second": trace["tokens_per_sec"],
        "ttft_ms": trace["ttft_ms"],
        "repeated_ngram_fraction_4": r["metrics"]["repeated_ngram_fraction_4"],
        "repeated_ngram_fraction_8": r["metrics"]["repeated_ngram_fraction_8"],
        "unique_token_ratio": r["metrics"]["unique_token_ratio"],
        "loop_diagnostics_json": json.dumps(r.get("loop"), sort_keys=True),
        "raw_evidence_reference": f"evidence/{study}-{arm}.tar.gz::{r['run_id']}/traces.jsonl#line={line}",
    }


def export(project: Path, output: Path, analysis_source_sha: str) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    # A concurrent research checkout must not silently change release semantics.
    package = Path(inspect.getfile(natural_final_status_of)).parents[1]
    module_hashes = {}
    for name in (
        "analysis/analyze.py",
        "analysis/compare.py",
        "analysis/paired.py",
        "analysis/draws.py",
        "analysis/stats.py",
        "analysis/survival.py",
        "reasoning/transitions.py",
        "evidence/run.py",
        "evidence/store.py",
        "config/identity.py",
    ):
        original = subprocess.check_output(
            ["git", "show", f"{analysis_source_sha}:src/ponderscope/{name}"], cwd=project
        )
        imported = (package / name).read_bytes()
        if imported != original:
            raise ValueError(
                f"Analysis source mismatch for {name}; use the declared immutable source on PYTHONPATH"
            )
        module_hashes[name] = digest(imported)
    output.mkdir(parents=True, exist_ok=False)
    for name in ("data", "metadata", "evidence"):
        (output / name).mkdir()
    coverage, pairs = [], []
    primary_results = {}
    task_sets = []
    for study, (bf16, q4, n, ntasks) in STUDIES.items():
        stores = [RunStore.load(project / "runs" / name) for name in (bf16, q4)]
        rows = []
        arm_rows = {}
        for arm, store in zip(("bf16", "q4"), stores, strict=True):
            coverage.append(
                public_bundle(store.path, output / "evidence" / f"{study}-{arm}.tar.gz")
            )
            traces = store.read_traces()
            if len(traces) != n or any(
                r["condition"]["mode"] != "sampled" or r["condition"]["repeat"] != 0 for r in traces
            ):
                raise ValueError("Unexpected stochastic-draw population or technical repeats")
            selected = [draw_row(r, study, arm, i) for i, r in enumerate(traces, 1)]
            arm_rows[arm] = selected
            rows.extend(selected)
        task_set = {r["structural_task_id"] for r in rows}
        if len(task_set) != ntasks:
            raise ValueError("Wrong structural task count")
        task_sets.append(task_set)
        paired = paired_outcomes(*stores, mode="sampled")
        if paired["n_matched_draws"] != n:
            raise ValueError("Incomplete matched population")
        for row in paired["draws"]:
            key = row["draw_key"]

            def find(
                arm: str, draw: list[Any], populations: dict[str, list[dict[str, Any]]]
            ) -> dict[str, Any]:
                hits = [
                    r
                    for r in populations[arm]
                    if (
                        r["presentation_id"],
                        r["structural_task_id"],
                        r["decoding_condition_id"],
                        r["sample_seed"],
                    )
                    == tuple(draw)
                ]
                if len(hits) != 1:
                    raise ValueError("Non-unique paired observation")
                return hits[0]

            a, b = find("bf16", key, arm_rows), find("q4", key, arm_rows)
            pairs.append(
                {
                    "study_population": study,
                    "structural_task_id": key[1],
                    "presentation_id": key[0],
                    "decoding_condition_id": key[2],
                    "sample_seed": key[3],
                    "task_family": a["task_family"],
                    "task_difficulty": a["task_difficulty"],
                    "bf16_trial_id": a["trial_id"],
                    "q4_trial_id": b["trial_id"],
                    "bf16_outcome": a["outcome_classification"],
                    "q4_outcome": b["outcome_classification"],
                    "closure_transition": row["closure_transition"],
                    "success_at_budget_transition": row.get(
                        "success_at_budget_transition", row.get("correctness_transition")
                    ),
                    "bf16_raw_evidence_reference": a["raw_evidence_reference"],
                    "q4_raw_evidence_reference": b["raw_evidence_reference"],
                }
            )
        pq.write_table(
            pa.Table.from_pylist(rows), output / "data" / f"{study}.parquet", compression="zstd"
        )
        comparison = compare_configs(*stores, mode="sampled", require_complete=True)
        normalized = {
            "direction": "Q4 minus BF16; source comparison JSON uses A(BF16) minus B(Q4)",
            "n_task_clusters": comparison["n_matched_tasks"],
            "n_matched_draws": n,
        }
        for metric in ("reasoning_closure_rate", "success_at_budget"):
            delta = comparison["metrics"][metric]["delta"]
            normalized[metric] = {
                "estimate": -delta["mean"],
                "ci95": [-delta["hi"], -delta["lo"]],
                "excludes_zero": delta["excludes_zero"],
            }
        rmst = comparison["rmst"]["endpoints"]["reasoning_closure"]
        normalized["restricted_mean_reasoning_closure_tokens"] = {
            "point": -rmst["point"],
            "bootstrap_mean": -rmst["mean"],
            "ci95": [-rmst["hi"], -rmst["lo"]],
            "excludes_zero": rmst["excludes_zero"],
            "horizon": rmst["tau"],
        }
        primary_results[study] = normalized
        write_json(output / "metadata" / f"{study}-comparison.json", sanitize(comparison, []))
        write_json(output / "metadata" / f"{study}-paired-outcomes.json", sanitize(paired, []))
        print(
            study,
            len(rows),
            "rows",
            n,
            "matched pairs",
            Counter(r["outcome_classification"] for r in rows),
        )
    if task_sets[0] & task_sets[1]:
        raise ValueError("Calibration/replication tasks overlap")
    pq.write_table(
        pa.Table.from_pylist(pairs), output / "data" / "paired-outcomes.parquet", compression="zstd"
    )
    for study in STUDIES:
        selected = [row for row in pairs if row["study_population"] == study]
        pq.write_table(
            pa.Table.from_pylist(selected),
            output / "data" / f"paired-{study}.parquet",
            compression="zstd",
        )
    write_json(output / "metadata" / "primary-results.json", primary_results)
    write_json(
        output / "metadata" / "export-provenance.json",
        {
            "release_date": "2026-10-12",
            "analysis_source_git_sha": analysis_source_sha,
            "analysis_module_sha256": module_hashes,
            "export_utility_sha256": digest(Path(__file__).read_bytes()),
            "pyarrow_version": pa.__version__,
            "no_new_research_runs": True,
            "bundle_note": "Source seals verified; sanitized derivative integrity seals verified separately.",
        },
    )
    write_json(output / "metadata" / "evidence-coverage.json", coverage)
    lock = project / "specs" / "phase1_6-population-lock.json"
    (output / "metadata" / lock.name).write_bytes(lock.read_bytes())
    write_json(
        output / "metadata" / "table-schemas.json",
        {p.name: str(pq.read_schema(p)) for p in sorted((output / "data").glob("*.parquet"))},
    )
    # Human privacy review is still required; fail closed on obvious secrets.
    for p in output.rglob("*.json"):
        if SECRET.search(p.read_text()) or HOME_PATH.search(p.read_text()):
            raise ValueError(f"Privacy scan failed: {p.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--analysis-source-sha", required=True)
    args = parser.parse_args()
    export(args.project, args.output, args.analysis_source_sha)
