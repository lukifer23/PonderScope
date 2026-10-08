from __future__ import annotations

import json
from pathlib import Path

import pytest

from ponderscope.config.identity import (
    DecodingPolicy,
    Deployment,
    RuntimeIdentity,
    SourceArtifactIdentity,
    WeightVariantIdentity,
)
from ponderscope.config.schema import ExperimentSpec
from ponderscope.evidence.run import RunStore, verify_run
from ponderscope.evidence.store import JsonlWriter, atomic_write_json, read_json


def _deployment() -> Deployment:
    return Deployment(
        model=WeightVariantIdentity(
            source=SourceArtifactIdentity(repo_id="fake/model", revision="deadbeef"),
            representation="original",
            precision="float32",
        ),
        runtime=RuntimeIdentity(
            runtime="fake",
            runtime_version="0",
            backend="fake",
            hardware="h",
            os="o",
            python_version="3.12",
        ),
        decoding=DecodingPolicy(mode="greedy", max_tokens=8),
    )


def test_atomic_write_and_read(tmp_path: Path):
    p = tmp_path / "a.json"
    atomic_write_json(p, {"x": 1})
    assert read_json(p) == {"x": 1}


def test_jsonl_writer_roundtrip(tmp_path: Path):
    p = tmp_path / "t.jsonl"
    with JsonlWriter(p).open() as w:
        w.append({"a": 1})
        w.append({"a": 2})
    lines = p.read_text().strip().splitlines()
    assert [json.loads(x)["a"] for x in lines] == [1, 2]


def test_run_no_overwrite(tmp_path: Path):
    spec = ExperimentSpec(name="t", task_pack="tasks-v1")
    store = RunStore.create(_deployment(), spec, runs_dir=tmp_path, created_utc="20260101T000000Z")
    assert store.path.exists()
    with pytest.raises(FileExistsError):
        RunStore.create(_deployment(), spec, runs_dir=tmp_path, created_utc="20260101T000000Z")


def test_manifest_serialization_roundtrip(tmp_path: Path):
    spec = ExperimentSpec(name="t2", task_pack="tasks-v1")
    store = RunStore.create(_deployment(), spec, runs_dir=tmp_path, created_utc="20260101T000001Z")
    loaded = RunStore.load(store.path)
    assert loaded.deployment_id == store.deployment_id
    assert loaded.artifact_id == store.artifact_id
    assert loaded.manifest["spec"]["name"] == "t2"


def test_run_evidence_roundtrip(tmp_path: Path):
    spec = ExperimentSpec(name="t3", task_pack="tasks-v1")
    store = RunStore.create(_deployment(), spec, runs_dir=tmp_path, created_utc="20260101T000002Z")
    store.write_tasks([{"task_id": "x"}])
    with store.open_traces() as w:
        w.append({"record_type": "generation", "task_id": "x"})
    with store.open_probes() as w:
        w.append({"task_id": "x", "prefix_len": 4})
    store.write_analysis({"run_id": store.run_id})
    store.write_summary("# r\n", "<h1>r</h1>")
    assert store.read_tasks() == [{"task_id": "x"}]
    assert len(store.read_traces()) == 1
    assert len(store.read_probes()) == 1
    assert store.has_analysis()
    assert (store.path / "summary.md").read_text() == "# r\n"


def test_latest_returns_most_recent(tmp_path: Path):
    spec = ExperimentSpec(name="a", task_pack="tasks-v1")
    s1 = RunStore.create(_deployment(), spec, runs_dir=tmp_path, created_utc="20260101T000000Z")
    s2 = RunStore.create(_deployment(), spec, runs_dir=tmp_path, created_utc="20260102T000000Z")
    assert RunStore.latest(tmp_path).run_id == s2.run_id
    assert s1.run_id != s2.run_id


def test_jsonl_writer_no_overwrite(tmp_path: Path):
    p = tmp_path / "t.jsonl"
    with JsonlWriter(p) as w:
        w.append({"a": 1})
    with pytest.raises(FileExistsError):
        JsonlWriter(p).open()


def test_jsonl_writer_open_is_idempotent(tmp_path: Path):
    p = tmp_path / "t.jsonl"
    w = JsonlWriter(p)
    w.open()
    w.open()  # must not create a second temp file / fd
    w.append({"a": 1})
    w.close()
    assert p.read_text().strip().splitlines() == ['{"a": 1}']


def test_task_file_is_create_once(tmp_path: Path):
    spec = ExperimentSpec(name="t4", task_pack="tasks-v1")
    store = RunStore.create(_deployment(), spec, runs_dir=tmp_path, created_utc="20260101T000003Z")
    store.write_tasks([{"task_id": "x"}])
    with pytest.raises(FileExistsError):
        store.write_tasks([{"task_id": "x"}])


def test_seal_hashes_raw_evidence_and_is_create_once(tmp_path: Path):
    spec = ExperimentSpec(name="seal", task_pack="tasks-v1")
    store = RunStore.create(_deployment(), spec, runs_dir=tmp_path, created_utc="20260101T000004Z")
    store.write_tasks([{"task_id": "x"}])
    with store.open_traces() as w:
        w.append({"task_id": "x"})
    with store.open_probes() as w:
        w.append({"task_id": "x"})
    assert store.is_sealed is False
    seal = store.seal()
    assert store.is_sealed is True
    assert store.manifest["status"]["run"] == "EVIDENCE_COMPLETE"
    assert "tasks.jsonl" in seal["files"]
    assert "traces.jsonl" in seal["files"]
    assert set(seal["files"]) >= {"environment.json", "tasks.jsonl", "traces.jsonl", "probes.jsonl"}
    with pytest.raises(FileExistsError):
        store.seal()


def test_sealed_manifest_cannot_be_mutated(tmp_path: Path):
    spec = ExperimentSpec(name="seal2", task_pack="tasks-v1")
    store = RunStore.create(_deployment(), spec, runs_dir=tmp_path, created_utc="20260101T000005Z")
    store.write_tasks([{"task_id": "x"}])
    with store.open_traces() as w:
        w.append({"task_id": "x"})
    with store.open_probes() as w:
        w.append({"task_id": "x"})
    store.seal()
    with pytest.raises(RuntimeError):
        store.update_status("RUNNING")


def _valid_trace(store: RunStore, *, trial_id: str = "trial-1", task_id: str = "x") -> dict:
    return {
        "record_type": "generation",
        "task_id": task_id,
        "condition_id": store.condition_ids[0],
        "deployment_id": store.deployment_id,
        "source_artifact_id": store.source_artifact_id,
        "weight_variant_id": store.weight_variant_id,
        "trial_id": trial_id,
        "condition": {"mode": "greedy", "seed": None, "repeat": 0},
        "termination": {"think_end_reached": True, "terminated_by_eos": True, "capped": False},
        "trace": {"token_ids": [1, 2, 3]},
    }


def _sealed_store(tmp_path: Path, name: str = "ver") -> RunStore:
    spec = ExperimentSpec(name=name, task_pack="tasks-v1")
    store = RunStore.create(
        _deployment(), spec, runs_dir=tmp_path, created_utc=f"20260101T0001{name[:2]}"
    )
    store.write_tasks([{"task_id": "x"}])
    with store.open_traces() as w:
        w.append(_valid_trace(store))
    with store.open_probes() as w:
        w.append({"task_id": "x"})
    store.seal()
    return store


def test_freshly_sealed_run_verifies_and_seal_matches_final_manifest(tmp_path: Path):
    store = _sealed_store(tmp_path)
    report = verify_run(store.path)
    assert report["pass"] is True, report["errors"]
    from ponderscope.config.identity import sha256_file

    assert store.read_seal()["manifest_sha256"] == sha256_file(str(store.path / "manifest.json"))


def test_manifest_mutation_fails_verify(tmp_path: Path):
    store = _sealed_store(tmp_path)
    manifest = read_json(store.path / "manifest.json")
    manifest["status"]["run"] = "TAMPERED"
    atomic_write_json(store.path / "manifest.json", manifest)
    report = verify_run(store.path)
    assert report["pass"] is False
    assert any("manifest.json" in e for e in report["errors"])


def test_raw_evidence_mutation_fails_verify(tmp_path: Path):
    for target in ("tasks.jsonl", "traces.jsonl", "probes.jsonl"):
        child = tmp_path / f"case-{target}"
        child.mkdir()
        store = _sealed_store(child)
        p = store.path / target
        p.write_text(p.read_text() + "\n")
        report = verify_run(store.path)
        assert report["pass"] is False, target
        assert any(target in e for e in report["errors"])


def test_manifest_condition_has_no_seed(tmp_path: Path):
    spec = ExperimentSpec(name="cond", task_pack="tasks-v1", sampled_seeds=[0, 1], max_tokens=8)
    store = RunStore.create(_deployment(), spec, runs_dir=tmp_path, created_utc="20260101T000900Z")
    for cond in store.manifest["conditions"]:
        assert "seed" not in cond["decoding"]


def test_failed_run_lifecycle_preserves_partial_evidence(tmp_path: Path):
    spec = ExperimentSpec(name="fail", task_pack="tasks-v1")
    store = RunStore.create(_deployment(), spec, runs_dir=tmp_path, created_utc="20260101T000006Z")
    store.write_tasks([{"task_id": "x"}])
    with store.open_traces() as w:
        w.append({"task_id": "x"})
    store.mark_failed("RuntimeError", "backend crashed", partial=True)
    assert store.manifest["status"]["run"] == "PARTIAL"
    assert store.manifest["status"]["error"]["message"] == "backend crashed"
    assert store.manifest["timestamps"]["completed_utc"] is not None
    assert store.is_sealed is False  # never labelled EVIDENCE_COMPLETE
    assert store.read_tasks() == [{"task_id": "x"}]  # partial evidence preserved


def _store_with_raw_traces(tmp_path: Path, name: str, raw: str) -> RunStore:
    """Create a sealed run whose traces.jsonl content is supplied verbatim."""
    spec = ExperimentSpec(name=name, task_pack="tasks-v1")
    store = RunStore.create(
        _deployment(), spec, runs_dir=tmp_path, created_utc=f"20260101T0002{name[:2]}"
    )
    store.write_tasks([{"task_id": "x"}])
    (store.path / "traces.jsonl").write_text(raw)
    with store.open_probes() as w:
        w.append({"task_id": "x"})
    store.seal()
    return store


def test_truncated_jsonl_fails_structural_verify(tmp_path: Path):
    store = _sealed_store(tmp_path, "trunc")
    raw = (store.path / "traces.jsonl").read_text() + '{"task_id": "x", "con'
    (store.path / "traces.jsonl").write_text(raw)
    # Re-seal so the byte hash matches the corrupt content; structural check
    # must still fail closed.
    (store.path / "evidence.json").unlink()
    store.manifest["status"]["run"] = "EVIDENCE_COMPLETE"
    store.seal()
    report = verify_run(store.path)
    assert report["pass"] is False
    assert report["structural_ok"] is False
    assert any("malformed" in e for e in report["structural_errors"])


def test_duplicate_trial_id_fails_structural_verify(tmp_path: Path):
    seed_store = _sealed_store(tmp_path, "dupseed")
    rec = _valid_trace(seed_store, trial_id="trial-dup")
    raw = json.dumps(rec) + "\n" + json.dumps(rec) + "\n"
    child = tmp_path / "dup"
    child.mkdir()
    store = _store_with_raw_traces(child, "dupx", raw)
    report = verify_run(store.path)
    assert report["pass"] is False
    assert any("duplicate trial_id" in e for e in report["structural_errors"])


def test_undeclared_condition_fails_structural_verify(tmp_path: Path):
    seed_store = _sealed_store(tmp_path, "undec0")
    rec = _valid_trace(seed_store)
    rec["condition_id"] = "cond-deadbeef"
    child = tmp_path / "undecl"
    child.mkdir()
    store = _store_with_raw_traces(child, "undec", json.dumps(rec) + "\n")
    report = verify_run(store.path)
    assert report["pass"] is False
    assert any("not declared in manifest" in e for e in report["structural_errors"])


def test_missing_condition_metadata_fails_structural_verify(tmp_path: Path):
    seed_store = _sealed_store(tmp_path, "nocond0")
    rec = _valid_trace(seed_store)
    del rec["condition"]
    child = tmp_path / "nocond"
    child.mkdir()
    store = _store_with_raw_traces(child, "nocond", json.dumps(rec) + "\n")
    report = verify_run(store.path)
    assert report["pass"] is False
    assert report["structural_ok"] is False
    assert any("condition" in e for e in report["structural_errors"])


def test_deployment_mismatch_fails_structural_verify(tmp_path: Path):
    seed_store = _sealed_store(tmp_path, "depmis0")
    rec = _valid_trace(seed_store)
    rec["deployment_id"] = "dep-ffffffffffff"
    child = tmp_path / "depmis"
    child.mkdir()
    store = _store_with_raw_traces(child, "depmis", json.dumps(rec) + "\n")
    report = verify_run(store.path)
    assert report["pass"] is False
    assert any("deployment_id mismatch" in e for e in report["structural_errors"])


def test_reordered_records_fail_verify(tmp_path: Path):
    seed_store = _sealed_store(tmp_path, "reord0")
    r0 = _valid_trace(seed_store, trial_id="trial-a", task_id="a")
    r1 = _valid_trace(seed_store, trial_id="trial-b", task_id="b")
    raw = json.dumps(r0) + "\n" + json.dumps(r1) + "\n"
    child = tmp_path / "reorder"
    child.mkdir()
    store = _store_with_raw_traces(child, "reorder", raw)
    assert verify_run(store.path)["pass"] is True
    lines = (store.path / "traces.jsonl").read_text().strip().splitlines()
    (store.path / "traces.jsonl").write_text("\n".join(reversed(lines)) + "\n")
    report = verify_run(store.path)
    assert report["pass"] is False  # byte hash no longer matches the seal
    assert any("traces.jsonl" in e for e in report["errors"])


def test_corrupt_seal_fails_verify(tmp_path: Path):
    store = _sealed_store(tmp_path, "corrupt")
    (store.path / "evidence.json").write_text("{not json")
    report = verify_run(store.path)
    assert report["pass"] is False


def test_missing_seal_fails_verify(tmp_path: Path):
    store = _sealed_store(tmp_path, "noseal")
    (store.path / "evidence.json").unlink()
    report = verify_run(store.path)
    assert report["pass"] is False
    assert any("no evidence seal" in e for e in report["errors"])


def test_reanalysis_without_raw_traces_refuses(tmp_path: Path):
    from ponderscope.analysis.analyze import analyze_run

    store = _sealed_store(tmp_path, "noraw")
    (store.path / "traces.jsonl").unlink()
    with pytest.raises(FileNotFoundError):
        analyze_run(store)


def test_raw_evidence_overwrite_refused(tmp_path: Path):
    store = _sealed_store(tmp_path, "overwrite")
    with pytest.raises(FileExistsError):
        store.open_traces()
    with pytest.raises(FileExistsError):
        store.write_tasks([{"task_id": "y"}])
