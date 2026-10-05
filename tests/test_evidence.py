from __future__ import annotations

import json
from pathlib import Path

import pytest

from ponderscope.config.identity import DecodingPolicy, Deployment, ModelIdentity, RuntimeIdentity
from ponderscope.config.schema import ExperimentSpec
from ponderscope.evidence.run import RunStore
from ponderscope.evidence.store import JsonlWriter, atomic_write_json, read_json


def _deployment() -> Deployment:
    return Deployment(
        model=ModelIdentity(repo_id="fake/model", revision="deadbeef", precision="float32"),
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
