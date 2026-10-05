from __future__ import annotations

from pathlib import Path

from ponderscope.analysis import analyze_run, compare_configs, generate_report
from ponderscope.config.schema import ExperimentSpec
from ponderscope.evidence.run import RunStore
from ponderscope.experiment import run_experiment


def _run(
    tmp_path: Path,
    fake_backend,
    monkeypatch,
    name="t",
    greedy=2,
    seeds=(0, 1),
    probe=True,
    code_state=None,
):
    monkeypatch.setattr("ponderscope.experiment.get_backend", lambda backend_name: fake_backend)
    spec = ExperimentSpec(
        name=name,
        task_pack="tasks-v1",
        families=["arith"],
        n_per_family=2,
        greedy_repeats=greedy,
        sampled_seeds=list(seeds),
        probe=probe,
        n_probes=3,
        max_tokens=16,
    )
    return run_experiment(
        spec,
        model_repo="fake/model",
        model_revision="deadbeef",
        runs_dir=tmp_path,
        code_state=code_state or {"version": "test", "git_sha": "0" * 40, "tracked_dirty": False},
    )


def test_run_records_expected_counts(tmp_path, fake_backend, monkeypatch):
    result = _run(tmp_path, fake_backend, monkeypatch)
    traces = result.store.read_traces()
    assert result.n_generations == 2 * (2 + 2)  # 2 tasks * (2 greedy + 2 sampled)
    assert len(traces) == result.n_generations
    assert result.n_probes > 0
    # each record has full layered identity
    assert all(r["deployment_id"].startswith("dep-") for r in traces)
    assert all(r["weight_variant_id"].startswith("wvar-") for r in traces)
    assert all(r["source_artifact_id"].startswith("src-") for r in traces)
    assert all(r["condition_id"].startswith("cond-") for r in traces)
    assert all(r["deployment"]["model"]["source"]["revision"] == "deadbeef" for r in traces)


def test_analysis_and_noise_floor(tmp_path, fake_backend, monkeypatch):
    result = _run(tmp_path, fake_backend, monkeypatch)
    analysis = analyze_run(result.store)
    assert analysis["n_generations"] == result.n_generations
    assert analysis["noise_floor"]["greedy_replay"]["token_identical_rate"] == 1.0
    # two different seeds (0, 1) give across-seed variation; there is no same-seed
    # repeat in this fixture, which is correctly reported as unavailable.
    assert analysis["noise_floor"]["across_seed"]["available"] is True
    assert analysis["noise_floor"]["same_seed_replay"]["available"] is False


def test_compare_identical_runs(tmp_path, fake_backend, monkeypatch):
    a = _run(tmp_path, fake_backend, monkeypatch, name="a")
    b = _run(tmp_path, fake_backend, monkeypatch, name="b")
    cmp = compare_configs(a.store, b.store, mode="greedy")
    assert cmp["refused"] is False
    assert cmp["n_matched_trials"] == 4  # 2 tasks x 2 greedy repeats
    acc = cmp["metrics"]["accuracy"]["delta"]
    assert acc["mean"] == 0.0


def test_compare_refuses_multi_dimension_confound(tmp_path, fake_backend, monkeypatch):
    a = _run(tmp_path, fake_backend, monkeypatch, name="a")
    b = _run(tmp_path, fake_backend, monkeypatch, name="b")
    # Change both artifact revision and runtime hardware: an uncontrolled,
    # multi-dimension change that cannot be attributed to one deployment variable.
    b.store.manifest["source_artifact"] = {
        **b.store.manifest["source_artifact"],
        "revision": "another-revision",
    }
    b.store.manifest["runtime"] = {
        **b.store.manifest["runtime"],
        "hardware": "some-other-host",
    }
    cmp = compare_configs(a.store, b.store, mode="greedy")
    assert cmp["refused"] is True
    assert any("simultaneously" in r for r in cmp["validity"]["reasons"])
    # explicit exploratory override still computes but is labelled.
    cmp2 = compare_configs(a.store, b.store, mode="greedy", allow_confounded=True)
    assert cmp2["refused"] is False
    assert cmp2["exploratory"] is True


def test_compare_allows_single_dimension_change(tmp_path, fake_backend, monkeypatch):
    a = _run(tmp_path, fake_backend, monkeypatch, name="a")
    b = _run(tmp_path, fake_backend, monkeypatch, name="b")
    b.store.manifest["source_artifact"] = {
        **b.store.manifest["source_artifact"],
        "revision": "another-revision",
    }
    cmp = compare_configs(a.store, b.store, mode="greedy")
    assert cmp["refused"] is False
    assert cmp["validity"]["model_changed"] == ["model.revision"]


def test_report_written(tmp_path, fake_backend, monkeypatch):
    result = _run(tmp_path, fake_backend, monkeypatch)
    analysis = analyze_run(result.store)
    md, html_text = generate_report(result.store, analysis)
    assert (result.store.path / "summary.md").exists()
    assert (result.store.path / "summary.html").exists()
    assert "# PonderScope report" in md
    assert "Deployment identity" in md
    assert "Repeatability / noise floor" in md
    assert "<html" in html_text


def test_run_records_task_pack_provenance_and_seals(tmp_path, fake_backend, monkeypatch):
    result = _run(tmp_path, fake_backend, monkeypatch)
    manifest = result.store.manifest
    assert manifest["task_pack"]["requested_pack"] == "tasks-v1"
    assert manifest["task_pack"]["n_tasks"] == len(result.store.read_tasks())
    assert manifest["spec_hash"]
    assert result.store.is_sealed
    assert manifest["status"]["run"] == "EVIDENCE_COMPLETE"
    seal = result.store.read_seal()
    assert "traces.jsonl" in seal["files"]


def test_failed_experiment_marks_partial(tmp_path, fake_backend, monkeypatch):
    monkeypatch.setattr("ponderscope.experiment.get_backend", lambda backend_name: fake_backend)
    calls = {"n": 0}
    original = fake_backend.generate

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] >= 2:
            raise RuntimeError("boom")
        return original(*args, **kwargs)

    monkeypatch.setattr(fake_backend, "generate", flaky)
    spec = ExperimentSpec(
        name="fail", task_pack="tasks-v1", families=["arith"], n_per_family=3, max_tokens=16
    )
    import pytest

    with pytest.raises(RuntimeError):
        run_experiment(
            spec,
            model_repo="fake/model",
            model_revision="deadbeef",
            runs_dir=tmp_path,
            code_state={"version": "test", "git_sha": "0" * 40, "tracked_dirty": False},
        )
    runs = [p for p in tmp_path.iterdir() if (p / "manifest.json").exists()]
    assert len(runs) == 1
    store = RunStore.load(runs[0])
    assert store.manifest["status"]["run"] == "PARTIAL"
    assert store.manifest["status"]["error"]["type"] == "RuntimeError"
    assert not store.is_sealed
