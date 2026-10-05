from __future__ import annotations

from pathlib import Path

from ponderscope.analysis import analyze_run, compare_configs, generate_report
from ponderscope.config.schema import ExperimentSpec
from ponderscope.experiment import run_experiment


def _run(tmp_path: Path, fake_backend, monkeypatch, name="t", greedy=2, seeds=(0, 1), probe=True):
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
        spec, model_repo="fake/model", model_revision="deadbeef", runs_dir=tmp_path
    )


def test_run_records_expected_counts(tmp_path, fake_backend, monkeypatch):
    result = _run(tmp_path, fake_backend, monkeypatch)
    traces = result.store.read_traces()
    assert result.n_generations == 2 * (2 + 2)  # 2 tasks * (2 greedy + 2 sampled)
    assert len(traces) == result.n_generations
    assert result.n_probes > 0
    # each record has full layered identity
    assert all(r["deployment_id"].startswith("dep-") for r in traces)
    assert all(r["artifact_id"].startswith("art-") for r in traces)
    assert all(r["condition_id"].startswith("cond-") for r in traces)
    assert all(r["deployment"]["model"]["revision"] == "deadbeef" for r in traces)


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
    b.store.manifest["artifact"] = {
        **b.store.manifest["artifact"],
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
    b.store.manifest["artifact"] = {
        **b.store.manifest["artifact"],
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
