from __future__ import annotations

from pathlib import Path

from ponderscope.analysis import analyze_run
from ponderscope.cli import main
from ponderscope.config.schema import ExperimentSpec
from ponderscope.experiment import run_experiment


def test_cli_generate(tmp_path: Path):
    out = tmp_path / "pack.jsonl"
    rc = main(
        ["generate", "--families", "arith", "logic", "--n-per-family", "2", "--out", str(out)]
    )
    assert rc == 0
    assert out.exists()
    assert len(out.read_text().strip().splitlines()) == 4


def _fake_run(tmp_path: Path, fake_backend, monkeypatch, name: str):
    monkeypatch.setattr("ponderscope.experiment.get_backend", lambda backend_name: fake_backend)
    spec = ExperimentSpec(
        name=name,
        task_pack="tasks-v1",
        families=["arith"],
        n_per_family=2,
        greedy_repeats=1,
        sampled_seeds=[0],
        max_tokens=16,
    )
    return run_experiment(
        spec, model_repo="fake/model", model_revision="deadbeef", runs_dir=tmp_path
    )


def test_cli_analyze_report_compare(tmp_path, fake_backend, monkeypatch):
    a = _fake_run(tmp_path, fake_backend, monkeypatch, "a")
    b = _fake_run(tmp_path, fake_backend, monkeypatch, "b")

    assert main(["analyze", "--run", str(a.store.path)]) == 0
    assert (a.store.path / "analysis.json").exists()

    assert main(["report", "--run", str(a.store.path)]) == 0
    assert (a.store.path / "summary.md").exists()

    assert (
        main(["compare", "--a", str(a.store.path), "--b", str(b.store.path), "--mode", "greedy"])
        == 0
    )
    assert (a.store.path / "comparison.json").exists()


def test_cli_analyze_latest(tmp_path, fake_backend, monkeypatch):
    _fake_run(tmp_path, fake_backend, monkeypatch, "old")
    latest = _fake_run(tmp_path, fake_backend, monkeypatch, "new")
    analyze_run(latest.store)
    assert main(["report", "--latest", "--runs-dir", str(tmp_path)]) == 0
    assert (latest.store.path / "summary.md").exists()
