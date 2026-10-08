"""Phase 1.6 tests: declared-trial-identity completeness, publication-grade
verification semantics, and consistent stochastic-draw statistical units.
"""

from __future__ import annotations

from pathlib import Path

from ponderscope.analysis.compare import (
    _expected_trial_keys,
    _key,
    _trial_population,
)
from ponderscope.evidence.run import RunStore


def _manifest(
    name,
    presentations,
    seeds,
    *,
    repeats=1,
    mode="sampled",
    condition_id="cond-x",
    greedy_repeats=0,
):
    return {
        "run_id": name,
        "spec": {
            "sampled_seeds": list(seeds),
            "sampled_repeats_per_seed": repeats,
            "greedy_repeats": greedy_repeats,
            "max_tokens": 16,
        },
        "task_pack": {"presentation_ids": list(presentations), "n_tasks": len(presentations)},
        "conditions": [{"condition_id": condition_id, "decoding": {"mode": mode}}],
        "deployment_id": "dep-x",
        "weight_variant_id": "wvar-x",
        "source_artifact_id": "src-x",
        "code": {"publication_grade": True},
    }


def _store(tmp_path: Path, manifest) -> RunStore:
    return RunStore(path=tmp_path, manifest=manifest)


def _tr(pres, task, cid, seed, repeat, tokens, mode="sampled"):
    return {
        "presentation_id": pres,
        "task_id": task,
        "condition_id": cid,
        "condition": {"mode": mode, "seed": seed, "repeat": repeat},
        "trace": {"token_ids": list(tokens)},
    }


def _pop(tmp_path, man_a, traces_a, man_b, traces_b):
    a = _store(tmp_path, man_a)
    b = _store(tmp_path, man_b)
    idx_a = {
        (
            r["presentation_id"],
            r["task_id"],
            r["condition_id"],
            r["condition"]["mode"],
            r["condition"]["seed"],
            r["condition"]["repeat"],
        )
        for r in traces_a
    }
    idx_b = {
        (
            r["presentation_id"],
            r["task_id"],
            r["condition_id"],
            r["condition"]["mode"],
            r["condition"]["seed"],
            r["condition"]["repeat"],
        )
        for r in traces_b
    }
    keys = sorted(idx_a & idx_b)
    return _trial_population(a, b, traces_a, traces_b, keys, "sampled")


PRES = [f"pres{i}" for i in range(10)]
TASKS = [f"task{i}" for i in range(10)]


def _full_traces(seeds, presentations=PRES, tasks=TASKS, cid="cond-x", repeats=1):
    out = []
    for p, t in zip(presentations, tasks, strict=True):
        for seed in seeds:
            for r in range(repeats):
                out.append(_tr(p, t, cid, seed, r, [1, 2, 3, seed, r]))
    return out


def test_expected_30_observed_20_both_incomplete(tmp_path):
    man = _manifest("a", PRES, [0, 1, 2])
    traces = _full_traces([0, 1])  # only 20
    pop = _pop(tmp_path, man, traces, man, traces)
    assert pop["expected_trials_a"] == 30
    assert pop["observed_executions_a"] == 20
    assert pop["missing_trials_a"] == 10
    assert pop["complete"] is False
    assert any("incomplete" in r for r in pop["incomplete_reasons"])


def test_expected_30_observed_30_complete(tmp_path):
    man = _manifest("a", PRES, [0, 1, 2])
    traces = _full_traces([0, 1, 2])
    pop = _pop(tmp_path, man, traces, man, traces)
    assert pop["expected_trials_a"] == 30
    assert pop["matched_pairs"] == 30
    assert pop["complete"] is True


def test_wrong_seed_identity_incomplete(tmp_path):
    man = _manifest("a", PRES, [0, 1, 2])
    traces = _full_traces([0, 1, 9])  # same count, wrong seed
    pop = _pop(tmp_path, man, traces, man, traces)
    assert pop["observed_executions_a"] == 30
    assert pop["unexpected_trials_a"] == 10
    assert pop["missing_trials_a"] == 10
    assert pop["complete"] is False


def test_missing_task_incomplete(tmp_path):
    man = _manifest("a", PRES, [0, 1, 2])
    traces = _full_traces([0, 1, 2], presentations=PRES[:9], tasks=TASKS[:9])
    pop = _pop(tmp_path, man, traces, man, traces)
    assert pop["complete"] is False
    assert pop["missing_trials_a"] == 3


def test_duplicate_identities_incomplete(tmp_path):
    man = _manifest("a", PRES, [0, 1, 2])
    traces = _full_traces([0, 1, 2])
    traces.append(dict(traces[0]))  # duplicate logical identity
    pop = _pop(tmp_path, man, traces, man, traces)  # a==b
    assert pop["duplicate_keys_a"] == 1
    assert pop["complete"] is False


def test_presentation_population_mismatch(tmp_path):
    man = _manifest("a", PRES, [0, 1, 2])
    traces = _full_traces([0, 1, 2], presentations=PRES[:5], tasks=TASKS[:5])
    man_obs = _manifest("a", PRES[:5], [0, 1, 2])
    pop = _pop(tmp_path, man, traces, man_obs, traces)
    assert pop["presentation_population_match_a"] is False
    assert pop["complete"] is False


def test_ambiguous_same_seed_repeats_incomplete(tmp_path):
    man = _manifest("a", PRES, [0, 1], repeats=2)
    traces = []
    for p, t in zip(PRES, TASKS, strict=True):
        for seed in (0, 1):
            traces.append(_tr(p, t, "cond-x", seed, 0, [1, 2, 3]))
            # divergent repeat for seed 0
            traces.append(_tr(p, t, "cond-x", seed, 1, [1, 2, 3] if seed else [9, 9, 9]))
    pop = _pop(tmp_path, man, traces, man, traces)
    assert pop["ambiguous_draws_a"] == 10
    assert pop["complete"] is False


def test_missing_metadata_fails_closed(tmp_path):
    man = _manifest("a", PRES, [0, 1, 2])
    del man["task_pack"]["presentation_ids"]
    traces = _full_traces([0, 1, 2])
    pop = _pop(tmp_path, man, traces, man, traces)
    assert pop["expected_known"] is False
    assert pop["complete"] is False


def test_same_population_required(tmp_path):
    a = _manifest("a", PRES, [0, 1])
    b = _manifest("b", PRES, [0, 1, 2])
    traces_a = _full_traces([0, 1])
    traces_b = _full_traces([0, 1, 2])
    pop = _pop(tmp_path, a, traces_a, b, traces_b)
    assert pop["complete"] is False
    assert any("different trial populations" in r for r in pop["incomplete_reasons"])


def test_condition_id_prevents_collision():
    a = _tr("pres0", "task0", "cond-a", 0, 0, [1])
    b = _tr("pres0", "task0", "cond-b", 0, 0, [1])
    assert _key(a) != _key(b)
    assert _key(a)[2] == "cond-a" and _key(b)[2] == "cond-b"


def test_expected_keys_requires_declared_population(tmp_path):
    man = _manifest("a", PRES, [0, 1, 2])
    keys, meta = _expected_trial_keys(_store(tmp_path, man), "sampled")
    assert keys is not None and len(keys) == 30 and meta["expected_total"] == 30
    man2 = _manifest("a", [], [0, 1, 2])
    keys2, meta2 = _expected_trial_keys(_store(tmp_path, man2), "sampled")
    assert keys2 is None and "presentation" in meta2["reason"]


# --------------------------------------------------------------------------- #
# Stage 2: publication-grade requires performed + passed verification
# --------------------------------------------------------------------------- #
from ponderscope.analysis import compare_configs  # noqa: E402
from ponderscope.config.schema import ExperimentSpec  # noqa: E402
from ponderscope.experiment import run_experiment  # noqa: E402

CLEAN = {"version": "test", "git_sha": "0" * 40, "tracked_dirty": False}


def _real_run(tmp_path, fake_backend, monkeypatch, name, seeds, repeats=1):
    monkeypatch.setattr("ponderscope.experiment.get_backend", lambda _n: fake_backend)
    spec = ExperimentSpec(
        name=name,
        task_pack="tasks-v1",
        families=["arith"],
        n_per_family=2,
        greedy_repeats=0,
        sampled_seeds=list(seeds),
        sampled_repeats_per_seed=repeats,
        max_tokens=16,
    )
    return run_experiment(
        spec,
        model_repo="fake/model",
        model_revision="deadbeef",
        runs_dir=tmp_path,
        code_state=CLEAN,
    )


def test_verify_false_is_not_publication_grade(tmp_path, fake_backend, monkeypatch):
    a = _real_run(tmp_path, fake_backend, monkeypatch, "a", [0, 1])
    b = _real_run(tmp_path, fake_backend, monkeypatch, "b", [0, 1])
    cmp = compare_configs(a.store, b.store, mode="sampled", n_resamples=100, verify=False)
    assert cmp["publication_grade"] is False
    assert cmp["exploratory"] is True
    assert "verification_skipped" in cmp["overrides_applied"]
    assert cmp["verification"]["checked"] is False


def test_verify_true_clean_complete_is_publication_grade(tmp_path, fake_backend, monkeypatch):
    a = _real_run(tmp_path, fake_backend, monkeypatch, "a", [0, 1])
    b = _real_run(tmp_path, fake_backend, monkeypatch, "b", [0, 1])
    cmp = compare_configs(a.store, b.store, mode="sampled", n_resamples=100)
    assert cmp["verification"]["checked"] is True
    assert cmp["publication_grade"] is True
    assert cmp["exploratory"] is False


def test_failed_seal_refused_by_default(tmp_path, fake_backend, monkeypatch):
    a = _real_run(tmp_path, fake_backend, monkeypatch, "a", [0, 1])
    b = _real_run(tmp_path, fake_backend, monkeypatch, "b", [0, 1])
    p = b.store.path / "manifest.json"
    p.write_text(p.read_text() + " ")
    cmp = compare_configs(a.store, b.store, mode="sampled")
    assert cmp["refused"] is True
    assert any("does not verify" in r for r in cmp["refusal_reasons"])


def test_override_is_not_publication_grade(tmp_path, fake_backend, monkeypatch):
    a = _real_run(tmp_path, fake_backend, monkeypatch, "a", [0, 1])
    b = _real_run(tmp_path, fake_backend, monkeypatch, "b", [0, 1])
    p = b.store.path / "manifest.json"
    p.write_text(p.read_text() + " ")
    cmp = compare_configs(a.store, b.store, mode="sampled", n_resamples=100, allow_unverified=True)
    assert cmp["refused"] is False
    assert cmp["publication_grade"] is False
    assert cmp["exploratory"] is True
    assert "unverified_evidence" in cmp["overrides_applied"]


def test_incomplete_population_is_not_publication_grade(tmp_path, fake_backend, monkeypatch):
    a = _real_run(tmp_path, fake_backend, monkeypatch, "a", [0])
    b = _real_run(tmp_path, fake_backend, monkeypatch, "b", [0, 1])
    cmp = compare_configs(a.store, b.store, mode="sampled", n_resamples=100)
    assert cmp["trial_population"]["complete"] is False
    assert cmp["publication_grade"] is False


# --------------------------------------------------------------------------- #
# Stage 3: consistent stochastic-draw statistical units
# --------------------------------------------------------------------------- #
from ponderscope.analysis.compare import _ambiguous_draw_count, _matched_draw_records  # noqa: E402


def test_repeats_change_execution_count_not_draw_population(tmp_path, fake_backend, monkeypatch):
    a = _real_run(tmp_path, fake_backend, monkeypatch, "a", [0, 1])
    b = _real_run(tmp_path, fake_backend, monkeypatch, "b", [0, 1], repeats=3)
    # A seed-0/repeat-0 comparison matches only repeat 0, so repeats never become
    # additional draws; the point estimate is unchanged.
    cmp = compare_configs(a.store, b.store, mode="sampled", n_resamples=100)
    assert cmp["analysis_population"]["n_draws_used_b"] == 4  # 2 tasks x 2 seeds
    assert cmp["metrics"]["success_at_budget"]["delta"]["mean"] == 0.0
    # The run-level draw collapse still counts all executions as diagnostics.
    from ponderscope.analysis.draws import collapse_to_stochastic_draws

    traces_b = b.store.read_traces()
    draws_b = collapse_to_stochastic_draws(traces_b)
    assert sum(d["n_executions"] for d in draws_b) == 12
    assert len(draws_b) == 4


def test_divergent_same_seed_repeats_excluded(tmp_path):
    # two executions of one draw with divergent tokens -> ambiguous, excluded
    man = _manifest("a", ["pres0"], [0], repeats=2)
    traces = [
        _tr("pres0", "task0", "cond-x", 0, 0, [1, 2, 3]),
        _tr("pres0", "task0", "cond-x", 0, 1, [1, 2, 9]),
    ]
    assert _ambiguous_draw_count(traces) == 1
    store = _store(tmp_path, man)
    keys = {_key(r) for r in traces}
    by_task, meta = _matched_draw_records(store, keys, traces=traces)
    assert meta["n_draws"] == 1
    assert meta["n_ambiguous_draws"] == 1
    assert by_task == {}  # ambiguous draw contributes no primary observation


def test_metrics_report_analysis_population(tmp_path, fake_backend, monkeypatch):
    a = _real_run(tmp_path, fake_backend, monkeypatch, "a", [0, 1])
    b = _real_run(tmp_path, fake_backend, monkeypatch, "b", [0, 1])
    cmp = compare_configs(a.store, b.store, mode="sampled", n_resamples=100)
    assert cmp["analysis_population"]["unit"].startswith("stochastic draw")
    for metric in cmp["metrics"].values():
        assert "analysis_population" in metric
        assert metric["analysis_population"]["n_draws_used_a"] == 4


def test_missing_condition_metadata_fails():
    bad = {"presentation_id": "p", "task_id": "t", "condition_id": "c"}
    try:
        _key(bad)
    except KeyError:
        return
    raise AssertionError("expected a fail-closed KeyError for missing condition metadata")


# --------------------------------------------------------------------------- #
# Stage 4: family reconciliation (closures vs EOS never conflated)
# --------------------------------------------------------------------------- #
def test_family_categories_reconcile(tmp_path, fake_backend, monkeypatch):
    from ponderscope.analysis import analyze_run

    a = _real_run(tmp_path, fake_backend, monkeypatch, "a", [0, 1])
    analysis = analyze_run(a.store)
    c = next(iter(analysis["configs"].values()))
    cats = c["categories"]
    fam = c["per_family"]
    assert sum(d["n"] for d in fam.values()) == cats["n"]
    assert (
        sum(d["categories"]["native_reasoning_closures"] for d in fam.values())
        == cats["native_reasoning_closures"]
    )
    assert (
        sum(d["categories"]["eos_terminations"] for d in fam.values()) == cats["eos_terminations"]
    )
    assert sum(d["categories"]["censored"] for d in fam.values()) == cats["censored"]
    assert sum(d["categories"]["correct_answers"] for d in fam.values()) == cats["correct_answers"]


def test_report_labels_closures_and_eos_distinctly(tmp_path, fake_backend, monkeypatch):
    from ponderscope.analysis import analyze_run, generate_report

    a = _real_run(tmp_path, fake_backend, monkeypatch, "a", [0, 1])
    analysis = analyze_run(a.store)
    md, _ = generate_report(a.store, analysis)
    assert "native closures" in md and "| EOS |" in md
    assert "closures vs EOS are distinct" in md
    # The ALL row must carry the native-closure count, not the EOS count.
    c = next(iter(analysis["configs"].values()))
    assert c["categories"]["native_reasoning_closures"] == 4  # FakeBackend always closes


# --------------------------------------------------------------------------- #
# Stage 5/6: replication specs, frozen population, preflight
# --------------------------------------------------------------------------- #
import dataclasses  # noqa: E402
import json  # noqa: E402

from ponderscope.preflight import run_preflight  # noqa: E402
from ponderscope.tasks import generate_pack  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ALL5 = ["arith", "path", "order", "logic", "sm"]


def test_replication_specs_identical_controlled_dimensions():
    bf16 = ExperimentSpec.from_dict(
        json.loads((ROOT / "specs" / "phase1_6-replication-bf16.json").read_text())
    )
    q4 = ExperimentSpec.from_dict(
        json.loads((ROOT / "specs" / "phase1_6-replication-q4.json").read_text())
    )
    for field in dataclasses.fields(ExperimentSpec):
        if field.name in ("name", "notes"):
            continue
        assert getattr(bf16, field.name) == getattr(q4, field.name), field.name
    # 5 families x 6 tasks x 2 seeds = 60 draws per arm.
    assert bf16.split == "test" and bf16.task_seed == 1729
    assert bf16.n_per_family == 6 and bf16.sampled_seeds == [0, 1]


def test_replication_population_is_disjoint_and_frozen():
    primary = generate_pack(
        families=ALL5,
        n_per_family=6,
        pack="tasks-v1",
        split="test",
        seed=1729,
        prompt_policy="pp-v1",
    )
    calibration = generate_pack(
        families=ALL5,
        n_per_family=2,
        pack="tasks-v1",
        split="calibration",
        seed=0,
        prompt_policy="pp-v1",
    )
    assert len(primary) == 30
    assert {t.task_id for t in primary}.isdisjoint({t.task_id for t in calibration})
    # deterministic regeneration
    again = generate_pack(
        families=ALL5,
        n_per_family=6,
        pack="tasks-v1",
        split="test",
        seed=1729,
        prompt_policy="pp-v1",
    )
    assert [t.task_id for t in primary] == [t.task_id for t in again]


def test_preflight_reports_population_and_expected_generations(tmp_path):
    spec = ExperimentSpec(
        name="pf",
        task_pack="tasks-v1",
        families=ALL5,
        n_per_family=6,
        task_seed=1729,
        split="test",
        greedy_repeats=0,
        sampled_seeds=[0, 1],
        max_tokens=2048,
    )
    report = run_preflight(
        spec,
        model_repo="fake/model",
        model_revision="deadbeef",
        runs_dir=tmp_path,
        per_generation_seconds=33.0,
        comparison_population=("calibration", 0, 2),
    )
    tp = report["checks"]["task_population"]
    assert tp["n_tasks"] == 30
    assert report["checks"]["expected_generations"]["total_executions"] == 60
    assert report["checks"]["expected_generations"]["sampled_draws"] == 60
    assert report["checks"]["disjointness"]["task_id_overlap"] == 0
    assert report["checks"]["disjointness"]["structural_overlap"] == 0
    assert report["checks"]["compute_estimate"]["total_hours"] == round(60 * 33.0 / 3600, 2)
    # fake source is not cached -> advisory problem, but checks still reported
    assert report["checks"]["source_cache"]["available"] is False


def test_cross_study_does_not_pool(tmp_path, fake_backend, monkeypatch):
    from ponderscope.analysis.compare import cross_study_summary

    a = _real_run(tmp_path, fake_backend, monkeypatch, "a", [0, 1])
    b = _real_run(tmp_path, fake_backend, monkeypatch, "b", [0, 1])
    cmp = compare_configs(a.store, b.store, mode="sampled", n_resamples=100)
    summary = cross_study_summary([cmp, cmp], labels=["calibration", "replication"])
    assert summary["pooled"] is None
    assert [s["label"] for s in summary["studies"]] == ["calibration", "replication"]
    assert summary["studies"][0]["n_matched_pairs"] == 4
