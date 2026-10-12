"""Phase 1.8 tests: saved-prefix replay, forced-outcome semantics, matched-checkpoint
eligibility, provenance, and paired recoverability.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ponderscope.analysis.paired import paired_outcomes
from ponderscope.analysis.recoverability import (
    checkpoint_keys,
    paired_recoverability,
    render_html,
    render_markdown,
)
from ponderscope.backends.base import CaptureSpec
from ponderscope.backends.processors import make_generated_history_presence_penalty
from ponderscope.config.identity import DecodingPolicy
from ponderscope.config.schema import ExperimentSpec
from ponderscope.evidence.bundle import bundle_run
from ponderscope.evidence.run import RunStore, verify_run
from ponderscope.experiment import run_experiment
from ponderscope.reasoning.outcomes import classify_forced_probe
from ponderscope.reasoning.probes import run_prefix_probes_at
from ponderscope.reasoning.replay import (
    ReplayError,
    SourceDraw,
    plan_draw,
    reasoning_tokens_only,
    run_budget_diagnostic,
    run_probe_existing,
    verify_source_run,
)
from ponderscope.reasoning.transitions import classify_forced_outcomes

CLEAN = {"version": "test", "git_sha": "0" * 40, "tracked_dirty": False}
THINK_END = 99
EOS = 42


# --------------------------------------------------------------------------- #
# Reasoning-prefix extraction: never a final-answer token or an EOS token
# --------------------------------------------------------------------------- #
def test_reasoning_prefix_closed_excludes_answer_and_eos():
    token_ids = [1, 2, 3, THINK_END, 200, 201, EOS]
    prefix = reasoning_tokens_only(token_ids, THINK_END, {EOS})
    assert prefix == [1, 2, 3]
    assert THINK_END not in prefix and EOS not in prefix and 200 not in prefix


def test_reasoning_prefix_unclosed_strips_trailing_eos():
    # A terminated-without-close trajectory ends in EOS, not a think-end token.
    token_ids = [1, 2, 3, EOS]
    prefix = reasoning_tokens_only(token_ids, THINK_END, {EOS})
    assert EOS not in prefix
    assert prefix == [1, 2, 3]


# --------------------------------------------------------------------------- #
# Forced-probe outcome taxonomy
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("forced_text", "gold", "capped", "error", "expected"),
    [
        ("Answer: 5", "5", False, None, "correct"),
        ("Answer: 6", "5", False, None, "scorable_incorrect"),
        ("no digits here", "5", False, None, "unparseable"),
        ("", "5", True, None, "censored"),
        # capped mid-generation with a stray extractable integer: still censored,
        # never an observed wrong answer
        ("To compute 18 * 10 ...", "5", True, None, "censored"),
        ("", "5", False, None, "unparseable"),
        ("anything", "5", False, "RuntimeError: boom", "error"),
    ],
)
def test_forced_outcome_matrix(forced_text, gold, capped, error, expected):
    result = classify_forced_probe(
        "arith",
        gold,
        forced_text,
        finish_reason="length" if capped else ("error" if error else "stop"),
        capped=capped,
        error=error,
        eos_observed=not capped and error is None,
    )
    assert result["outcome"] == expected


def test_capped_with_extractable_answer_is_censored_not_wrong():
    result = classify_forced_probe(
        "arith",
        "5",
        "To compute the value ... 18",
        finish_reason="length",
        capped=True,
        eos_observed=False,
    )
    assert result["outcome"] == "censored"
    assert result["censored_with_extractable_answer"] is True
    assert result["pre_censor_outcome"] == "scorable_incorrect"


def test_censored_probe_is_not_a_wrong_answer():
    result = classify_forced_probe("arith", "5", "", finish_reason="length", capped=True)
    assert result["outcome"] == "censored"
    assert result["answer_scorable"] is False
    # a censored probe must never materialize as a scored incorrect answer
    assert result["outcome"] != "scorable_incorrect"


def test_unparseable_is_distinct_from_incorrect():
    unscorable = classify_forced_probe(
        "arith", "5", "I cannot decide", finish_reason="stop", capped=False
    )
    wrong = classify_forced_probe("arith", "5", "Answer: 6", finish_reason="stop", capped=False)
    assert unscorable["outcome"] == "unparseable"
    assert wrong["outcome"] == "scorable_incorrect"


# --------------------------------------------------------------------------- #
# Outcome-based transition classification: missing/indeterminate are not wrong
# --------------------------------------------------------------------------- #
def test_missing_checkpoint_is_not_wrong():
    state = classify_forced_outcomes(
        {"0": "scorable_incorrect", "256": None, "512": None},
        checkpoint_order=["0", "256", "512"],
    )
    assert state["n_missing"] == 2
    assert state["state"] == "never_correct"
    # missing checkpoints never appear as observed scorable outcomes
    assert set(state["observed_scorable"]) == {"0"}


def test_indeterminate_probe_does_not_create_a_flip():
    state = classify_forced_outcomes(
        {"0": "correct", "256": "censored", "512": "correct"},
        checkpoint_order=["0", "256", "512"],
    )
    # the censored middle checkpoint is not an observed flip to "wrong"
    assert state["n_indeterminate"] == 1
    assert state["flips"] == 0
    assert state["state"] == "stable_correct"
    assert state["stable_through_end"] is True


# --------------------------------------------------------------------------- #
# Checkpoint plan: exact eligibility, no substitution
# --------------------------------------------------------------------------- #
def _draw(rlen: int) -> SourceDraw:
    return SourceDraw(
        task_id="t1",
        presentation_id="p1",
        family="arith",
        difficulty={"level": "easy"},
        prompt="q",
        prompt_policy="pp-v1",
        split="test",
        seed=0,
        mode="sampled",
        condition_id="cond-x",
        deployment_id="dep-x",
        stochastic_draw_id="draw-x",
        trial_id="trial-x",
        trace_sha256="0" * 64,
        reasoning_token_ids=list(range(rlen)),
    )


def test_plan_draw_eligibility_and_exact_alignment():
    plan = plan_draw(_draw(300), [0, 256, 512], include_full=True)
    assert plan["checkpoint_eligible"]["0"] is True
    assert plan["checkpoint_eligible"]["256"] is True
    assert plan["checkpoint_eligible"]["512"] is False
    assert plan["ineligible_checkpoints"] == ["512"]
    lengths = [p["checkpoint"] for p in plan["planned"]]
    assert lengths == [0, 256, 300]
    assert plan["planned"][-1]["checkpoint_label"] == "full_prefix"
    assert plan["planned"][-1]["is_full_prefix"] is True


def test_plan_draw_full_prefix_coinciding_with_a_checkpoint():
    plan = plan_draw(_draw(256), [0, 256], include_full=True)
    entry = plan["planned"][-1]
    assert entry["checkpoint"] == 256
    assert entry["checkpoint_label"] == "256"
    assert entry["is_full_prefix"] is True
    # no duplicate probe is scheduled for the same token length
    assert [p["checkpoint"] for p in plan["planned"]] == [0, 256]


def test_run_prefix_probes_at_rejects_length_beyond_reasoning(fake_backend):
    with pytest.raises(ValueError):
        run_prefix_probes_at(
            fake_backend,
            family="arith",
            answer="1",
            prompt_token_ids=[10, 20, 30],
            prefix_token_ids=[1, 2, 3],
            lengths=[4],
            probe_decoding=DecodingPolicy(mode="greedy", max_tokens=8),
            capture=CaptureSpec.minimal(),
        )


# --------------------------------------------------------------------------- #
# Replay/logits-processor boundary semantics
# --------------------------------------------------------------------------- #
def test_generated_history_presence_penalizes_replayed_prefix():
    import mlx.core as mx

    # history = prompt (2) + replayed prefix (3) + new (1); prompt_len=2
    processor = make_generated_history_presence_penalty(1.5, prompt_len=2)
    tokens = mx.array([10, 11, 50, 51, 52, 60])
    logits = mx.zeros((1, 100))
    out = processor(tokens, logits)
    mx.eval(out)
    arr = out.tolist()[0]
    # the replayed prefix tokens 50/51/52 must be penalized (they are generated
    # history, not prompt); the true prompt tokens 10/11 must not.
    for tid in (50, 51, 52, 60):
        assert arr[tid] < 0
    for tid in (10, 11):
        assert arr[tid] == 0


def test_probe_run_passes_true_prompt_length_to_processors(monkeypatch):
    import sys

    import mlx_lm.generate  # noqa: F401  (ensure the submodule is imported)

    from ponderscope.backends.mlx_backend import MlxBackend

    backend = MlxBackend()
    backend._model = object()  # bypass the "not loaded" guard for this unit test

    class _StubTokenizer:
        think_end_id = 99
        eos_token_ids = {42}

        def decode(self, token_ids):
            return " ".join(str(t) for t in token_ids)

    backend._tokenizer = _StubTokenizer()

    def fake_generate_step(*_args, **_kwargs):
        return iter(())

    generate_module = sys.modules["mlx_lm.generate"]
    monkeypatch.setattr(generate_module, "generate_step", fake_generate_step, raising=False)
    decoding = DecodingPolicy(
        mode="greedy",
        max_tokens=4,
        presence_penalty=1.5,
        presence_scope="generated_history",
    )
    trace = backend._run(
        [1, 2, 3],
        decoding,
        CaptureSpec.minimal(),
        extra_tokens=[7, 8],
        processor_prompt_len=3,
    )
    assert trace.extra["processor_prompt_len"] == 3
    assert trace.extra["extra_tokens_are_generated_history"] is True
    assert trace.extra["logits_processors"]["base_prompt_tokens"] == 3


# --------------------------------------------------------------------------- #
# End-to-end replay over a fake sealed source run
# --------------------------------------------------------------------------- #
def _make_source_run(tmp_path, fake_backend, monkeypatch, name, sample_seeds=(0,)):
    monkeypatch.setattr("ponderscope.experiment.get_backend", lambda _n: fake_backend)
    spec = ExperimentSpec(
        name=name,
        task_pack="tasks-v1",
        families=["arith"],
        n_per_family=2,
        greedy_repeats=0,
        sampled_seeds=list(sample_seeds),
        max_tokens=32,
    )
    result = run_experiment(
        spec,
        model_repo="fake/model",
        model_revision="deadbeef",
        runs_dir=tmp_path,
        code_state=CLEAN,
    )
    return result.store


def _probe(tmp_path, source_store, fake_backend, label, checkpoints=(0, 5, 10, 15)):
    return run_probe_existing(
        source_store.path,
        backend=fake_backend,
        model_repo="fake/model",
        model_revision="deadbeef",
        run_label=label,
        checkpoints=list(checkpoints),
        probe_max_tokens=8,
        seeds=[0],
        runs_dir=tmp_path,
        code_state=CLEAN,
    )


def test_probe_existing_seals_and_preserves_source(tmp_path, fake_backend, monkeypatch):
    source = _make_source_run(tmp_path, fake_backend, monkeypatch, "source-bf16")
    before = {p.name: p.read_bytes() for p in source.path.iterdir() if p.is_file()}
    before_seal = source.read_seal()

    result = _probe(tmp_path, source, fake_backend, "phase1-8-probe-bf16")
    store = result.store

    # source run is byte-for-byte untouched
    after = {p.name: p.read_bytes() for p in source.path.iterdir() if p.is_file()}
    assert before == after
    assert source.read_seal() == before_seal

    # probe run is a distinct, sealed, verifiable run
    assert store.path != source.path
    assert verify_run(store.path)["pass"] is True
    assert store.manifest["run_kind"] == "prefix_probe_replay"

    records = store.read_probes()
    assert len(records) == result.n_probes > 0
    for record in records:
        # provenance lineage
        assert record["source_run_id"] == source.run_id
        assert record["source_evidence_seal_sha256"]
        assert record["source_trace_sha256"]
        assert record["source_stochastic_draw_id"]
        assert record["source_trial_id"]
        # exact checkpoint alignment
        assert record["reasoning_prefix_tokens"] == record["checkpoint"]
        # explicit outcome taxonomy
        assert record["outcome"] in {
            "correct",
            "scorable_incorrect",
            "unparseable",
            "censored",
            "error",
        }
        # no final-answer/EOS leakage into the replayed prefix hash
        prefix_hash = record["prefix_token_ids_sha256"]
        assert isinstance(prefix_hash, str) and len(prefix_hash) == 64


def test_probe_existing_refuses_source_without_seal(tmp_path, fake_backend, monkeypatch):
    source = _make_source_run(tmp_path, fake_backend, monkeypatch, "source-noseal")
    (source.path / "evidence.json").unlink()
    with pytest.raises(ReplayError):
        verify_source_run(source.path)


def test_probe_existing_refuses_prompt_retokenization_mismatch(tmp_path, fake_backend, monkeypatch):
    source = _make_source_run(tmp_path, fake_backend, monkeypatch, "source-tok")
    original = fake_backend.tokenize_prompt

    def bad_tokenize(messages, enable_thinking):
        return original(messages, enable_thinking) + [999]

    fake_backend.tokenize_prompt = bad_tokenize  # type: ignore[method-assign]
    try:
        with pytest.raises(ReplayError):
            _probe(tmp_path, source, fake_backend, "phase1-8-probe-bad")
    finally:
        fake_backend.tokenize_prompt = original  # type: ignore[method-assign]


def test_probe_existing_refuses_artifact_identity_mismatch(tmp_path, fake_backend, monkeypatch):
    source = _make_source_run(tmp_path, fake_backend, monkeypatch, "source-bf16-id")
    with pytest.raises(ReplayError):
        run_probe_existing(
            source.path,
            backend=fake_backend,
            model_repo="fake/model",
            model_revision="deadbeef",
            artifact_path="some/derived/artifact",
            run_label="phase1-8-probe-mismatch",
            checkpoints=[0, 5],
            probe_max_tokens=8,
            seeds=[0],
            runs_dir=tmp_path,
            code_state=CLEAN,
        )


def test_probe_bundle_is_reproducible(tmp_path, fake_backend, monkeypatch):
    source = _make_source_run(tmp_path, fake_backend, monkeypatch, "source-bundle")
    store = _probe(tmp_path, source, fake_backend, "phase1-8-probe-bundle").store
    first = bundle_run(store.path, tmp_path / "a.tar.gz")
    second = bundle_run(store.path, tmp_path / "b.tar.gz")
    assert first["sha256"] == second["sha256"]


# --------------------------------------------------------------------------- #
# Paired recoverability (offline; no inference)
# --------------------------------------------------------------------------- #
def test_paired_recoverability_reconciles_and_detects_degenerate(
    tmp_path, fake_backend, monkeypatch
):
    source = _make_source_run(tmp_path, fake_backend, monkeypatch, "source-pair")
    bf16 = _probe(tmp_path, source, fake_backend, "phase1-8-probe-bf16").store
    q4 = _probe(tmp_path, source, fake_backend, "phase1-8-probe-q4").store

    analysis = paired_recoverability(bf16, q4)
    assert analysis["analysis_version"] == "phase1.8"
    assert analysis["delta_orientation"].startswith("Q4 - BF16")
    keys = analysis["checkpoints"]
    assert keys == checkpoint_keys(bf16)
    for key in keys:
        cp = analysis["per_checkpoint"][key]
        n = cp["n_jointly_eligible"]
        # paired transition buckets are exhaustive over the jointly eligible set
        assert sum(cp["paired_correctness_transitions"].values()) == n
        assert sum(cp["paired_indeterminate_transitions"].values()) == n
        delta = cp["delta_recoverable_correct_q4_minus_bf16"]
        # identical arms -> no distinguishable difference -> degenerate, no CI
        if n >= 2:
            assert delta["mean"] == 0.0
            assert delta["degenerate"] is True
            assert delta["lo"] is None and delta["hi"] is None


def test_paired_recoverability_is_idempotent(tmp_path, fake_backend, monkeypatch):
    source = _make_source_run(tmp_path, fake_backend, monkeypatch, "source-idem")
    bf16 = _probe(tmp_path, source, fake_backend, "phase1-8-probe-bf16").store
    q4 = _probe(tmp_path, source, fake_backend, "phase1-8-probe-q4").store
    first = paired_recoverability(bf16, q4)
    second = paired_recoverability(bf16, q4)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    md = render_markdown(first)
    html = render_html(first)
    assert "EXPLORATORY" in md and "<html" in html


def test_recoverability_refuses_unverified_probe_run(tmp_path, fake_backend, monkeypatch):
    source = _make_source_run(tmp_path, fake_backend, monkeypatch, "source-corrupt")
    bf16 = _probe(tmp_path, source, fake_backend, "phase1-8-probe-bf16").store
    q4 = _probe(tmp_path, source, fake_backend, "phase1-8-probe-q4").store
    with open(q4.path / "probes.jsonl", "a", encoding="utf-8") as fh:
        fh.write("{}\n")
    with pytest.raises(RuntimeError):
        paired_recoverability(bf16, q4)


def test_probe_budget_diagnostic_records_outcomes(tmp_path, fake_backend, monkeypatch):
    source = _make_source_run(tmp_path, fake_backend, monkeypatch, "source-budget")
    task_id = source.read_tasks()[0]["task_id"]
    result = run_budget_diagnostic(
        source.path,
        backend=fake_backend,
        model_repo="fake/model",
        model_revision="deadbeef",
        arm_label="phase1-8-budget-bf16",
        task_ids=[task_id],
        seed=0,
        checkpoint="full_prefix",
        budgets=[4, 8],
    )
    assert result.checkpoint == "full_prefix"
    assert [row["probe_max_tokens"] for row in result.per_task[0]["per_budget"]] == [4, 8]
    for row in result.per_task[0]["per_budget"]:
        assert row["outcome"] in {
            "correct",
            "scorable_incorrect",
            "unparseable",
            "censored",
            "error",
        }


# --------------------------------------------------------------------------- #
# Paired outcome nomenclature (Phase 1.8 rename)
# --------------------------------------------------------------------------- #
def test_paired_success_at_budget_naming(tmp_path, fake_backend, monkeypatch):
    a = _make_source_run(tmp_path, fake_backend, monkeypatch, "po-a")
    b = _make_source_run(tmp_path, fake_backend, monkeypatch, "po-b")
    out = paired_outcomes(a, b, mode="sampled")
    assert "success_at_budget_transitions" in out
    assert "correctness_transitions" in out  # legacy alias preserved
    assert "natural_final_status_transition_counts" in out
    assert out["analysis_version"] == "phase1.8"


def test_historical_and_new_seals_verify(tmp_path, fake_backend, monkeypatch):
    source = _make_source_run(tmp_path, fake_backend, monkeypatch, "source-seal")
    assert verify_run(source.path)["pass"] is True
    probe = _probe(tmp_path, source, fake_backend, "phase1-8-probe-seal").store
    assert verify_run(probe.path)["pass"] is True


def test_probe_records_have_no_final_answer_tokens(tmp_path, fake_backend, monkeypatch):
    source = _make_source_run(tmp_path, fake_backend, monkeypatch, "source-leak")
    # first source trace's own token stream ends with think_end + answer tokens
    traces = source.read_traces()
    assert any(rec["trace"]["token_ids"][-1] != THINK_END for rec in traces)
    store = _probe(tmp_path, source, fake_backend, "phase1-8-probe-leak").store
    for record in store.read_probes():
        assert record["reasoning_prefix_tokens"] <= record["source_reasoning_tokens"]
        assert record["reasoning_prefix_tokens"] == record["checkpoint"]


def test_runstore_paths(tmp_path, fake_backend, monkeypatch):
    source = _make_source_run(tmp_path, fake_backend, monkeypatch, "source-path")
    store = _probe(tmp_path, source, fake_backend, "phase1-8-probe-path").store
    assert Path(store.path).exists()
    assert RunStore.load(store.path).run_id == store.run_id
