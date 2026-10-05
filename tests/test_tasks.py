from __future__ import annotations

import pytest

from ponderscope.tasks import (
    ALL_FAMILIES,
    Task,
    TaskError,
    generate_pack,
    make_task,
    normalize,
    score,
    task_id_for,
    validate_task,
)


def test_generation_is_deterministic():
    a = [t.to_dict() for t in generate_pack(n_per_family=3, seed=0)]
    b = [t.to_dict() for t in generate_pack(n_per_family=3, seed=0)]
    assert a == b


def test_generation_changes_with_seed():
    a = [t.to_dict() for t in generate_pack(n_per_family=3, seed=0)]
    b = [t.to_dict() for t in generate_pack(n_per_family=3, seed=1)]
    assert a != b


def test_all_families_generate_and_score():
    tasks = generate_pack(n_per_family=2, seed=0)
    families = {t.family for t in tasks}
    assert families == set(ALL_FAMILIES)
    for t in tasks:
        assert score(t.family, f"Reasoning here.\nAnswer: {t.answer}", t.answer)
        assert normalize(t.family, f"Answer: {t.answer}") is not None


def test_answer_not_leaked_at_cue():
    base = make_task("arith", 0)
    leaked = Task(**{**base.to_dict(), "prompt": base.prompt + f"\nAnswer: {base.answer}"})
    with pytest.raises(TaskError):
        validate_task(leaked)


def test_task_id_independent_of_wording():
    a = make_task("logic", 1)
    b = Task(**{**a.to_dict(), "prompt": "completely different wording"})
    assert a.task_id == b.task_id


def test_task_id_stable_function():
    assert task_id_for("tasks-v1", "arith", "dev", 0, 0, None) == make_task("arith", 0).task_id


def test_splits_are_disjoint():
    dev = generate_pack(n_per_family=2, split="dev", seed=0)
    test = generate_pack(n_per_family=2, split="test", seed=0)
    assert {t.task_id for t in dev}.isdisjoint({t.task_id for t in test})


def test_difficulty_metadata_present():
    for t in generate_pack(n_per_family=3, seed=0):
        assert "level" in t.difficulty
        assert t.difficulty["level"] in {"easy", "medium", "hard"}


def test_structural_variant_changes_structure_but_not_id():
    base = make_task("arith", 0, variant=None)
    variant = make_task("arith", 0, variant="v2")
    assert base.task_id != variant.task_id  # variant is part of identity
    assert base.structural != variant.structural


def test_scorer_rejects_wrong_answer():
    t = make_task("arith", 0)
    wrong = str(int(t.answer) + 1) if t.answer.lstrip("-").isdigit() else "nope"
    assert not score(t.family, f"Answer: {wrong}", t.answer)


def test_normalize_order_and_logic():
    assert normalize("order", "Answer: Cedar, Amber, Dune") == "cedar,amber,dune"
    assert normalize("logic", "Answer: True, False") == "true, false"
    assert normalize("sm", "Answer: s12") == "s12"
    assert normalize("arith", "the answer is 42") == "42"
