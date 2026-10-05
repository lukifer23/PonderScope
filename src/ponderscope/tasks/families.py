"""Procedural task families with exact ground truth.

Five mechanically scored families. No LLM judge is used anywhere. Each family
returns a prompt (problem statement), a canonical answer string, structural
difficulty metadata, and the structural parameters that define the instance
(so held-out structural variants can be generated later without changing task
identity).
"""

from __future__ import annotations

import heapq
import random
from collections.abc import Callable
from typing import Any

DIFFICULTY_LEVELS = ("easy", "medium", "hard")


def _level(difficulty: str) -> int:
    return {"easy": 0, "medium": 1, "hard": 2}.get(difficulty, 1)


# --------------------------------------------------------------------------- #
# 1. Multi-step integer arithmetic
# --------------------------------------------------------------------------- #
def make_arith(rng: random.Random, difficulty: str) -> dict[str, Any]:
    ops = 2 + 2 * _level(difficulty)
    expr, value = _arith_expr(rng, ops)
    prompt = (
        "Compute the value of the following expression. "
        "Reply with a single integer on the last line in the form 'Answer: <integer>'.\n\n"
        f"{expr}"
    )
    return {
        "prompt": prompt,
        "answer": str(value),
        "difficulty": {"operations": ops},
        "structural": {"operations": ops, "expression": expr, "family": "arith"},
    }


def _arith_expr(rng: random.Random, ops: int) -> tuple[str, int]:
    value = rng.randint(2, 20)
    expr = str(value)
    for _ in range(ops):
        op = rng.choice(["+", "-", "*"])
        operand = rng.randint(2, 12)
        if op == "*":
            value = value * operand
            expr = f"({expr} * {operand})"
        elif op == "+":
            value = value + operand
            expr = f"({expr} + {operand})"
        else:
            value = value - operand
            expr = f"({expr} - {operand})"
    return expr, value


# --------------------------------------------------------------------------- #
# 2. Weighted shortest path
# --------------------------------------------------------------------------- #
def make_path(rng: random.Random, difficulty: str) -> dict[str, Any]:
    lvl = _level(difficulty)
    n = 4 + lvl * 2
    labels = [chr(ord("A") + i) for i in range(n)]
    edges: dict[tuple[str, str], int] = {}
    # spanning chain guarantees connectivity
    for i in range(n - 1):
        edges[(labels[i], labels[i + 1])] = rng.randint(1, 9)
    extra = 1 + lvl * 2
    for _ in range(extra):
        a, b = rng.sample(range(n), 2)
        if a > b:
            a, b = b, a
        if (labels[a], labels[b]) not in edges:
            edges[(labels[a], labels[b])] = rng.randint(1, 9)
    edge_list = sorted(edges.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    undirected: dict[str, list[tuple[str, int]]] = {lab: [] for lab in labels}
    for (u, v), w in edge_list:
        undirected[u].append((v, w))
        undirected[v].append((u, w))
    src, dst = labels[0], labels[-1]
    dist = _dijkstra(undirected, src)
    lines = "\n".join(f"  {u} -- {v} : weight {w}" for (u, v), w in edge_list)
    prompt = (
        "The following undirected graph has weighted edges. The cost of a path is "
        "the sum of its edge weights. What is the minimum cost from "
        f"{src} to {dst}?\n\n{lines}\n\n"
        "Reply with a single integer on the last line in the form 'Answer: <integer>'."
    )
    return {
        "prompt": prompt,
        "answer": str(dist[dst]),
        "difficulty": {"nodes": n, "edges": len(edge_list)},
        "structural": {
            "nodes": n,
            "edges": [[a, b, w] for (a, b), w in edge_list],
            "source": src,
            "target": dst,
            "family": "path",
        },
    }


def _dijkstra(graph: dict[str, list[tuple[str, int]]], src: str) -> dict[str, int]:
    dist = dict.fromkeys(graph, 10**9)
    dist[src] = 0
    pq = [(0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in graph[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist


# --------------------------------------------------------------------------- #
# 3. Constraint / ordering
# --------------------------------------------------------------------------- #
def make_order(rng: random.Random, difficulty: str) -> dict[str, Any]:
    lvl = _level(difficulty)
    n = 3 + lvl
    words = rng.sample(["amber", "basil", "cedar", "dune", "ember", "frost", "gale", "haven"], n)
    order = words[:]
    rng.shuffle(order)
    constraints: list[tuple[str, str]] = []
    for i in range(n - 1):
        constraints.append((order[i], order[i + 1]))
    # redundant (implied) distractors
    if n >= 4:
        constraints.append((order[0], order[2]))
    rng.shuffle(constraints)
    lines = "\n".join(f"  {a} must come before {b}" for a, b in constraints)
    prompt = (
        "A set of ordering constraints is given. Exactly one ordering of the items "
        "satisfies all constraints. List the items in that order, separated by "
        "commas, on the last line in the form 'Answer: item1, item2, ...'.\n\n"
        f"Items: {', '.join(words)}\nConstraints:\n{lines}"
    )
    return {
        "prompt": prompt,
        "answer": ",".join(order),
        "difficulty": {"items": n, "constraints": len(constraints)},
        "structural": {
            "items": words,
            "constraints": constraints,
            "order": order,
            "family": "order",
        },
    }


# --------------------------------------------------------------------------- #
# 4. Boolean / logic
# --------------------------------------------------------------------------- #
def make_logic(rng: random.Random, difficulty: str) -> dict[str, Any]:
    lvl = _level(difficulty)
    n = 2 + lvl
    names = ["P", "Q", "R", "S"][:n]
    target = {name: rng.choice([True, False]) for name in names}
    clauses = _logic_clauses(rng, names, target, n + 2 + lvl)
    # REQUIRE a unique satisfying assignment. Add clauses until uniqueness holds;
    # fail loudly rather than emitting "exactly one assignment" without proof.
    attempts = 0
    while not _unique_solution(clauses, names, target):
        attempts += 1
        if attempts > 500:
            raise ValueError("could not generate a uniquely satisfiable logic task")
        clauses.extend(_logic_clauses(rng, names, target, 1))
    lines = "\n".join("  (" + " or ".join(lit) + ")" for lit in clauses)
    prompt = (
        "The following logical clauses must all be true, where a variable may be "
        "true or false and 'not X' means X is false. Exactly one assignment of the "
        "variables satisfies every clause.\n\n"
        f"{lines}\n\n"
        "Give the value of "
        + ", ".join(names)
        + " in that order, each as true or false, separated by commas, on the last "
        "line in the form 'Answer: <value>, <value>, ...'."
    )
    answer = ", ".join("true" if target[name] else "false" for name in names)
    return {
        "prompt": prompt,
        "answer": answer,
        "difficulty": {"variables": n, "clauses": len(clauses)},
        "structural": {
            "variables": names,
            "clauses": clauses,
            "solution": dict(target),
            "family": "logic",
        },
    }


def _logic_clauses(
    rng: random.Random, names: list[str], target: dict[str, bool], count: int
) -> list[list[str]]:
    clauses: list[list[str]] = []
    for _ in range(count):
        k = rng.randint(1, 2)
        vars_ = rng.sample(names, k)
        clause: list[str] = []
        for v in vars_:
            # choose a literal that is satisfied by target
            if target[v]:
                clause.append(v)
            else:
                clause.append(f"not {v}")
        clauses.append(clause)
    return clauses


def _unique_solution(clauses: list[list[str]], names: list[str], target: dict[str, bool]) -> bool:
    # ``target`` satisfies every clause by construction, so at least one solution
    # always exists. Uniqueness means no second assignment satisfies all clauses.
    solutions = 0
    n = len(names)
    for mask in range(1 << n):
        assignment = {names[i]: bool((mask >> i) & 1) for i in range(n)}
        if all(_clause_true(clause, assignment) for clause in clauses):
            solutions += 1
            if solutions > 1:
                return False
    return solutions == 1


def _clause_true(clause: list[str], assignment: dict[str, bool]) -> bool:
    for lit in clause:
        if lit.startswith("not "):
            if not assignment[lit[4:]]:
                return True
        elif assignment[lit]:
            return True
    return False


# --------------------------------------------------------------------------- #
# 5. Deterministic state machine
# --------------------------------------------------------------------------- #
def make_sm(rng: random.Random, difficulty: str) -> dict[str, Any]:
    lvl = _level(difficulty)
    n_states = 3 + lvl
    length = 4 + 2 * lvl
    states = [f"s{i}" for i in range(n_states)]
    alphabet = ["0", "1"]
    table: dict[str, dict[str, str]] = {}
    for s in states:
        table[s] = {sym: rng.choice(states) for sym in alphabet}
    start = states[0]
    input_str = "".join(rng.choice(alphabet) for _ in range(length))
    current = start
    for ch in input_str:
        current = table[current][ch]
    lines = "\n".join(f"  {s} -- {sym} --> {table[s][sym]}" for s in states for sym in alphabet)
    prompt = (
        "The following deterministic state machine starts in state "
        f"{start}. Process the input string {input_str} one symbol at a time, "
        "moving according to the transition table. What is the final state?\n\n"
        f"{lines}\n\n"
        "Reply with the state name (for example 'Answer: <state name>') on the last line."
    )
    return {
        "prompt": prompt,
        "answer": current,
        "difficulty": {"states": n_states, "input_length": length},
        "structural": {
            "states": states,
            "table": table,
            "start": start,
            "input": input_str,
            "final": current,
            "family": "sm",
        },
    }


FAMILY_MAKERS: dict[str, Callable[[random.Random, str], dict[str, Any]]] = {
    "arith": make_arith,
    "path": make_path,
    "order": make_order,
    "logic": make_logic,
    "sm": make_sm,
}

ALL_FAMILIES: tuple[str, ...] = tuple(FAMILY_MAKERS)


def make_family_task(family: str, rng: random.Random, difficulty: str) -> dict[str, Any]:
    if family not in FAMILY_MAKERS:
        raise ValueError(f"unknown family: {family!r}")
    if difficulty not in DIFFICULTY_LEVELS:
        raise ValueError(f"unknown difficulty: {difficulty!r}")
    return FAMILY_MAKERS[family](rng, difficulty)
