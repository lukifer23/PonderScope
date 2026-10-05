"""Independent per-family invariants and structural collision auditing.

Every generated task's ground truth is re-derived from its own structural
parameters, independently of the generator's arithmetic. Ground truth is never
trusted because it was emitted by the same code path that produced the prompt.

Structural signatures allow a collision audit across pools: id-disjoint pools
are only "structurally held out" if their problem structures do not collide.
"""

from __future__ import annotations

import hashlib
import heapq
import itertools
from typing import Any

from .models import Task, TaskError


def _arith_value(expression: str) -> int:
    import ast
    import operator

    ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def ev(node: ast.AST) -> int:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return int(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](ev(node.left), ev(node.right))  # type: ignore[operator]
        if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
            return ops[type(node.op)](ev(node.operand))  # type: ignore[operator]
        raise ValueError("unsupported arithmetic expression")

    return ev(ast.parse(expression, mode="eval"))


def _dijkstra(edges: list[list[Any]], source: str, target: str) -> int:
    graph: dict[str, list[tuple[str, int]]] = {}
    for a, b, w in edges:
        graph.setdefault(a, []).append((b, int(w)))
        graph.setdefault(b, []).append((a, int(w)))
    dist: dict[str, int] = dict.fromkeys(graph, 10**9)
    dist[source] = 0
    pq = [(0, source)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in graph[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist[target]


def _logic_solutions(clauses: list[list[str]], names: list[str]) -> list[dict[str, bool]]:
    sols: list[dict[str, bool]] = []
    for mask in range(1 << len(names)):
        assignment = {names[i]: bool((mask >> i) & 1) for i in range(len(names))}
        ok = True
        for clause in clauses:
            sat = False
            for lit in clause:
                if lit.startswith("not "):
                    sat = sat or (not assignment[lit[4:]])
                else:
                    sat = sat or assignment[lit]
            if not sat:
                ok = False
                break
        if ok:
            sols.append(assignment)
    return sols


def verify_invariants(task: Task) -> None:
    """Re-derive the answer from structural parameters; raise on mismatch."""
    s = task.structural
    family = task.family
    if family == "arith":
        value = _arith_value(str(s["expression"]))
        if str(value) != task.answer.strip():
            raise TaskError(f"{task.task_id}: arith answer {task.answer!r} != {value}")
    elif family == "path":
        value = _dijkstra([list(e) for e in s["edges"]], str(s["source"]), str(s["target"]))
        if str(value) != task.answer.strip():
            raise TaskError(f"{task.task_id}: path answer {task.answer!r} != {value}")
    elif family == "order":
        constraints = [tuple(c) for c in s["constraints"]]
        claimed = [w.strip() for w in task.answer.split(",") if w.strip()]
        if claimed != [str(x) for x in s["order"]]:
            raise TaskError(f"{task.task_id}: order answer {task.answer!r} != stored order")
        items = [str(x) for x in s["items"]]
        for a, b in constraints:
            if claimed.index(a) >= claimed.index(b):
                raise TaskError(f"{task.task_id}: constraint {a} before {b} violated")
        # uniqueness: exactly one permutation satisfies all constraints
        satisfying = 0
        for perm in itertools.permutations(items):
            pos = {w: i for i, w in enumerate(perm)}
            if all(pos[a] < pos[b] for a, b in constraints):
                satisfying += 1
        if satisfying != 1:
            raise TaskError(f"{task.task_id}: order constraints not unique ({satisfying})")
    elif family == "logic":
        names = [str(v) for v in s["variables"]]
        clauses = [list(c) for c in s["clauses"]]
        sols = _logic_solutions(clauses, names)
        if len(sols) != 1:
            raise TaskError(f"{task.task_id}: logic has {len(sols)} solutions, not 1")
        expected = ", ".join("true" if sols[0][v] else "false" for v in names)
        if task.answer.strip().lower() != expected:
            raise TaskError(f"{task.task_id}: logic answer {task.answer!r} != {expected!r}")
    elif family == "sm":
        table = s["table"]
        current = str(s["start"])
        for ch in str(s["input"]):
            current = str(table[current][ch])
        if current != task.answer.strip():
            raise TaskError(f"{task.task_id}: sm answer {task.answer!r} != {current}")
        if str(s.get("final")) != current:
            raise TaskError(f"{task.task_id}: sm stored final {s.get('final')!r} != {current}")
    else:
        raise TaskError(f"{task.task_id}: unknown family {family!r}")


def structural_signature(task: Task) -> tuple:
    """A hashable, wording-independent fingerprint of the problem structure."""
    s = task.structural
    family = task.family
    if family == "arith":
        return (family, int(s["operations"]), str(s["expression"]))
    if family == "path":
        edges = tuple(sorted(tuple(e) for e in s["edges"]))
        return (family, str(s["source"]), str(s["target"]), edges)
    if family == "order":
        return (
            family,
            tuple(sorted(str(x) for x in s["items"])),
            tuple(sorted(tuple(c) for c in s["constraints"])),
        )
    if family == "logic":
        clauses = tuple(sorted(tuple(sorted(str(lit) for lit in c)) for c in s["clauses"]))
        return (family, tuple(sorted(str(v) for v in s["variables"])), clauses)
    if family == "sm":
        table = s["table"]
        edges = tuple(
            sorted((str(st), str(sym), str(table[st][sym])) for st in table for sym in table[st])
        )
        return (family, edges, str(s["start"]), str(s["input"]))
    return (family,)


def signature_hash(task: Task) -> str:
    import json

    payload = json.dumps(structural_signature(task), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def collision_audit(pools: dict[str, list[Task]]) -> dict[str, Any]:
    """Find structural collisions within and across named pools."""
    index: dict[str, list[str]] = {}
    for pool_name, tasks in pools.items():
        for task in tasks:
            index.setdefault(signature_hash(task), []).append(pool_name)
    cross: dict[str, list[str]] = {}
    within: dict[str, list[str]] = {}
    for sig, owners in index.items():
        if len(owners) > 1 and len(set(owners)) > 1:
            cross[sig] = sorted(set(owners))
        elif len(owners) > 1:
            within[sig] = [owners[0]]
    return {
        "n_pools": len(pools),
        "n_unique_structures": len(index),
        "cross_pool_collisions": cross,
        "within_pool_collisions": within,
        "clean": not cross and not within,
    }
