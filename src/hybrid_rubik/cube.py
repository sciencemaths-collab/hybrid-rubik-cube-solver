from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from heapq import heappush, heappop
from math import ceil
from typing import Dict, List, Sequence, Tuple

Vec = Tuple[int, int, int]
Slot = Tuple[Vec, Vec]

FACE_NORMALS: Dict[str, Vec] = {
    "U": (0, 1, 0),
    "D": (0, -1, 0),
    "R": (1, 0, 0),
    "L": (-1, 0, 0),
    "F": (0, 0, 1),
    "B": (0, 0, -1),
}
NORMAL_FACE = {v: k for k, v in FACE_NORMALS.items()}

MOVE_SPECS = {
    "U": ((0, 1, 0), 1, -1),
    "D": ((0, 1, 0), -1, 1),
    "R": ((1, 0, 0), 1, -1),
    "L": ((1, 0, 0), -1, 1),
    "F": ((0, 0, 1), 1, -1),
    "B": ((0, 0, 1), -1, 1),
}


def _rot90(v: Vec, axis: Vec, sign: int) -> Vec:
    x, y, z = v
    if axis == (1, 0, 0):
        return (x, -sign * z, sign * y)
    if axis == (0, 1, 0):
        return (sign * z, y, -sign * x)
    if axis == (0, 0, 1):
        return (-sign * y, sign * x, z)
    raise ValueError(axis)


def _slots() -> List[Slot]:
    out: List[Slot] = []
    for face in ("U", "R", "F", "D", "L", "B"):
        n = FACE_NORMALS[face]
        for r in (-1, 0, 1):
            for c in (-1, 0, 1):
                if face == "U":
                    pos = (c, 1, r)
                elif face == "D":
                    pos = (c, -1, -r)
                elif face == "F":
                    pos = (c, -r, 1)
                elif face == "B":
                    pos = (-c, -r, -1)
                elif face == "R":
                    pos = (1, -r, -c)
                else:
                    pos = (-1, -r, c)
                out.append((pos, n))
    return out


SLOTS = _slots()
SLOT_INDEX = {s: i for i, s in enumerate(SLOTS)}
SOLVED = tuple(NORMAL_FACE[n] for _, n in SLOTS)


def _build_perm(face: str) -> Tuple[int, ...]:
    axis, layer, sign = MOVE_SPECS[face]
    perm = list(range(54))
    ai = 0 if axis[0] else (1 if axis[1] else 2)
    for old_idx, (pos, normal) in enumerate(SLOTS):
        if pos[ai] == layer:
            new_slot = (_rot90(pos, axis, sign), _rot90(normal, axis, sign))
            perm[old_idx] = SLOT_INDEX[new_slot]
    return tuple(perm)


BASE_PERMS = {m: _build_perm(m) for m in MOVE_SPECS}


def _apply_perm(state: Tuple[str, ...], perm: Tuple[int, ...]) -> Tuple[str, ...]:
    new = [""] * 54
    for old_i, new_i in enumerate(perm):
        new[new_i] = state[old_i]
    return tuple(new)


def apply_move(state: Tuple[str, ...], move: str) -> Tuple[str, ...]:
    base = move[0]
    turns = 1
    if len(move) > 1:
        if move[1] == "2":
            turns = 2
        elif move[1] == "'":
            turns = 3
    out = state
    for _ in range(turns):
        out = _apply_perm(out, BASE_PERMS[base])
    return out


MOVES = tuple(f + suffix for f in "URFDLB" for suffix in ("", "'", "2"))


def scramble(sequence: Sequence[str]) -> Tuple[str, ...]:
    s = SOLVED
    for m in sequence:
        s = apply_move(s, m)
    return s


def misplaced_facelets(state: Tuple[str, ...]) -> int:
    centers = {4, 13, 22, 31, 40, 49}
    return sum(1 for i, (a, b) in enumerate(zip(state, SOLVED)) if i not in centers and a != b)


def heuristic(state: Tuple[str, ...]) -> int:
    return ceil(misplaced_facelets(state) / 20)


def _allowed(prev: str | None, nxt: str) -> bool:
    return not prev or prev[0] != nxt[0]


@dataclass
class SearchResult:
    method: str
    solved: bool
    moves: List[str]
    nodes_expanded: int


def astar(start: Tuple[str, ...], max_nodes: int = 250_000) -> SearchResult:
    if start == SOLVED:
        return SearchResult("A*", True, [], 0)
    counter = 0
    pq = []
    heappush(pq, (heuristic(start), 0, counter, start, [], None))
    best_g = {start: 0}
    expanded = 0
    while pq and expanded < max_nodes:
        _, g, _, state, path, prev = heappop(pq)
        if state == SOLVED:
            return SearchResult("A*", True, path, expanded)
        if g != best_g.get(state):
            continue
        expanded += 1
        for move in MOVES:
            if not _allowed(prev, move):
                continue
            ns = apply_move(state, move)
            ng = g + 1
            if ng >= best_g.get(ns, 1 << 30):
                continue
            best_g[ns] = ng
            counter += 1
            heappush(pq, (ng + heuristic(ns), ng, counter, ns, path + [move], move))
    return SearchResult("A*", False, [], expanded)


def bfs(start: Tuple[str, ...], max_nodes: int = 250_000) -> SearchResult:
    if start == SOLVED:
        return SearchResult("BFS", True, [], 0)
    q = deque([(start, [], None)])
    seen = {start}
    expanded = 0
    while q and expanded < max_nodes:
        state, path, prev = q.popleft()
        expanded += 1
        for move in MOVES:
            if not _allowed(prev, move):
                continue
            ns = apply_move(state, move)
            if ns in seen:
                continue
            npth = path + [move]
            if ns == SOLVED:
                return SearchResult("BFS", True, npth, expanded)
            seen.add(ns)
            q.append((ns, npth, move))
    return SearchResult("BFS", False, [], expanded)
