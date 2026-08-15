from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from hybrid_rubik.cube import MOVES, astar, bfs, scramble


def random_scramble(depth: int, rng: random.Random) -> list[str]:
    out = []
    while len(out) < depth:
        m = rng.choice(MOVES)
        if out and out[-1][0] == m[0]:
            continue
        out.append(m)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", default="1,2,3,4")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-nodes", type=int, default=250000)
    ap.add_argument("--out", default="benchmarks/cube_search_results.csv")
    ap.add_argument("--fig-dir", default="docs/figures")
    args = ap.parse_args()

    depths = [int(x) for x in args.depths.split(",")]
    rng = random.Random(args.seed)
    rows = []

    for depth in depths:
        for trial in range(args.trials):
            seq = random_scramble(depth, rng)
            state = scramble(seq)
            for name, solver in (("BFS", bfs), ("A*", astar)):
                t0 = time.perf_counter()
                result = solver(state, max_nodes=args.max_nodes)
                dt = time.perf_counter() - t0
                rows.append({
                    "depth": depth,
                    "trial": trial,
                    "scramble": " ".join(seq),
                    "solver": name,
                    "solved": result.solved,
                    "solution_length": len(result.moves) if result.solved else None,
                    "nodes_expanded": result.nodes_expanded,
                    "time_s": dt,
                })

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    summary = df.groupby(["depth", "solver"], as_index=False).agg(
        solve_rate=("solved", "mean"),
        nodes_mean=("nodes_expanded", "mean"),
        time_mean=("time_s", "mean"),
        solution_len_mean=("solution_length", "mean"),
    )
    summary.to_csv(out.with_name("cube_search_summary.csv"), index=False)

    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    for solver, sub in summary.groupby("solver"):
        ax.plot(sub["depth"], sub["nodes_mean"], marker="o", label=solver)
    ax.set_yscale("log")
    ax.set_xlabel("Scramble depth")
    ax.set_ylabel("Mean nodes expanded (log scale)")
    ax.set_title("Cube Search Efficiency")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "cube_nodes_expanded.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    for solver, sub in summary.groupby("solver"):
        ax.plot(sub["depth"], sub["time_mean"], marker="o", label=solver)
    ax.set_xlabel("Scramble depth")
    ax.set_ylabel("Mean runtime (s)")
    ax.set_title("Cube Search Runtime")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "cube_runtime.svg")
    plt.close(fig)

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
