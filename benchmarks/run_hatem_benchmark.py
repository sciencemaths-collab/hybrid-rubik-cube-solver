"""Reproducible HATEM benchmark runner."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from hybrid_rubik.hatem import (
    SCENARIOS,
    TinyPolicyNet,
    hybrid_adaptive,
    knot_continuous,
    sample_curriculum_scenario,
    sample_target_family,
)


def train_policy(n: int, episodes: int, budget: int, ground_states: int):
    rng = np.random.default_rng(123)
    policy = TinyPolicyNet(in_dim=11, hid=14, seed=999)
    rows = []
    for ep in range(episodes):
        stage = ep / max(1, episodes - 1)
        scen_name = sample_curriculum_scenario(rng, stage)
        kind = sample_target_family(rng, stage)
        target_seed = int(rng.integers(0, max(1, ground_states)))
        res, policy = hybrid_adaptive(
            n=n,
            scen=SCENARIOS[scen_name].copy(),
            seed=1000 + ep * 13,
            budget=budget,
            kind=kind,
            target_seed=target_seed,
            policy=policy,
            learn=True,
        )
        rows.append({"episode": ep + 1, "score": res.score, "auc": res.auc})
    return policy, pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=36)
    ap.add_argument("--episodes", type=int, default=80)
    ap.add_argument("--train-budget", type=int, default=600)
    ap.add_argument("--ground-states", type=int, default=200)
    ap.add_argument("--budget", type=int, default=1600)
    ap.add_argument("--seeds", default="3,7,11,19,23")
    ap.add_argument("--out-dir", default="benchmarks/reproduced")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    policy, train_df = train_policy(args.n, args.episodes, args.train_budget, args.ground_states)
    train_df.to_csv(out / "train_log.csv", index=False)

    rows = []
    for scenario, scen in SCENARIOS.items():
        for seed in [int(x) for x in args.seeds.split(",")]:
            c = knot_continuous(args.n, scen.copy(), seed=seed, budget=args.budget, kind="trefoil", target_seed=0)
            h, _ = hybrid_adaptive(args.n, scen.copy(), seed=seed, budget=args.budget, kind="trefoil", target_seed=0, policy=policy, learn=False)
            for name, r in (("Knot-Continuous", c), ("Hybrid-Adaptive (trained)", h)):
                rows.append({"scenario": scenario, "seed": seed, "solver": name, "score": r.score, "auc": r.auc, "time_s": r.time_s})

    df = pd.DataFrame(rows)
    df.to_csv(out / "results.csv", index=False)
    summary = df.groupby("solver", as_index=False).agg(
        score_mean=("score", "mean"),
        score_std=("score", "std"),
        auc_mean=("auc", "mean"),
        time_mean=("time_s", "mean"),
    )
    summary.to_csv(out / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
