"""Leaky replication (demonstration only — DO NOT compare with honest rows).

Rebuilds the paper-style setup: season-average team features computed over
the *full* season (including the match itself and later matches) plus a
random 80/20 split that shuffles matches across seasons. Both choices leak
the future by construction; that is the point. Reports accuracy only, on the
held-out 20%, and appends a clearly labelled row
(model=leaky_random80, feature_set=leaky_season_avg) to reports/results.csv.

Usage: python -m epl.leaky
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from epl.evaluate import SEED

MATCHES_PATH = Path("data/processed/matches.parquet")
RESULTS_PATH = Path("reports/results.csv")

POINTS = {"H": (3, 0), "D": (1, 1), "A": (0, 3)}


def season_average_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-match features from FULL-season team averages (leaky on purpose).

    Each team's points/goals per game are computed over all 38 matches of
    the season, i.e. they include the current match and matches played after
    it. A walk-forward model never sees any of this.
    """
    df = df.copy()
    pts = {}
    scored = {}
    conceded = {}
    n = {}
    for row in df.itertuples():
        hp, ap = POINTS[row.FTR]
        for team, p, s, c in ((row.HomeTeam, hp, row.FTHG, row.FTAG),
                              (row.AwayTeam, ap, row.FTAG, row.FTHG)):
            key = (row.season, team)
            pts[key] = pts.get(key, 0) + p
            scored[key] = scored.get(key, 0) + s
            conceded[key] = conceded.get(key, 0) + c
            n[key] = n.get(key, 0) + 1
    df["home_season_ppg"] = [pts[(s, t)] / n[(s, t)]
                             for s, t in zip(df["season"], df["HomeTeam"])]
    df["away_season_ppg"] = [pts[(s, t)] / n[(s, t)]
                             for s, t in zip(df["season"], df["AwayTeam"])]
    df["home_season_gs"] = [scored[(s, t)] / n[(s, t)]
                            for s, t in zip(df["season"], df["HomeTeam"])]
    df["away_season_gs"] = [scored[(s, t)] / n[(s, t)]
                            for s, t in zip(df["season"], df["AwayTeam"])]
    df["home_season_gc"] = [conceded[(s, t)] / n[(s, t)]
                            for s, t in zip(df["season"], df["HomeTeam"])]
    df["away_season_gc"] = [conceded[(s, t)] / n[(s, t)]
                            for s, t in zip(df["season"], df["AwayTeam"])]
    df["ppg_diff"] = df["home_season_ppg"] - df["away_season_ppg"]
    df["gs_diff"] = df["home_season_gs"] - df["away_season_gs"]
    df["gc_diff"] = df["home_season_gc"] - df["away_season_gc"]
    return df


def main() -> None:
    """Entry point for `python -m epl.leaky` (one-off; not part of make all)."""
    df = pd.read_parquet(MATCHES_PATH)
    df = season_average_features(df)
    cols = ["home_season_ppg", "away_season_ppg", "ppg_diff",
            "gs_diff", "gc_diff"]
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(df))
    cut = int(0.8 * len(df))
    tr, te = df.iloc[idx[:cut]], df.iloc[idx[cut:]]
    scaler = StandardScaler().fit(tr[cols].to_numpy(dtype=float))
    x_tr = scaler.transform(tr[cols].to_numpy(dtype=float))
    x_te = scaler.transform(te[cols].to_numpy(dtype=float))
    est = LogisticRegression(max_iter=1000, random_state=SEED)
    est.fit(x_tr, tr["target"].to_numpy())
    pred = est.predict(x_te)
    y_te = te["target"].to_numpy()
    acc = float((pred == y_te).mean())
    tp = int(((pred == 1) & (y_te == 1)).sum())
    draw_rec = tp / max(int((y_te == 1).sum()), 1)
    print(f"LEAKY replication (random 80/20, season-average features): "
          f"accuracy={acc:.4f}, draw_recall={draw_rec:.4f}")
    print("Compare with honest walk-forward logreg/P1 accuracy=0.5384. "
          "The gap is the leakage premium: future information (full-season "
          "averages) plus shuffled time splits.")
    row = pd.DataFrame([{"model": "leaky_random80",
                         "feature_set": "leaky_season_avg",
                         "test_season": "pooled",
                         "log_loss": float("nan"), "brier": float("nan"),
                         "accuracy": acc, "draw_recall": draw_rec,
                         "seed": SEED}])
    existing = pd.read_csv(RESULTS_PATH)
    merged = pd.concat([existing, row], ignore_index=True).drop_duplicates(
        subset=["model", "feature_set", "test_season"], keep="last")
    merged.to_csv(RESULTS_PATH, index=False)
    print(f"appended leaky row; {RESULTS_PATH} now {len(merged)} rows")


if __name__ == "__main__":
    main()
