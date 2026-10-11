"""Walk-forward evaluation by season + benchmark rows.

Splits (per docs/evaluation.md): for each test season from 2014/15 through
the latest season in the data, train on all earlier seasons *except* the one
directly before the test season (that one is the validation season for
tuning in milestone 4), then score the test season once. Never shuffles.

Benchmarks (no training, no tuning):
- always_home: P(H)=0.98, P(D)=P(A)=0.01 (a hard 1.0 would give infinite log
  loss; the 1% floor is documented here and applied identically every season).
- class_priors: class distribution of the *training* seasons only.
- bookmaker: de-overrounded Bet365 implied probabilities from matches.parquet.

Metrics: log loss (primary), Brier score vs one-hot (primary), accuracy and
per-class precision/recall (secondary). results.csv columns:
model, feature_set, test_season, log_loss, brier, accuracy, draw_recall, seed.

Usage: python -m epl.evaluate
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FEATURES_PATH = Path("data/processed/features.parquet")
MATCHES_PATH = Path("data/processed/matches.parquet")
RESULTS_PATH = Path("reports/results.csv")

SEED = 42
FIRST_TEST_SEASON = "2014/15"
ALWAYS_HOME_EPS = 0.01
CLASSES = [0, 1, 2]


def get_splits(seasons: list[str]) -> list[dict]:
    """Walk-forward splits: test from FIRST_TEST_SEASON onward."""
    splits = []
    for i, test in enumerate(seasons):
        if test < FIRST_TEST_SEASON:
            continue
        valid = seasons[i - 1]
        train = [s for s in seasons[:i] if s != valid]
        splits.append({"test": test, "train": train, "valid": valid})
    return splits


def log_loss_score(y_true: np.ndarray, proba: np.ndarray, eps: float = 1e-15) -> float:
    """Multiclass log loss with probability clipping."""
    proba = np.clip(proba, eps, 1.0)
    proba = proba / proba.sum(axis=1, keepdims=True)
    return float(-np.log(proba[np.arange(len(y_true)), y_true]).mean())


def brier_score(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Mean squared error between predicted probabilities and one-hot truth."""
    one_hot = np.zeros_like(proba)
    one_hot[np.arange(len(y_true)), y_true] = 1.0
    return float(((proba - one_hot) ** 2).sum(axis=1).mean())


def accuracy_score(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Fraction of argmax predictions matching the truth."""
    return float((proba.argmax(axis=1) == y_true).mean())


def per_class_pr(y_true: np.ndarray, proba: np.ndarray) -> dict[str, dict[int, float]]:
    """Precision and recall per class from argmax predictions."""
    pred = proba.argmax(axis=1)
    out: dict[str, dict[int, float]] = {"precision": {}, "recall": {}}
    for c in CLASSES:
        tp = int(((pred == c) & (y_true == c)).sum())
        out["precision"][c] = tp / max(int((pred == c).sum()), 1)
        out["recall"][c] = tp / max(int((y_true == c).sum()), 1)
    return out


def score_all(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    """All spec metrics for one (y_true, proba) pair."""
    pr = per_class_pr(y_true, proba)
    return {
        "log_loss": log_loss_score(y_true, proba),
        "brier": brier_score(y_true, proba),
        "accuracy": accuracy_score(y_true, proba),
        "draw_recall": pr["recall"][1],
    }


def always_home_proba(n: int) -> np.ndarray:
    """Constant [0.98, 0.01, 0.01] rows (see module docstring for the floor)."""
    e = ALWAYS_HOME_EPS
    return np.tile([1.0 - 2 * e, e, e], (n, 1))


def prior_proba(train: pd.DataFrame, n: int) -> np.ndarray:
    """Class priors from training seasons only, tiled to n rows."""
    dist = train["target"].value_counts(normalize=True).reindex(CLASSES).fillna(0)
    return np.tile(dist.to_numpy(), (n, 1))


def bookmaker_proba(test_matches: pd.DataFrame) -> np.ndarray:
    """De-overrounded Bet365 probabilities, H/D/A column order."""
    return test_matches[
        ["b365_home_prob", "b365_draw_prob", "b365_away_prob"]
    ].to_numpy()


def run_benchmarks(
    feats: pd.DataFrame, matches: pd.DataFrame
) -> pd.DataFrame:
    """One results row per benchmark per test season."""
    seasons = sorted(feats["season"].unique())
    book = matches.set_index("match_id")[
        ["b365_home_prob", "b365_draw_prob", "b365_away_prob"]
    ]
    rows = []
    for split in get_splits(seasons):
        test = feats[feats["season"] == split["test"]]
        train = feats[feats["season"].isin(split["train"])]
        y = test["target"].to_numpy()
        candidates = {
            "always_home": always_home_proba(len(test)),
            "class_priors": prior_proba(train, len(test)),
            "bookmaker": book.loc[test["match_id"]].to_numpy(),
        }
        for model, proba in candidates.items():
            metrics = score_all(y, proba)
            rows.append({
                "model": model, "feature_set": "none",
                "test_season": split["test"], **metrics, "seed": SEED,
            })
    return pd.DataFrame(rows)


def main() -> None:
    """Entry point for `make eval` (milestone 3: benchmarks only)."""
    feats = pd.read_parquet(FEATURES_PATH)
    matches = pd.read_parquet(MATCHES_PATH)
    fresh = run_benchmarks(feats, matches)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if RESULTS_PATH.exists():
        # Idempotent merge: replace benchmark rows, keep any model rows, so
        # `make all` (train before eval) never wipes trained results.
        existing = pd.read_csv(RESULTS_PATH)
        keep = existing[~existing["model"].isin(fresh["model"].unique())]
        results = pd.concat([keep, fresh], ignore_index=True)
    else:
        results = fresh
    results = results.sort_values(
        ["model", "feature_set", "test_season"]).reset_index(drop=True)
    results.to_csv(RESULTS_PATH, index=False)
    print(f"wrote {RESULTS_PATH} ({len(results)} rows)")
    print("-- mean log_loss / accuracy per benchmark --")
    for model, g in fresh.groupby("model"):
        print(f"{model}: log_loss={g['log_loss'].mean():.4f} "
              f"(std {g['log_loss'].std():.4f}), "
              f"accuracy={g['accuracy'].mean():.4f} "
              f"(std {g['accuracy'].std():.4f}), n={len(g)}")


if __name__ == "__main__":
    main()
