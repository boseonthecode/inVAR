"""No-leakage tests: features for match M use only strictly earlier matches.

 Strategy (per docs/featuresandmodels.md): recompute features after deleting
 later matches and assert earlier rows are unchanged; plus a same-date
 isolation check (flipping one match's result must not change other matches
 kicked off the same day) and a debut check (first-ever matches are all-NaN
 rolling with init Elo).
"""

from pathlib import Path

import pandas as pd
import pytest

from epl.features import build_features

MATCHES = Path("data/processed/matches.parquet")
CUTOFF = "2018-01-01"

FEATURE_COLS = [
    "elo_home", "elo_away", "elo_diff",
    "home_ppg_5", "away_ppg_5", "ppg_diff_5",
    "home_gs_5", "away_gs_5", "gs_diff_5",
    "home_gc_5", "away_gc_5", "gc_diff_5",
    "home_home_ppg_5", "away_away_ppg_5", "split_diff_5",
    "home_shots_5", "away_shots_5", "shots_diff_5",
    "home_sot_5", "away_sot_5", "sot_diff_5",
    "home_acc_5", "away_acc_5", "acc_diff_5",
    "home_corners_5", "away_corners_5", "corners_diff_5",
    "home_ppg_10", "away_ppg_10", "ppg_diff_10",
    "home_rest", "away_rest", "rest_diff",
    "h2h_home_share5", "h2h_count",
    "home_prev_pos", "away_prev_pos", "prev_pos_diff",
]


@pytest.fixture(scope="module")
def matches() -> pd.DataFrame:
    if not MATCHES.exists():
        pytest.skip("matches.parquet missing; run `make data` first")
    return pd.read_parquet(MATCHES)


@pytest.fixture(scope="module")
def full_features(matches: pd.DataFrame) -> pd.DataFrame:
    return build_features(matches)


def assert_frames_equal(a: pd.DataFrame, b: pd.DataFrame) -> None:
    a = a.sort_values("match_id").reset_index(drop=True)
    b = b.sort_values("match_id").reset_index(drop=True)
    assert list(a["match_id"]) == list(b["match_id"])
    for col in FEATURE_COLS:
        va, vb = a[col].to_numpy(), b[col].to_numpy()
        # NaN-aware equality
        assert ((va == vb) | (pd.isna(va) & pd.isna(vb))).all(), f"col {col} changed"


def test_truncation_invariance(matches: pd.DataFrame, full_features: pd.DataFrame) -> None:
    """Deleting all matches on/after CUTOFF must not change earlier features."""
    early_ids = full_features.loc[full_features["Date"] < CUTOFF, "match_id"]
    truncated = matches[matches["Date"] < CUTOFF].reset_index(drop=True)
    rebuilt = build_features(truncated)
    assert len(rebuilt) == len(early_ids)
    assert_frames_equal(
        full_features[full_features["match_id"].isin(early_ids)][["match_id", *FEATURE_COLS]],
        rebuilt[["match_id", *FEATURE_COLS]],
    )


def test_same_date_isolation(matches: pd.DataFrame, full_features: pd.DataFrame) -> None:
    """Flipping one match's result must not change same-date matches' features."""
    counts = matches.groupby("Date").size()
    date = counts[counts >= 2].index[0]
    day = matches[matches["Date"] == date]
    victim_id = day["match_id"].iloc[0]
    other_ids = day["match_id"].iloc[1:].tolist()
    tampered = matches.copy()
    old = tampered.loc[tampered["match_id"] == victim_id, "FTR"].iloc[0]
    new = {"H": "A", "A": "H", "D": "H"}[old]
    tampered.loc[tampered["match_id"] == victim_id, "FTR"] = new
    rebuilt = build_features(tampered)
    assert_frames_equal(
        full_features[full_features["match_id"].isin(other_ids)][["match_id", *FEATURE_COLS]],
        rebuilt[rebuilt["match_id"].isin(other_ids)][["match_id", *FEATURE_COLS]],
    )


def test_debut_match_is_clean(full_features: pd.DataFrame) -> None:
    """A team's first-ever match: NaN rolling/rest, init Elo 1500, NaN prev pos."""
    first_date = full_features["Date"].min()
    openers = full_features[full_features["Date"] == first_date]
    assert len(openers) > 0
    assert (openers[["elo_home", "elo_away"]].to_numpy() == 1500.0).all()
    rolling = [c for c in FEATURE_COLS if c.startswith(("home_ppg", "away_ppg", "home_gs"))]
    assert openers[rolling].isna().all().all()
    assert openers[["home_rest", "away_rest"]].isna().all().all()
    assert openers[["home_prev_pos", "away_prev_pos"]].isna().all().all()
