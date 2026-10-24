"""Tests for the live fixture demo and the unplayed-row support it needs."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from epl.data import AVG_COLS, B365_COLS, STAT_COLS, TARGET_MAP
from epl.features import build_features

MATCHES = Path("data/processed/matches.parquet")

KEY_COLS = {"match_id", "Date", "season", "HomeTeam", "AwayTeam", "FTR", "target"}
ODDS_PROB_COLS = [
    "b365_home_prob", "b365_draw_prob", "b365_away_prob",
    "avg_home_prob", "avg_draw_prob", "avg_away_prob",
]


@pytest.fixture(scope="module")
def matches() -> pd.DataFrame:
    if not MATCHES.exists():
        pytest.skip("matches.parquet missing; run `make data` first")
    return pd.read_parquet(MATCHES)


@pytest.fixture(scope="module")
def full_features(matches: pd.DataFrame) -> pd.DataFrame:
    return build_features(matches)


def _fixture_row(
    base: pd.Series,
    date: str,
    home: str,
    away: str,
    season: str = "2026/27",
    ftr: str | None = None,
    fthg: float | None = None,
    ftag: float | None = None,
) -> pd.Series:
    row = base.copy()
    row["Date"] = pd.Timestamp(date)
    row["season"] = season
    row["HomeTeam"] = home
    row["AwayTeam"] = away
    row["match_id"] = f"{season.replace('/', '-')}_{date}_{home}_{away}"
    row["FTR"] = ftr if ftr is not None else np.nan
    row["target"] = TARGET_MAP[ftr] if ftr is not None else np.nan
    row["FTHG"] = fthg if fthg is not None else np.nan
    row["FTAG"] = ftag if ftag is not None else np.nan
    for col in [*STAT_COLS, *B365_COLS, *AVG_COLS, *ODDS_PROB_COLS]:
        row[col] = np.nan
    return row


def _combined(matches: pd.DataFrame, rows: list[pd.Series]) -> pd.DataFrame:
    return pd.concat([matches, pd.DataFrame(rows)], ignore_index=True)


def _assert_feature_rows_equal(a: pd.DataFrame, b: pd.DataFrame) -> None:
    a = a.sort_values("match_id").reset_index(drop=True)
    b = b.sort_values("match_id").reset_index(drop=True)
    assert list(a["match_id"]) == list(b["match_id"])
    cols = [c for c in a.columns if c not in KEY_COLS]
    for col in cols:
        va, vb = a[col].to_numpy(), b[col].to_numpy()
        assert ((va == vb) | (pd.isna(va) & pd.isna(vb))).all(), f"col {col} changed"


def test_unplayed_row_does_not_change_past_features(
    matches: pd.DataFrame, full_features: pd.DataFrame
) -> None:
    base = matches.iloc[-1]
    upcoming = _fixture_row(
        base, "2026-08-15", "Brighton", "Manchester United"
    )
    combined = build_features(_combined(matches, [upcoming]))
    original = combined[combined["match_id"].isin(full_features["match_id"])]
    _assert_feature_rows_equal(
        full_features[["match_id", *_feature_cols(full_features)]],
        original[["match_id", *_feature_cols(full_features)]],
    )


def test_unplayed_row_does_not_update_later_history(matches: pd.DataFrame) -> None:
    base = matches.iloc[-1]
    first = _fixture_row(base, "2026-08-15", "Brighton", "Manchester United")
    later = _fixture_row(base, "2026-08-20", "Brighton", "Fulham")
    feats = build_features(_combined(matches, [first, later]))
    first_home = feats.loc[feats["match_id"] == first["match_id"]].iloc[0]
    later_home = feats.loc[feats["match_id"] == later["match_id"]].iloc[0]
    assert later_home["elo_home"] == pytest.approx(first_home["elo_home"])
    assert pd.isna(later_home["home_rest"])
    assert later_home["h2h_count"] == first_home["h2h_count"]


def test_completed_row_updates_later_history(matches: pd.DataFrame) -> None:
    base = matches.iloc[-1]
    played = _fixture_row(
        base, "2026-08-15", "Brighton", "Manchester United",
        ftr="H", fthg=2.0, ftag=0.0,
    )
    later = _fixture_row(base, "2026-08-20", "Brighton", "Fulham")
    with_played = build_features(_combined(matches, [played, later]))
    without_played = build_features(_combined(matches, [later]))
    a = with_played.loc[with_played["match_id"] == later["match_id"]].iloc[0]
    b = without_played.loc[without_played["match_id"] == later["match_id"]].iloc[0]
    assert a["elo_home"] > b["elo_home"]
    assert a["home_rest"] == pytest.approx(5.0)
    assert pd.isna(b["home_rest"])


def test_incomplete_new_season_seeds_newcomers_at_init(matches: pd.DataFrame) -> None:
    base = matches.iloc[-1]
    row = _fixture_row(
        base, "2026-08-15", "Newly Promoted FC", "Brighton"
    )
    feats = build_features(_combined(matches, [row]))
    out = feats.loc[feats["match_id"] == row["match_id"]].iloc[0]
    assert out["elo_home"] == pytest.approx(1500.0)
    assert out["elo_away"] > 0.0


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in KEY_COLS]
