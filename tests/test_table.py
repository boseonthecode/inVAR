"""Tests for previous-season table features and the table-higher-wins rule."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from epl.evaluate import table_higher_proba
from epl.features import build_features

MATCHES = Path("data/processed/matches.parquet")
POINTS = {"H": (3, 0), "D": (1, 1), "A": (0, 3)}
FIRST_TEST_SEASON = "2014/15"


@pytest.fixture(scope="module")
def matches() -> pd.DataFrame:
    if not MATCHES.exists():
        pytest.skip("matches.parquet missing; run `make data` first")
    return pd.read_parquet(MATCHES)


@pytest.fixture(scope="module")
def full_features(matches: pd.DataFrame) -> pd.DataFrame:
    return build_features(matches)


def _final_table(matches: pd.DataFrame, season: str) -> pd.DataFrame:
    """Independent final table for one season, sorted like the feature builder."""
    pts: dict[str, int] = {}
    gd: dict[str, int] = {}
    gf: dict[str, int] = {}
    for row in matches[matches["season"] == season].itertuples():
        hp, ap = POINTS[row.FTR]
        pts[row.HomeTeam] = pts.get(row.HomeTeam, 0) + hp
        pts[row.AwayTeam] = pts.get(row.AwayTeam, 0) + ap
        gd[row.HomeTeam] = gd.get(row.HomeTeam, 0) + row.FTHG - row.FTAG
        gd[row.AwayTeam] = gd.get(row.AwayTeam, 0) + row.FTAG - row.FTHG
        gf[row.HomeTeam] = gf.get(row.HomeTeam, 0) + row.FTHG
        gf[row.AwayTeam] = gf.get(row.AwayTeam, 0) + row.FTAG
    table = pd.DataFrame({"team": list(pts), "pts": [pts[t] for t in pts],
                          "gd": [gd[t] for t in pts], "gf": [gf[t] for t in pts]})
    table = table.sort_values(["pts", "gd", "gf"], ascending=False)
    table = table.reset_index(drop=True)
    table["rank"] = table.index + 1
    return table


def test_prev_table_matches_prior_season(
    matches: pd.DataFrame, full_features: pd.DataFrame
) -> None:
    """Non-promoted teams carry their true prior-season points/rank/GD."""
    season = "2019/20"
    prior = _final_table(matches, "2018/19").set_index("team")
    rows = full_features[full_features["season"] == season]
    checked = 0
    for row in rows.itertuples():
        if row.HomeTeam in prior.index and row.AwayTeam in prior.index:
            h, a = prior.loc[row.HomeTeam], prior.loc[row.AwayTeam]
            assert row.home_prev_pos == h["rank"]
            assert row.home_prev_pts == h["pts"]
            assert row.away_prev_pos == a["rank"]
            assert row.away_prev_pts == a["pts"]
            assert row.prev_pts_diff == h["pts"] - a["pts"]
            assert row.prev_pos_gap == a["rank"] - h["rank"]
            checked += 1
    assert checked > 0


def test_promoted_teams_get_proxy(
    matches: pd.DataFrame, full_features: pd.DataFrame
) -> None:
    """Promoted sides get ranks 21-23 and the prior 20th-placed totals."""
    seasons = sorted(matches["season"].unique())
    for i, season in enumerate(seasons):
        if season < FIRST_TEST_SEASON:
            continue
        now = set(matches.loc[matches["season"] == season, "HomeTeam"]) | set(
            matches.loc[matches["season"] == season, "AwayTeam"]
        )
        before = set(matches.loc[matches["season"] == seasons[i - 1], "HomeTeam"]) | set(
            matches.loc[matches["season"] == seasons[i - 1], "AwayTeam"]
        )
        promoted = sorted(now - before)
        if not promoted:
            continue
        team = promoted[0]
        prior = _final_table(matches, seasons[i - 1]).set_index("team")
        last = prior.iloc[-1]
        row = full_features[
            (full_features["season"] == season)
            & ((full_features["HomeTeam"] == team) | (full_features["AwayTeam"] == team))
        ].iloc[0]
        rank = row.home_prev_pos if row.HomeTeam == team else row.away_prev_pos
        pts = row.home_prev_pts if row.HomeTeam == team else row.away_prev_pts
        assert 21 <= rank <= 23
        assert pts == last["pts"]
        return
    pytest.skip("no promoted teams found in test seasons")


def test_rule_known_cases() -> None:
    """Higher rank wins; ties and missing ranks fall back to home."""
    df = pd.DataFrame({
        "home_prev_pos": [1.0, 20.0, 10.0, np.nan],
        "away_prev_pos": [20.0, 1.0, 11.0, 5.0],
    })
    proba = table_higher_proba(df)
    assert proba.argmax(axis=1).tolist() == [0, 2, 0, 0]
    assert np.allclose(proba.sum(axis=1), 1.0)
    # Never predicts a draw.
    assert (proba[:, 1] == 0.01).all()


def test_rule_matches_raw_rank_picks(
    matches: pd.DataFrame, full_features: pd.DataFrame
) -> None:
    """Rule agrees with an independent prior-table calculation."""
    season = "2022/23"
    prior = _final_table(matches, "2021/22").set_index("team")
    rows = full_features[full_features["season"] == season]
    mask = rows["HomeTeam"].isin(prior.index) & rows["AwayTeam"].isin(prior.index)
    subset = rows[mask]
    pred = table_higher_proba(subset).argmax(axis=1)
    expected = np.where(
        subset["AwayTeam"].map(prior["rank"]).to_numpy()
        < subset["HomeTeam"].map(prior["rank"]).to_numpy(),
        2, 0,
    )
    assert (pred == expected).all()
    assert len(subset) > 0
