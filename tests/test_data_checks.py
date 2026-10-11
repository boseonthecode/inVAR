"""Data checks from docs/dataspec.md (run via pytest or `make data`)."""

from pathlib import Path

import pandas as pd
import pytest

MATCHES = Path("data/processed/matches.parquet")


@pytest.fixture
def df() -> pd.DataFrame:
    if not MATCHES.exists():
        pytest.skip("matches.parquet missing; run `make data` first")
    return pd.read_parquet(MATCHES)


def test_season_counts(df: pd.DataFrame) -> None:
    counts = df.groupby("season").size()
    assert (counts == 380).all(), counts[counts != 380].to_dict()


def test_no_duplicates(df: pd.DataFrame) -> None:
    assert df.duplicated(subset=["Date", "HomeTeam", "AwayTeam"]).sum() == 0
    assert df["match_id"].is_unique


def test_target_encoding(df: pd.DataFrame) -> None:
    assert set(df["FTR"].unique()) <= {"H", "D", "A"}
    assert ((df["FTR"] == "H").astype(int) * 0 + df["target"]).isin([0, 1, 2]).all()
    mapping = df.drop_duplicates("FTR").set_index("FTR")["target"].to_dict()
    assert mapping == {"H": 0, "D": 1, "A": 2}


def test_sorted_by_date(df: pd.DataFrame) -> None:
    assert df["Date"].is_monotonic_increasing


def test_no_leaky_columns(df: pd.DataFrame) -> None:
    leaky = [c for c in df.columns if "HT" in c or "C>" in c or "CA" in c or "xG" in c
             or c in ("HTHG", "HTAG", "HTR")]
    assert leaky == [], leaky


def test_canonical_names(df: pd.DataFrame) -> None:
    raw_aliases = {"Man United", "Man City", "Spurs", "QPR", "Nott'm Forest"}
    teams = set(df["HomeTeam"]) | set(df["AwayTeam"])
    assert not (raw_aliases & teams), raw_aliases & teams


def test_probabilities(df: pd.DataFrame) -> None:
    b365 = df[["b365_home_prob", "b365_draw_prob", "b365_away_prob"]]
    assert b365.notna().all().all(), "every row must have de-overrounded B365 probs"
    sums = b365.sum(axis=1)
    assert ((sums - 1.0).abs() < 1e-9).all()
