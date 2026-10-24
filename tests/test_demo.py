"""Tests for the live fixture demo and the unplayed-row support it needs."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from epl.data import AVG_COLS, B365_COLS, STAT_COLS, TARGET_MAP
from epl.demo import (
    TEMPLATE_PATH,
    combine_matches,
    infer_season,
    load_fixtures,
    predict_fixtures,
    summarize,
)
from epl.features import build_features
from epl.models import feature_sets

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


def test_load_fixtures_parses_schema(tmp_path: Path) -> None:
    path = tmp_path / "fixtures.csv"
    path.write_text(
        "date,home_team,away_team,home_goals,away_goals,result,"
        "b365_home,b365_draw,b365_away\n"
        "15/08/2026,Man City,Chelsea,,,,2.00,3.50,3.80\n"
        "2026-08-16,liverpool,Arsenal,1,2,,3.00,3.40,2.30\n"
    )
    fx = load_fixtures(path)
    assert len(fx) == 2
    assert list(fx["season"].unique()) == ["2026/27"]
    assert fx.iloc[0]["HomeTeam"] == "Manchester City"
    assert fx.iloc[0]["AwayTeam"] == "Chelsea"
    assert pd.isna(fx.iloc[0]["FTR"])
    assert fx.iloc[1]["FTR"] == "A"
    assert fx.iloc[1]["target"] == 2
    assert fx.iloc[0]["b365_home_prob"] > 0
    probs = fx[["b365_home_prob", "b365_draw_prob", "b365_away_prob"]].sum(axis=1)
    assert ((probs - 1.0).abs() < 1e-9).all()


def test_load_fixtures_default_season(tmp_path: Path) -> None:
    path = tmp_path / "fixtures.csv"
    path.write_text("date,home_team,away_team\n2026-08-15,Arsenal,Chelsea\n")
    fx = load_fixtures(path, default_season="2026/27")
    assert fx.iloc[0]["season"] == "2026/27"


def test_load_fixtures_requires_columns(tmp_path: Path) -> None:
    path = tmp_path / "fixtures.csv"
    path.write_text("date,home_team\n2026-08-15,Arsenal\n")
    with pytest.raises(ValueError, match="missing required"):
        load_fixtures(path)


def test_load_fixtures_rejects_bad_date(tmp_path: Path) -> None:
    path = tmp_path / "fixtures.csv"
    path.write_text("date,home_team,away_team\nnot-a-date,Arsenal,Chelsea\n")
    with pytest.raises(ValueError, match="invalid or missing date"):
        load_fixtures(path)


def test_template_loads() -> None:
    fx = load_fixtures(TEMPLATE_PATH)
    assert len(fx) == 3
    assert list(fx["season"].unique()) == ["2026/27"]


def test_infer_season() -> None:
    assert infer_season(pd.Timestamp("2026-08-15")) == "2026/27"
    assert infer_season(pd.Timestamp("2026-05-24")) == "2025/26"


def _direct_p1_fit(
    feats: pd.DataFrame, pred: pd.DataFrame, date: pd.Timestamp
) -> np.ndarray:
    cols = feature_sets()["P1"]
    train = feats[(feats["Date"] < date) & feats["target"].notna()]
    med = np.nanmedian(train[cols].to_numpy(dtype=float), axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    x = np.where(
        np.isnan(train[cols].to_numpy(dtype=float)), med,
        train[cols].to_numpy(dtype=float),
    )
    scaler = StandardScaler().fit(x)
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(scaler.transform(x), train["target"].to_numpy(dtype=int))
    return clf.predict_proba(scaler.transform(np.where(
        np.isnan(pred[cols].to_numpy(dtype=float)), med,
        pred[cols].to_numpy(dtype=float),
    )))


def test_prediction_matches_direct_fit(
    matches: pd.DataFrame, full_features: pd.DataFrame
) -> None:
    row = matches[matches["season"] == "2025/26"].iloc[[100]].reset_index(drop=True)
    preds = predict_fixtures(full_features, row, matches, "P1", 42)
    assert len(preds) == 1
    frow = full_features[full_features["match_id"] == row.iloc[0]["match_id"]]
    expected = _direct_p1_fit(full_features, frow, row.iloc[0]["Date"])[0]
    got = preds.iloc[0]
    assert np.allclose(
        [got["p_home"], got["p_draw"], got["p_away"]], expected, atol=1e-10
    )
    assert got["actual"] == row.iloc[0]["FTR"]
    assert got["correct"] == (got["pick"] == got["actual"])
    assert got["cutoff"] < got["date"]


def test_prediction_ignores_later_matches(
    matches: pd.DataFrame, full_features: pd.DataFrame
) -> None:
    row = matches[matches["season"] == "2025/26"].iloc[[100]].reset_index(drop=True)
    full = predict_fixtures(full_features, row, matches, "P1", 42).iloc[0]
    truncated = matches[matches["Date"] <= row.iloc[0]["Date"]]
    feats = build_features(truncated)
    short = predict_fixtures(feats, row, truncated, "P1", 42).iloc[0]
    assert np.allclose(
        [full["p_home"], full["p_draw"], full["p_away"]],
        [short["p_home"], short["p_draw"], short["p_away"]],
        atol=1e-10,
    )


def test_upcoming_fixture_prediction(matches: pd.DataFrame, tmp_path: Path) -> None:
    path = tmp_path / "fixtures.csv"
    path.write_text(
        "date,home_team,away_team\n2026-08-15,Arsenal,Chelsea\n"
    )
    fixtures = load_fixtures(path)
    combined = combine_matches(matches, fixtures)
    feats = build_features(combined)
    preds = predict_fixtures(feats, fixtures, combined, "P1", 42)
    assert len(preds) == 1
    row = preds.iloc[0]
    assert np.isclose(row["p_home"] + row["p_draw"] + row["p_away"], 1.0)
    assert pd.isna(row["actual"])
    assert row["cutoff"] < row["date"]
    assert row["train_rows"] == len(matches)
    assert summarize(preds)["n_upcoming"] == 1

