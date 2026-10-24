"""Live demo: predict current-season fixtures supplied as a CSV.

The canonical dataset ends at 2025/26. This module accepts a separate fixture
file (default ``data/fixtures.csv``) for the season being played, combines it
with the canonical history in memory, and predicts each fixture using only
matches dated strictly before kickoff. Completed rows in the fixture file are
fed back into Elo/form for later fixtures, but are never used for their own
prediction.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from epl.data import (
    AVG_COLS,
    B365_COLS,
    STAT_COLS,
    TARGET_MAP,
    canonical_name,
    deoverround,
)
from epl.evaluate import SEED

DEFAULT_FIXTURES = Path("data/fixtures.csv")
DEFAULT_HISTORY = Path("data/processed/matches.parquet")
TEMPLATE_PATH = Path("data/fixtures.example.csv")

REQUIRED_COLUMNS = ["date", "home_team", "away_team"]
STAT_MAP = {
    "home_shots": "HS",
    "away_shots": "AS",
    "home_sot": "HST",
    "away_sot": "AST",
    "home_corners": "HC",
    "away_corners": "AC",
}
ODDS_PROB_COLS = [
    "b365_home_prob",
    "b365_draw_prob",
    "b365_away_prob",
    "avg_home_prob",
    "avg_draw_prob",
    "avg_away_prob",
]
MATCH_COLUMNS = [
    "Date",
    "season",
    "match_id",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR",
    "target",
    *STAT_COLS,
    *B365_COLS,
    *AVG_COLS,
    *ODDS_PROB_COLS,
]


def parse_fixture_dates(series: pd.Series) -> pd.Series:
    """Parse ISO or day-first dates; raise with row numbers on bad values."""
    text = series.astype("string").str.strip()
    parsed = pd.to_datetime(text, format="%Y-%m-%d", errors="coerce")
    for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y/%m/%d"):
        mask = parsed.isna() & text.notna()
        if not mask.any():
            break
        parsed.loc[mask] = pd.to_datetime(text[mask], format=fmt, errors="coerce")
    bad = parsed.isna()
    if bad.any():
        rows = [str(i + 2) for i in series.index[bad]]
        raise ValueError(f"invalid or missing date in CSV row(s): {', '.join(rows)}")
    return parsed


def infer_season(date: pd.Timestamp) -> str:
    """Infer ``YYYY/YY`` from a date (July starts a new season)."""
    year = date.year if date.month >= 7 else date.year - 1
    return f"{year}/{(year + 1) % 100:02d}"


def _optional_numeric(df: pd.DataFrame, source: str) -> pd.Series:
    if source not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[source], errors="coerce")


def load_fixtures(path: str | Path, default_season: str | None = None) -> pd.DataFrame:
    """Load and normalise a fixture CSV to the canonical match schema."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"fixture file not found: {path}\n"
            f"Copy {TEMPLATE_PATH} to {DEFAULT_FIXTURES} and fill it in."
        )
    raw = pd.read_csv(path, encoding="utf-8-sig")
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"fixture CSV missing required column(s): {', '.join(missing)}")
    if raw.empty:
        raise ValueError(f"fixture CSV has no rows: {path}")

    df = raw.copy()
    df["Date"] = parse_fixture_dates(df["date"])
    df["HomeTeam"] = df["home_team"].astype(str).str.strip().map(canonical_name)
    df["AwayTeam"] = df["away_team"].astype(str).str.strip().map(canonical_name)
    if (df["HomeTeam"] == "").any() or (df["AwayTeam"] == "").any():
        raise ValueError("fixture CSV has an empty team name")
    same = df["HomeTeam"] == df["AwayTeam"]
    if same.any():
        rows = [str(i + 2) for i in df.index[same]]
        raise ValueError(f"home_team equals away_team in CSV row(s): {', '.join(rows)}")

    inferred = df["Date"].map(infer_season)
    if "season" in df.columns:
        explicit = df["season"].astype("string").str.strip()
        explicit = explicit.where(explicit.notna() & (explicit != ""), pd.NA)
    else:
        explicit = pd.Series(pd.NA, index=df.index, dtype="string")
    if default_season:
        explicit = explicit.fillna(default_season)
    season = explicit.fillna(inferred)

    df["FTHG"] = _optional_numeric(df, "home_goals")
    df["FTAG"] = _optional_numeric(df, "away_goals")
    have_goals = df["FTHG"].notna() & df["FTAG"].notna()
    derived = pd.Series(pd.NA, index=df.index, dtype="string")
    derived.loc[have_goals & (df["FTHG"] > df["FTAG"])] = "H"
    derived.loc[have_goals & (df["FTHG"] < df["FTAG"])] = "A"
    derived.loc[have_goals & (df["FTHG"] == df["FTAG"])] = "D"

    if "result" in df.columns:
        explicit_result = df["result"].astype("string").str.strip().str.upper()
        explicit_result = explicit_result.where(
            explicit_result.notna() & (explicit_result != ""), pd.NA
        )
        invalid = explicit_result.notna() & ~explicit_result.isin(["H", "D", "A"])
        if invalid.any():
            rows = [str(i + 2) for i in df.index[invalid]]
            raise ValueError(f"result must be H, D or A in CSV row(s): {', '.join(rows)}")
    else:
        explicit_result = pd.Series(pd.NA, index=df.index, dtype="string")
    conflict = (
        explicit_result.notna() & derived.notna() & (explicit_result != derived)
    )
    if conflict.any():
        warnings.warn(
            "result disagrees with goals; using goals for the affected row(s)",
            stacklevel=2,
        )
    result = derived.where(derived.notna(), explicit_result)

    date_text = df["Date"].dt.strftime("%Y-%m-%d")
    df["match_id"] = (
        season.astype(str).str.replace("/", "-", regex=False)
        + "_" + date_text
        + "_" + df["HomeTeam"]
        + "_" + df["AwayTeam"]
    )

    for src, dst in STAT_MAP.items():
        df[dst] = _optional_numeric(df, src)
    for col in STAT_COLS:
        if col not in df.columns:
            df[col] = np.nan
    for col in AVG_COLS:
        df[col] = np.nan
    df["B365H"] = _optional_numeric(df, "b365_home")
    df["B365D"] = _optional_numeric(df, "b365_draw")
    df["B365A"] = _optional_numeric(df, "b365_away")

    for col, out in zip(
        B365_COLS,
        ["b365_home_prob", "b365_draw_prob", "b365_away_prob"],
    ):
        df[out] = np.nan
    valid = df[B365_COLS].notna().all(axis=1) & (df[B365_COLS] > 0).all(axis=1)
    if valid.any():
        probs = deoverround(df.loc[valid, B365_COLS])
        for i, out in enumerate(
            ["b365_home_prob", "b365_draw_prob", "b365_away_prob"]
        ):
            df.loc[valid, out] = probs.iloc[:, i].to_numpy()
    for out in ["avg_home_prob", "avg_draw_prob", "avg_away_prob"]:
        df[out] = np.nan

    out = pd.DataFrame({
        "Date": df["Date"],
        "season": season.astype(str),
        "match_id": df["match_id"],
        "HomeTeam": df["HomeTeam"],
        "AwayTeam": df["AwayTeam"],
        "FTHG": df["FTHG"],
        "FTAG": df["FTAG"],
        "FTR": result.astype(object).where(result.notna(), np.nan),
        "target": result.astype(object).where(result.notna(), np.nan).map(TARGET_MAP),
    })
    for col in [*STAT_COLS, *B365_COLS, *AVG_COLS, *ODDS_PROB_COLS]:
        out[col] = df[col]
    out = out[MATCH_COLUMNS].sort_values("Date").reset_index(drop=True)

    dups = out.duplicated(subset=["Date", "HomeTeam", "AwayTeam"], keep="first")
    if dups.any():
        warnings.warn(
            f"dropped {int(dups.sum())} duplicate fixture row(s)",
            stacklevel=2,
        )
        out = out.loc[~dups].reset_index(drop=True)
    return out


def load_history(path: str | Path = DEFAULT_HISTORY) -> pd.DataFrame:
    """Load the canonical processed matches, with a helpful error if missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"history file not found: {path}\nRun `make data` first."
        )
    return pd.read_parquet(path)


def combine_matches(history: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    """Concatenate history and fixtures, dropping duplicate matches."""
    hist = history.copy()
    fix = fixtures.copy()
    for col in MATCH_COLUMNS:
        if col not in hist.columns:
            hist[col] = np.nan
        if col not in fix.columns:
            fix[col] = np.nan
    duplicate_ids = set(hist["match_id"]) & set(fix["match_id"])
    if duplicate_ids:
        warnings.warn(
            f"{len(duplicate_ids)} fixture(s) already exist in history; "
            "canonical rows take precedence",
            stacklevel=2,
        )
    combined = pd.concat(
        [hist[MATCH_COLUMNS], fix[MATCH_COLUMNS]], ignore_index=True
    )
    combined = combined.drop_duplicates(subset=["match_id"], keep="first")
    combined = combined.drop_duplicates(
        subset=["Date", "HomeTeam", "AwayTeam"], keep="first"
    )
    return combined.sort_values("Date").reset_index(drop=True)


def unknown_teams(fixtures: pd.DataFrame, history: pd.DataFrame) -> list[str]:
    """Teams in the fixture file with no canonical history."""
    known = set(history["HomeTeam"]) | set(history["AwayTeam"])
    current = set(fixtures["HomeTeam"]) | set(fixtures["AwayTeam"])
    return sorted(current - known)


def print_template() -> None:
    """Print the example fixture CSV to stdout."""
    print(TEMPLATE_PATH.read_text())


def main() -> None:
    """Entry point. Prediction/rendering is added in the next commit."""
    parser = argparse.ArgumentParser(description="EPL live fixture demo")
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--season", default=None)
    parser.add_argument("--template", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.template:
        print_template()
        return
    fixtures = load_fixtures(args.fixtures, args.season)
    history = load_history(args.history)
    missing = unknown_teams(fixtures, history)
    if missing:
        warnings.warn(f"unknown teams (Elo starts at 1500): {', '.join(missing)}")
    print(f"loaded {len(fixtures)} fixture(s) from {args.fixtures}")
    print(
        fixtures[["Date", "season", "HomeTeam", "AwayTeam", "FTR"]]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
