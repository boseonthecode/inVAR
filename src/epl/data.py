"""Download and clean EPL match data from football-data.co.uk.

Single source: one E0 CSV per season, 2009/10 through the latest completed
season (currently 2025/26; 2026/27 is in progress and excluded).

Verified live 2026-10-11:
- URL pattern https://www.football-data.co.uk/mmz4281/{YYZZ}/E0.csv
  with YYZZ in 0910..2526.
- Columns include Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR, HS, AS,
  HST, AST, HC, AC, HF, AF, HY, AY, HR, AR, B365H/D/A.
  AvgH/D/A present from ~2010s onward; absent in 2009/10 (kept as NaN).

Usage: python -m epl.data
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

BASE_URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"

# (file code, season label). 2627 excluded: in progress (~50 rows on 2026-10-11).
SEASONS: list[tuple[str, str]] = [
    ("0910", "2009/10"),
    ("1011", "2010/11"),
    ("1112", "2011/12"),
    ("1213", "2012/13"),
    ("1314", "2013/14"),
    ("1415", "2014/15"),
    ("1516", "2015/16"),
    ("1617", "2016/17"),
    ("1718", "2017/18"),
    ("1819", "2018/19"),
    ("1920", "2019/20"),
    ("2021", "2020/21"),
    ("2122", "2021/22"),
    ("2223", "2022/23"),
    ("2324", "2023/24"),
    ("2425", "2024/25"),
    ("2526", "2025/26"),
]

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
MATCHES_PATH = PROCESSED_DIR / "matches.parquet"
DATA_CARD_PATH = PROCESSED_DIR / "data_card.md"

# Short football-data.co.uk names -> canonical full names.
# The source is self-consistent (same short forms every season), so this map
# is mostly a safety net plus documentation. Observed across 0910..2526.
TEAM_ALIASES: dict[str, str] = {
    "Man United": "Manchester United",
    "Man City": "Manchester City",
    "Tottenham": "Tottenham Hotspur",
    "Newcastle": "Newcastle United",
    "West Ham": "West Ham United",
    "Wolves": "Wolverhampton Wanderers",
    "West Brom": "West Bromwich Albion",
    "QPR": "Queens Park Rangers",
    "Nott'm Forest": "Nottingham Forest",
    "Sheffield United": "Sheffield United",
    "West Bromwich Albion": "West Bromwich Albion",
}

STAT_COLS = ["HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR"]
B365_COLS = ["B365H", "B365D", "B365A"]
AVG_COLS = ["AvgH", "AvgD", "AvgA"]
TARGET_MAP = {"H": 0, "D": 1, "A": 2}


def canonical_name(name: str) -> str:
    """Map a raw team name to its canonical form."""
    return TEAM_ALIASES.get(name.strip(), name.strip())


def download_season(code: str, raw_dir: Path = RAW_DIR) -> Path:
    """Download one season CSV into raw_dir (skip if already present)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / f"E0_{code}.csv"
    if dest.exists():
        return dest
    url = BASE_URL.format(code=code)
    req = urllib.request.Request(url, headers={"User-Agent": "inVAR-epl/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as f:
        f.write(resp.read())
    return dest


def parse_dates(series: pd.Series) -> pd.Series:
    """Parse dd/mm/yyyy or dd/mm/yy explicitly; never rely on file order."""
    parsed = pd.to_datetime(series, format="%d/%m/%Y", errors="coerce")
    mask = parsed.isna()
    parsed.loc[mask] = pd.to_datetime(series[mask], format="%d/%m/%y", errors="coerce")
    return parsed


def deoverround(odds: pd.DataFrame) -> pd.DataFrame:
    """Convert decimal odds to implied probabilities with overround removed."""
    inv = 1.0 / odds
    return inv.div(inv.sum(axis=1), axis=0)


def clean_season(df: pd.DataFrame, season: str) -> pd.DataFrame:
    """Clean one raw season frame per docs/dataspec.md."""
    df = df.copy()
    df["Date"] = parse_dates(df["Date"].astype(str).str.strip())
    df = df.dropna(subset=["Date"])
    df["FTR"] = df["FTR"].astype(str).str.strip()
    df = df[df["FTR"].isin(TARGET_MAP)]
    df["HomeTeam"] = df["HomeTeam"].astype(str).map(canonical_name)
    df["AwayTeam"] = df["AwayTeam"].astype(str).map(canonical_name)
    df["season"] = season
    for col in ["FTHG", "FTAG", *STAT_COLS, *B365_COLS, *AVG_COLS]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = float("nan")
    df["target"] = df["FTR"].map(TARGET_MAP).astype(int)
    b365 = df[B365_COLS]
    valid = b365.notna().all(axis=1) & (b365 > 0).all(axis=1)
    for i, out in enumerate(["b365_home_prob", "b365_draw_prob", "b365_away_prob"]):
        df[out] = float("nan")
        df.loc[valid, out] = deoverround(b365[valid]).iloc[:, i].to_numpy()
    avg = df[AVG_COLS]
    valid_avg = avg.notna().all(axis=1) & (avg > 0).all(axis=1)
    for i, out in enumerate(["avg_home_prob", "avg_draw_prob", "avg_away_prob"]):
        df[out] = float("nan")
        df.loc[valid_avg, out] = deoverround(avg[valid_avg]).iloc[:, i].to_numpy()
    df["match_id"] = (
        df["season"].str.replace("/", "-")
        + "_"
        + df["Date"].dt.strftime("%Y-%m-%d")
        + "_"
        + df["HomeTeam"]
        + "_"
        + df["AwayTeam"]
    )
    # Keep only the spec columns + derived fields. In particular this drops
    # half-time results (HTHG/HTAG/HTR), closing odds (*CH/*CD/*CA), over/under
    # and Asian-handicap columns, and xG: all are in-match or post-kickoff
    # information that must never reach the feature builder.
    keep = [
        "Date", "season", "match_id", "HomeTeam", "AwayTeam",
        "FTHG", "FTAG", "FTR", "target",
        *STAT_COLS, *B365_COLS, *AVG_COLS,
        "b365_home_prob", "b365_draw_prob", "b365_away_prob",
        "avg_home_prob", "avg_draw_prob", "avg_away_prob",
    ]
    return df.sort_values("Date").reset_index(drop=True)[keep]


def load_all(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Download (if needed) and clean every season; return merged frame."""
    frames = []
    for code, season in SEASONS:
        path = download_season(code, raw_dir)
        raw = pd.read_csv(path)
        frames.append(clean_season(raw, season))
    merged = pd.concat(frames, ignore_index=True)
    return merged.sort_values(["Date", "season"]).reset_index(drop=True)


def run_checks(df: pd.DataFrame) -> None:
    """Run the four spec checks; print results; raise on hard failures."""
    print("== data checks ==")
    counts = df.groupby("season").size()
    print("-- rows per season --")
    for season, n in counts.items():
        flag = "" if n == 380 else "  <-- EXPECTED 380"
        print(f"{season}: {n}{flag}")
    bad_counts = counts[counts != 380]
    if len(bad_counts):
        raise ValueError(f"Seasons without 380 matches: {bad_counts.to_dict()}")
    dups = df.duplicated(subset=["Date", "HomeTeam", "AwayTeam"]).sum()
    print(f"duplicate (date, home, away) rows: {dups}")
    if dups:
        raise ValueError(f"{dups} duplicate rows")
    balance = df["FTR"].value_counts(normalize=True).reindex(["H", "D", "A"])
    print("-- class balance (expect ~.46 / .25 / .29) --")
    for k in ["H", "D", "A"]:
        print(f"{k}: {balance[k]:.3f} (n={int((df['FTR'] == k).sum())})")
    if not (0.42 <= balance["H"] <= 0.50 and 0.21 <= balance["D"] <= 0.29 and 0.25 <= balance["A"] <= 0.33):
        print("WARNING: class balance outside expected bands")
    seasons_per_team = df.melt(
        id_vars="season", value_vars=["HomeTeam", "AwayTeam"], value_name="team"
    ).drop_duplicates()
    n_seasons = seasons_per_team.groupby("team")["season"].nunique()
    single = n_seasons[n_seasons < 2]
    print(f"teams appearing in <2 seasons: {len(single)}")
    if len(single):
        # Genuine one-season clubs (e.g. Blackpool 2010/11, Luton 2023/24) exist,
        # so this is a warning, not a failure. Each should have 38 appearances
        # (19 home + 19 away); far fewer would suggest a name-alias typo.
        counts = df.melt(value_vars=["HomeTeam", "AwayTeam"], value_name="team")[
            "team"
        ].value_counts()
        for team in single.index:
            print(f"  WARNING single-season team: {team} "
                  f"({int(counts[team])} appearances)")
        if (counts[single.index] < 38).any():
            raise ValueError("team with <38 appearances: likely name-alias typo")
    if df["match_id"].duplicated().any():
        raise ValueError("duplicate match_id")
    if not df["Date"].is_monotonic_increasing:
        raise ValueError("frame not sorted by date")
    print(f"total rows: {len(df)}, seasons: {df['season'].nunique()}, "
          f"date range: {df['Date'].min().date()}..{df['Date'].max().date()}")
    print("all checks passed")


def write_data_card(df: pd.DataFrame, path: Path = DATA_CARD_PATH) -> None:
    """Write row counts, date ranges, balance and the team-name map."""
    lines = ["# Data card", ""]
    lines.append(f"Seasons: {df['season'].nunique()}, rows: {len(df)}")
    lines.append("")
    lines.append("| season | rows | first | last | H | D | A |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for season, g in df.groupby("season", sort=True):
        bal = (g["FTR"].value_counts(normalize=True).reindex(["H", "D", "A"]).fillna(0))
        lines.append(
            f"| {season} | {len(g)} | {g['Date'].min().date()} | {g['Date'].max().date()} "
            f"| {bal['H']:.3f} | {bal['D']:.3f} | {bal['A']:.3f} |"
        )
    lines += ["", "## Team-name map (raw -> canonical)", ""]
    for raw, canon in sorted(TEAM_ALIASES.items()):
        lines.append(f"- {raw} -> {canon}")
    lines += ["", "Source: football-data.co.uk E0 CSVs, codes "
              + ", ".join(c for c, _ in SEASONS) + ". 2627 excluded (in progress)."]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}")


def main() -> None:
    """Entry point for `make data` / `python -m epl.data`."""
    df = load_all()
    run_checks(df)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(MATCHES_PATH, index=False)
    print(f"wrote {MATCHES_PATH} ({len(df)} rows)")
    write_data_card(df)


if __name__ == "__main__":
    main()
