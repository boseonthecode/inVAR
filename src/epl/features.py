"""Rolling pre-match features, nested sets P1-P4.

Every feature for match M uses only matches with a date strictly before M.
Same-date matches are processed as a batch (read all features, then append
results to histories), so e.g. two Saturday 15:00 kickoffs never see each
other. Teams with fewer than N prior matches get NaN (never zeros or the
current match); a train-fit imputer handles NaNs in milestone 3+.

Column naming: ``home_*`` = home team's value, ``away_*`` = away team's
value, ``*_diff`` = home minus away. ``gs``/``gc`` = goals scored/conceded
per game. ``home_home_ppg_N`` = home team's points per game over its last N
*home* matches; ``away_away_ppg_N`` mirrors for the away team.
``h2h_home_share5`` = share of the last 5 head-to-head meetings won by the
current home team (draws count half); NaN when fewer than 5 prior meetings,
with ``h2h_count`` recording how many exist. ``*_prev_pos`` = previous
season's final league rank; promoted sides get 21/22/23 by order of first
appearance in the new season (true promotion order is not in our data).

Usage: python -m epl.features
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from epl.elo import compute_elo

MATCHES_PATH = Path("data/processed/matches.parquet")
FEATURES_PATH = Path("data/processed/features.parquet")

WINDOWS = (5, 10)

POINTS = {"H": (3, 0), "D": (1, 1), "A": (0, 3)}


def _mean_or_nan(values: list[float], n: int) -> float:
    if len(values) < n:
        return float("nan")
    return sum(values[-n:]) / n


def _prev_season_positions(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Final league rank per team per season, keyed by the *next* season.

    Returns {season: {team: rank}} where rank applies to matches played in
    ``season`` based on the table of the season before it. Promoted sides
    (absent the prior season) get 21/22/23 by first-appearance order.
    The first season maps every team to NaN.
    """
    df = df.sort_values("Date").reset_index(drop=True)
    tables: dict[str, pd.DataFrame] = {}
    for season, g in df.groupby("season", sort=False):
        pts: dict[str, float] = {}
        gd: dict[str, float] = {}
        gf: dict[str, float] = {}
        for row in g.itertuples():
            hp, ap = POINTS[row.FTR]
            pts[row.HomeTeam] = pts.get(row.HomeTeam, 0) + hp
            pts[row.AwayTeam] = pts.get(row.AwayTeam, 0) + ap
            gd[row.HomeTeam] = gd.get(row.HomeTeam, 0) + row.FTHG - row.FTAG
            gd[row.AwayTeam] = gd.get(row.AwayTeam, 0) + row.FTAG - row.FTHG
            gf[row.HomeTeam] = gf.get(row.HomeTeam, 0) + row.FTHG
            gf[row.AwayTeam] = gf.get(row.AwayTeam, 0) + row.FTAG
        table = pd.DataFrame(
            {"team": list(pts), "pts": [pts[t] for t in pts],
             "gd": [gd[t] for t in pts], "gf": [gf[t] for t in pts]}
        ).sort_values(["pts", "gd", "gf"], ascending=False).reset_index(drop=True)
        table["rank"] = table.index + 1
        tables[season] = table
    seasons = list(df["season"].unique())
    out: dict[str, dict[str, float]] = {}
    for i, season in enumerate(seasons):
        if i == 0:
            out[season] = {}
            continue
        prev = tables[seasons[i - 1]]
        ranks = dict(zip(prev["team"], prev["rank"]))
        squad_now = set(df.loc[df["season"] == season, "HomeTeam"]) | set(
            df.loc[df["season"] == season, "AwayTeam"]
        )
        squad_prev = set(prev["team"])
        promoted = sorted(
            squad_now - squad_prev,
            key=lambda t: (
                df.loc[((df["HomeTeam"] == t) | (df["AwayTeam"] == t))
                        & (df["season"] == season), "Date"].min(),
                t,
            ),
        )
        mapping = dict(ranks)
        for j, team in enumerate(promoted):
            mapping[team] = 21 + j
        out[season] = mapping
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build P1-P4 features for every row of a cleaned matches frame."""
    df = df.sort_values("Date").reset_index(drop=True).copy()
    df = compute_elo(df)
    pos_by_season = _prev_season_positions(df)

    history: dict[str, list[dict]] = {}
    last_date: dict[str, pd.Timestamp] = {}
    h2h: dict[frozenset, list[tuple]] = {}
    rows: list[dict] = []
    cur_season: str | None = None

    for date, group in df.groupby("Date", sort=True):
        season = str(group["season"].iloc[0])
        if season != cur_season:
            # Summer break: rest days reset each season, so every team's
            # first match of a season is NaN (train-fit impute).
            last_date.clear()
            cur_season = season
        # Read phase.
        for idx in group.index:
            home, away = df.loc[idx, "HomeTeam"], df.loc[idx, "AwayTeam"]
            season = df.loc[idx, "season"]
            hh = history.get(home, [])
            ah = history.get(away, [])
            feat: dict = {
                "match_id": df.loc[idx, "match_id"],
                "Date": date,
                "season": season,
                "HomeTeam": home,
                "AwayTeam": away,
                "FTR": df.loc[idx, "FTR"],
                "target": df.loc[idx, "target"],
                "elo_home": df.loc[idx, "elo_home"],
                "elo_away": df.loc[idx, "elo_away"],
                "elo_diff": df.loc[idx, "elo_diff"],
            }
            for n in WINDOWS:
                hp = [r["pts"] for r in hh]
                ap = [r["pts"] for r in ah]
                feat[f"home_ppg_{n}"] = _mean_or_nan(hp, n)
                feat[f"away_ppg_{n}"] = _mean_or_nan(ap, n)
                feat[f"ppg_diff_{n}"] = feat[f"home_ppg_{n}"] - feat[f"away_ppg_{n}"]
                hs = [r["scored"] for r in hh]
                aws = [r["scored"] for r in ah]
                feat[f"home_gs_{n}"] = _mean_or_nan(hs, n)
                feat[f"away_gs_{n}"] = _mean_or_nan(aws, n)
                feat[f"gs_diff_{n}"] = feat[f"home_gs_{n}"] - feat[f"away_gs_{n}"]
                hc = [r["conceded"] for r in hh]
                ac = [r["conceded"] for r in ah]
                feat[f"home_gc_{n}"] = _mean_or_nan(hc, n)
                feat[f"away_gc_{n}"] = _mean_or_nan(ac, n)
                feat[f"gc_diff_{n}"] = feat[f"home_gc_{n}"] - feat[f"away_gc_{n}"]
                hhp = [r["pts"] for r in hh if r["venue"] == "home"]
                aap = [r["pts"] for r in ah if r["venue"] == "away"]
                feat[f"home_home_ppg_{n}"] = _mean_or_nan(hhp, n)
                feat[f"away_away_ppg_{n}"] = _mean_or_nan(aap, n)
                feat[f"split_diff_{n}"] = (
                    feat[f"home_home_ppg_{n}"] - feat[f"away_away_ppg_{n}"]
                )
                hsh = [r["shots"] for r in hh]
                ash = [r["shots"] for r in ah]
                feat[f"home_shots_{n}"] = _mean_or_nan(hsh, n)
                feat[f"away_shots_{n}"] = _mean_or_nan(ash, n)
                feat[f"shots_diff_{n}"] = feat[f"home_shots_{n}"] - feat[f"away_shots_{n}"]
                hst = [r["sot"] for r in hh]
                ast = [r["sot"] for r in ah]
                feat[f"home_sot_{n}"] = _mean_or_nan(hst, n)
                feat[f"away_sot_{n}"] = _mean_or_nan(ast, n)
                feat[f"sot_diff_{n}"] = feat[f"home_sot_{n}"] - feat[f"away_sot_{n}"]
                if len(hh) >= n and sum(r["shots"] for r in hh[-n:]) > 0:
                    feat[f"home_acc_{n}"] = sum(r["sot"] for r in hh[-n:]) / sum(
                        r["shots"] for r in hh[-n:]
                    )
                else:
                    feat[f"home_acc_{n}"] = float("nan")
                if len(ah) >= n and sum(r["shots"] for r in ah[-n:]) > 0:
                    feat[f"away_acc_{n}"] = sum(r["sot"] for r in ah[-n:]) / sum(
                        r["shots"] for r in ah[-n:]
                    )
                else:
                    feat[f"away_acc_{n}"] = float("nan")
                feat[f"acc_diff_{n}"] = feat[f"home_acc_{n}"] - feat[f"away_acc_{n}"]
                hco = [r["corners"] for r in hh]
                aco = [r["corners"] for r in ah]
                feat[f"home_corners_{n}"] = _mean_or_nan(hco, n)
                feat[f"away_corners_{n}"] = _mean_or_nan(aco, n)
                feat[f"corners_diff_{n}"] = (
                    feat[f"home_corners_{n}"] - feat[f"away_corners_{n}"]
                )
            feat["home_rest"] = (
                float((date - last_date[home]).days) if home in last_date else float("nan")
            )
            feat["away_rest"] = (
                float((date - last_date[away]).days) if away in last_date else float("nan")
            )
            feat["rest_diff"] = feat["home_rest"] - feat["away_rest"]
            meetings = h2h.get(frozenset({home, away}), [])
            feat["h2h_count"] = min(len(meetings), 5)
            if len(meetings) >= 5:
                last5 = meetings[-5:]
                feat["h2h_home_share5"] = sum(
                    1.0 if w == home else (0.5 if w == "D" else 0.0) for _, w in last5
                ) / 5.0
            else:
                feat["h2h_home_share5"] = float("nan")
            pos = pos_by_season.get(season, {})
            feat["home_prev_pos"] = pos.get(home, float("nan"))
            feat["away_prev_pos"] = pos.get(away, float("nan"))
            feat["prev_pos_diff"] = feat["home_prev_pos"] - feat["away_prev_pos"]
            rows.append(feat)
        # Append phase: results become visible only to strictly later dates.
        for idx in group.index:
            home, away = df.loc[idx, "HomeTeam"], df.loc[idx, "AwayTeam"]
            ftr = df.loc[idx, "FTR"]
            hp, ap = POINTS[ftr]
            history.setdefault(home, []).append({
                "pts": hp, "scored": df.loc[idx, "FTHG"],
                "conceded": df.loc[idx, "FTAG"], "venue": "home",
                "shots": df.loc[idx, "HS"], "sot": df.loc[idx, "HST"],
                "corners": df.loc[idx, "HC"],
            })
            history.setdefault(away, []).append({
                "pts": ap, "scored": df.loc[idx, "FTAG"],
                "conceded": df.loc[idx, "FTHG"], "venue": "away",
                "shots": df.loc[idx, "AS"], "sot": df.loc[idx, "AST"],
                "corners": df.loc[idx, "AC"],
            })
            last_date[home] = date
            last_date[away] = date
            winner = home if ftr == "H" else (away if ftr == "A" else "D")
            h2h.setdefault(frozenset({home, away}), []).append((date, winner))
    return pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)


def main() -> None:
    """Entry point for `make features` / `python -m epl.features`."""
    df = pd.read_parquet(MATCHES_PATH)
    feats = build_features(df)
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(FEATURES_PATH, index=False)
    p1 = [c for c in feats.columns if c.startswith("elo")]
    print(f"wrote {FEATURES_PATH} ({len(feats)} rows, {len(feats.columns)} cols)")
    print(f"P1 cols ({len(p1)}): {p1}")
    for n in WINDOWS:
        ncols = [c for c in feats.columns if c.endswith(f"_{n}")]
        print(f"N={n} rolling cols: {len(ncols)}")
    p4 = ["home_rest", "away_rest", "rest_diff", "h2h_home_share5",
          "h2h_count", "home_prev_pos", "away_prev_pos", "prev_pos_diff"]
    print(f"P4 cols: {p4}")
    print("null rates (first season should be ~1.0 for rolling):")
    first = feats["season"] == feats["season"].iloc[0]
    print(f"  home_ppg_5 null in 2009/10: {feats.loc[first, 'home_ppg_5'].isna().mean():.3f}")


if __name__ == "__main__":
    main()
