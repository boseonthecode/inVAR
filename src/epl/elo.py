"""Elo ratings computed strictly from past matches.

Pre-match rating for match M uses only matches with a date strictly before
M's date. Matches sharing the same date are processed as a batch: they all
read the pre-date ratings, and updates are applied afterwards. This prevents
same-day leakage (e.g. two Saturday 15:00 kickoffs seeing each other).

Defaults per docs/featuresandmodels.md: init 1500, K=20, home advantage h=60.
Promoted teams start at the mean of the three relegated teams' end-of-season
ratings (fallback 1500 if there is no prior season).
"""

from __future__ import annotations

import pandas as pd

INIT_RATING = 1500.0
K = 20.0
HOME_ADV = 60.0


def expected_home(r_home: float, r_away: float, h: float = HOME_ADV) -> float:
    """Expected home score from pre-match ratings."""
    return 1.0 / (1.0 + 10.0 ** (-(r_home + h - r_away) / 400.0))


def compute_elo(
    df: pd.DataFrame,
    k: float = K,
    h: float = HOME_ADV,
    init: float = INIT_RATING,
) -> pd.DataFrame:
    """Add pre-match ``elo_home`` / ``elo_away`` / ``elo_diff`` columns.

    The input must contain Date, season, HomeTeam, AwayTeam, FTR and be
    sorted by Date (sorted again here defensively). The frame itself is
    never shuffled; updates are applied in date order.
    """
    df = df.sort_values("Date").reset_index(drop=True).copy()
    squads: dict[str, set[str]] = {
        season: set(g["HomeTeam"]) | set(g["AwayTeam"])
        for season, g in df.groupby("season", sort=False)
    }
    ratings: dict[str, float] = {}
    prev_season: str | None = None
    elo_home: list[float] = []
    elo_away: list[float] = []
    for date, group in df.groupby("Date", sort=True):
        season = str(group["season"].iloc[0])
        if season != prev_season:
            # Season boundary: seed newcomers (promoted sides).
            if prev_season is None:
                for team in squads[season]:
                    ratings.setdefault(team, init)
            else:
                relegated = squads[prev_season] - squads[season]
                if relegated and all(t in ratings for t in relegated):
                    seed = sum(ratings[t] for t in relegated) / len(relegated)
                else:
                    seed = init
                for team in squads[season] - squads[prev_season]:
                    ratings.setdefault(team, seed)
                for team in squads[season]:
                    ratings.setdefault(team, init)
            prev_season = season
        # Read phase: every match on this date sees pre-date ratings only.
        rows = [
            (ratings[group["HomeTeam"].iloc[i]], ratings[group["AwayTeam"].iloc[i]])
            for i in range(len(group))
        ]
        elo_home.extend(r for r, _ in rows)
        elo_away.extend(r for _, r in rows)
        # Update phase: apply results afterwards.
        for idx, ftr in zip(group.index, group["FTR"]):
            ht, at = df.loc[idx, "HomeTeam"], df.loc[idx, "AwayTeam"]
            rh, ra = ratings[ht], ratings[at]
            e_home = expected_home(rh, ra, h)
            s_home = {"H": 1.0, "D": 0.5, "A": 0.0}[ftr]
            ratings[ht] = rh + k * (s_home - e_home)
            ratings[at] = ra + k * ((1.0 - s_home) - (1.0 - e_home))
    df["elo_home"] = elo_home
    df["elo_away"] = elo_away
    df["elo_diff"] = df["elo_home"] - df["elo_away"]
    return df
