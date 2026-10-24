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

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from epl.data import (
    AVG_COLS,
    B365_COLS,
    STAT_COLS,
    TARGET_MAP,
    canonical_name,
    deoverround,
)
from epl.evaluate import SEED, score_all
from epl.features import build_features
from epl.models import feature_sets

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.table import Table
except ImportError:  # pragma: no cover - optional dependency
    Console = Panel = Prompt = Table = None

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


BOOK_PROB_COLS = ["b365_home_prob", "b365_draw_prob", "b365_away_prob"]


def _impute_scale_fit(
    x_train: np.ndarray, x_predict: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    med = np.nanmedian(x_train, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    x_train = np.where(np.isnan(x_train), med, x_train)
    x_predict = np.where(np.isnan(x_predict), med, x_predict)
    scaler = StandardScaler().fit(x_train)
    return scaler.transform(x_train), scaler.transform(x_predict)


def _prior_proba(y: np.ndarray, n: int) -> np.ndarray:
    counts = np.bincount(np.asarray(y, dtype=int), minlength=3).astype(float) + 0.5
    return np.tile(counts / counts.sum(), (n, 1))


def _predict_proba(
    train: pd.DataFrame, pred: pd.DataFrame, cols: list[str], seed: int
) -> np.ndarray:
    if train.empty:
        return np.tile([1.0 / 3.0] * 3, (len(pred), 1))
    y = train["target"].to_numpy(dtype=int)
    if len(np.unique(y)) < 2:
        return _prior_proba(y, len(pred))
    x_train = train[cols].to_numpy(dtype=float)
    x_pred = pred[cols].to_numpy(dtype=float)
    x_train, x_pred = _impute_scale_fit(x_train, x_pred)
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
    clf.fit(x_train, y)
    raw = clf.predict_proba(x_pred)
    full = np.zeros((len(pred), 3))
    full[:, clf.classes_.astype(int)] = raw
    return full / full.sum(axis=1, keepdims=True)


def predict_fixtures(
    feats: pd.DataFrame,
    fixtures: pd.DataFrame,
    combined: pd.DataFrame,
    feature_set: str = "P1",
    seed: int = SEED,
) -> pd.DataFrame:
    """Predict every fixture as of its own date, using only earlier matches."""
    fsets = feature_sets()
    if feature_set not in fsets:
        raise ValueError(f"unknown feature set {feature_set!r}")
    cols = fsets[feature_set]
    fixture_ids = set(fixtures["match_id"])
    pred_feats = feats[feats["match_id"].isin(fixture_ids)].copy()
    if pred_feats.empty:
        raise ValueError("no fixture rows found after combining with history")
    meta = combined.drop_duplicates("match_id").set_index("match_id")

    rows: list[dict] = []
    for date, group in pred_feats.groupby("Date", sort=True):
        train = feats[(feats["Date"] < date) & feats["target"].notna()]
        if len(train) and pd.Timestamp(train["Date"].max()) >= pd.Timestamp(date):
            raise AssertionError("training data reached the fixture date")
        proba = _predict_proba(train, group, cols, seed)
        cutoff = train["Date"].max() if len(train) else pd.NaT
        n_train = len(train)
        for (_, frow), prob in zip(group.iterrows(), proba):
            mid = frow["match_id"]
            actual = meta.at[mid, "FTR"] if mid in meta.index else np.nan
            book = (
                meta.loc[mid, BOOK_PROB_COLS].to_numpy(dtype=float)
                if mid in meta.index else np.full(3, np.nan)
            )
            pick = "HDA"[int(prob.argmax())]
            correct = bool(pick == actual) if not pd.isna(actual) else np.nan
            rows.append({
                "match_id": mid,
                "date": frow["Date"],
                "home_team": frow["HomeTeam"],
                "away_team": frow["AwayTeam"],
                "p_home": float(prob[0]),
                "p_draw": float(prob[1]),
                "p_away": float(prob[2]),
                "pick": pick,
                "confidence": float(prob.max()),
                "b365_home_prob": float(book[0]),
                "b365_draw_prob": float(book[1]),
                "b365_away_prob": float(book[2]),
                "actual": actual,
                "correct": correct,
                "cutoff": cutoff,
                "train_rows": n_train,
            })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def summarize(preds: pd.DataFrame) -> dict:
    """Aggregate metrics for the completed fixtures in a prediction table."""
    out: dict = {
        "n": len(preds),
        "n_played": int(preds["actual"].notna().sum()),
        "n_upcoming": int(preds["actual"].isna().sum()),
    }
    played = preds[preds["actual"].notna()]
    if len(played):
        y = played["actual"].map(TARGET_MAP).to_numpy(dtype=int)
        proba = played[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
        out.update(score_all(y, proba))
        out["correct"] = int(played["correct"].sum())
        valid = played[BOOK_PROB_COLS].notna().all(axis=1)
        if valid.any():
            book = played.loc[valid, BOOK_PROB_COLS].to_numpy(dtype=float)
            yb = played.loc[valid, "actual"].map(TARGET_MAP).to_numpy(dtype=int)
            out["bookmaker"] = score_all(yb, book)
            out["bookmaker_n"] = int(valid.sum())
    return out


def _prob_text(row: pd.Series, cols: list[str]) -> str:
    values = row[cols].to_numpy(dtype=float)
    if np.isnan(values).any():
        return "-"
    return f"{values[0] * 100:.0f}/{values[1] * 100:.0f}/{values[2] * 100:.0f}"


def _market_text(row: pd.Series) -> str:
    return _prob_text(row, BOOK_PROB_COLS)


def _history_text(preds: pd.DataFrame) -> str:
    cutoffs = preds["cutoff"].dropna()
    rows = preds["train_rows"]
    if cutoffs.empty:
        return "no prior matches (uniform probabilities)"
    lo, hi = cutoffs.min().date(), cutoffs.max().date()
    cutoff = str(lo) if lo == hi else f"{lo} .. {hi}"
    n_lo, n_hi = int(rows.min()), int(rows.max())
    n = str(n_lo) if n_lo == n_hi else f"{n_lo}..{n_hi}"
    return f"{n} training rows, cutoff {cutoff}"


def _render_plain(
    preds: pd.DataFrame, fixtures_path: Path, feature_set: str, seed: int
) -> None:
    summary = summarize(preds)
    print(f"EPL live demo | logreg/{feature_set} | seed={seed}")
    print(f"fixtures: {fixtures_path}")
    has_market = preds[BOOK_PROB_COLS].notna().all(axis=1).any()
    header = f"{'Date':<11} {'Home':<18} {'Away':<16} {'H/D/A':>8} {'Pick':>4}"
    if has_market:
        header += f" {'Market':>8}"
    header += f" {'Actual':>7}"
    print(header)
    for _, row in preds.iterrows():
        actual = "-" if pd.isna(row["actual"]) else f"{row['actual']} {'OK' if row['correct'] else 'X'}"
        line = (
            f"{row['date'].date()!s:<11} {row['home_team']:<18} {row['away_team']:<16} "
            f"{_prob_text(row, ['p_home', 'p_draw', 'p_away']):>8} {row['pick']:>4}"
        )
        if has_market:
            line += f" {_market_text(row):>8}"
        line += f" {actual:>7}"
        print(line)
    print(f"history: {_history_text(preds)}")
    print("every prediction uses only matches strictly before its date")
    if summary["n_played"]:
        print(
            f"played: {summary['n_played']} | accuracy {summary['accuracy']:.3f} | "
            f"log loss {summary['log_loss']:.4f} | brier {summary['brier']:.4f}"
        )
        if "bookmaker" in summary:
            b = summary["bookmaker"]
            print(
                f"bookmaker ({summary['bookmaker_n']}): accuracy {b['accuracy']:.3f} | "
                f"log loss {b['log_loss']:.4f} | brier {b['brier']:.4f}"
            )
    else:
        print("no completed fixtures yet; predictions only")


def _render_rich(
    preds: pd.DataFrame,
    fixtures_path: Path,
    feature_set: str,
    seed: int,
    no_color: bool,
) -> None:
    summary = summarize(preds)
    console = Console(no_color=no_color)
    console.print(Panel.fit(
        f"[bold]EPL live demo[/bold]  logreg/{feature_set}  seed={seed}\n"
        f"fixtures: {fixtures_path}",
        border_style="cyan",
    ))
    table = Table(
        title="Pre-match predictions",
        header_style="bold cyan",
        padding=(0, 1),
        pad_edge=False,
    )
    table.add_column("Date", no_wrap=True)
    table.add_column("Home", no_wrap=True)
    table.add_column("Away", no_wrap=True)
    table.add_column("H/D/A", justify="center", no_wrap=True)
    table.add_column("Pick", justify="center", no_wrap=True)
    has_market = preds[BOOK_PROB_COLS].notna().all(axis=1).any()
    if has_market:
        table.add_column("Mkt H/D/A", justify="center", no_wrap=True)
    has_actual = preds["actual"].notna().any()
    if has_actual:
        table.add_column("Act", justify="center", no_wrap=True)
    for _, row in preds.iterrows():
        cells = [
            str(row["date"].date()),
            row["home_team"],
            row["away_team"],
            _prob_text(row, ["p_home", "p_draw", "p_away"]),
            f"[bold]{row['pick']}[/bold]",
        ]
        if has_market:
            cells.append(_market_text(row))
        if has_actual:
            if pd.isna(row["actual"]):
                cells.append("-")
            else:
                mark = "[green]OK[/green]" if row["correct"] else "[red]X[/red]"
                cells.append(f"{row['actual']} {mark}")
        table.add_row(*cells)
    console.print(table)

    lines = [
        f"training history: {_history_text(preds)}",
        "each prediction sees only matches strictly before its date",
        f"fixtures: {summary['n']} total, {summary['n_played']} played, "
        f"{summary['n_upcoming']} upcoming",
    ]
    if summary["n_played"]:
        lines.append(
            f"[bold]model[/bold]: accuracy {summary['accuracy']:.3f} | "
            f"log loss {summary['log_loss']:.4f} | brier {summary['brier']:.4f} | "
            f"draw recall {summary['draw_recall']:.3f}"
        )
        if "bookmaker" in summary:
            b = summary["bookmaker"]
            lines.append(
                f"[bold]bookmaker[/bold] ({summary['bookmaker_n']}): accuracy "
                f"{b['accuracy']:.3f} | log loss {b['log_loss']:.4f} | "
                f"brier {b['brier']:.4f}"
            )
    else:
        lines.append("no completed fixtures yet; predictions only")
    console.print(Panel("\n".join(lines), title="Summary", border_style="green"))


def render_predictions(
    preds: pd.DataFrame,
    fixtures_path: Path,
    feature_set: str,
    seed: int,
    no_color: bool = False,
) -> None:
    if Table is None:
        _render_plain(preds, fixtures_path, feature_set, seed)
    else:
        _render_rich(preds, fixtures_path, feature_set, seed, no_color)


def run_demo(
    fixtures_path: Path,
    history_path: Path,
    season: str | None,
    feature_set: str,
    seed: int,
    no_color: bool,
) -> int:
    try:
        fixtures = load_fixtures(fixtures_path, season)
        history = load_history(history_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}")
        return 1
    missing = unknown_teams(fixtures, history)
    if missing:
        warnings.warn(
            f"unknown teams (Elo starts at 1500): {', '.join(missing)}",
            stacklevel=2,
        )
    print(f"building features for {len(fixtures)} fixture(s)...")
    combined = combine_matches(history, fixtures)
    feats = build_features(combined)
    preds = predict_fixtures(feats, fixtures, combined, feature_set, seed)
    render_predictions(preds, fixtures_path, feature_set, seed, no_color)
    return 0


def _ask(prompt: str, default: str) -> str:
    if Prompt is not None:
        return Prompt.ask(prompt, default=default)
    answer = input(f"{prompt} [{default}]: ").strip()
    return answer or default


def interactive_menu(args: argparse.Namespace) -> None:
    while True:
        print("\nEPL live demo")
        print("  1. Predict fixtures from a CSV")
        print("  2. Quit")
        choice = _ask("Choose", "1")
        if choice == "1":
            path = _ask("Fixture CSV", str(DEFAULT_FIXTURES))
            run_demo(
                Path(path), args.history, args.season, args.features,
                args.seed, args.no_color,
            )
        elif choice in ("2", "q", "quit", "exit"):
            break
        else:
            print("unknown choice")


def print_template() -> None:
    """Print the example fixture CSV to stdout."""
    print(TEMPLATE_PATH.read_text())


def main() -> None:
    """Entry point for `python -m epl.demo`."""
    parser = argparse.ArgumentParser(description="EPL live fixture demo")
    parser.add_argument("--fixtures", type=Path, default=None)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--season", default=None)
    parser.add_argument(
        "--features", default="P1", choices=["P1", "P2", "P3", "P4"]
    )
    parser.add_argument("--template", action="store_true")
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.template:
        print_template()
        return
    if args.fixtures is not None:
        raise SystemExit(run_demo(
            args.fixtures, args.history, args.season, args.features,
            args.seed, args.no_color,
        ))
    interactive_menu(args)


if __name__ == "__main__":
    main()
