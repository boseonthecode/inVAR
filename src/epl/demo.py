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
RESULTS_PATH = Path("reports/results.csv")

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

    for out in ["b365_home_prob", "b365_draw_prob", "b365_away_prob"]:
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
    preds: pd.DataFrame, source: str, feature_set: str, seed: int
) -> None:
    summary = summarize(preds)
    print(f"EPL live demo | logreg/{feature_set} | seed={seed}")
    print(f"source: {source}")
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
    source: str,
    feature_set: str,
    seed: int,
    no_color: bool,
) -> None:
    summary = summarize(preds)
    console = Console(no_color=no_color)
    console.print(Panel.fit(
        f"[bold]EPL live demo[/bold]  logreg/{feature_set}  seed={seed}\n"
        f"source: {source}",
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
    source: str,
    feature_set: str,
    seed: int,
    no_color: bool = False,
) -> None:
    if Table is None:
        _render_plain(preds, source, feature_set, seed)
    else:
        _render_rich(preds, source, feature_set, seed, no_color)


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
    render_predictions(preds, str(fixtures_path), feature_set, seed, no_color)
    return 0


def show_leakage(
    results_path: str | Path = RESULTS_PATH, no_color: bool = False
) -> int:
    """Show the honest walk-forward result next to the leaky replication."""
    results_path = Path(results_path)
    if not results_path.exists():
        print(f"error: {results_path} missing; run `make all` first")
        return 1
    results = pd.read_csv(results_path)
    honest = results[(results["model"] == "logreg")
                     & (results["feature_set"] == "P1")]
    book = results[results["model"] == "bookmaker"]
    leaky = results[results["model"] == "leaky_random80"]
    if honest.empty:
        print("error: results.csv has no honest logreg/P1 rows")
        return 1
    honest_ll = float(honest["log_loss"].mean())
    honest_acc = float(honest["accuracy"].mean())
    honest_lo = float(honest["accuracy"].min())
    honest_hi = float(honest["accuracy"].max())
    book_ll = float(book["log_loss"].mean()) if not book.empty else float("nan")
    book_acc = float(book["accuracy"].mean()) if not book.empty else float("nan")
    leaky_acc = float(leaky["accuracy"].iloc[0]) if not leaky.empty else float("nan")
    premium = leaky_acc - honest_acc if not np.isnan(leaky_acc) else float("nan")

    rows = [
        (f"honest logreg/P1 ({len(honest)} test seasons)", honest_ll, honest_acc),
        ("bookmaker benchmark", book_ll, book_acc),
    ]
    if not np.isnan(leaky_acc):
        rows.append(("leaky random 80/20 (pooled)", float("nan"), leaky_acc))

    if Table is None:
        print("honest vs leaky")
        for label, ll, acc in rows:
            ll_text = "n/a" if np.isnan(ll) else f"{ll:.4f}"
            print(f"  {label:<42} log loss {ll_text:>7}  accuracy {acc:.3f}")
        print(f"honest per-season accuracy range: {honest_lo:.3f}..{honest_hi:.3f}")
        if not np.isnan(premium):
            print(f"leakage premium: +{premium * 100:.1f} accuracy points")
        print("why: full-season averages include the match itself and later games; "
              "the random split shuffles time.")
    else:
        console = Console(no_color=no_color)
        table = Table(
            title="Honest walk-forward vs leaky replication",
            header_style="bold cyan",
            padding=(0, 1),
        )
        table.add_column("row")
        table.add_column("log loss", justify="right")
        table.add_column("accuracy", justify="right")
        for label, ll, acc in rows:
            ll_text = "n/a" if np.isnan(ll) else f"{ll:.4f}"
            table.add_row(label, ll_text, f"{acc:.3f}")
        console.print(table)
        lines = [
            f"honest per-season accuracy range: {honest_lo:.3f}..{honest_hi:.3f}",
        ]
        if not np.isnan(premium):
            lines.append(
                f"[bold]leakage premium[/bold]: +{premium * 100:.1f} accuracy points"
            )
        lines += [
            "leaky features are full-season team averages: they include the",
            "match itself and matches played after it, and the 80/20 split",
            "shuffles matches across seasons. The extra points are future",
            "information, not skill. See src/epl/leaky.py and",
            "tests/test_no_leakage.py for the honest guards.",
        ]
        console.print(Panel("\n".join(lines), title="Why the leaky row is higher",
                            border_style="yellow"))
    return 0


def show_about(no_color: bool = False) -> int:
    """Print the methodology and headline numbers for a presentation."""
    honest_ll = honest_acc = book_ll = book_acc = leaky_acc = float("nan")
    if RESULTS_PATH.exists():
        results = pd.read_csv(RESULTS_PATH)
        honest = results[(results["model"] == "logreg")
                         & (results["feature_set"] == "P1")]
        book = results[results["model"] == "bookmaker"]
        leaky = results[results["model"] == "leaky_random80"]
        if not honest.empty:
            honest_ll = float(honest["log_loss"].mean())
            honest_acc = float(honest["accuracy"].mean())
        if not book.empty:
            book_ll = float(book["log_loss"].mean())
            book_acc = float(book["accuracy"].mean())
        if not leaky.empty:
            leaky_acc = float(leaky["accuracy"].iloc[0])

    lines = [
        "Goal: predict H/D/A from everything known strictly before kickoff,",
        "and compare honestly with de-overrounded bookmaker probabilities.",
        "",
        "Protocol: walk-forward by season. For each test season, train on",
        "earlier seasons, tune on the season just before it, and score the",
        "test season once. No shuffling across seasons.",
        "",
        "Guards: date-ordered Elo/rolling features with same-date batching;",
        "train-only imputation and scaling; test season never used for tuning;",
        "bookmaker odds are a benchmark, never a training feature.",
        "",
        f"Headline (12 test seasons): logreg/P1 {honest_ll:.4f} log loss / "
        f"{honest_acc:.1%} accuracy; bookmaker {book_ll:.4f} / {book_acc:.1%}.",
        f"Leaky replication: {leaky_acc:.1%} accuracy vs {honest_acc:.1%} honest.",
        "No honest model beats the bookmaker.",
        "",
        "This live demo uses logreg/P1 with fixed C=1.0, trained on all",
        "matches before each fixture. It is an inference tool, not a new",
        "headline result; reports/results.csv remains the evaluation.",
    ]
    if Panel is None:
        print("\n".join(lines))
    else:
        Console(no_color=no_color).print(
            Panel("\n".join(lines), title="Methodology", border_style="cyan")
        )
    return 0


def run_historical(
    history_path: str | Path = DEFAULT_HISTORY,
    season: str | None = "2025/26",
    match_id: str | None = None,
    feature_set: str = "P1",
    seed: int = SEED,
    no_color: bool = False,
) -> int:
    """Rehearse a prediction on a completed match, revealing the result."""
    try:
        history = load_history(history_path)
    except FileNotFoundError as exc:
        print(f"error: {exc}")
        return 1
    if match_id is not None:
        row = history[history["match_id"] == match_id]
        if row.empty:
            print(f"error: match_id not found: {match_id}")
            return 1
    else:
        season = season or "2025/26"
        candidates = history[history["season"] == season]
        if candidates.empty:
            seasons = ", ".join(sorted(history["season"].unique()))
            print(f"error: no matches for {season}; available: {seasons}")
            return 1
        row = candidates.sort_values("Date").tail(1)

    print(f"building features for historical match {row.iloc[0]['match_id']}...")
    feats = build_features(history)
    preds = predict_fixtures(feats, row, history, feature_set, seed)
    label = f"historical rehearsal: {row.iloc[0]['season']}"
    render_predictions(preds, label, feature_set, seed, no_color)
    return 0


def _render_match_choices(
    matches: pd.DataFrame, season: str, no_color: bool
) -> None:
    if Table is None:
        print(f"last {len(matches)} matches in {season}")
        for i, (_, row) in enumerate(matches.iterrows(), start=1):
            print(f"  {i:>2}. {row['Date'].date()}  {row['HomeTeam']} vs "
                  f"{row['AwayTeam']}")
        return
    console = Console(no_color=no_color)
    table = Table(
        title=f"Last {len(matches)} matches in {season}",
        header_style="bold cyan",
        padding=(0, 1),
    )
    table.add_column("#", justify="right")
    table.add_column("Date")
    table.add_column("Home")
    table.add_column("Away")
    for i, (_, row) in enumerate(matches.iterrows(), start=1):
        table.add_row(
            str(i), str(row["Date"].date()), row["HomeTeam"], row["AwayTeam"]
        )
    console.print(table)


def interactive_historical(args: argparse.Namespace) -> None:
    season = _ask("Season", "2025/26")
    try:
        history = load_history(args.history)
    except FileNotFoundError as exc:
        print(f"error: {exc}")
        return
    candidates = history[history["season"] == season]
    if candidates.empty:
        seasons = ", ".join(sorted(history["season"].unique()))
        print(f"no matches for {season}; available: {seasons}")
        return
    tail = (
        candidates.sort_values("Date").tail(10).reset_index(drop=True)
    )
    _render_match_choices(tail, season, args.no_color)
    answer = _ask("Match number", str(len(tail)))
    try:
        idx = int(answer) - 1
    except ValueError:
        print("error: match number must be an integer")
        return
    if not 0 <= idx < len(tail):
        print(f"error: choose 1..{len(tail)}")
        return
    match_id = tail.iloc[idx]["match_id"]
    run_historical(
        args.history, season, match_id, args.features, args.seed, args.no_color
    )


def _ask(prompt: str, default: str) -> str:
    if Prompt is not None:
        return Prompt.ask(prompt, default=default)
    answer = input(f"{prompt} [{default}]: ").strip()
    return answer or default


def interactive_menu(args: argparse.Namespace) -> None:
    while True:
        print("\nEPL live demo")
        print("  1. Predict fixtures from a CSV")
        print("  2. Rehearse on a historical match")
        print("  3. Leakage contrast")
        print("  4. Methodology / talking points")
        print("  5. Quit")
        choice = _ask("Choose", "1")
        if choice == "1":
            path = _ask("Fixture CSV", str(DEFAULT_FIXTURES))
            run_demo(
                Path(path), args.history, args.season, args.features,
                args.seed, args.no_color,
            )
        elif choice == "2":
            interactive_historical(args)
        elif choice == "3":
            show_leakage(no_color=args.no_color)
        elif choice == "4":
            show_about(no_color=args.no_color)
        elif choice in ("5", "q", "quit", "exit"):
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
    parser.add_argument("--leakage", action="store_true")
    parser.add_argument("--about", action="store_true")
    parser.add_argument(
        "--historical", nargs="?", const="2025/26", default=None,
        metavar="SEASON",
    )
    parser.add_argument("--match-id", default=None)
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.template:
        print_template()
        return
    if args.leakage:
        raise SystemExit(show_leakage(no_color=args.no_color))
    if args.about:
        raise SystemExit(show_about(no_color=args.no_color))
    if args.historical is not None:
        raise SystemExit(run_historical(
            args.history, args.historical, args.match_id, args.features,
            args.seed, args.no_color,
        ))
    if args.fixtures is not None:
        raise SystemExit(run_demo(
            args.fixtures, args.history, args.season, args.features,
            args.seed, args.no_color,
        ))
    interactive_menu(args)


if __name__ == "__main__":
    main()
