"""Models on P1-P4 and PT with walk-forward evaluation.

Models (each returns H/D/A probabilities):
- logreg: multinomial logistic regression, L2, scaled inputs; tune C.
- svm_rbf: SVM, RBF kernel, probability outputs; tune C (gamma='scale').
- svm_poly: SVM, degree-5 polynomial kernel (paper replication); tune C.
- lgbm: LightGBM multiclass, small trees, early stopping on the valid season.
- mlp: one-hidden-layer MLP; tune hidden width.

Tuning uses the validation season only (the season before each test season);
the final model is refit on train+valid with the chosen hyperparameters and
scored once on the test season. Imputation (median) and scaling are fit on
training data only (refit on train+valid for the final fit; never on test).

Draw handling (per docs/featuresandmodels.md):
- class-weighted variants (logreg_draww, lgbm_draww): upweight draws with
  weight w tuned on valid log loss.
- decision-rule rows (<base>_drawrule): same probabilities as the base model
  (identical log loss/Brier), but predict a draw when |pH - pA| < margin,
  with the margin tuned on valid accuracy (tie-break: draw recall). Margin
  0.0 is in the grid, so the rule can always recover plain argmax.

All random seeds are 42. SVM grids are deliberately small: each
SVC(probability=True) fit costs ~10s at full-train size.

Usage:
  python -m epl.models [--models logreg,svm_rbf] [--sets P1,P2]
Appends (idempotently) to reports/results.csv.
"""

from __future__ import annotations

import argparse
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from epl.evaluate import SEED, get_splits, score_all

FEATURES_PATH = "data/processed/features.parquet"
RESULTS_PATH = "reports/results.csv"

KEY_COLS = ["match_id", "Date", "season", "HomeTeam", "AwayTeam", "FTR", "target"]
P1_COLS = ["elo_home", "elo_away", "elo_diff"]
TABLE_COLS = [
    "home_prev_pos", "away_prev_pos", "prev_pos_diff", "prev_pos_gap",
    "home_prev_pts", "away_prev_pts", "prev_pts_diff", "prev_gd_diff",
]


def _aspect_cols(aspect: str, n: int) -> list[str]:
    if aspect == "split":
        return [f"home_home_ppg_{n}", f"away_away_ppg_{n}", f"split_diff_{n}"]
    stem = {"ppg": "ppg", "gs": "gs", "gc": "gc", "shots": "shots",
            "sot": "sot", "acc": "acc", "corners": "corners"}[aspect]
    if aspect == "ppg":
        return [f"home_ppg_{n}", f"away_ppg_{n}", f"ppg_diff_{n}"]
    return [f"home_{stem}_{n}", f"away_{stem}_{n}", f"{stem}_diff_{n}"]


def feature_sets() -> dict[str, list[str]]:
    """Nested P1-P4 column lists plus the table-only PT set."""
    p2 = P1_COLS + [c for n in (5, 10) for a in ("ppg", "gs", "gc", "split")
                    for c in _aspect_cols(a, n)]
    p3 = p2 + [c for n in (5, 10) for a in ("shots", "sot", "acc", "corners")
               for c in _aspect_cols(a, n)]
    p4 = p3 + ["home_rest", "away_rest", "rest_diff", "h2h_home_share5",
               "h2h_count", "home_prev_pos", "away_prev_pos", "prev_pos_diff"]
    return {"P1": P1_COLS, "P2": p2, "P3": p3, "P4": p4, "PT": TABLE_COLS}


NEEDS_SCALING = {"logreg", "logreg_draww", "svm_rbf", "svm_poly", "mlp"}


def prepare(df: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Feature matrix (raw, NaNs kept) and target vector."""
    return df[cols].to_numpy(dtype=float), df["target"].to_numpy()


def impute_scale(
    x_tr: np.ndarray, x_va: np.ndarray, x_te: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Median-impute + standardize, statistics from the training block only."""
    med = np.nanmedian(x_tr, axis=0)
    x_tr = np.where(np.isnan(x_tr), med, x_tr)
    x_va = np.where(np.isnan(x_va), med, x_va)
    x_te = np.where(np.isnan(x_te), med, x_te)
    scaler = StandardScaler().fit(x_tr)
    return scaler.transform(x_tr), scaler.transform(x_va), scaler.transform(x_te)


def sample_weights(y: np.ndarray, cw: dict | None) -> np.ndarray | None:
    """Per-sample weights from a {class: weight} dict (None keeps uniform)."""
    if cw is None:
        return None
    return np.array([cw[int(c)] for c in y], dtype=float)


def fit_predict(model: str, params: dict, x_tr: np.ndarray, y_tr: np.ndarray,
                x_va: np.ndarray | None, y_va: np.ndarray | None,
                x_te: np.ndarray) -> np.ndarray:
    """Fit on train (early-stop on valid for lgbm/mlp) and predict test."""
    if model in ("logreg", "logreg_draww"):
        est = LogisticRegression(max_iter=1000,
                                 random_state=SEED, **params)
        est.fit(x_tr, y_tr)
        return est.predict_proba(x_te)
    if model in ("svm_rbf", "svm_poly"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est = SVC(probability=True, random_state=SEED, **params)
            est.fit(x_tr, y_tr)
        return est.predict_proba(x_te)
    if model in ("lgbm", "lgbm_draww"):
        cw = params.get("class_weight")
        train_set = lgb.Dataset(x_tr, label=y_tr, weight=sample_weights(y_tr, cw))
        valid_set = lgb.Dataset(x_va, label=y_va, reference=train_set)
        lgb_params = {"objective": "multiclass", "num_class": 3,
                      "learning_rate": 0.05, "min_data_in_leaf": 50,
                      "verbose": -1, "seed": SEED,
                      **{k: v for k, v in params.items() if k != "class_weight"}}
        booster = lgb.train(lgb_params, train_set, num_boost_round=1000,
                            valid_sets=[valid_set],
                            callbacks=[lgb.early_stopping(50, verbose=False)])
        params["_best_iter"] = booster.best_iteration
        return booster.predict(x_te)
    if model == "mlp":
        est = MLPClassifier(hidden_layer_sizes=params["hidden"], max_iter=500,
                            early_stopping=True, n_iter_no_change=20,
                            random_state=SEED)
        est.fit(x_tr, y_tr)
        return est.predict_proba(x_te)
    raise ValueError(model)


def refit_predict(model: str, params: dict, x_trv: np.ndarray, y_trv: np.ndarray,
                  x_te: np.ndarray) -> np.ndarray:
    """Final fit on train+valid with chosen hyperparameters, predict test."""
    params = {k: v for k, v in params.items() if not k.startswith("_")}
    if model in ("logreg", "logreg_draww"):
        est = LogisticRegression(max_iter=1000,
                                 random_state=SEED, **params)
        est.fit(x_trv, y_trv)
        return est.predict_proba(x_te)
    if model in ("svm_rbf", "svm_poly"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est = SVC(probability=True, random_state=SEED, **params)
            est.fit(x_trv, y_trv)
        return est.predict_proba(x_te)
    if model in ("lgbm", "lgbm_draww"):
        n = params.pop("_best_iter", 200)
        cw = params.pop("class_weight", None)
        train_set = lgb.Dataset(x_trv, label=y_trv,
                                weight=sample_weights(y_trv, cw))
        lgb_params = {"objective": "multiclass", "num_class": 3,
                      "learning_rate": 0.05, "min_data_in_leaf": 50,
                      "verbose": -1, "seed": SEED, **params}
        booster = lgb.train(lgb_params, train_set, num_boost_round=n)
        return booster.predict(x_te)
    if model == "mlp":
        est = MLPClassifier(hidden_layer_sizes=params["hidden"], max_iter=500,
                            early_stopping=True, n_iter_no_change=20,
                            random_state=SEED)
        est.fit(x_trv, y_trv)
        return est.predict_proba(x_te)
    raise ValueError(model)


GRIDS: dict[str, list[dict]] = {
    "logreg": [{"C": c} for c in (0.5, 1.0, 4.0)],
    "logreg_draww": [{"C": c, "class_weight": {0: 1.0, 1: w, 2: 1.0}}
                     for c in (1.0,) for w in (2.0, 4.0)],
    "svm_rbf": [{"kernel": "rbf", "C": c, "gamma": "scale"} for c in (1, 10)],
    "svm_poly": [{"kernel": "poly", "degree": 5, "C": c} for c in (0.1, 1.0)],
    "lgbm": [{"num_leaves": v} for v in (15, 31)],
    "lgbm_draww": [{"num_leaves": 15, "class_weight": {0: 1.0, 1: w, 2: 1.0}}
                   for w in (2.0, 4.0)],
    "mlp": [{"hidden": (w,)} for w in (32, 64)],
}

BASE_MODELS = ["logreg", "svm_rbf", "svm_poly", "lgbm", "mlp"]
MARGIN_GRID = [round(x, 2) for x in np.arange(0.0, 0.31, 0.05)]


def tune_margin(y_va: np.ndarray, p_va: np.ndarray) -> float:
    """Draw margin maximizing valid accuracy (tie-break: draw recall)."""
    best = (0.0, -1.0, -1.0)  # margin, accuracy, draw recall
    for m in MARGIN_GRID:
        pred = apply_margin(p_va, m)
        acc = float((pred == y_va).mean())
        tp = int(((pred == 1) & (y_va == 1)).sum())
        rec = tp / max(int((y_va == 1).sum()), 1)
        if (acc, rec) > (best[1], best[2]):
            best = (m, acc, rec)
    return best[0]


def apply_margin(proba: np.ndarray, margin: float) -> np.ndarray:
    """Argmax, except predict draw when |pH - pA| < margin."""
    pred = proba.argmax(axis=1)
    draw_mask = np.abs(proba[:, 0] - proba[:, 2]) < margin
    return np.where(draw_mask, 1, pred)


def run_models(feats: pd.DataFrame, models: list[str], sets: list[str]) -> pd.DataFrame:
    """Full walk-forward run; returns one row per model/set/test season
    (base + draw-rule rows)."""
    fsets = feature_sets()
    seasons = sorted(feats["season"].unique())
    rows = []
    for split in get_splits(seasons):
        tr = feats[feats["season"].isin(split["train"])]
        va = feats[feats["season"] == split["valid"]]
        te = feats[feats["season"] == split["test"]]
        y_tr, y_va, y_te = (tr["target"].to_numpy(), va["target"].to_numpy(),
                            te["target"].to_numpy())
        for s in sets:
            cols = fsets[s]
            x_tr_raw, _ = prepare(tr, cols)
            x_va_raw, _ = prepare(va, cols)
            x_te_raw, _ = prepare(te, cols)
            for model in models:
                if model in NEEDS_SCALING:
                    x_tr, x_va, x_te = impute_scale(x_tr_raw, x_va_raw, x_te_raw)
                else:
                    x_tr, x_va, x_te = x_tr_raw, x_va_raw, x_te_raw
                # Tune on valid log loss.
                best_params, best_ll = GRIDS[model][0], float("inf")
                for params in GRIDS[model]:
                    p_va = fit_predict(model, dict(params), x_tr, y_tr, x_va, y_va, x_va)
                    ll = score_all(y_va, p_va)["log_loss"]
                    if ll < best_ll:
                        best_params, best_ll = params, ll
                # Refit on train+valid, score test once.
                if model in NEEDS_SCALING:
                    med = np.nanmedian(np.vstack([x_tr_raw, x_va_raw]), axis=0)
                    x_trv = np.where(np.isnan(np.vstack([x_tr_raw, x_va_raw])), med,
                                     np.vstack([x_tr_raw, x_va_raw]))
                    x_te_f = np.where(np.isnan(x_te_raw), med, x_te_raw)
                    scaler = StandardScaler().fit(x_trv)
                    x_trv, x_te_f = scaler.transform(x_trv), scaler.transform(x_te_f)
                else:
                    x_trv = np.vstack([x_tr, x_va])
                    x_te_f = x_te
                y_trv = np.concatenate([y_tr, y_va])
                tuned = dict(best_params)
                if model in ("lgbm", "lgbm_draww"):
                    # Recover the early-stopping iteration on train/valid.
                    fit_predict(model, tuned, x_tr, y_tr, x_va, y_va, x_va)
                p_te = refit_predict(model, tuned, x_trv, y_trv, x_te_f)
                metrics = score_all(y_te, p_te)
                rows.append({"model": model, "feature_set": s,
                             "test_season": split["test"], **metrics, "seed": SEED})
                # Draw-rule row (base models only; same probs, new decisions).
                if model in BASE_MODELS:
                    p_va_best = fit_predict(model, dict(best_params), x_tr, y_tr,
                                            x_va, y_va, x_va)
                    margin = tune_margin(y_va, p_va_best)
                    pred = apply_margin(p_te, margin)
                    tp = int(((pred == 1) & (y_te == 1)).sum())
                    rows.append({"model": f"{model}_drawrule", "feature_set": s,
                                 "test_season": split["test"],
                                 "log_loss": metrics["log_loss"],
                                 "brier": metrics["brier"],
                                 "accuracy": float((pred == y_te).mean()),
                                 "draw_recall": tp / max(int((y_te == 1).sum()), 1),
                                 "seed": SEED})
        print(f"split {split['test']}: done {sets}", flush=True)
    return pd.DataFrame(rows)


def append_results(new: pd.DataFrame, path: str = RESULTS_PATH) -> pd.DataFrame:
    """Idempotent append: replace rows with the same (model, set, season)."""
    existing = pd.read_csv(path)
    key = ["model", "feature_set", "test_season"]
    merged = pd.concat([existing, new], ignore_index=True)
    merged = merged.drop_duplicates(subset=key, keep="last").reset_index(drop=True)
    merged = merged.sort_values(key).reset_index(drop=True)
    merged.to_csv(path, index=False)
    return merged


def main() -> None:
    """Entry point for `make train` / `python -m epl.models`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(
        BASE_MODELS + ["logreg_draww", "lgbm_draww"]))
    parser.add_argument("--sets", default="P1,P2,P3,P4")
    args = parser.parse_args()
    models = [m for m in args.models.split(",") if m]
    sets = [s for s in args.sets.split(",") if s]
    feats = pd.read_parquet(FEATURES_PATH)
    new = run_models(feats, models, sets)
    merged = append_results(new)
    print(f"appended {len(new)} rows; {RESULTS_PATH} now {len(merged)} rows")


if __name__ == "__main__":
    main()
