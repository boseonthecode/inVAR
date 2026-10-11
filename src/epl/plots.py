"""Figures and ablation table from reports/results.csv.

Outputs (all under reports/):
- figures/logloss_bar.png: mean log loss by base model and feature set,
  bookmaker as a horizontal line.
- figures/calibration_best.png: per-class calibration of the best model
  (lowest mean log loss: logreg on P1), probabilities pooled across all
  test seasons (each scored exactly once, walk-forward).
- figures/confusion_best.png: confusion matrix of the best model, rows
  normalized by true class.
- figures/importance_lgbm_P4.png: permutation importance (log-loss based,
  10 repeats) for LightGBM on P4, fit on seasons <= 2023/24, early-stopped
  on 2024/25, explained on the 2025/26 test season.
- ablation.csv: mean log loss per base model and set, plus P1->P2->P3->P4
  deltas (negative = improvement).

Usage: python -m epl.plots
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix

from epl.evaluate import get_splits, log_loss_score
from epl.models import (
    GRIDS,
    SEED,
    feature_sets,
    fit_predict,
    impute_scale,
    refit_predict,
)

RESULTS_PATH = Path("reports/results.csv")
FEATURES_PATH = Path("data/processed/features.parquet")
FIG_DIR = Path("reports/figures")
ABLATION_PATH = Path("reports/ablation.csv")

BASE_MODELS = ["logreg", "svm_rbf", "svm_poly", "lgbm", "mlp"]
SETS = ["P1", "P2", "P3", "P4"]
CLASS_NAMES = ["H", "D", "A"]


def best_model(results: pd.DataFrame) -> tuple[str, str]:
    """(model, set) with the lowest mean log loss among base models."""
    base = results[results["model"].isin(BASE_MODELS)]
    means = base.groupby(["model", "feature_set"])["log_loss"].mean()
    return means.idxmin()


def collect_test_proba(feats: pd.DataFrame, model: str, fset: str
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Walk-forward test probabilities, mirroring epl.models (valid-tuned)."""
    cols = feature_sets()[fset]
    seasons = sorted(feats["season"].unique())
    ys, ps = [], []
    for split in get_splits(seasons):
        tr = feats[feats["season"].isin(split["train"])]
        va = feats[feats["season"] == split["valid"]]
        te = feats[feats["season"] == split["test"]]
        y_tr, y_va, y_te = (tr["target"].to_numpy(), va["target"].to_numpy(),
                            te["target"].to_numpy())
        x_tr_raw = tr[cols].to_numpy(dtype=float)
        x_va_raw = va[cols].to_numpy(dtype=float)
        x_te_raw = te[cols].to_numpy(dtype=float)
        x_tr, x_va, x_te = impute_scale(x_tr_raw, x_va_raw, x_te_raw)
        best_params, best_ll = GRIDS[model][0], float("inf")
        for params in GRIDS[model]:
            p_va = fit_predict(model, dict(params), x_tr, y_tr, x_va, y_va, x_va)
            ll = log_loss_score(y_va, p_va)
            if ll < best_ll:
                best_ll, best_params = ll, params
        med = np.nanmedian(np.vstack([x_tr_raw, x_va_raw]), axis=0)
        x_trv = np.where(np.isnan(np.vstack([x_tr_raw, x_va_raw])), med,
                         np.vstack([x_tr_raw, x_va_raw]))
        x_te_f = np.where(np.isnan(x_te_raw), med, x_te_raw)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(x_trv)
        p_te = refit_predict(model, dict(best_params),
                             scaler.transform(x_trv),
                             np.concatenate([y_tr, y_va]),
                             scaler.transform(x_te_f))
        ys.append(y_te)
        ps.append(p_te)
    return np.concatenate(ys), np.concatenate(ps)


def plot_logloss_bar(results: pd.DataFrame) -> None:
    """Grouped bars of mean log loss; bookmaker as a horizontal line."""
    means = results[results["model"].isin(BASE_MODELS)].groupby(
        ["model", "feature_set"])["log_loss"].mean().unstack("feature_set")
    means = means.reindex(BASE_MODELS)
    book = results.loc[results["model"] == "bookmaker", "log_loss"].mean()
    x = np.arange(len(BASE_MODELS))
    width = 0.18
    _, ax = plt.subplots(figsize=(10, 5))
    for i, s in enumerate(SETS):
        ax.bar(x + (i - 1.5) * width, means[s].to_numpy(), width, label=s)
    ax.axhline(book, color="black", linestyle="--",
               label=f"bookmaker ({book:.3f})")
    ax.set_xticks(x)
    ax.set_xticklabels(BASE_MODELS, rotation=15)
    ax.set_ylabel("mean log loss (12 test seasons)")
    ax.set_title("Mean log loss by model and feature set")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "logloss_bar.png", dpi=150)
    plt.close()


def plot_calibration(y: np.ndarray, proba: np.ndarray, tag: str) -> None:
    """One calibration curve per class (one-vs-rest, 10 bins)."""
    _, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k:", label="perfect")
    for c, name in enumerate(CLASS_NAMES):
        frac_pos, mean_pred = calibration_curve(y == c, proba[:, c], n_bins=10)
        ax.plot(mean_pred, frac_pos, marker="o", label=f"class {name}")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("empirical win rate")
    ax.set_title(f"Calibration ({tag}, pooled test seasons)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "calibration_best.png", dpi=150)
    plt.close()


def plot_confusion(y: np.ndarray, proba: np.ndarray, tag: str) -> None:
    """Confusion matrix of argmax predictions, rows normalized by truth."""
    cm = confusion_matrix(y, proba.argmax(axis=1), labels=[0, 1, 2],
                          normalize="true")
    _, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, vmin=0, vmax=1)
    ax.set_xticks([0, 1, 2], CLASS_NAMES)
    ax.set_yticks([0, 1, 2], CLASS_NAMES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Confusion matrix ({tag}, normalized by true class)")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "confusion_best.png", dpi=150)
    plt.close()


def lgbm_p4_importance(feats: pd.DataFrame) -> pd.DataFrame:
    """Permutation importance for LightGBM/P4, explained on 2025/26."""
    cols = feature_sets()["P4"]
    tr = feats[feats["season"] <= "2023/24"]
    va = feats[feats["season"] == "2024/25"]
    te = feats[feats["season"] == "2025/26"]
    x_tr, y_tr = tr[cols].to_numpy(dtype=float), tr["target"].to_numpy()
    x_va, y_va = va[cols].to_numpy(dtype=float), va["target"].to_numpy()
    x_te, y_te = te[cols].to_numpy(dtype=float), te["target"].to_numpy()
    train_set = lgb.Dataset(x_tr, label=y_tr)
    valid_set = lgb.Dataset(x_va, label=y_va, reference=train_set)
    booster = lgb.train({"objective": "multiclass", "num_class": 3,
                         "learning_rate": 0.05, "num_leaves": 15,
                         "min_data_in_leaf": 50, "verbose": -1, "seed": SEED},
                        train_set, num_boost_round=1000, valid_sets=[valid_set],
                        callbacks=[lgb.early_stopping(50, verbose=False)])

    baseline = log_loss_score(y_te, booster.predict(x_te))
    rng = np.random.default_rng(SEED)
    rows = []
    for j, col in enumerate(cols):
        drops = []
        for _ in range(10):
            x_perm = x_te.copy()
            x_perm[:, j] = rng.permutation(x_perm[:, j])
            drops.append(log_loss_score(y_te, booster.predict(x_perm)) - baseline)
        rows.append({"feature": col, "importance": float(np.mean(drops)),
                     "std": float(np.std(drops))})
    out = pd.DataFrame(rows)
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


def plot_importance(imp: pd.DataFrame, top: int = 20) -> None:
    """Horizontal bars of the top-N permutation importances."""
    top = imp.head(top).iloc[::-1]
    _, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["importance"], xerr=top["std"])
    ax.set_xlabel("mean drop in log loss when shuffled (2025/26)")
    ax.set_title("Permutation importance: LightGBM on P4")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "importance_lgbm_P4.png", dpi=150)
    plt.close()


def write_ablation(results: pd.DataFrame) -> pd.DataFrame:
    """Mean log loss per base model/set plus P1->P4 step deltas."""
    means = results[results["model"].isin(BASE_MODELS)].groupby(
        ["model", "feature_set"])["log_loss"].mean().unstack("feature_set")
    means = means.reindex(BASE_MODELS)[SETS]
    abl = means.copy()
    abl["d_P2-P1"] = means["P2"] - means["P1"]
    abl["d_P3-P2"] = means["P3"] - means["P2"]
    abl["d_P4-P3"] = means["P4"] - means["P3"]
    abl.round(4).to_csv(ABLATION_PATH)
    return abl


def main() -> None:
    """Entry point for `make plots` / `python -m epl.plots`."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(RESULTS_PATH)
    feats = pd.read_parquet(FEATURES_PATH)
    model, fset = best_model(results)
    tag = f"{model}/{fset}"
    print(f"best model by mean log loss: {tag}")
    plot_logloss_bar(results)
    y, proba = collect_test_proba(feats, model, fset)
    logged = results[(results["model"] == model)
                     & (results["feature_set"] == fset)]["log_loss"].mean()
    print(f"recollected LL={log_loss_score(y, proba):.4f} vs logged {logged:.4f}")
    plot_calibration(y, proba, tag)
    plot_confusion(y, proba, tag)
    imp = lgbm_p4_importance(feats)
    plot_importance(imp)
    print("top-5 LGBM/P4 features:")
    print(imp.head(5).round(4).to_string(index=False))
    abl = write_ablation(results)
    print("ablation (mean log loss):")
    print(abl.round(4).to_string())
    print(f"wrote {sorted(p.name for p in FIG_DIR.glob('*.png'))} + {ABLATION_PATH}")


if __name__ == "__main__":
    main()
