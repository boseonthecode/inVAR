# inVAR

Predict English Premier League results (H/D/A) from pre-match information
only, and compare honestly against bookmaker odds. Every feature for match M
uses only matches dated strictly before M; evaluation is walk-forward by
season (train earlier, tune on the season before the test season, score the
test season once).

## Reproduce

```bash
pip install -e ".[dev]"
make all          # data -> features -> train -> eval -> plots
python -m pytest  # 10 checks (data + no-leakage)
```

`make data` downloads the E0 season CSVs (2009/10–2025/26) from
football-data.co.uk into `data/raw` (read-only), writes
`data/processed/matches.parquet` (6,460 rows) plus a data card.
`make train` runs the full model grid (~50 min, mostly SVMs); per-model
chunks work too, e.g. `python -m epl.models --models logreg --sets P1`.
`python -m epl.leaky` is a one-off leakage demonstration (not part of
`make all`). Dependencies are pinned in `pyproject.toml`; all seeds are 42.
Detailed specs live in `docs/` (local-only, not committed).

## Results (mean over 12 walk-forward test seasons, 2014/15–2025/26)

| model | set | log_loss | accuracy |
| --- | --- | --- | --- |
| bookmaker | none | 0.9611 | 0.5478 |
| logreg | P1 | 0.9804 | 0.5384 |
| mlp | P1 | 0.9834 | 0.5357 |
| logreg | P2 | 0.9850 | 0.5351 |
| logreg | P3 | 0.9878 | 0.5384 |
| logreg | P4 | 0.9909 | 0.5366 |
| mlp | P2 | 1.0004 | 0.5202 |
| svm_rbf | P3 | 1.0042 | 0.5303 |
| svm_rbf | P2 | 1.0066 | 0.5289 |
| svm_rbf | P4 | 1.0063 | 0.5316 |
| mlp | P3 | 1.0064 | 0.5235 |
| svm_rbf | P1 | 1.0093 | 0.5311 |
| mlp | P4 | 1.0211 | 0.5116 |
| lgbm | P1 | 1.0262 | 0.5134 |
| svm_poly | P3 | 1.0260 | 0.4868 |
| svm_poly | P1 | 1.0301 | 0.4904 |
| svm_poly | P2 | 1.0328 | 0.4785 |
| lgbm | P4 | 1.0367 | 0.5092 |
| lgbm | P3 | 1.0415 | 0.5154 |
| lgbm | P2 | 1.0424 | 0.5042 |
| class_priors | none | 1.0685 | 0.4441 |
| svm_poly | P4 | 1.0804 | 0.4884 |
| always_home | none | 2.5691 | 0.4441 |

No model beats the bookmaker on log loss or accuracy. The simplest feature
set (P1, Elo only) wins or ties for nearly every model; the P1→P4 ablation
(`reports/ablation.csv`) shows added form/shot/context features hurting more
often than helping. Full per-season rows: `reports/results.csv`.

![Mean log loss by model and feature set](reports/figures/logloss_bar.png)
![Calibration of logreg/P1](reports/figures/calibration_best.png)
![Confusion matrix of logreg/P1](reports/figures/confusion_best.png)
![Permutation importance, LightGBM on P4](reports/figures/importance_lgbm_P4.png)

Draw handling: base models almost never predict draws (draw recall 0–12%).
Class-weighted variants (`logreg_draww`, `lgbm_draww`) lift draw recall to
33–54% at a ~8-point accuracy cost; the `|pH−pA| < margin` decision rule
(margin tuned on the validation season) adds small recall gains for free.
See `reports/results.csv` (`*_drawrule`, `*_draww` rows).

## Leakage discussion

Guards: date-ordered Elo/rolling computation with same-date batching (same-day
kickoffs never see each other); median imputation and scaling fit on training
seasons only; hyperparameters tuned on the validation season, never the test
season; bookmaker odds used as a benchmark, never a training feature;
`tests/test_no_leakage.py` enforces truncation invariance (deleting later
matches leaves earlier features unchanged), same-date isolation, and clean
debut matches. Tripwires held: best accuracy 0.5478 (bookmaker) < 0.60, and
no model's log loss went below the bookmaker's 0.9611.

Leaky replication (`leaky_random80` row, `python -m epl.leaky`): random 80/20
split with full-season-average team features scores **0.5642** accuracy —
2.6 points above the honest walk-forward logreg/P1 (0.5384). The premium
comes from two future leaks: season averages that include the match itself
and later matches, plus shuffled time splits. (The paper's reported 58.89%
uses richer features; ours is a minimal demonstration of the mechanism, not
a like-for-like comparison.)

## Limitations

- Single league (EPL only); no test of transferability.
- No lineups, injuries, suspensions, or managerial changes.
- No expected-goals data for most seasons (shots/SoT used as proxies).
- Closing odds excluded from features by design; only B365/market-average
  pre-match odds used, as a benchmark.
- 380 matches per test season ⇒ single-season accuracy SE ≈ 2.5 points;
  gaps smaller than that across seasons are noise.
