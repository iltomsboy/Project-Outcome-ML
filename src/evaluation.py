"""
Evaluation of the five models on the simulated project data.

- Stratified train/test split + stratified K-fold cross-validation (on train only)
- Metrics: ROC-AUC, PR-AUC, F1, accuracy, confusion matrix
- Calibration: Brier score + calibration plot (out-of-fold predictions)
- Feature importance: permutation importance (all models) + SHAP (XGBoost),
  compared with the TRUE weights used in the simulation
- Final comparison table, with two reference rows:
    * constant baseline (always predicts the base rate)
    * oracle (true probabilities from the simulation = theoretical ceiling)

Outputs are saved in reports/ (figures + CSV tables). Run from the repo root:
    python src/evaluation.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures to file, no pop-up windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.calibration import CalibrationDisplay
from sklearn.inspection import permutation_importance
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, average_precision_score,
                             brier_score_loss, confusion_matrix, f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, train_test_split

from models import FEATURES, NUMERIC_FEATURES, TARGET, get_models
from simulated_data import (PROJECT_TYPE_EFFECT, TRUE_WEIGHTS, TYPE_PROBS,
                            simulate_project_data)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "simulated_data.csv"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"

SEED = 42
N_SAMPLES = 1500      # must match the size/seed used to create the CSV (needed for the oracle)
TEST_SIZE = 0.2
N_FOLDS = 5
THRESHOLD = 0.5       # decision threshold for F1 / accuracy / confusion matrix


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true, p, threshold=THRESHOLD):
    """Threshold-free (ROC-AUC, PR-AUC, Brier) and threshold-based metrics."""
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "Test ROC-AUC": roc_auc_score(y_true, p),
        "Test PR-AUC": average_precision_score(y_true, p),
        "Test F1": f1_score(y_true, pred, zero_division=0),
        "Test Accuracy": accuracy_score(y_true, pred),
        "Test Brier": brier_score_loss(y_true, p),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    }


def oracle_probabilities(df):
    """True P(success) from the simulation (None if the CSV does not match seed/size)."""
    sim, true_p = simulate_project_data(N_SAMPLES, SEED, return_true_prob=True)
    if len(sim) != len(df) or not np.array_equal(sim[TARGET].to_numpy(), df[TARGET].to_numpy()):
        print("WARNING: CSV does not match the simulation seed/size -> oracle row skipped.")
        return None
    return pd.Series(true_p, index=df.index)


def evaluate_models(models, X_tr, X_te, y_tr, y_te, true_p_test=None):
    """CV on train, then fit on train and score on test. Returns fitted models, test probs, table."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    rows, fitted, test_probs = {}, {}, {}

    for name, model in models.items():
        cv = cross_validate(model, X_tr, y_tr, cv=skf, scoring="roc_auc")["test_score"]
        model.fit(X_tr, y_tr)
        p = model.predict_proba(X_te)[:, 1]
        fitted[name], test_probs[name] = model, p
        rows[name] = {"CV ROC-AUC": f"{cv.mean():.3f} ± {cv.std():.3f}", **compute_metrics(y_te, p)}

    # Reference rows
    base_rate = y_tr.mean()
    rows["Baseline (constant)"] = {"CV ROC-AUC": "-", **compute_metrics(y_te, np.full(len(y_te), base_rate))}
    if true_p_test is not None:
        rows["Oracle (true prob.)"] = {"CV ROC-AUC": "-", **compute_metrics(y_te, true_p_test.to_numpy())}

    return fitted, test_probs, pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# Plots: confusion matrices + calibration
# ---------------------------------------------------------------------------
def plot_confusion_matrices(test_probs, y_te):
    fig, axes = plt.subplots(1, len(test_probs), figsize=(3.6 * len(test_probs), 3.6))
    for ax, (name, p) in zip(axes, test_probs.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_te, (p >= THRESHOLD).astype(int), ax=ax, colorbar=False, cmap="Blues",
            display_labels=["failure", "success"])
        ax.set_title(name)
    fig.suptitle(f"Confusion matrices (test set, threshold = {THRESHOLD})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_matrices.png", dpi=150)
    plt.close(fig)


def plot_calibration(models, X, y):
    """Calibration curves from 5-fold OUT-OF-FOLD predictions (1500 points: more stable
    than the 300-row test set). Returns the out-of-fold Brier score per model."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    briers = {}
    for name, model in models.items():
        p = cross_val_predict(model, X, y, cv=skf, method="predict_proba")[:, 1]
        briers[name] = brier_score_loss(y, p)
        CalibrationDisplay.from_predictions(
            y, p, n_bins=10, strategy="quantile", ax=ax, ref_line=False,
            name=f"{name} (Brier {briers[name]:.3f})")
    ax.set_title("Calibration curves (out-of-fold predictions, 10 quantile bins)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "calibration.png", dpi=150)
    plt.close(fig)
    return pd.Series(briers, name="OOF Brier")


# ---------------------------------------------------------------------------
# Feature importance vs TRUE simulation weights
# ---------------------------------------------------------------------------
def true_importance():
    """True |effect| of each feature on the logit (per 1 SD), from the simulation.
    team_size: |quadratic weight|; project_type: weighted SD of the category effects."""
    w = TRUE_WEIGHTS
    p, eff = np.array(TYPE_PROBS), np.array(list(PROJECT_TYPE_EFFECT.values()))
    type_sd = np.sqrt(np.sum(p * eff ** 2) - np.sum(p * eff) ** 2)
    imp = {
        "budget": w["budget"], "manager_experience": w["manager_experience"],
        "team_experience_avg": w["team_experience_avg"], "duration": abs(w["duration"]),
        "team_size": abs(w["team_size_sq"]), "planned_roi": w["planned_roi"],
        "pct_personnel": w["pct_personnel"], "pct_equipment": abs(w["pct_equipment"]),
        "project_type": type_sd,
    }
    return pd.Series(imp).reindex(FEATURES)


def _share(s):
    """Normalise importances to shares summing to 1 (negative values clipped to 0)."""
    s = s.clip(lower=0)
    return s / s.sum()


def permutation_table(fitted, X_te, y_te):
    """Permutation importance (drop in ROC-AUC on the test set) for every model.
    Works on the raw columns, so project_type is permuted as ONE feature."""
    out = {}
    for name, model in fitted.items():
        r = permutation_importance(model, X_te, y_te, scoring="roc_auc",
                                   n_repeats=20, random_state=SEED)
        out[name] = pd.Series(r.importances_mean, index=X_te.columns)
    return pd.DataFrame(out).reindex(FEATURES)


def shap_xgboost(model, X_te):
    """Mean |SHAP| per feature for XGBoost (logit units); saves the SHAP summary plot.
    SHAP values of the one-hot dummies are summed back into `project_type`."""
    prep, clf = model.named_steps["prep"], model.named_steps["clf"]
    raw_names = prep.get_feature_names_out()
    clean = ["project_type" if n.startswith("cat__project_type") else n.replace("num__", "")
             for n in raw_names]
    Xt = pd.DataFrame(prep.transform(X_te), columns=raw_names)
    sv = shap.TreeExplainer(clf).shap_values(Xt)

    grouped = pd.DataFrame(sv, columns=clean).T.groupby(level=0).sum().T
    importance = grouped.abs().mean().reindex(FEATURES)

    shap.summary_plot(sv, Xt.set_axis(raw_names, axis=1), show=False)
    plt.title("SHAP summary - XGBoost (test set)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_summary_xgboost.png", dpi=150)
    plt.close()
    return importance


def plot_importance_vs_true(imp_table, true_imp):
    """One panel per model: bars = normalised learned importance, diamonds = true weights.
    Returns Spearman rank correlation between learned and true importance."""
    true_share = _share(true_imp)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    y_pos, rhos = np.arange(len(FEATURES)), {}
    for ax, (name, s) in zip(axes.ravel(), imp_table.items()):
        share = _share(s).reindex(FEATURES)
        rhos[name] = spearmanr(share, true_share.reindex(FEATURES)).statistic
        ax.barh(y_pos, share, color="tab:blue", alpha=0.7)
        ax.scatter(true_share.reindex(FEATURES), y_pos, color="k", marker="D", zorder=3,
                   label="True (simulation)")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(FEATURES)
        ax.invert_yaxis()
        ax.set_title(f"{name}  (Spearman ρ = {rhos[name]:.2f})")
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("Learned feature importance (share of total) vs true simulation weights")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "importance_vs_true.png", dpi=150)
    plt.close(fig)
    return pd.Series(rhos, name="Spearman rho vs true")


def lr_coefficients_vs_true(lr_model):
    """Logistic-regression coefficients on standardised features are on the same scale as the
    simulation weights (logit per 1 SD) -> direct comparison (numeric features only)."""
    prep, clf = lr_model.named_steps["prep"], lr_model.named_steps["clf"]
    coefs = pd.Series(clf.coef_[0], index=prep.get_feature_names_out())
    w = TRUE_WEIGHTS
    true_linear = {"budget": w["budget"], "manager_experience": w["manager_experience"],
                   "team_experience_avg": w["team_experience_avg"], "duration": w["duration"],
                   "team_size": 0.0,  # true effect is quadratic (inverted-U): no linear part
                   "planned_roi": w["planned_roi"], "pct_personnel": w["pct_personnel"],
                   "pct_equipment": w["pct_equipment"]}
    return pd.DataFrame({
        "LR coef (per SD)": [coefs[f"num__{f}"] for f in NUMERIC_FEATURES],
        "True weight": [true_linear[f] for f in NUMERIC_FEATURES],
    }, index=NUMERIC_FEATURES)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    X, y = df[FEATURES], df[TARGET]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    print(f"Train: {len(X_tr)} rows (success rate {y_tr.mean():.3f}) | "
          f"Test: {len(X_te)} rows (success rate {y_te.mean():.3f})")

    true_p = oracle_probabilities(df)
    models = get_models(SEED)
    fitted, test_probs, table = evaluate_models(
        models, X_tr, X_te, y_tr, y_te,
        true_p_test=None if true_p is None else true_p.loc[y_te.index])

    # 1) Comparison table
    cols = ["CV ROC-AUC", "Test ROC-AUC", "Test PR-AUC", "Test F1", "Test Accuracy", "Test Brier"]
    print("\n=== Model comparison (CV on train, metrics on test; threshold 0.5) ===")
    print(table[cols].to_string(float_format=lambda v: f"{v:.3f}"))
    table.to_csv(REPORT_DIR / "model_comparison.csv")

    # 2) Confusion matrices + calibration
    plot_confusion_matrices(test_probs, y_te)
    oof_brier = plot_calibration(get_models(SEED), X, y)
    print("\n=== Out-of-fold Brier score (lower = better calibrated/more accurate) ===")
    print(oof_brier.round(3).to_string())

    # 3) Feature importance vs true weights
    imp = permutation_table(fitted, X_te, y_te)
    imp["XGBoost (SHAP)"] = shap_xgboost(fitted["XGBoost"], X_te)
    true_imp = true_importance()
    rhos = plot_importance_vs_true(imp, true_imp)
    print("\n=== Rank agreement with true weights (Spearman) ===")
    print(rhos.round(2).to_string())
    pd.concat([imp, true_imp.rename("True |weight|")], axis=1).to_csv(REPORT_DIR / "feature_importance.csv")

    print("\n=== Logistic Regression coefficients vs true weights (logit per 1 SD) ===")
    print(lr_coefficients_vs_true(fitted["Logistic Regression"]).round(2).to_string())

    print(f"\nSaved tables and figures in {REPORT_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()