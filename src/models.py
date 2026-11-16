"""
Five classifiers for the "project success" task, all usable with the
categorical feature `project_type` and all exposing predict_proba().

1. Naive Bayes         (sklearn GaussianNB, one-hot for project_type, no scaling)
2. Logistic Regression  (scaling + one-hot)
3. Random Forest        (one-hot, no scaling)
4. XGBoost              (one-hot, no scaling)
5. SVM, RBF kernel      (scaling + one-hot, probability estimates)
"""

from pathlib import Path

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

NUMERIC_FEATURES = [
    "budget", "pct_personnel", "pct_equipment", "duration",
    "team_size", "manager_experience", "team_experience_avg", "planned_roi",
]
CATEGORICAL_FEATURES = ["project_type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "success"


# ---------------------------------------------------------------------------
# PREPROCESSING + MODELS (all sklearn / xgboost pipelines)
# ---------------------------------------------------------------------------
def _scaled_preprocessor():
    """Scaling + one-hot (needed by LR and SVM)."""
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ])


def _tree_preprocessor():
    """One-hot only, numeric features pass through unscaled (trees are scale-invariant)."""
    return ColumnTransformer([
        ("num", "passthrough", NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ])


def get_models(random_state=42):
    """Return {model name: unfitted estimator}. All support fit / predict_proba."""
    return {
        "Naive Bayes": Pipeline([
            ("prep", _tree_preprocessor()),  # one-hot only; Gaussian NB is scale-invariant
            ("clf", GaussianNB()),
        ]),
        "Logistic Regression": Pipeline([
            ("prep", _scaled_preprocessor()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000)),
        ]),
        "Random Forest": Pipeline([
            ("prep", _tree_preprocessor()),
            ("clf", RandomForestClassifier(
                n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=random_state)),
        ]),
        "XGBoost": Pipeline([
            ("prep", _tree_preprocessor()),
            ("clf", XGBClassifier(
                n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, eval_metric="logloss", random_state=random_state)),
        ]),
        "SVM (RBF)": Pipeline([
            ("prep", _scaled_preprocessor()),
            # Platt scaling (sigmoid) fitted with internal 5-fold CV -> calibrated probabilities.
            # Replaces SVC(probability=True), deprecated in recent scikit-learn versions.
            ("clf", CalibratedClassifierCV(
                SVC(kernel="rbf", C=1.0, gamma="scale", random_state=random_state),
                method="sigmoid", cv=5, ensemble=False)),
        ]),
    }


if __name__ == "__main__":
    # Quick smoke test only; the real evaluation lives in evaluation.py
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    data_path = Path(__file__).resolve().parents[1] / "data" / "simulated_data.csv"
    df = pd.read_csv(data_path)
    X_tr, X_te, y_tr, y_te = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.2, stratify=df[TARGET], random_state=42)

    for name, model in get_models().items():
        model.fit(X_tr, y_tr)
        auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
        print(f"{name:<20} ROC-AUC (test): {auc:.3f}")