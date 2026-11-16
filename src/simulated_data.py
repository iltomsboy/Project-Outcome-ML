"""
Simulation of project-portfolio data for the "project success" prediction task.

Extends the original thesis script (Dataset_Simulation.py):
- more features (budget composition, team experience, planned ROI, project type)
- correlations between features (via a latent "complexity" factor)
- a richer hidden logit model (non-linearity, interaction, latent noise)

All features are known BEFORE the project starts -> no data leakage.
The target `success` is the only column generated from the outcome model.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# TRUE DATA-GENERATING WEIGHTS (hidden from the models)
# Effects are expressed on the logit scale, per +1 standard deviation of the
# (sample-standardised) feature. Keep in sync with the code below: the
# evaluation step compares learned feature importances against these values.
# ---------------------------------------------------------------------------
TRUE_WEIGHTS = {
    "budget": +1.0,
    "manager_experience": +0.8,
    "team_experience_avg": +0.7,
    "duration": -0.6,
    "team_size_sq": -0.3,           # inverted-U: -0.3 * z^2 (optimum at mean team size)
    "planned_roi": +0.15,
    "pct_personnel": +0.10,
    "pct_equipment": -0.10,
    "interaction_long_duration_low_mgr_exp": -0.3,  # 1 if duration z>0 AND manager_exp z<0
}
PROJECT_TYPE_EFFECT = {              # logit shift per category
    "Tech/IT": -0.2,
    "Operations": +0.2,
    "Marketing & Sales": 0.0,
    "HR/Organizational": 0.0,
    "R&D/Product development": -0.3,
}
LATENT_NOISE_SD = 0.7                # unobserved factors -> no model reaches 100%
INTERCEPT = -0.55                    # tuned so the base success rate is ~35-40%

PROJECT_TYPES = list(PROJECT_TYPE_EFFECT)
TYPE_PROBS = [0.25, 0.25, 0.20, 0.15, 0.15]
# Type-specific shift of the latent complexity (R&D / Tech more complex)
TYPE_COMPLEXITY = {"Tech/IT": 0.2, "Operations": -0.3, "Marketing & Sales": 0.0,
                   "HR/Organizational": -0.2, "R&D/Product development": 0.4}
# Dirichlet alphas for (personnel, equipment, other) budget shares per type
TYPE_BUDGET_ALPHA = {"Tech/IT": (5, 3, 4), "Operations": (4, 5, 3),
                     "Marketing & Sales": (4, 1.5, 6), "HR/Organizational": (7, 1, 4),
                     "R&D/Product development": (5, 4, 3)}


def _z(x):
    """Standardise to mean 0, sd 1 (sample statistics)."""
    return (x - x.mean()) / x.std()


def simulate_project_data(n=1500, random_state=42, return_true_prob=False):
    """Simulate n projects. Returns a DataFrame (plus the true P(success) if requested)."""
    rng = np.random.default_rng(random_state)

    # --- Project type and latent complexity (drives correlations) -----------
    project_type = rng.choice(PROJECT_TYPES, size=n, p=TYPE_PROBS)
    complexity = np.array([TYPE_COMPLEXITY[t] for t in project_type]) + rng.normal(0, 1, n)

    # --- Numeric features ----------------------------------------------------
    # budget & duration both grow with complexity (corr ~ 0.25)
    budget = 250 + 80 * (0.5 * complexity + np.sqrt(0.75) * rng.normal(0, 1, n))
    duration = 8 + 3 * (0.5 * complexity + np.sqrt(0.75) * rng.normal(0, 1, n))
    # team size grows with budget (corr ~ 0.6)
    team_size = 10 + 4 * (0.6 * _z(budget) + 0.8 * rng.normal(0, 1, n))
    # team experience correlated with manager experience (corr ~ 0.5)
    manager_experience = 7 + 3 * rng.normal(0, 1, n)
    team_experience_avg = 5 + 2 * (0.5 * _z(manager_experience) + np.sqrt(0.75) * rng.normal(0, 1, n))
    # planned ROI: from the business case, independent of the outcome
    planned_roi = 30 + 15 * rng.normal(0, 1, n)

    # --- Budget composition (Dirichlet => personnel + equipment <= 1) --------
    shares = np.array([rng.dirichlet(TYPE_BUDGET_ALPHA[t]) for t in project_type])
    pct_personnel, pct_equipment = shares[:, 0], shares[:, 1]

    # --- Clip unrealistic values --------------------------------------------
    budget = np.clip(budget, 10, None)
    duration = np.clip(duration, 1, None)
    team_size = np.round(np.clip(team_size, 2, None)).astype(int)
    manager_experience = np.clip(manager_experience, 0, None)
    team_experience_avg = np.clip(team_experience_avg, 0, None)
    planned_roi = np.clip(planned_roi, 0, 80)

    # --- Hidden outcome model ------------------------------------------------
    z_bud, z_dur, z_ts = _z(budget), _z(duration), _z(team_size)
    z_mgr, z_team = _z(manager_experience), _z(team_experience_avg)
    interaction = ((z_dur > 0) & (z_mgr < 0)).astype(float)
    w = TRUE_WEIGHTS

    logit = (
        INTERCEPT
        + w["budget"] * z_bud
        + w["manager_experience"] * z_mgr
        + w["team_experience_avg"] * z_team
        + w["duration"] * z_dur
        + w["team_size_sq"] * z_ts ** 2
        + w["planned_roi"] * _z(planned_roi)
        + w["pct_personnel"] * _z(pct_personnel)
        + w["pct_equipment"] * _z(pct_equipment)
        + w["interaction_long_duration_low_mgr_exp"] * interaction
        + np.array([PROJECT_TYPE_EFFECT[t] for t in project_type])
        + rng.normal(0, LATENT_NOISE_SD, n)          # latent unobserved noise
    )
    prob = 1 / (1 + np.exp(-logit))
    success = rng.binomial(1, prob)

    df = pd.DataFrame({
        "budget": budget,
        "pct_personnel": pct_personnel,
        "pct_equipment": pct_equipment,
        "duration": duration,
        "team_size": team_size,
        "manager_experience": manager_experience,
        "team_experience_avg": team_experience_avg,
        "planned_roi": planned_roi,
        "project_type": project_type,
        "success": success,
    })
    return (df, prob) if return_true_prob else df


if __name__ == "__main__":
    df = simulate_project_data(n=1500)
    print(df.head(7).round(2))
    print("\nClass balance:")
    print(df["success"].value_counts(normalize=True).rename({0: "failure", 1: "success"}).round(3))
    print("\nSuccess rate by project_type:")
    print(df.groupby("project_type")["success"].agg(["mean", "count"]).round(3))

    out_path = Path(__file__).resolve().parents[1] / "data" / "simulated_data.csv"
    out_path.parent.mkdir(exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path.relative_to(out_path.parents[1])}")