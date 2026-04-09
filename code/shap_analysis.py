"""
shap_analysis.py
----------------
SHAP (SHapley Additive exPlanations) analysis of the DML nuisance functions
to understand which confounders are most important for predicting R&D
expenditure and TFP growth.

Input:  ../data/processed/analysis_crosssection.csv
Output: SHAP figures in ../paper/figures/
"""

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
FIG  = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# ── 1. Load data ─────────────────────────────────────────────────────────────
print("Loading data ...")
cs = pd.read_csv(DATA / "analysis_crosssection.csv")

treatment = "mean_rd_gdp"
outcome   = "mean_tfp_growth"

controls = [c for c in cs.columns
            if c.startswith("mean_") and c not in
            [treatment, outcome, "mean_gdppc_growth"]
            and not c.endswith("_sq")]
if "initial_ln_gdppc" in cs.columns:
    controls.append("initial_ln_gdppc")
for gc in ["governance_pc1", "governance_pc2"]:
    if gc in cs.columns and gc not in controls:
        controls.append(gc)

cols = [outcome, treatment] + controls
sample = cs[cols].dropna().copy()
print(f"  Sample: {len(sample)} countries, {len(controls)} controls")

X = sample[controls]
D = sample[treatment]
Y = sample[outcome]

# Cleaner feature names for plots
name_map = {
    "mean_hc": "Human Capital",
    "mean_inv_share": "Investment Share",
    "mean_gov_share": "Govt. Consumption",
    "mean_trade_openness": "Trade Openness",
    "mean_inflation": "Inflation",
    "mean_fdi_gdp": "FDI (% GDP)",
    "mean_credit_gdp": "Credit (% GDP)",
    "mean_pop_growth": "Pop. Growth",
    "mean_educ_exp_gdp": "Education Exp.",
    "mean_internet_users": "Internet Users",
    "mean_urban_pop": "Urban Pop.",
    "mean_voice_accountability": "Voice & Account.",
    "mean_pol_stability": "Political Stability",
    "mean_gov_effectiveness": "Govt. Effectiveness",
    "mean_reg_quality": "Regulatory Quality",
    "mean_rule_of_law": "Rule of Law",
    "mean_control_corruption": "Control Corruption",
    "initial_ln_gdppc": "Initial GDP p.c.",
    "governance_pc1": "Governance (PC1)",
    "governance_pc2": "Governance (PC2)",
    "mean_gdppc_growth": "GDP p.c. Growth",
}

X_display = X.rename(columns=name_map)

# ── 2. Fit GBM for outcome nuisance (E[Y|X]) ───────────────────────────────
print("\nFitting outcome model E[Y|X] ...")
gbm_y = GradientBoostingRegressor(
    n_estimators=500, max_depth=3, learning_rate=0.05,
    min_samples_leaf=5, subsample=0.8, random_state=42,
)
gbm_y.fit(X, Y)
r2_y = cross_val_score(gbm_y, X, Y, cv=5, scoring="r2").mean()
print(f"  CV R² (outcome): {r2_y:.4f}")

# ── 3. Fit GBM for treatment nuisance (E[D|X]) ─────────────────────────────
print("Fitting treatment model E[D|X] ...")
gbm_d = GradientBoostingRegressor(
    n_estimators=500, max_depth=3, learning_rate=0.05,
    min_samples_leaf=5, subsample=0.8, random_state=42,
)
gbm_d.fit(X, D)
r2_d = cross_val_score(gbm_d, X, D, cv=5, scoring="r2").mean()
print(f"  CV R² (treatment): {r2_d:.4f}")

# ── 4. SHAP values for outcome model ────────────────────────────────────────
print("\nComputing SHAP values for outcome model ...")
explainer_y = shap.TreeExplainer(gbm_y)
shap_values_y = explainer_y.shap_values(X)

# Summary plot
fig, ax = plt.subplots(figsize=(10, 7))
shap.summary_plot(shap_values_y, X_display, show=False, max_display=15)
plt.title("SHAP Values: Outcome Model E[TFP Growth | X]", fontsize=12)
plt.tight_layout()
plt.savefig(FIG / "shap_outcome.pdf", dpi=300, bbox_inches="tight")
plt.close()
print(f"  Figure saved: {FIG / 'shap_outcome.pdf'}")

# Bar plot (mean absolute SHAP)
fig, ax = plt.subplots(figsize=(8, 6))
shap.summary_plot(shap_values_y, X_display, plot_type="bar", show=False,
                  max_display=15)
plt.title("Mean |SHAP|: Outcome Model", fontsize=12)
plt.tight_layout()
plt.savefig(FIG / "shap_outcome_bar.pdf", dpi=300, bbox_inches="tight")
plt.close()
print(f"  Figure saved: {FIG / 'shap_outcome_bar.pdf'}")

# ── 5. SHAP values for treatment model ──────────────────────────────────────
print("Computing SHAP values for treatment model ...")
explainer_d = shap.TreeExplainer(gbm_d)
shap_values_d = explainer_d.shap_values(X)

fig, ax = plt.subplots(figsize=(10, 7))
shap.summary_plot(shap_values_d, X_display, show=False, max_display=15)
plt.title("SHAP Values: Treatment Model E[R&D | X]", fontsize=12)
plt.tight_layout()
plt.savefig(FIG / "shap_treatment.pdf", dpi=300, bbox_inches="tight")
plt.close()
print(f"  Figure saved: {FIG / 'shap_treatment.pdf'}")

fig, ax = plt.subplots(figsize=(8, 6))
shap.summary_plot(shap_values_d, X_display, plot_type="bar", show=False,
                  max_display=15)
plt.title("Mean |SHAP|: Treatment Model", fontsize=12)
plt.tight_layout()
plt.savefig(FIG / "shap_treatment_bar.pdf", dpi=300, bbox_inches="tight")
plt.close()
print(f"  Figure saved: {FIG / 'shap_treatment_bar.pdf'}")

# ── 6. Save SHAP importance table ───────────────────────────────────────────
shap_imp = pd.DataFrame({
    "feature": controls,
    "feature_label": [name_map.get(c, c) for c in controls],
    "shap_outcome": np.abs(shap_values_y).mean(axis=0),
    "shap_treatment": np.abs(shap_values_d).mean(axis=0),
}).sort_values("shap_outcome", ascending=False)

print("\nSHAP Feature Importance Table:")
print(shap_imp.to_string(index=False))

shap_imp.to_csv(DATA / "shap_importance.csv", index=False)

# ── 7. Summary stats ────────────────────────────────────────────────────────
print(f"\n  Nuisance model fit:")
print(f"    E[Y|X] CV R²: {r2_y:.4f}")
print(f"    E[D|X] CV R²: {r2_d:.4f}")

print("\n✓ SHAP analysis complete.")
