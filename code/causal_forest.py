"""
causal_forest.py
----------------
Heterogeneous Treatment Effects (HTE) analysis using Causal Forest
from econml, extending the DML analysis to explore how the effect of
R&D on productivity varies across country characteristics.

Input:  ../data/processed/analysis_crosssection.csv
Output: figures + tables in ../paper/figures/
"""

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

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

# Controls (level terms only)
controls = [c for c in cs.columns
            if c.startswith("mean_") and c not in
            [treatment, outcome, "mean_gdppc_growth"]
            and not c.endswith("_sq")]
if "initial_ln_gdppc" in cs.columns:
    controls.append("initial_ln_gdppc")
for gc in ["governance_pc1", "governance_pc2"]:
    if gc in cs.columns and gc not in controls:
        controls.append(gc)

# Clean sample
cols = [outcome, treatment] + controls + ["iso3c", "country"]
sample = cs[cols].dropna(subset=[outcome, treatment] + controls).copy()
print(f"  Sample: {len(sample)} countries, {len(controls)} controls")

Y = sample[outcome].values
T = sample[treatment].values
X = sample[controls].values

# ── 2. Causal Forest via econml ──────────────────────────────────────────────
print("\nFitting Causal Forest ...")
from econml.dml import CausalForestDML

cf = CausalForestDML(
    model_y=GradientBoostingRegressor(
        n_estimators=500, max_depth=3, learning_rate=0.05,
        min_samples_leaf=5, subsample=0.8, random_state=42,
    ),
    model_t=GradientBoostingRegressor(
        n_estimators=500, max_depth=3, learning_rate=0.05,
        min_samples_leaf=5, subsample=0.8, random_state=42,
    ),
    n_estimators=2000,
    min_samples_leaf=5,
    max_depth=None,
    random_state=42,
    cv=5,
)
cf.fit(Y, T, X=X)

# ── 3. Individual Treatment Effects ─────────────────────────────────────────
print("Estimating individual treatment effects ...")
tau = cf.effect(X)
tau_ci = cf.effect_interval(X, alpha=0.05)

sample["cate"] = tau
sample["cate_lower"] = tau_ci[0]
sample["cate_upper"] = tau_ci[1]

print(f"  ATE (Causal Forest): {tau.mean():.6f}")
print(f"  CATE range: [{tau.min():.6f}, {tau.max():.6f}]")
print(f"  Std of CATE: {tau.std():.6f}")

# ── 4. Feature importance ───────────────────────────────────────────────────
print("\nFeature importances ...")
importances = cf.feature_importances_
feat_imp = pd.DataFrame({
    "feature": controls,
    "importance": importances,
}).sort_values("importance", ascending=False)
print(feat_imp.head(10).to_string(index=False))

# Plot feature importances
fig, ax = plt.subplots(figsize=(8, 6))
top_n = min(12, len(feat_imp))
top = feat_imp.head(top_n)
ax.barh(range(top_n), top["importance"].values, color="steelblue")
ax.set_yticks(range(top_n))
ax.set_yticklabels(top["feature"].values)
ax.invert_yaxis()
ax.set_xlabel("Feature Importance")
ax.set_title("Causal Forest: Feature Importances for Treatment Heterogeneity")
plt.tight_layout()
plt.savefig(FIG / "cf_feature_importance.pdf", dpi=300, bbox_inches="tight")
plt.close()
print(f"  Figure saved: {FIG / 'cf_feature_importance.pdf'}")

# ── 5. HTE by development level ─────────────────────────────────────────────
print("\nHTE by development level ...")
# Classify countries by initial income
median_gdppc = sample["initial_ln_gdppc"].median()
sample["income_group"] = np.where(
    sample["initial_ln_gdppc"] >= median_gdppc, "High Income", "Low Income"
)

hte_groups = sample.groupby("income_group").agg(
    mean_cate=("cate", "mean"),
    std_cate=("cate", "std"),
    n=("cate", "count"),
    mean_rd=("mean_rd_gdp", "mean"),
).reset_index()
hte_groups["se_cate"] = hte_groups["std_cate"] / np.sqrt(hte_groups["n"])
print(hte_groups.to_string(index=False))

# Plot CATE by income group
fig, ax = plt.subplots(figsize=(7, 5))
colors = ["#2166ac", "#b2182b"]
for i, (_, row) in enumerate(hte_groups.iterrows()):
    ax.bar(i, row["mean_cate"], yerr=1.96*row["se_cate"],
           color=colors[i], capsize=8, width=0.6, alpha=0.8)
ax.set_xticks([0, 1])
ax.set_xticklabels(hte_groups["income_group"].values)
ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
ax.set_ylabel("CATE (Effect of R&D on TFP Growth)")
ax.set_title("Heterogeneous Effects by Development Level")
plt.tight_layout()
plt.savefig(FIG / "cf_hte_income.pdf", dpi=300, bbox_inches="tight")
plt.close()
print(f"  Figure saved: {FIG / 'cf_hte_income.pdf'}")

# ── 6. CATE scatter vs initial GDP ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
sc = ax.scatter(
    sample["initial_ln_gdppc"], sample["cate"],
    c=sample["mean_rd_gdp"], cmap="viridis", s=60, alpha=0.7, edgecolors="gray"
)
plt.colorbar(sc, ax=ax, label="R&D (% GDP)")
ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)
ax.set_xlabel("Initial Log GDP per capita (2000)")
ax.set_ylabel("CATE: Effect of R&D on TFP Growth")
ax.set_title("Causal Forest: Heterogeneous Treatment Effects")

# Label notable countries
for _, row in sample.iterrows():
    if row["iso3c"] in ["USA", "CHN", "DEU", "JPN", "KOR", "ITA", "GBR",
                         "ISR", "FIN", "IND", "BRA", "NGA", "ETH"]:
        ax.annotate(row["iso3c"], (row["initial_ln_gdppc"], row["cate"]),
                   fontsize=7, alpha=0.7, xytext=(3, 3),
                   textcoords="offset points")

plt.tight_layout()
plt.savefig(FIG / "cf_cate_scatter.pdf", dpi=300, bbox_inches="tight")
plt.close()
print(f"  Figure saved: {FIG / 'cf_cate_scatter.pdf'}")

# ── 7. Summary table ────────────────────────────────────────────────────────
# Top & bottom CATE countries
top10 = sample.nlargest(10, "cate")[["iso3c", "country", "cate", "cate_lower",
                                      "cate_upper", "mean_rd_gdp", "initial_ln_gdppc"]]
bot10 = sample.nsmallest(10, "cate")[["iso3c", "country", "cate", "cate_lower",
                                       "cate_upper", "mean_rd_gdp", "initial_ln_gdppc"]]

print("\nTop 10 countries (highest CATE):")
print(top10.to_string(index=False))
print("\nBottom 10 countries (lowest CATE):")
print(bot10.to_string(index=False))

# Save results
sample[["iso3c", "country", "cate", "cate_lower", "cate_upper",
        "mean_rd_gdp", "initial_ln_gdppc", "income_group"]].to_csv(
    DATA / "causal_forest_cate.csv", index=False
)
feat_imp.to_csv(DATA / "causal_forest_importance.csv", index=False)

print("\n✓ Causal Forest analysis complete.")
