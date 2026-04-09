"""
experimental_extensions.py
--------------------------
Experimental approaches to improve nuisance model quality and explore
latent structure in the R&D → TFP growth relationship.

Five extensions:
  1. Super Learner (cross-validated stacking of all base learners)
  2. Factor-Augmented DML (PCA factors instead of raw controls)
  3. Gaussian Process Regression as nuisance learner
  4. Spatial/Peer-Group Lags (regional and income-group averages)
  5. Latent Mixture Regimes (GMM clustering → regime-specific DML)

DISCLAIMER: These extensions are exploratory and should be interpreted
with caution given the small sample size (n ≈ 104). They are included
to probe the limits of what ML methods can extract from cross-country
growth data, not to claim definitive causal estimates.

Input:  ../data/processed/analysis_crosssection.csv
Output: results tables + figures in ../paper/figures/
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

from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    StackingRegressor, HistGradientBoostingRegressor,
)
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.mixture import GaussianMixture
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.base import clone, BaseEstimator, RegressorMixin

import doubleml as dml
from doubleml import DoubleMLPLR, DoubleMLData

import statsmodels.api as sm

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading cross-section data ...")
cs = pd.read_csv(DATA / "analysis_crosssection.csv")

treatment = "mean_rd_gdp"
outcome_tfp = "mean_tfp_growth"
outcome_gdp = "mean_gdppc_growth"

mean_controls_level = [c for c in cs.columns
                       if c.startswith("mean_") and c not in
                       [treatment, outcome_tfp, outcome_gdp]
                       and not c.endswith("_sq")]
if "initial_ln_gdppc" in cs.columns:
    mean_controls_level.append("initial_ln_gdppc")
for gc in ["governance_pc1", "governance_pc2"]:
    if gc in cs.columns and gc not in mean_controls_level:
        mean_controls_level.append(gc)

controls_nonlinear = mean_controls_level


def prepare_sample(df, y_col, d_col, x_cols):
    cols = [y_col, d_col] + x_cols
    cols = [c for c in cols if c in df.columns]
    sample = df[cols].dropna()
    return sample


def run_dml_plr(df, y_col, d_col, x_cols, learner_name, ml_l, ml_m,
                n_folds=5, n_rep=10):
    sample = prepare_sample(df, y_col, d_col, x_cols)
    dml_data = DoubleMLData(sample, y_col=y_col, d_cols=d_col, x_cols=x_cols)
    plr = DoubleMLPLR(dml_data, ml_l=ml_l, ml_m=ml_m,
                      n_folds=n_folds, n_rep=n_rep, score="partialling out")
    plr.fit()
    coef = plr.coef[0]
    se = plr.se[0]
    pval = plr.pval[0]
    ci = plr.confint(level=0.95).values[0]
    return {"method": f"DML-{learner_name}", "coef": coef, "se": se,
            "pval": pval, "ci_lower": ci[0], "ci_upper": ci[1],
            "n": len(sample), "n_controls": len(x_cols)}


# ══════════════════════════════════════════════════════════════════════════════
# EXTENSION 1: SUPER LEARNER (Cross-Validated Stacking)
# ══════════════════════════════════════════════════════════════════════════════
def make_super_learner():
    """
    Create a Super Learner (stacking ensemble) combining all base learners.

    The Super Learner (van der Laan et al., 2007) uses cross-validated stacking
    to find the optimal weighted combination of heterogeneous base learners.
    Instead of picking the single best learner or simple averaging, it learns
    meta-weights via a Ridge meta-regressor using the out-of-fold predictions
    of each base learner.

    Base learners:
      - LASSO, Ridge, Elastic Net (linear, different regularization)
      - Random Forest (bagging, variance reduction)
      - Gradient Boosting (boosting, bias reduction)
      - BART-approx (regularized boosting)

    We exclude Neural Net (too unstable for stacking with n=104) and CART
    (dominated by RF in every scenario).
    """
    base_estimators = [
        ("lasso", Pipeline([
            ("scaler", StandardScaler()),
            ("model", LassoCV(cv=5, max_iter=10000, random_state=42))
        ])),
        ("ridge", Pipeline([
            ("scaler", StandardScaler()),
            ("model", RidgeCV(cv=5))
        ])),
        ("enet", Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNetCV(cv=5, max_iter=10000,
                                   l1_ratio=[0.1, 0.5, 0.7, 0.9],
                                   random_state=42))
        ])),
        ("rf", RandomForestRegressor(
            n_estimators=500, max_features="sqrt", min_samples_leaf=5,
            random_state=42, n_jobs=-1
        )),
        ("gbm", GradientBoostingRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.02,
            min_samples_leaf=10, subsample=0.7, random_state=42
        )),
        ("bart", HistGradientBoostingRegressor(
            max_iter=200, max_depth=2, learning_rate=0.03,
            min_samples_leaf=10, l2_regularization=5.0, random_state=42
        )),
    ]

    # Meta-learner: Ridge regression over the base learner predictions
    # Ridge (not OLS) because base learner predictions are correlated
    super_learner = StackingRegressor(
        estimators=base_estimators,
        final_estimator=RidgeCV(cv=5),
        cv=5,
        n_jobs=-1,
    )
    return super_learner


def run_super_learner(df, y_col):
    """Run DML with Super Learner as nuisance estimator."""
    print("\n" + "=" * 70)
    print(f"EXTENSION 1: SUPER LEARNER — {y_col}")
    print("=" * 70)

    x_cols = [c for c in controls_nonlinear if c in df.columns]
    sample = prepare_sample(df, y_col, treatment, x_cols)
    X = sample[x_cols].values
    Y = sample[y_col].values
    D = sample[treatment].values

    sl = make_super_learner()

    # CV R² diagnostics
    r2_y = cross_val_score(clone(sl), X, Y, cv=5, scoring="r2").mean()
    r2_d = cross_val_score(clone(sl), X, D, cv=5, scoring="r2").mean()
    print(f"  Super Learner CV R²: outcome={r2_y:.4f}, treatment={r2_d:.4f}")

    # DML estimation
    res = run_dml_plr(df, y_col, treatment, x_cols,
                      "Super Learner", clone(sl), clone(sl))
    print(f"  θ = {res['coef']:.6f}  (SE = {res['se']:.6f}, p = {res['pval']:.4f})")

    return {"extension": "Super Learner", "r2_outcome": r2_y,
            "r2_treatment": r2_d, **res}


# ══════════════════════════════════════════════════════════════════════════════
# EXTENSION 2: FACTOR-AUGMENTED DML
# ══════════════════════════════════════════════════════════════════════════════
def run_factor_augmented(df, y_col):
    """
    Run DML with latent factors extracted from the control matrix.

    Instead of using 12 raw (noisy) controls, we extract k latent factors
    that capture the shared structure. The idea: economic development,
    institutional quality, and financial depth are all manifestations of
    a few underlying "development dimensions."

    We use both PCA (maximizes variance explained) and Factor Analysis
    (models the latent structure explicitly). We keep k factors that
    explain ≥90% of the variance, typically 4-5.
    """
    print("\n" + "=" * 70)
    print(f"EXTENSION 2: FACTOR-AUGMENTED DML — {y_col}")
    print("=" * 70)

    x_cols = [c for c in controls_nonlinear if c in df.columns]
    sample = prepare_sample(df, y_col, treatment, x_cols)

    X_raw = sample[x_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Determine number of factors (≥90% variance)
    pca_full = PCA().fit(X_scaled)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_factors = int(np.searchsorted(cumvar, 0.90) + 1)
    n_factors = max(n_factors, 3)  # at least 3
    print(f"  PCA: {n_factors} factors explain {cumvar[n_factors-1]*100:.1f}% of variance")

    # Create factor-augmented dataset
    pca = PCA(n_components=n_factors)
    X_factors = pca.fit_transform(X_scaled)

    factor_names = [f"factor_{i+1}" for i in range(n_factors)]
    sample_fa = sample[[y_col, treatment]].copy()
    for i, fn in enumerate(factor_names):
        sample_fa[fn] = X_factors[:, i]

    results = []

    # --- PCA factors + GBM learner ---
    gbm = GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.02,
        min_samples_leaf=10, subsample=0.7, random_state=42
    )
    Y_fa = sample_fa[y_col].values
    D_fa = sample_fa[treatment].values
    X_fa = sample_fa[factor_names].values

    r2_y = cross_val_score(clone(gbm), X_fa, Y_fa, cv=5, scoring="r2").mean()
    r2_d = cross_val_score(clone(gbm), X_fa, D_fa, cv=5, scoring="r2").mean()
    print(f"  Factor-GBM CV R²: outcome={r2_y:.4f}, treatment={r2_d:.4f}")

    res = run_dml_plr(sample_fa, y_col, treatment, factor_names,
                      "Factor-GBM", clone(gbm), clone(gbm))
    print(f"  θ = {res['coef']:.6f}  (SE = {res['se']:.6f}, p = {res['pval']:.4f})")
    results.append({"extension": "Factor-GBM", "n_factors": n_factors,
                    "r2_outcome": r2_y, "r2_treatment": r2_d, **res})

    # --- PCA factors + RF learner ---
    rf = RandomForestRegressor(
        n_estimators=500, max_features="sqrt", min_samples_leaf=5,
        random_state=42, n_jobs=-1
    )
    r2_y_rf = cross_val_score(clone(rf), X_fa, Y_fa, cv=5, scoring="r2").mean()
    r2_d_rf = cross_val_score(clone(rf), X_fa, D_fa, cv=5, scoring="r2").mean()
    print(f"  Factor-RF  CV R²: outcome={r2_y_rf:.4f}, treatment={r2_d_rf:.4f}")

    res_rf = run_dml_plr(sample_fa, y_col, treatment, factor_names,
                         "Factor-RF", clone(rf), clone(rf))
    print(f"  θ = {res_rf['coef']:.6f}  (SE = {res_rf['se']:.6f}, p = {res_rf['pval']:.4f})")
    results.append({"extension": "Factor-RF", "n_factors": n_factors,
                    "r2_outcome": r2_y_rf, "r2_treatment": r2_d_rf, **res_rf})

    # --- Factor Analysis (explicit latent model) + GBM ---
    try:
        fa = FactorAnalysis(n_components=n_factors, random_state=42)
        X_fa_explicit = fa.fit_transform(X_scaled)

        sample_fa2 = sample[[y_col, treatment]].copy()
        fa_names = [f"latent_{i+1}" for i in range(n_factors)]
        for i, fn in enumerate(fa_names):
            sample_fa2[fn] = X_fa_explicit[:, i]

        Y_fa2 = sample_fa2[y_col].values
        X_fa2 = sample_fa2[fa_names].values
        r2_y_fa = cross_val_score(clone(gbm), X_fa2, Y_fa2, cv=5, scoring="r2").mean()
        r2_d_fa = cross_val_score(clone(gbm), X_fa2, D_fa, cv=5, scoring="r2").mean()
        print(f"  FactorAnalysis-GBM CV R²: outcome={r2_y_fa:.4f}, treatment={r2_d_fa:.4f}")

        res_fa = run_dml_plr(sample_fa2, y_col, treatment, fa_names,
                             "FA-GBM", clone(gbm), clone(gbm))
        print(f"  θ = {res_fa['coef']:.6f}  (SE = {res_fa['se']:.6f}, p = {res_fa['pval']:.4f})")
        results.append({"extension": "FA-GBM", "n_factors": n_factors,
                        "r2_outcome": r2_y_fa, "r2_treatment": r2_d_fa, **res_fa})
    except Exception as e:
        print(f"  Factor Analysis failed: {e}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# EXTENSION 3: GAUSSIAN PROCESS REGRESSION
# ══════════════════════════════════════════════════════════════════════════════
def make_gp_learner():
    """
    Create a Gaussian Process Regressor for nuisance functions.

    GP is a Bayesian nonparametric method ideal for small n:
    - Naturally regularized by the prior
    - Captures smooth nonlinear functions
    - Provides posterior uncertainty (not used here, but available)
    - Computational cost O(n³) — trivial for n=104

    We use a composite kernel: constant × Matérn(ν=2.5) + WhiteKernel.
    The Matérn kernel is more flexible than RBF for economic data (allows
    less smooth functions). WhiteKernel models noise.
    """
    kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1.0)

    gp = Pipeline([
        ("scaler", StandardScaler()),
        ("model", GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            alpha=1e-6,  # numerical stability
            normalize_y=True,
            random_state=42,
        ))
    ])
    return gp


def run_gp_extension(df, y_col):
    """Run DML with Gaussian Process as nuisance estimator."""
    print("\n" + "=" * 70)
    print(f"EXTENSION 3: GAUSSIAN PROCESS — {y_col}")
    print("=" * 70)

    x_cols = [c for c in controls_nonlinear if c in df.columns]
    sample = prepare_sample(df, y_col, treatment, x_cols)
    X = sample[x_cols].values
    Y = sample[y_col].values
    D = sample[treatment].values

    gp = make_gp_learner()

    r2_y = cross_val_score(clone(gp), X, Y, cv=5, scoring="r2").mean()
    r2_d = cross_val_score(clone(gp), X, D, cv=5, scoring="r2").mean()
    print(f"  GP CV R²: outcome={r2_y:.4f}, treatment={r2_d:.4f}")

    res = run_dml_plr(df, y_col, treatment, x_cols,
                      "Gaussian Process", clone(gp), clone(gp))
    print(f"  θ = {res['coef']:.6f}  (SE = {res['se']:.6f}, p = {res['pval']:.4f})")

    return {"extension": "Gaussian Process", "r2_outcome": r2_y,
            "r2_treatment": r2_d, **res}


# ══════════════════════════════════════════════════════════════════════════════
# EXTENSION 4: SPATIAL / PEER-GROUP LAGS
# ══════════════════════════════════════════════════════════════════════════════
# World Bank income groups and regions for peer-group construction
INCOME_GROUPS = {
    "High income": ["AUS", "AUT", "BEL", "CAN", "CHE", "CHL", "CZE", "DEU",
                     "DNK", "ESP", "EST", "FIN", "FRA", "GBR", "GRC", "HKG",
                     "HRV", "HUN", "IRL", "ISL", "ISR", "ITA", "JPN", "KOR",
                     "LTU", "LUX", "LVA", "NLD", "NOR", "NZL", "PAN", "POL",
                     "PRT", "QAT", "SAU", "SGP", "SVK", "SVN", "SWE", "TTO",
                     "URY", "USA", "ARE", "BHR", "BRN", "CYP", "KWT", "MLT",
                     "OMN"],
    "Upper middle": ["ALB", "ARG", "ARM", "AZE", "BGR", "BIH", "BLR", "BRA",
                      "BWA", "CHN", "COL", "CRI", "DOM", "DZA", "ECU", "GAB",
                      "GEO", "GTM", "IDN", "IRN", "IRQ", "JAM", "JOR", "KAZ",
                      "LBN", "LBY", "MEX", "MKD", "MNE", "MUS", "MYS", "NAM",
                      "PER", "PRY", "ROM", "ROU", "RUS", "SRB", "SUR", "THA",
                      "TKM", "TUN", "TUR", "VEN", "ZAF"],
    "Lower middle": ["AGO", "BEN", "BGD", "BOL", "BTN", "CIV", "CMR", "COG",
                      "COM", "CPV", "EGY", "GHA", "GIN", "HND", "HTI", "IND",
                      "KEN", "KGZ", "KHM", "LAO", "LKA", "MAR", "MDA", "MMR",
                      "MNG", "MOZ", "MRT", "NGA", "NIC", "NPL", "PAK", "PHL",
                      "PNG", "SEN", "SLV", "SWZ", "TJK", "TZA", "UGA", "UKR",
                      "UZB", "VNM", "ZMB", "ZWE"],
    "Low income": ["AFG", "BDI", "BFA", "CAF", "COD", "ERI", "ETH", "GMB",
                    "GNB", "LBR", "MDG", "MLI", "MWI", "NER", "RWA", "SDN",
                    "SLE", "SOM", "SSD", "TCD", "TGO", "YEM"],
}

REGIONS = {
    "East Asia & Pacific": ["AUS", "BRN", "CHN", "HKG", "IDN", "JPN", "KHM",
                             "KOR", "LAO", "MMR", "MNG", "MYS", "NZL", "PHL",
                             "PNG", "SGP", "THA", "VNM"],
    "Europe & Central Asia": ["ALB", "ARM", "AUT", "AZE", "BEL", "BGR", "BIH",
                               "BLR", "CHE", "CYP", "CZE", "DEU", "DNK", "ESP",
                               "EST", "FIN", "FRA", "GBR", "GEO", "GRC", "HRV",
                               "HUN", "IRL", "ISL", "ITA", "KAZ", "KGZ", "LTU",
                               "LUX", "LVA", "MDA", "MKD", "MLT", "MNE", "NLD",
                               "NOR", "POL", "PRT", "ROM", "ROU", "RUS", "SRB",
                               "SVK", "SVN", "SWE", "TJK", "TKM", "TUR", "UKR",
                               "UZB"],
    "Latin America": ["ARG", "BOL", "BRA", "CHL", "COL", "CRI", "DOM", "ECU",
                       "GTM", "HND", "HTI", "JAM", "MEX", "NIC", "PAN", "PER",
                       "PRY", "SLV", "SUR", "TTO", "URY", "VEN"],
    "Middle East & North Africa": ["ARE", "BHR", "DZA", "EGY", "IRN", "IRQ",
                                    "ISR", "JOR", "KWT", "LBN", "LBY", "MAR",
                                    "OMN", "QAT", "SAU", "TUN", "YEM"],
    "South Asia": ["AFG", "BGD", "BTN", "IND", "LKA", "NPL", "PAK"],
    "Sub-Saharan Africa": ["AGO", "BDI", "BEN", "BFA", "BWA", "CAF", "CIV",
                            "CMR", "COD", "COG", "COM", "CPV", "ERI", "ETH",
                            "GAB", "GHA", "GIN", "GMB", "GNB", "KEN", "LBR",
                            "MDG", "MLI", "MOZ", "MRT", "MUS", "MWI", "NAM",
                            "NER", "NGA", "RWA", "SDN", "SEN", "SLE", "SOM",
                            "SSD", "SWZ", "TCD", "TGO", "TZA", "UGA", "ZAF",
                            "ZMB", "ZWE"],
    "North America": ["CAN", "USA"],
}


def assign_groups(df):
    """Assign income group and region based on iso3c."""
    iso_to_income = {}
    for group, codes in INCOME_GROUPS.items():
        for c in codes:
            iso_to_income[c] = group

    iso_to_region = {}
    for region, codes in REGIONS.items():
        for c in codes:
            iso_to_region[c] = region

    df = df.copy()
    df["income_group"] = df["iso3c"].map(iso_to_income).fillna("Unknown")
    df["region"] = df["iso3c"].map(iso_to_region).fillna("Unknown")
    return df


def add_peer_lags(df, y_col, d_col, x_cols):
    """
    Add spatial/peer-group lag features.

    For each country, compute the average of key variables among:
      (a) countries in the same income group (excluding self)
      (b) countries in the same region (excluding self)

    These "peer lags" capture technology diffusion, regional shocks,
    and other spatial/network effects that domestic controls miss.
    """
    df = assign_groups(df)

    # Variables to compute peer averages for
    peer_vars = [y_col, d_col, "mean_hc", "mean_inv_share", "initial_ln_gdppc"]
    peer_vars = [v for v in peer_vars if v in df.columns]

    lag_features = []

    for group_col in ["income_group", "region"]:
        for var in peer_vars:
            col_name = f"peer_{group_col[:3]}_{var}"
            # Mean of peers (same group, excluding self)
            group_means = df.groupby(group_col)[var].transform("mean")
            group_counts = df.groupby(group_col)[var].transform("count")
            # Leave-one-out mean: (sum - self) / (count - 1)
            peer_mean = (group_means * group_counts - df[var]) / (group_counts - 1)
            df[col_name] = peer_mean
            lag_features.append(col_name)

    return df, lag_features


def run_spatial_lags(df, y_col):
    """Run DML with peer-group lag features added to controls."""
    print("\n" + "=" * 70)
    print(f"EXTENSION 4: SPATIAL / PEER-GROUP LAGS — {y_col}")
    print("=" * 70)

    x_base = [c for c in controls_nonlinear if c in df.columns]
    df_aug, lag_features = add_peer_lags(df, y_col, treatment, x_base)

    x_augmented = x_base + lag_features
    sample = prepare_sample(df_aug, y_col, treatment, x_augmented)

    X = sample[x_augmented].values
    Y = sample[y_col].values
    D = sample[treatment].values

    print(f"  Added {len(lag_features)} peer-lag features → {len(x_augmented)} total controls")

    # Use GBM (best performer from main analysis)
    gbm = GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.02,
        min_samples_leaf=10, subsample=0.7, random_state=42
    )

    r2_y = cross_val_score(clone(gbm), X, Y, cv=5, scoring="r2").mean()
    r2_d = cross_val_score(clone(gbm), X, D, cv=5, scoring="r2").mean()
    print(f"  Spatial-GBM CV R²: outcome={r2_y:.4f}, treatment={r2_d:.4f}")

    res = run_dml_plr(df_aug, y_col, treatment, x_augmented,
                      "Spatial-GBM", clone(gbm), clone(gbm))
    print(f"  θ = {res['coef']:.6f}  (SE = {res['se']:.6f}, p = {res['pval']:.4f})")

    # Also try RF with spatial features
    rf = RandomForestRegressor(
        n_estimators=500, max_features="sqrt", min_samples_leaf=5,
        random_state=42, n_jobs=-1
    )
    r2_y_rf = cross_val_score(clone(rf), X, Y, cv=5, scoring="r2").mean()
    print(f"  Spatial-RF  CV R²: outcome={r2_y_rf:.4f}")

    res_rf = run_dml_plr(df_aug, y_col, treatment, x_augmented,
                         "Spatial-RF", clone(rf), clone(rf))
    print(f"  θ = {res_rf['coef']:.6f}  (SE = {res_rf['se']:.6f}, p = {res_rf['pval']:.4f})")

    return [
        {"extension": "Spatial-GBM", "r2_outcome": r2_y,
         "r2_treatment": r2_d, **res},
        {"extension": "Spatial-RF", "r2_outcome": r2_y_rf,
         "r2_treatment": r2_d, **res_rf},
    ]


# ══════════════════════════════════════════════════════════════════════════════
# EXTENSION 5: LATENT MIXTURE REGIMES
# ══════════════════════════════════════════════════════════════════════════════
def run_mixture_regimes(df, y_col):
    """
    Identify latent country regimes via Gaussian Mixture Model, then
    run DML separately for each regime.

    The idea: pooling all 104 countries may mask regime-specific effects.
    Maybe R&D helps TFP in "innovation-capable" countries but not in others.
    A GMM on the control variables discovers these regimes without supervision.

    We select the number of clusters k via BIC, then estimate DML-RF
    within each cluster.
    """
    print("\n" + "=" * 70)
    print(f"EXTENSION 5: LATENT MIXTURE REGIMES — {y_col}")
    print("=" * 70)

    x_cols = [c for c in controls_nonlinear if c in df.columns]
    sample = prepare_sample(df, y_col, treatment, x_cols)

    X = sample[x_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Select k via BIC (max 3 to ensure viable regime sizes with n~100-120)
    bics = []
    for k in range(2, 4):
        gmm = GaussianMixture(n_components=k, covariance_type="full",
                               random_state=42, n_init=10)
        gmm.fit(X_scaled)
        bics.append((k, gmm.bic(X_scaled)))
        print(f"  GMM k={k}: BIC={gmm.bic(X_scaled):.1f}")

    best_k = min(bics, key=lambda x: x[1])[0]
    print(f"  Best k={best_k} (by BIC)")

    # Fit best GMM
    gmm = GaussianMixture(n_components=best_k, covariance_type="full",
                           random_state=42, n_init=10)
    labels = gmm.fit_predict(X_scaled)
    sample["regime"] = labels

    results = []

    for regime in range(best_k):
        mask = sample["regime"] == regime
        n_regime = mask.sum()
        print(f"\n  Regime {regime}: n={n_regime}")

        if n_regime < 30:
            print(f"    Too few observations for DML (need ≥30), skipping.")
            # Still report descriptive stats
            regime_data = sample[mask]
            avg_rd = regime_data[treatment].mean()
            avg_y = regime_data[y_col].mean()
            avg_gdp = regime_data["initial_ln_gdppc"].mean() if "initial_ln_gdppc" in regime_data.columns else np.nan
            print(f"    Mean R&D/GDP: {avg_rd:.3f}, Mean {y_col}: {avg_y:.4f}, Mean log GDP p.c.: {avg_gdp:.2f}")
            results.append({
                "extension": f"Regime {regime}",
                "n": n_regime,
                "mean_rd": avg_rd,
                "mean_outcome": avg_y,
                "coef": np.nan,
                "se": np.nan,
                "pval": np.nan,
                "note": "too few obs for DML"
            })
            continue

        regime_sample = sample[mask].copy()
        regime_x = x_cols

        X_r = regime_sample[regime_x].values
        Y_r = regime_sample[y_col].values
        D_r = regime_sample[treatment].values

        # Use RF (more stable with smaller n than GBM)
        rf = RandomForestRegressor(
            n_estimators=500, max_features="sqrt", min_samples_leaf=3,
            random_state=42, n_jobs=-1
        )

        r2_y = cross_val_score(clone(rf), X_r, Y_r, cv=3, scoring="r2").mean()
        r2_d = cross_val_score(clone(rf), X_r, D_r, cv=3, scoring="r2").mean()
        print(f"    RF CV R²: outcome={r2_y:.4f}, treatment={r2_d:.4f}")

        avg_rd = regime_sample[treatment].mean()
        avg_y = regime_sample[y_col].mean()
        avg_gdp = regime_sample["initial_ln_gdppc"].mean() if "initial_ln_gdppc" in regime_sample.columns else np.nan
        print(f"    Mean R&D/GDP: {avg_rd:.3f}, Mean {y_col}: {avg_y:.4f}, Mean log GDP p.c.: {avg_gdp:.2f}")

        try:
            # Use 3 folds for smaller subsamples
            n_folds = 3 if n_regime < 60 else 5
            res = run_dml_plr(regime_sample, y_col, treatment, regime_x,
                              f"Regime-{regime}", clone(rf), clone(rf),
                              n_folds=n_folds, n_rep=10)
            print(f"    θ = {res['coef']:.6f}  (SE = {res['se']:.6f}, p = {res['pval']:.4f})")
            results.append({
                "extension": f"Regime {regime}",
                "r2_outcome": r2_y, "r2_treatment": r2_d,
                "n": n_regime, "mean_rd": avg_rd,
                "mean_outcome": avg_y, "mean_gdp": avg_gdp,
                **res,
            })
        except Exception as e:
            print(f"    DML failed: {e}")
            results.append({
                "extension": f"Regime {regime}",
                "n": n_regime, "coef": np.nan,
                "note": str(e)
            })

    # Characterize regimes
    print("\n  Regime characterization:")
    for regime in range(best_k):
        mask = sample["regime"] == regime
        regime_data = sample[mask]
        print(f"    Regime {regime} (n={mask.sum()}):")
        for col in ["initial_ln_gdppc", "mean_hc", treatment, y_col]:
            if col in regime_data.columns:
                print(f"      {col}: mean={regime_data[col].mean():.4f}, "
                      f"std={regime_data[col].std():.4f}")

    # List some countries in each regime
    if "country" in df.columns:
        sample_with_country = sample.copy()
        # Get country names from original df
        country_map = dict(zip(df["iso3c"], df["country"]))
        if "iso3c" in df.columns:
            original_idx = sample.index
            countries = df.loc[original_idx, "country"] if "country" in df.columns else None
            if countries is not None:
                sample_with_country["country"] = countries.values
                for regime in range(best_k):
                    mask = sample_with_country["regime"] == regime
                    names = sample_with_country.loc[mask, "country"].tolist()[:10]
                    print(f"    Regime {regime} examples: {', '.join(names)}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════
def plot_extension_comparison(all_results, y_label, filename):
    """Plot CV R² and θ estimates for all extensions vs baseline."""
    df = pd.DataFrame(all_results).dropna(subset=["coef"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: CV R² outcome
    if "r2_outcome" in df.columns:
        ax = axes[0]
        valid = df.dropna(subset=["r2_outcome"])
        y_pos = range(len(valid))
        colors = ["#2ecc71" if r > 0.25 else "#e74c3c" if r < 0 else "#f39c12"
                  for r in valid["r2_outcome"]]
        ax.barh(list(y_pos), valid["r2_outcome"], color=colors, alpha=0.8)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(valid["extension"], fontsize=9)
        ax.set_xlabel("CV R² (outcome model)")
        ax.set_title(f"Nuisance Quality: {y_label}")
        ax.axvline(x=0, color="black", linewidth=0.5)
        # Add baseline reference
        ax.axvline(x=0.26, color="gray", linestyle="--", alpha=0.5, label="GBM baseline")
        ax.legend(fontsize=8)

    # Panel B: θ estimates
    ax = axes[1]
    y_pos = range(len(df))
    ax.errorbar(df["coef"], list(y_pos),
                xerr=1.96 * df["se"],
                fmt="o", color="#3498db", capsize=4, markersize=6)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(df["extension"], fontsize=9)
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.set_xlabel("θ (R&D effect)")
    ax.set_title(f"DML Estimates: {y_label}")

    plt.tight_layout()
    plt.savefig(FIG / filename, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Figure saved: {FIG / filename}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print("EXPERIMENTAL EXTENSIONS")
    print("Probing Latent Structure in R&D → Productivity Growth")
    print("=" * 70)

    all_results_tfp = []
    all_results_gdp = []

    for y_col, y_label, results_list in [
        (outcome_tfp, "TFP Growth", all_results_tfp),
        (outcome_gdp, "GDP p.c. Growth", all_results_gdp),
    ]:
        print(f"\n\n{'#' * 70}")
        print(f"# {y_label.upper()}")
        print(f"{'#' * 70}")

        # Extension 1: Super Learner
        try:
            res = run_super_learner(cs, y_col)
            results_list.append(res)
        except Exception as e:
            print(f"  Super Learner FAILED: {e}")

        # Extension 2: Factor-Augmented
        try:
            res_list = run_factor_augmented(cs, y_col)
            results_list.extend(res_list)
        except Exception as e:
            print(f"  Factor-Augmented FAILED: {e}")

        # Extension 3: Gaussian Process
        try:
            res = run_gp_extension(cs, y_col)
            results_list.append(res)
        except Exception as e:
            print(f"  Gaussian Process FAILED: {e}")

        # Extension 4: Spatial Lags
        try:
            res_list = run_spatial_lags(cs, y_col)
            results_list.extend(res_list)
        except Exception as e:
            print(f"  Spatial Lags FAILED: {e}")

        # Extension 5: Mixture Regimes
        try:
            res_list = run_mixture_regimes(cs, y_col)
            results_list.extend(res_list)
        except Exception as e:
            print(f"  Mixture Regimes FAILED: {e}")

        # Plot comparison
        if results_list:
            plot_extension_comparison(
                results_list, y_label,
                f"experimental_{y_col}.pdf"
            )

    # ── Save summary table ───────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("SUMMARY — TFP Growth Extensions")
    print("=" * 70)
    if all_results_tfp:
        summary_tfp = pd.DataFrame(all_results_tfp)
        cols_to_show = ["extension", "coef", "se", "pval", "r2_outcome", "n"]
        cols_to_show = [c for c in cols_to_show if c in summary_tfp.columns]
        print(summary_tfp[cols_to_show].to_string(index=False))
        summary_tfp.to_csv(DATA / "experimental_tfp.csv", index=False)

    print("\n\nSUMMARY — GDP p.c. Growth Extensions")
    print("=" * 70)
    if all_results_gdp:
        summary_gdp = pd.DataFrame(all_results_gdp)
        cols_to_show = [c for c in cols_to_show if c in summary_gdp.columns]
        print(summary_gdp[cols_to_show].to_string(index=False))
        summary_gdp.to_csv(DATA / "experimental_gdp.csv", index=False)

    print("\n✓ Experimental extensions complete.")
