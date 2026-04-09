"""
dml_analysis.py
---------------
Double Machine Learning estimation of the effect of R&D expenditure
on productivity growth, following the Baiardi & Naghi (2024) framework.

Models:
  - Partially Linear Regression (PLR): Y = θ·D + g(X) + ε
  - Multiple ML learners for nuisance functions: LASSO, Ridge, Elastic Net,
    Random Forest, Gradient Boosting, Neural Network, BART (if available)

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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.base import clone, BaseEstimator, RegressorMixin

import doubleml as dml
from doubleml import DoubleMLPLR, DoubleMLData

import statsmodels.api as sm

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


# ── 1. Load data ─────────────────────────────────────────────────────────────
print("Loading cross-section data ...")
cs = pd.read_csv(DATA / "analysis_crosssection.csv")

# ── 2. Define variables ──────────────────────────────────────────────────────
treatment = "mean_rd_gdp"
outcome_tfp = "mean_tfp_growth"
outcome_gdp = "mean_gdppc_growth"

# Controls (level terms only for tree methods; level + squared for linear)
mean_controls_level = [c for c in cs.columns
                       if c.startswith("mean_") and c not in
                       [treatment, outcome_tfp, outcome_gdp]
                       and not c.endswith("_sq")]
mean_controls_level = [c for c in mean_controls_level if c in cs.columns]

# Include initial GDP for convergence
if "initial_ln_gdppc" in cs.columns:
    mean_controls_level.append("initial_ln_gdppc")

# Include governance PCA components
for gc in ["governance_pc1", "governance_pc2"]:
    if gc in cs.columns and gc not in mean_controls_level:
        mean_controls_level.append(gc)

# Squared terms (for linear/LASSO models)
squared_controls = [c for c in cs.columns if c.endswith("_sq")]

controls_linear = mean_controls_level + squared_controls
controls_nonlinear = mean_controls_level  # tree models don't need squares

print(f"  Treatment: {treatment}")
print(f"  Outcomes:  {outcome_tfp}, {outcome_gdp}")
print(f"  Controls (level): {len(mean_controls_level)}")
print(f"  Controls (+ squared): {len(controls_linear)}")


# ── 3. Prepare clean sample ─────────────────────────────────────────────────
def prepare_sample(df, y_col, d_col, x_cols):
    """Drop rows with missing values in key variables."""
    cols = [y_col, d_col] + x_cols
    cols = [c for c in cols if c in df.columns]
    sample = df[cols].dropna()
    print(f"  Sample size: {len(sample)} countries")
    return sample


# ── 4. OLS baseline ─────────────────────────────────────────────────────────
def run_ols_baseline(df, y_col, d_col, x_cols):
    """Simple OLS regression for comparison."""
    sample = prepare_sample(df, y_col, d_col, x_cols)
    y = sample[y_col]
    X = sm.add_constant(sample[[d_col] + x_cols])
    model = sm.OLS(y, X).fit(cov_type="HC1")

    coef = model.params[d_col]
    se = model.bse[d_col]
    pval = model.pvalues[d_col]
    return {"method": "OLS", "coef": coef, "se": se, "pval": pval,
            "n": len(sample), "n_controls": len(x_cols)}


# ── 5. DML estimation ───────────────────────────────────────────────────────

# ── Optuna-based hyperparameter tuning ──────────────────────────────────────
def tune_nn_optuna(X, y, n_trials=80):
    """
    Tune Neural Network (MLPRegressor) via Optuna.

    The key challenge: n=104 with a (64,32) architecture = ~3800 parameters.
    That's 36× more parameters than observations → catastrophic overfitting.
    Solution: search over MUCH smaller architectures + strong regularization.
    """
    def objective(trial):
        # Architecture: 1 or 2 hidden layers, very small
        n_layers = trial.suggest_int("n_layers", 1, 2)
        layer1 = trial.suggest_int("layer1", 2, 16)
        if n_layers == 2:
            layer2 = trial.suggest_int("layer2", 2, layer1)
            hidden = (layer1, layer2)
        else:
            hidden = (layer1,)

        # Strong L2 regularization (alpha)
        alpha = trial.suggest_float("alpha", 1e-2, 10.0, log=True)
        lr_init = trial.suggest_float("lr_init", 1e-4, 1e-2, log=True)
        activation = trial.suggest_categorical("activation", ["relu", "tanh"])

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(
                hidden_layer_sizes=hidden,
                activation=activation,
                alpha=alpha,   # L2 penalty — key regularizer
                learning_rate_init=lr_init,
                max_iter=2000,
                early_stopping=True,
                validation_fraction=0.2,
                n_iter_no_change=20,
                random_state=42,
            ))
        ])
        scores = cross_val_score(model, X, y, cv=5, scoring="r2")
        return scores.mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    bp = study.best_params
    n_layers = bp["n_layers"]
    hidden = (bp["layer1"], bp["layer2"]) if n_layers == 2 else (bp["layer1"],)

    best_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPRegressor(
            hidden_layer_sizes=hidden,
            activation=bp["activation"],
            alpha=bp["alpha"],
            learning_rate_init=bp["lr_init"],
            max_iter=2000,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=20,
            random_state=42,
        ))
    ])
    print(f"    NN Optuna best: hidden={hidden}, alpha={bp['alpha']:.4f}, "
          f"lr={bp['lr_init']:.5f}, act={bp['activation']}, "
          f"CV R²={study.best_value:.4f}")
    return best_model, study.best_value


def tune_rf_optuna(X, y, n_trials=60):
    """
    Tune Random Forest via Optuna.

    RF is already the best performer (CV R²≈0.14 for outcome). Tuning focuses
    on preventing overfitting: deeper trees can memorize noise in n=104.
    """
    def objective(trial):
        model = RandomForestRegressor(
            n_estimators=trial.suggest_int("n_estimators", 100, 1000, step=100),
            max_depth=trial.suggest_int("max_depth", 2, 10),
            max_features=trial.suggest_categorical("max_features",
                                                    ["sqrt", "log2", 0.3, 0.5]),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 3, 20),
            min_samples_split=trial.suggest_int("min_samples_split", 4, 20),
            random_state=42, n_jobs=-1,
        )
        return cross_val_score(model, X, y, cv=5, scoring="r2").mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    bp = study.best_params

    best_model = RandomForestRegressor(
        n_estimators=bp["n_estimators"],
        max_depth=bp["max_depth"],
        max_features=bp["max_features"],
        min_samples_leaf=bp["min_samples_leaf"],
        min_samples_split=bp["min_samples_split"],
        random_state=42, n_jobs=-1,
    )
    print(f"    RF Optuna best: n_est={bp['n_estimators']}, depth={bp['max_depth']}, "
          f"max_feat={bp['max_features']}, leaf={bp['min_samples_leaf']}, "
          f"CV R²={study.best_value:.4f}")
    return best_model, study.best_value


def tune_gbm_optuna(X, y, n_trials=60):
    """
    Tune Gradient Boosting via Optuna.

    GBM's default (500 trees, depth=3, lr=0.05) gave CV R²≈-0.04 for outcome.
    Key risk: too many trees + too high learning rate → overfitting.
    Solution: search for fewer trees, lower learning rate, higher min_leaf.
    """
    def objective(trial):
        model = GradientBoostingRegressor(
            n_estimators=trial.suggest_int("n_estimators", 50, 500, step=50),
            max_depth=trial.suggest_int("max_depth", 1, 5),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 25),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            random_state=42,
        )
        return cross_val_score(model, X, y, cv=5, scoring="r2").mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    bp = study.best_params

    best_model = GradientBoostingRegressor(
        n_estimators=bp["n_estimators"],
        max_depth=bp["max_depth"],
        learning_rate=bp["learning_rate"],
        min_samples_leaf=bp["min_samples_leaf"],
        subsample=bp["subsample"],
        random_state=42,
    )
    print(f"    GBM Optuna best: n_est={bp['n_estimators']}, depth={bp['max_depth']}, "
          f"lr={bp['learning_rate']:.4f}, leaf={bp['min_samples_leaf']}, "
          f"sub={bp['subsample']:.2f}, CV R²={study.best_value:.4f}")
    return best_model, study.best_value


def tune_bart_optuna(X, y, n_trials=40):
    """
    Tune BART-approx (HistGradientBoosting) via Optuna.

    BART's strength: regularization through many shallow trees with small
    learning rate. Tuning focuses on the n_trees vs learning_rate tradeoff.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    def objective(trial):
        model = HistGradientBoostingRegressor(
            max_iter=trial.suggest_int("max_iter", 50, 500, step=50),
            max_depth=trial.suggest_int("max_depth", 1, 5),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 30),
            l2_regularization=trial.suggest_float("l2_reg", 0.0, 10.0),
            random_state=42,
        )
        return cross_val_score(model, X, y, cv=5, scoring="r2").mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    bp = study.best_params

    best_model = HistGradientBoostingRegressor(
        max_iter=bp["max_iter"],
        max_depth=bp["max_depth"],
        learning_rate=bp["learning_rate"],
        min_samples_leaf=bp["min_samples_leaf"],
        l2_regularization=bp["l2_reg"],
        random_state=42,
    )
    print(f"    BART Optuna best: iter={bp['max_iter']}, depth={bp['max_depth']}, "
          f"lr={bp['learning_rate']:.4f}, leaf={bp['min_samples_leaf']}, "
          f"l2={bp['l2_reg']:.2f}, CV R²={study.best_value:.4f}")
    return best_model, study.best_value


def make_learners(X_linear=None, y_linear=None, X_nonlinear=None, y_nonlinear=None):
    """
    Create dictionary of ML learners for nuisance functions.

    If X/y arrays are provided, Optuna tunes RF, GBM, BART, and NN on
    those arrays. Otherwise, uses default hyperparameters.
    """
    learners = {}
    tuning_log = {}
    do_tune = X_nonlinear is not None and y_nonlinear is not None

    # ── 1. LASSO ──────────────────────────────────────────────────────────
    # L1 penalty selects a sparse model. With p=25 features and n=104,
    # LASSO will typically select ~5-10 variables. CV selects λ automatically.
    # Outcome CV R² is negative because TFP growth has no strong linear
    # relationship with controls — this is a DATA limitation, not a model bug.
    learners["LASSO"] = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LassoCV(cv=5, max_iter=10000, random_state=42))
    ])

    # ── 2. Ridge ──────────────────────────────────────────────────────────
    # L2 penalty keeps ALL features but shrinks coefficients.
    # Ridge tends to produce LARGER estimates than LASSO because it doesn't
    # zero out correlated features — it spreads the effect across them.
    # This explains why Ridge gives θ=0.006 vs LASSO's θ=0.002.
    learners["Ridge"] = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RidgeCV(cv=5))
    ])

    # ── 3. Elastic Net ────────────────────────────────────────────────────
    # Combines L1 + L2: gets LASSO's sparsity with Ridge's stability for
    # correlated features. For our data, it behaves more like LASSO (sparse).
    learners["Elastic Net"] = Pipeline([
        ("scaler", StandardScaler()),
        ("model", ElasticNetCV(cv=5, max_iter=10000, l1_ratio=[0.1, 0.5, 0.7, 0.9],
                               random_state=42))
    ])

    # ── 4. CART (Decision Tree) ───────────────────────────────────────────
    # A single tree: interpretable but high variance. With n=104 and
    # max_depth=5, each leaf has ~3 observations → noisy predictions.
    # Included as a baseline for ensemble methods and for interpretability
    # (you can visualize the tree structure).
    from sklearn.tree import DecisionTreeRegressor
    learners["CART"] = DecisionTreeRegressor(
        max_depth=5, min_samples_leaf=10, random_state=42
    )

    # ── 5. Random Forest (Optuna-tuned) ───────────────────────────────────
    # Averages many decorrelated trees → reduces variance vs CART.
    # Already the best performer for outcome (R²≈0.14). Optuna fine-tunes
    # the depth/leaf-size tradeoff: deeper trees capture more signal but
    # risk overfitting leaves with ~2-3 observations.
    if do_tune:
        print("  Tuning Random Forest with Optuna ...")
        learners["Random Forest"], r2 = tune_rf_optuna(X_nonlinear, y_nonlinear)
        tuning_log["Random Forest"] = r2
    else:
        learners["Random Forest"] = RandomForestRegressor(
            n_estimators=500, max_features="sqrt", min_samples_leaf=5,
            random_state=42, n_jobs=-1
        )

    # ── 6. Gradient Boosting (Optuna-tuned) ───────────────────────────────
    # Builds trees sequentially, each correcting the previous residuals.
    # More prone to overfitting than RF because trees are correlated.
    # Key dial: learning_rate × n_estimators. Lower lr needs more trees
    # but generalizes better. Optuna finds the sweet spot.
    if do_tune:
        print("  Tuning Gradient Boosting with Optuna ...")
        learners["Gradient Boosting"], r2 = tune_gbm_optuna(X_nonlinear, y_nonlinear)
        tuning_log["Gradient Boosting"] = r2
    else:
        learners["Gradient Boosting"] = GradientBoostingRegressor(
            n_estimators=500, max_depth=3, learning_rate=0.05,
            min_samples_leaf=5, subsample=0.8, random_state=42
        )

    # ── 7. BART-approx (Optuna-tuned) ────────────────────────────────────
    # BART regularizes by using many small trees with a strong prior toward
    # shrinkage. Our HistGBM approximation adds L2 regularization.
    # Expected to do well in small samples due to built-in regularization.
    if do_tune:
        print("  Tuning BART-approx with Optuna ...")
        learners["BART-approx"], r2 = tune_bart_optuna(X_nonlinear, y_nonlinear)
        tuning_log["BART-approx"] = r2
    else:
        from sklearn.ensemble import HistGradientBoostingRegressor
        learners["BART-approx"] = HistGradientBoostingRegressor(
            max_iter=200, max_depth=3, learning_rate=0.1,
            min_samples_leaf=5, random_state=42
        )

    # ── 8. Neural Network (Optuna-tuned) ──────────────────────────────────
    # BEFORE tuning: (64,32) = ~3800 params with n=104 → CV R² = -428.
    # The NN memorized training folds and predicted garbage on held-out data.
    # Optuna searches for TINY architectures (2-16 neurons) with STRONG
    # L2 regularization (alpha up to 10). This constrains the model to
    # behave more like a regularized nonlinear regression.
    if do_tune:
        print("  Tuning Neural Network with Optuna ...")
        learners["Neural Net"], r2 = tune_nn_optuna(X_nonlinear, y_nonlinear)
        tuning_log["Neural Net"] = r2
    else:
        learners["Neural Net"] = Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(
                hidden_layer_sizes=(64, 32), activation="relu",
                max_iter=1000, early_stopping=True, random_state=42
            ))
        ])

    return learners, tuning_log


def make_ensemble_learner(rf_model=None, gbm_model=None):
    """
    Create an Ensemble learner (average of RF + GBM), following B&N 2024.

    If tuned RF/GBM models are provided, uses those instead of defaults.
    The ensemble averages predictions from the two base learners,
    combining RF's variance reduction with GBM's bias reduction.
    """
    class EnsembleAverage(BaseEstimator, RegressorMixin):
        def __init__(self, estimators=None):
            self.estimators = estimators or [
                RandomForestRegressor(n_estimators=500, max_features="sqrt",
                                      min_samples_leaf=5, random_state=42, n_jobs=-1),
                GradientBoostingRegressor(n_estimators=500, max_depth=3, learning_rate=0.05,
                                          min_samples_leaf=5, subsample=0.8, random_state=42),
            ]
        def fit(self, X, y):
            self.fitted_ = [clone(e).fit(X, y) for e in self.estimators]
            return self
        def predict(self, X):
            preds = np.column_stack([e.predict(X) for e in self.fitted_])
            return preds.mean(axis=1)

    estimators = []
    if rf_model is not None:
        estimators.append(clone(rf_model))
    else:
        estimators.append(RandomForestRegressor(
            n_estimators=500, max_features="sqrt",
            min_samples_leaf=5, random_state=42, n_jobs=-1))
    if gbm_model is not None:
        estimators.append(clone(gbm_model))
    else:
        estimators.append(GradientBoostingRegressor(
            n_estimators=500, max_depth=3, learning_rate=0.05,
            min_samples_leaf=5, subsample=0.8, random_state=42))

    return EnsembleAverage(estimators=estimators)


def run_dml_plr(df, y_col, d_col, x_cols, learner_name, ml_l, ml_m,
                n_folds=5, n_rep=10):
    """
    Run DoubleML Partially Linear Regression.

    PLR model:  Y = θ·D + g(X) + U,  E[U|X,D] = 0
                D = m(X) + V,         E[V|X]   = 0

    ml_l: learner for E[Y|X] (outcome nuisance)
    ml_m: learner for E[D|X] (treatment nuisance)
    """
    sample = prepare_sample(df, y_col, d_col, x_cols)

    # DoubleML requires specific data object
    dml_data = DoubleMLData(
        sample, y_col=y_col, d_cols=d_col, x_cols=x_cols
    )

    # Fit PLR
    plr = DoubleMLPLR(
        dml_data,
        ml_l=ml_l,
        ml_m=ml_m,
        n_folds=n_folds,
        n_rep=n_rep,
        score="partialling out",
    )
    plr.fit()

    # Extract results
    coef = plr.coef[0]
    se = plr.se[0]
    pval = plr.pval[0]
    ci = plr.confint(level=0.95).values[0]

    return {
        "method": f"DML-{learner_name}",
        "coef": coef,
        "se": se,
        "pval": pval,
        "ci_lower": ci[0],
        "ci_upper": ci[1],
        "n": len(sample),
        "n_controls": len(x_cols),
        "n_folds": n_folds,
        "n_rep": n_rep,
    }


# ── 6. Run all estimations ──────────────────────────────────────────────────
def run_full_analysis(df, y_col, y_label):
    """Run OLS + DML with all learners for a given outcome."""
    print(f"\n{'='*70}")
    print(f"OUTCOME: {y_label} ({y_col})")
    print(f"{'='*70}")

    results = []

    # OLS baselines
    print("\n--- OLS (no controls) ---")
    res_ols0 = run_ols_baseline(df, y_col, treatment, [])
    res_ols0["method"] = "OLS (no controls)"
    results.append(res_ols0)

    print("\n--- OLS (all controls) ---")
    ols_x = [c for c in mean_controls_level if c in
             prepare_sample(df, y_col, treatment, mean_controls_level).columns]
    res_ols = run_ols_baseline(df, y_col, treatment, ols_x)
    results.append(res_ols)

    # DML with each learner
    # Prepare data for Optuna tuning (use nonlinear controls, outcome model)
    sample_tune = prepare_sample(df, y_col, treatment, controls_nonlinear)
    X_tune = sample_tune[[c for c in controls_nonlinear if c in sample_tune.columns]].values
    Y_tune = sample_tune[y_col].values

    print("\n  ── Optuna hyperparameter tuning ──")
    learners, tuning_log = make_learners(
        X_nonlinear=X_tune, y_nonlinear=Y_tune
    )
    if tuning_log:
        print(f"  Tuning log: {tuning_log}")

    # Store tuned RF/GBM for ensemble
    tuned_rf = learners.get("Random Forest")
    tuned_gbm = learners.get("Gradient Boosting")

    # Nuisance model diagnostics
    nuisance_diag = []

    for name, learner in learners.items():
        print(f"\n--- DML: {name} ---")
        # Use squared terms for linear methods, level only for tree methods
        if name in ("LASSO", "Ridge", "Elastic Net"):
            x_use = [c for c in controls_linear if c in df.columns]
        else:
            x_use = [c for c in controls_nonlinear if c in df.columns]

        try:
            from sklearn.base import clone

            # Nuisance model CV R² diagnostics
            sample_diag = prepare_sample(df, y_col, treatment, x_use)
            X_diag = sample_diag[x_use].values
            Y_diag = sample_diag[y_col].values
            D_diag = sample_diag[treatment].values
            r2_y = cross_val_score(clone(learner), X_diag, Y_diag, cv=5, scoring='r2').mean()
            r2_d = cross_val_score(clone(learner), X_diag, D_diag, cv=5, scoring='r2').mean()
            nuisance_diag.append({"learner": name, "r2_outcome": r2_y, "r2_treatment": r2_d})
            print(f"  Nuisance CV R²: outcome={r2_y:.4f}, treatment={r2_d:.4f}")

            res = run_dml_plr(
                df, y_col, treatment, x_use,
                learner_name=name,
                ml_l=clone(learner),
                ml_m=clone(learner),
                n_folds=5,
                n_rep=10,
            )
            results.append(res)
            print(f"  θ = {res['coef']:.6f}  (SE = {res['se']:.6f}, p = {res['pval']:.4f})")
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({"method": f"DML-{name}", "coef": np.nan,
                           "se": np.nan, "pval": np.nan, "n": 0})

    # Ensemble learner (average of Optuna-tuned RF + GBM, following B&N 2024)
    print(f"\n--- DML: Ensemble (RF + GBM average) ---")
    try:
        ensemble = make_ensemble_learner(rf_model=tuned_rf, gbm_model=tuned_gbm)
        x_ens = [c for c in controls_nonlinear if c in df.columns]

        sample_diag = prepare_sample(df, y_col, treatment, x_ens)
        X_diag = sample_diag[x_ens].values
        Y_diag = sample_diag[y_col].values
        D_diag = sample_diag[treatment].values
        r2_y = cross_val_score(clone(ensemble), X_diag, Y_diag, cv=5, scoring='r2').mean()
        r2_d = cross_val_score(clone(ensemble), X_diag, D_diag, cv=5, scoring='r2').mean()
        nuisance_diag.append({"learner": "Ensemble", "r2_outcome": r2_y, "r2_treatment": r2_d})
        print(f"  Nuisance CV R²: outcome={r2_y:.4f}, treatment={r2_d:.4f}")

        res = run_dml_plr(
            df, y_col, treatment, x_ens,
            learner_name="Ensemble",
            ml_l=clone(ensemble),
            ml_m=clone(ensemble),
            n_folds=5,
            n_rep=10,
        )
        results.append(res)
        print(f"  θ = {res['coef']:.6f}  (SE = {res['se']:.6f}, p = {res['pval']:.4f})")
    except Exception as e:
        print(f"  Ensemble FAILED: {e}")

    # Save nuisance diagnostics
    if nuisance_diag:
        diag_df = pd.DataFrame(nuisance_diag)
        diag_df.to_csv(DATA / f"nuisance_diagnostics_{y_col}.csv", index=False)
        print(f"\n  Nuisance diagnostics saved.")
        print(diag_df.to_string(index=False))

    return pd.DataFrame(results)


# ── 7. Visualization ────────────────────────────────────────────────────────
def plot_coefficient_comparison(results_df, outcome_label, filename):
    """Create a coefficient plot comparing all methods (like B&N Table 1)."""
    df = results_df.dropna(subset=["coef"]).copy()
    df["ci_lower"] = df.get("ci_lower", df["coef"] - 1.96 * df["se"])
    df["ci_upper"] = df.get("ci_upper", df["coef"] + 1.96 * df["se"])

    # Fill CI for OLS
    mask = df["ci_lower"].isna()
    df.loc[mask, "ci_lower"] = df.loc[mask, "coef"] - 1.96 * df.loc[mask, "se"]
    df.loc[mask, "ci_upper"] = df.loc[mask, "coef"] + 1.96 * df.loc[mask, "se"]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = range(len(df))

    ax.errorbar(
        df["coef"], y_pos,
        xerr=[df["coef"] - df["ci_lower"], df["ci_upper"] - df["coef"]],
        fmt="o", color="navy", capsize=4, markersize=8, linewidth=1.5
    )
    ax.axvline(x=0, color="red", linestyle="--", alpha=0.5)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(df["method"])
    ax.set_xlabel(f"Effect of R&D (% GDP) on {outcome_label}")
    ax.set_title(f"DML Estimates: R&D → {outcome_label}")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(FIG / filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {FIG / filename}")


def plot_scatter_rd_tfp(df, filename="scatter_rd_tfp.pdf"):
    """Scatter plot of R&D vs TFP growth with country labels."""
    sample = df[[treatment, outcome_tfp, "iso3c", "country"]].dropna()

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(sample[treatment], sample[outcome_tfp], alpha=0.6, s=50, color="navy")

    # Label selected countries
    for _, row in sample.iterrows():
        if row[treatment] > 2.5 or abs(row[outcome_tfp]) > 0.02 or \
           row["iso3c"] in ["USA", "CHN", "DEU", "JPN", "KOR", "ITA", "GBR",
                            "ISR", "FIN", "IND", "BRA"]:
            ax.annotate(row["iso3c"], (row[treatment], row[outcome_tfp]),
                       fontsize=7, alpha=0.7,
                       xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel("R&D Expenditure (% GDP)")
    ax.set_ylabel("Mean TFP Growth (log difference)")
    ax.set_title("R&D Expenditure and Productivity Growth (2000–2019 averages)")
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)

    # OLS fit line
    from numpy.polynomial.polynomial import polyfit
    mask = sample[[treatment, outcome_tfp]].notna().all(axis=1)
    b, m = polyfit(sample.loc[mask, treatment], sample.loc[mask, outcome_tfp], 1)
    x_line = np.linspace(sample[treatment].min(), sample[treatment].max(), 100)
    ax.plot(x_line, b + m * x_line, "--", color="red", alpha=0.5, label="OLS fit")
    ax.legend()

    plt.tight_layout()
    plt.savefig(FIG / filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {FIG / filename}")


# ── 8. Sensitivity analysis ─────────────────────────────────────────────────
def sensitivity_nfolds(df, y_col, x_cols, folds_list=[3, 5, 7, 10]):
    """Check sensitivity of DML estimates to number of cross-fitting folds."""
    print("\n--- Sensitivity: Number of folds ---")
    from sklearn.base import clone

    rf = RandomForestRegressor(
        n_estimators=500, max_features="sqrt", min_samples_leaf=5,
        random_state=42, n_jobs=-1
    )
    results = []
    for nf in folds_list:
        try:
            res = run_dml_plr(df, y_col, treatment, x_cols,
                             f"RF-{nf}folds", clone(rf), clone(rf),
                             n_folds=nf, n_rep=10)
            results.append({"folds": nf, **res})
            print(f"  K={nf}: θ={res['coef']:.6f} (SE={res['se']:.6f})")
        except Exception as e:
            print(f"  K={nf}: FAILED ({e})")
    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print("DOUBLE MACHINE LEARNING ANALYSIS")
    print("R&D Expenditure → Productivity Growth")
    print("=" * 70)

    # ── Descriptive scatter ──────────────────────────────────────────────
    print("\n1. Descriptive plots ...")
    plot_scatter_rd_tfp(cs)

    # ── Main DML: TFP growth ─────────────────────────────────────────────
    results_tfp = run_full_analysis(cs, outcome_tfp, "TFP Growth")
    print("\n\nRESULTS TABLE — TFP Growth:")
    print(results_tfp[["method", "coef", "se", "pval", "n"]].to_string(index=False))
    results_tfp.to_csv(DATA / "results_dml_tfp.csv", index=False)

    plot_coefficient_comparison(results_tfp, "TFP Growth", "coef_plot_tfp.pdf")

    # ── Main DML: GDP per capita growth ──────────────────────────────────
    results_gdp = run_full_analysis(cs, outcome_gdp, "GDP p.c. Growth")
    print("\n\nRESULTS TABLE — GDP p.c. Growth:")
    print(results_gdp[["method", "coef", "se", "pval", "n"]].to_string(index=False))
    results_gdp.to_csv(DATA / "results_dml_gdp.csv", index=False)

    plot_coefficient_comparison(results_gdp, "GDP p.c. Growth", "coef_plot_gdp.pdf")

    # ── Sensitivity analysis ─────────────────────────────────────────────
    print("\n\n3. Sensitivity analysis ...")
    x_sens = [c for c in controls_nonlinear if c in cs.columns]
    sens = sensitivity_nfolds(cs, outcome_tfp, x_sens)
    if len(sens) > 0:
        sens.to_csv(DATA / "sensitivity_folds.csv", index=False)

    print("\n✓ DML analysis complete.")
