"""
data_preparation.py
--------------------
Merge PWT 10.01, World Bank WDI, and WGI into a single cross-country panel,
then build the analysis sample for the DML estimation.

Expected raw files in ../data/raw/:
  - pwt1001.dta           (Penn World Table 10.01, Stata format)
  - wdi/WDICSV.csv        (World Development Indicators bulk CSV)
  - wgi.xlsx              (Worldwide Governance Indicators 2025)

Output: ../data/processed/analysis_sample.csv
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

# ── 1. Penn World Table ──────────────────────────────────────────────────────
print("Loading Penn World Table 10.01 ...")
pwt_path = RAW / "pwt1001.dta"
if not pwt_path.exists():
    sys.exit(f"ERROR: {pwt_path} not found. Download from https://www.rug.nl/ggdc/productivity/pwt/")

pwt = pd.read_stata(pwt_path)

# Keep relevant columns
pwt_cols = {
    "countrycode": "iso3c",
    "country": "country",
    "year": "year",
    "rgdpna": "rgdp",          # Real GDP at constant national prices (mil 2017 USD)
    "rtfpna": "tfp",           # TFP at constant national prices (2017=1)
    "pop": "pop",              # Population (millions)
    "emp": "emp",              # Number of persons engaged (millions)
    "hc": "hc",                # Human capital index (based on schooling + returns)
    "csh_i": "inv_share",      # Share of gross capital formation at current PPPs
    "csh_g": "gov_share",      # Share of government consumption at current PPPs
    "csh_x": "export_share",   # Share of merchandise exports at current PPPs
    "csh_m": "import_share",   # Share of merchandise imports at current PPPs
}
pwt = pwt[list(pwt_cols.keys())].rename(columns=pwt_cols)

# Compute GDP per capita and trade openness
pwt["gdppc"] = pwt["rgdp"] / pwt["pop"]
pwt["trade_openness"] = pwt["export_share"] + pwt["import_share"]
pwt["ln_gdppc"] = np.log(pwt["gdppc"])

# Compute TFP growth (log difference)
pwt = pwt.sort_values(["iso3c", "year"])
pwt["tfp_growth"] = pwt.groupby("iso3c")["tfp"].transform(
    lambda x: np.log(x).diff()
)
pwt["gdppc_growth"] = pwt.groupby("iso3c")["ln_gdppc"].diff()

print(f"  PWT: {len(pwt)} rows, {pwt['iso3c'].nunique()} countries")

# ── 2. World Development Indicators ─────────────────────────────────────────
print("Loading World Bank WDI ...")
wdi_path = RAW / "wdi" / "WDICSV.csv"
if not wdi_path.exists():
    sys.exit(f"ERROR: {wdi_path} not found. Download WDI_CSV.zip from World Bank and extract.")

wdi_raw = pd.read_csv(wdi_path, low_memory=False)

# Indicators we need
indicators = {
    "GB.XPD.RSDV.GD.ZS": "rd_gdp",           # R&D expenditure (% GDP) — TREATMENT
    "FP.CPI.TOTL.ZG": "inflation",            # Inflation, consumer prices (annual %)
    "BX.KLT.DINV.WD.GD.ZS": "fdi_gdp",       # Foreign direct investment, net inflows (% GDP)
    "FS.AST.PRVT.GD.ZS": "credit_gdp",        # Domestic credit to private sector (% GDP)
    "SP.POP.GROW": "pop_growth",               # Population growth (annual %)
    "SE.XPD.TOTL.GD.ZS": "educ_exp_gdp",      # Government expenditure on education (% GDP)
    "NY.GDP.MKTP.KD.ZG": "gdp_growth_wb",     # GDP growth (annual %) — for validation
    "NE.TRD.GNFS.ZS": "trade_gdp_wb",         # Trade (% GDP) — alternative
    "IT.NET.USER.ZS": "internet_users",        # Individuals using the Internet (% population)
    "SP.URB.TOTL.IN.ZS": "urban_pop",          # Urban population (% of total)
}

wdi_filt = wdi_raw[wdi_raw["Indicator Code"].isin(indicators.keys())].copy()

# Melt year columns to long format
year_cols = [c for c in wdi_filt.columns if c.isdigit()]
wdi_long = wdi_filt.melt(
    id_vars=["Country Code", "Indicator Code"],
    value_vars=year_cols,
    var_name="year",
    value_name="value",
)
wdi_long["year"] = wdi_long["year"].astype(int)
wdi_long["indicator"] = wdi_long["Indicator Code"].map(indicators)

# Pivot to wide
wdi_wide = wdi_long.pivot_table(
    index=["Country Code", "year"],
    columns="indicator",
    values="value",
    aggfunc="first",
).reset_index()
wdi_wide = wdi_wide.rename(columns={"Country Code": "iso3c"})

print(f"  WDI: {len(wdi_wide)} rows, {wdi_wide['iso3c'].nunique()} countries")

# ── 3. Worldwide Governance Indicators ───────────────────────────────────────
print("Loading Worldwide Governance Indicators ...")
wgi_path = RAW / "wgi.xlsx"
if not wgi_path.exists():
    sys.exit(f"ERROR: {wgi_path} not found. Download from World Bank WGI page.")

# WGI 2025 format: one sheet per dimension (va, pv, ge, rq, rl, cc)
# Each sheet has: Economy (code), Year, Governance estimate, etc.
gov_indicators = {
    "va": "voice_accountability",
    "pv": "pol_stability",
    "ge": "gov_effectiveness",
    "rq": "reg_quality",
    "rl": "rule_of_law",
    "cc": "control_corruption",
}

wgi_frames = []
for sheet, long_name in gov_indicators.items():
    tmp = pd.read_excel(wgi_path, sheet_name=sheet)
    tmp = tmp.rename(columns={
        "Economy (code)": "iso3c",
        "Year": "year",
        "Governance estimate (approx. -2.5 to +2.5)": long_name,
    })
    tmp["year"] = pd.to_numeric(tmp["year"], errors="coerce")
    tmp[long_name] = pd.to_numeric(tmp[long_name], errors="coerce")
    tmp = tmp[["iso3c", "year", long_name]].dropna(subset=["year"])
    tmp["year"] = tmp["year"].astype(int)
    wgi_frames.append(tmp)

wgi = wgi_frames[0]
for frame in wgi_frames[1:]:
    wgi = wgi.merge(frame, on=["iso3c", "year"], how="outer")

print(f"  WGI: {len(wgi)} rows, columns: {list(wgi.columns)}")

# ── 4. Merge all datasets ───────────────────────────────────────────────────
print("Merging datasets ...")
df = pwt.merge(wdi_wide, on=["iso3c", "year"], how="left")
df = df.merge(wgi, on=["iso3c", "year"], how="left")

# ── 5. Build analysis sample ────────────────────────────────────────────────
# Focus on 2000-2019 (best R&D coverage, pre-COVID)
df = df[(df["year"] >= 2000) & (df["year"] <= 2019)].copy()

# Drop micro-states (pop < 0.3 million)
df = df[df["pop"] >= 0.3].copy()

# Our key variables
treatment = "rd_gdp"
outcomes = ["tfp_growth", "gdppc_growth"]

# Report coverage
n_total = len(df)
n_rd = df[treatment].notna().sum()
n_tfp = df["tfp_growth"].notna().sum()
n_both = df[[treatment, "tfp_growth"]].dropna().shape[0]

print(f"\n  Analysis window: 2000-2019")
print(f"  Total rows: {n_total}")
print(f"  Rows with R&D data: {n_rd}")
print(f"  Rows with TFP growth: {n_tfp}")
print(f"  Rows with both: {n_both}")
print(f"  Countries with both: {df.dropna(subset=[treatment, 'tfp_growth'])['iso3c'].nunique()}")

# ── 6. Create lagged controls (avoid simultaneity) ──────────────────────────
# Use t-1 values for controls, t for outcome
control_cols = [
    "hc", "inv_share", "gov_share", "trade_openness", "ln_gdppc",
    "inflation", "fdi_gdp", "credit_gdp", "pop_growth", "educ_exp_gdp",
    "internet_users", "urban_pop",
]
# Add governance if available
gov_cols = [c for c in gov_indicators.values() if c in df.columns]
control_cols += gov_cols

# Create lags
df = df.sort_values(["iso3c", "year"])
for col in control_cols:
    df[f"L1_{col}"] = df.groupby("iso3c")[col].shift(1)

# Lagged treatment too (to check robustness)
df["L1_rd_gdp"] = df.groupby("iso3c")[treatment].shift(1)

# Also create initial GDP (for convergence — use year 2000 value per country)
initial_gdp = df[df["year"] == 2000][["iso3c", "ln_gdppc"]].rename(
    columns={"ln_gdppc": "initial_ln_gdppc"}
)
df = df.merge(initial_gdp, on="iso3c", how="left")

# ── 7. Polynomial / interaction features for DML ───────────────────────────
# We'll let the ML models handle nonlinearities, but create some explicit
# features for the LASSO/linear baseline
lag_controls = [f"L1_{c}" for c in control_cols] + ["initial_ln_gdppc"]

# Squares of continuous variables
for col in lag_controls:
    if col in df.columns:
        df[f"{col}_sq"] = df[col] ** 2

# ── 8. Save ─────────────────────────────────────────────────────────────────
# Full panel
out_path = OUT / "analysis_panel.csv"
df.to_csv(out_path, index=False)
print(f"\nFull panel saved: {out_path}")

# Cross-sectional averages (2000-2019) for simpler DML like B&N/Djankov
cs = df.groupby("iso3c").agg(
    country=("country", "first"),
    mean_rd_gdp=(treatment, "mean"),
    mean_tfp_growth=("tfp_growth", "mean"),
    mean_gdppc_growth=("gdppc_growth", "mean"),
    **{f"mean_{c}": (f"L1_{c}", "mean") for c in control_cols if f"L1_{c}" in df.columns},
    initial_ln_gdppc=("initial_ln_gdppc", "first"),
    n_years=("year", "count"),
).reset_index()

# Drop countries with too few observations
cs = cs[cs["n_years"] >= 5].copy()

# ── Fix multicollinearity ─────────────────────────────────────────────────
# 1. Drop mean_ln_gdppc (r=0.987 with initial_ln_gdppc)
if "mean_ln_gdppc" in cs.columns:
    cs = cs.drop(columns=["mean_ln_gdppc"])
    print("  Dropped mean_ln_gdppc (collinear with initial_ln_gdppc)")

# 2. Replace 6 WGI indicators with PCA governance index
from sklearn.decomposition import PCA
wgi_cols = [c for c in cs.columns if any(x in c for x in
            ["voice_accountability", "pol_stability", "gov_effectiveness",
             "reg_quality", "rule_of_law", "control_corruption"])
            and not c.endswith("_sq")]
if len(wgi_cols) >= 2:
    wgi_data = cs[wgi_cols].copy()
    wgi_valid = wgi_data.dropna()
    pca = PCA(n_components=2)
    pca_vals = pca.fit_transform(wgi_valid)
    print(f"  WGI PCA: PC1 explains {pca.explained_variance_ratio_[0]:.1%}, "
          f"PC2 explains {pca.explained_variance_ratio_[1]:.1%}")
    cs["governance_pc1"] = np.nan
    cs["governance_pc2"] = np.nan
    cs.loc[wgi_valid.index, "governance_pc1"] = pca_vals[:, 0]
    cs.loc[wgi_valid.index, "governance_pc2"] = pca_vals[:, 1]
    cs = cs.drop(columns=wgi_cols)
    print(f"  Replaced {len(wgi_cols)} WGI columns with governance_pc1, governance_pc2")

# Report
n_cs = len(cs)
n_cs_complete = cs[["mean_rd_gdp", "mean_tfp_growth"]].dropna().shape[0]
print(f"\nCross-section (country averages, >=5 years):")
print(f"  Countries: {n_cs}")
print(f"  Complete cases (R&D + TFP): {n_cs_complete}")

# Squared terms for cross-section
mean_controls = [c for c in cs.columns if c.startswith("mean_") and c not in
                 ["mean_rd_gdp", "mean_tfp_growth", "mean_gdppc_growth"]]
# Also include governance PCA and initial_ln_gdppc for squaring
extra_sq = [c for c in cs.columns if c.startswith("governance_pc")]
for col in mean_controls + extra_sq:
    cs[f"{col}_sq"] = cs[col] ** 2

cs_path = OUT / "analysis_crosssection.csv"
cs.to_csv(cs_path, index=False)
print(f"Cross-section saved: {cs_path}")

# ── 9. Summary statistics ──────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY STATISTICS (Cross-section)")
print("=" * 70)
key_vars = ["mean_rd_gdp", "mean_tfp_growth", "mean_gdppc_growth",
            "initial_ln_gdppc"]
key_vars += [c for c in mean_controls[:8]]
desc = cs[key_vars].describe().T[["count", "mean", "std", "min", "50%", "max"]]
desc.columns = ["N", "Mean", "Std", "Min", "Median", "Max"]
print(desc.to_string())

print("\n✓ Data preparation complete.")
