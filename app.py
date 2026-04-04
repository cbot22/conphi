#!/usr/bin/env python3
"""
Con φ v1 — Consumption Distribution Explorer
=============================================
Streamlit dashboard for Con φ v1 model outputs.

PATH CONFIGURATION
------------------
Two path variables at the top of this file control where the app
reads its data from:

  COLAB_BASE_DIR  — the Google Drive path used when generating outputs
                    in Colab.  Kept here for reference; not used at
                    runtime on your local machine.

  DATA_DIR        — the LOCAL path the app actually reads from.
                    Change this if you move the report folder.
                    Default: G:\\My Drive\\conphi\\outputs\\conphi_v1_report

To run:
  cd C:\\Users\\chris\\Documents\\conphi_dashboard
  streamlit run app.py
"""

from pathlib import Path

# ============================================================
# PATH CONFIGURATION  ← change DATA_DIR if your local path differs
# ============================================================
VERSION = "v1"

# Google Drive path used in Colab (reference only — not used at runtime)
COLAB_BASE_DIR = Path("/content/drive/MyDrive/conphi")

# Local path — Google Drive for Desktop synced folder
DATA_DIR = Path(r"G:\My Drive\conphi\outputs\conphi_v1_report")

# Diagnostic subdirectories (relative to DATA_DIR)
DIAG_USE_DIR  = DATA_DIR / "diagnostics_use"
DIAG_WASE_DIR = DATA_DIR / "diagnostics_wase"

# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ============================================================
# COLUMN NAME MAP  (matches conphi_v1_report_prep.py)
# ============================================================
COL = {
    "country_code":                  "Country Code",
    "country_name":                  "Country",
    "year":                          "Year",
    "prediction_year":               "Prediction Year",
    "percentile":                    "Percentile",
    "horizon":                       "Horizon",
    "model_type":                    "Model Type",
    "prediction_type":               "Prediction Type",
    "data_type":                     "Data Type",
    "region":                        "Region",
    "sub_region":                    "Sub-Region",
    "anchor_year":                   "Anchor Year",
    "dt":                            "Years Since Survey",
    "observed_log_consumption":      "Observed Log Consumption",
    "observed_consumption":          "Observed Consumption",
    "predicted_log_consumption":     "Predicted Log Consumption",
    "predicted_consumption":         "Predicted Consumption",
    "lower_band_log":                "Lower Band Log (5th)",
    "upper_band_log":                "Upper Band Log (95th)",
    "lower_band":                    "Lower Band (5th)",
    "upper_band":                    "Upper Band (95th)",
    "lower_predictive_band_log":     "Lower Predictive Band Log (5th)",
    "upper_predictive_band_log":     "Upper Predictive Band Log (95th)",
    "lower_predictive_band":         "Lower Predictive Band (5th)",
    "upper_predictive_band":         "Upper Predictive Band (95th)",
    "residual":                      "Residual",
    "absolute_error":                "Absolute Error",
    "percentage_error":              "Percentage Error",
    "log_residual":                  "Log Residual",
    "log_absolute_error":            "Log Absolute Error",
    "income_group":                  "Income Group",
    "wfp_country":                   "WFP Country",
    "country_scope":                 "Country Scope",
}

METRIC_COLS = {
    "n_obs":           "Observations",
    "n_countries":     "Countries",
    "mae_cons":        "MAE (Consumption)",
    "rmse_cons":       "RMSE (Consumption)",
    "bias_cons":       "Bias (Consumption)",
    "mape_pct":        "MAPE %",
    "r2_cons":         "R² (Consumption)",
    "mae_log":         "MAE (Log)",
    "rmse_log":        "RMSE (Log)",
    "bias_log":        "Bias (Log)",
    "r2_log":          "R² (Log)",
    "coverage_90_mu":  "Coverage 90% (Mean)",
    "coverage_90_y":   "Coverage 90% (Predictive)",
    "mahler_loss":     "Mahler Loss (×100)",
    "mean_pct_bias":   "Mean % Bias",
    "median_pct_bias": "Median % Bias",
}
MC = METRIC_COLS

# ============================================================
# COLOUR PALETTE
# ============================================================
MUSTARD     = "#E0AA49"
WARM_BEIGE  = "#D7D3C8"
LIGHT_GRAY  = "#D0CFC7"
CREAM       = "#EFEAE0"
ESPRESSO    = "#5D4B46"
LIGHT_SLATE = "#95AEBA"
SLATE_BLUE  = "#56798D"
TERRACOTTA  = "#B86653"

OBS_COLOUR     = "#1f77b4"
USE_COLOUR     = TERRACOTTA
WASE_COLOUR    = MUSTARD
CI_COLOUR_USE  = "rgba(184,102,83,0.15)"
CI_COLOUR_WASE = "rgba(224,170,73,0.15)"
CI_COLOUR      = "rgba(184,102,83,0.15)"
MODEL_COLOURS  = {"USE": USE_COLOUR, "WASE": WASE_COLOUR}
CI_COLOURS     = {"USE": CI_COLOUR_USE, "WASE": CI_COLOUR_WASE}

WB_PALETTE = {
    "East Asia & Pacific":        "#1f77b4",
    "Europe & Central Asia":      "#aec7e8",
    "Latin America & Caribbean":  "#ff7f0e",
    "Middle East & North Africa": "#ffbb78",
    "South Asia":                 "#2ca02c",
    "Sub-Saharan Africa":         "#d62728",
    "Unknown":                    "#999999",
}

INCOME_ORDER = [
    "Low income", "Lower middle income",
    "Upper middle income", "High income", "Unknown",
]
INCOME_PALETTE = {
    "Low income":          "#d62728",
    "Lower middle income": "#ff7f0e",
    "Upper middle income": "#2ca02c",
    "High income":         "#1f77b4",
    "Unknown":             "#999999",
}

# ============================================================
# PARAMETER MAPS  (must match diagnostic script output)
# ============================================================
PARAM_MAP_USE = {
    "beta0_pos":  "β⁺ Expansion Passthrough",
    "beta0_neg":  "β⁻ Contraction Passthrough",
    "beta_p_pos": "β_p⁺ Distributional Tilt (Expansion)",
    "beta_p_neg": "β_p⁻ Distributional Tilt (Contraction)",
    "sigma":      "σ Observation Noise",
    "nu":         "ν Degrees of Freedom",
}

PARAM_MAP_WASE = {
    "intercept":               "Intercept (Global Baseline)",
    "gdp_elasticity":          "GDP p.c. Elasticity",
    "gdp_curvature":           "GDP Curvature (Non-linear)",
    "gdp_growth_effect":       "GDP Growth Effect",
    "u5_mortality_effect":     "Under-5 Mortality Effect",
    "rural_share_effect":      "Rural Population Share Effect",
    "female_education_effect": "Female Education Effect",
    "gov_rev_effect":          "Government Revenue Effect",
    "res_rents_effect":        "Resource Rents Effect",
    "gov_exp_effect":          "Government Expenditure Effect",
    "baseline_inequality":     "Baseline Inequality (Shape)",
    "noise_baseline":          "Observation Noise (Baseline)",
    "noise_tail_widening":     "Observation Noise (Tail Widening)",
    "rbf_weight_1":            "RBF Spline Weight 1",
    "rbf_weight_2":            "RBF Spline Weight 2",
    "rbf_weight_3":            "RBF Spline Weight 3",
}

LOG_CLIP = (-20.0, 20.0)

SUB_REGION_COL = "Sub-Region"


# ============================================================
# UTILITIES
# ============================================================
def safe_exp(x):
    return np.exp(np.clip(x, *LOG_CLIP))


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def count_surveys(df):
    val = df.dropna(subset=[COL["observed_consumption"]])
    if len(val) == 0:
        return 0
    return val[[COL["country_code"], COL["prediction_year"]]].drop_duplicates().shape[0]


# ============================================================
# DATA LOADERS
# ============================================================
@st.cache_data
def load_fact():
    fp = DATA_DIR / "fact_predictions.parquet"
    if not fp.exists():
        return None
    return pd.read_parquet(fp)


@st.cache_data
def load_country_dim():
    fp = DATA_DIR / "dim_country.parquet"
    if not fp.exists():
        return None
    return pd.read_parquet(fp)


@st.cache_data
def load_diag_parquet(model: str, name: str):
    base = DIAG_USE_DIR if model == "USE" else DIAG_WASE_DIR
    fp   = base / name
    if not fp.exists():
        return None
    return pd.read_parquet(fp)


@st.cache_data
def load_diag_residuals(model: str):
    prefix = "use" if model == "USE" else "wase"
    return load_diag_parquet(model, f"{prefix}_diag_residuals.parquet")


@st.cache_data
def load_diag_params(model: str):
    if model == "USE":
        return load_diag_parquet(model, "use_diag_params_over_time.parquet")
    else:
        return load_diag_parquet(model, "wase_diag_params_forest.parquet")


@st.cache_data
def load_diag_coverage(model: str):
    prefix = "use" if model == "USE" else "wase"
    return load_diag_parquet(model, f"{prefix}_diag_coverage.parquet")


@st.cache_data
def load_diag_country_mae(model: str):
    prefix = "use" if model == "USE" else "wase"
    return load_diag_parquet(model, f"{prefix}_diag_country_mae.parquet")


# ============================================================
# APP CONFIG
# ============================================================
st.set_page_config(
    page_title="Con φ — Consumption Explorer",
    page_icon="φ",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    html, body, [class*="css"] {{ font-family: 'Source Sans 3', sans-serif; }}
    code, .stCode, pre {{ font-family: 'JetBrains Mono', monospace !important; }}

    .block-container {{ padding-top: 1rem !important; }}
    [data-testid="stAppViewBlockContainer"] {{ padding-top: 1rem !important; }}

    .main-header {{ padding: 0.8rem 0 0.6rem 0; border-bottom: 3px solid {LIGHT_SLATE}; margin-bottom: 0.8rem; }}
    .main-header h1 {{ font-size: 1.8rem; font-weight: 700; color: {ESPRESSO}; margin: 0; }}
    .main-header p {{ font-size: 0.9rem; color: {SLATE_BLUE}; margin: 0.2rem 0 0 0; }}

    .metric-badge {{ display: inline-block; background: {CREAM}; border: 1px solid {LIGHT_GRAY}; border-radius: 6px; padding: 0.35rem 0.7rem; margin-right: 0.5rem; font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; }}
    .metric-badge .label {{ color: {SLATE_BLUE}; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.3px; }}
    .metric-badge .value {{ font-weight: 600; color: {ESPRESSO}; }}

    .section-header {{ font-size: 1.05rem; font-weight: 700; color: {ESPRESSO}; margin: 1.5rem 0 0.5rem 0; padding-bottom: 0.3rem; border-bottom: 2px solid {LIGHT_SLATE}; }}
    .section-sub {{ font-size: 0.82rem; color: {SLATE_BLUE}; margin: -0.3rem 0 0.7rem 0; line-height: 1.5; }}

    .summary-table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; margin: 1rem 0; }}
    .summary-table th {{ background: {ESPRESSO}; color: {CREAM}; padding: 0.6rem 0.8rem; text-align: right; font-weight: 600; font-size: 0.75rem; text-transform: uppercase; }}
    .summary-table th:first-child, .summary-table th:nth-child(2) {{ text-align: left; }}
    .summary-table td {{ padding: 0.5rem 0.8rem; text-align: right; border-bottom: 1px solid {LIGHT_GRAY}; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: {ESPRESSO}; }}
    .summary-table td:first-child, .summary-table td:nth-child(2) {{ text-align: left; font-family: 'Source Sans 3', sans-serif; font-weight: 600; }}
    .summary-table tr:hover {{ background: {CREAM}; }}

    [data-testid="stSidebar"] {{ background: {CREAM}; }}

    [data-testid="stSlider"] [role="slider"] {{
        background-color: {SLATE_BLUE} !important;
        border-color: {SLATE_BLUE} !important;
        box-shadow: none !important;
    }}
    [data-testid="stToggle"] label > span:first-of-type {{ background-color: {LIGHT_GRAY} !important; }}
    [data-testid="stToggle"] input:checked + label > span:first-of-type,
    [data-testid="stToggle"] label[data-checked="true"] > span:first-of-type {{
        background-color: {SLATE_BLUE} !important;
    }}

    .methods-content {{ color: {ESPRESSO}; line-height: 1.7; }}
    .methods-content h2 {{ color: {ESPRESSO}; font-weight: 700; margin-top: 1.5rem; border-bottom: 2px solid {LIGHT_SLATE}; padding-bottom: 0.3rem; }}
    .methods-content h3 {{ color: {SLATE_BLUE}; font-weight: 600; margin-top: 1.2rem; }}
    .methods-content p {{ margin-bottom: 0.8rem; }}

    #MainMenu {{visibility: hidden;}} footer {{visibility: hidden;}} header {{visibility: hidden;}}
</style>
""", unsafe_allow_html=True)


# ── Load core data ────────────────────────────────────────────
fact        = load_fact()
country_dim = load_country_dim()

if fact is None:
    st.error(
        f"Could not load `fact_predictions.parquet` from:\n\n"
        f"`{DATA_DIR}`\n\n"
        "Check that `DATA_DIR` at the top of `app.py` points to your "
        "local `conphi_v1_report/` folder."
    )
    st.stop()

all_isos = sorted(fact[COL["country_code"]].unique())
if country_dim is not None and COL["country_name"] in country_dim.columns:
    code_to_name = country_dim.set_index(COL["country_code"])[COL["country_name"]].to_dict()
else:
    code_to_name = {c: c for c in all_isos}
name_to_code = {v: k for k, v in code_to_name.items()}

all_model_types = sorted(fact[COL["model_type"]].unique())
all_pred_types  = sorted(
    fact[COL["prediction_type"]].unique(),
    key=lambda x: 0 if x == "Nowcast" else 1,
)
all_regions    = sorted(fact[COL["region"]].dropna().unique())    if COL["region"]     in fact.columns else []
all_subregions = sorted(fact[COL["sub_region"]].dropna().unique()) if COL["sub_region"] in fact.columns else []

if "selected_country" not in st.session_state:
    st.session_state.selected_country = "All Countries"
if "selected_year" not in st.session_state:
    st.session_state.selected_year = "All Years"


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## φ Controls")
    selected_model     = st.selectbox("Model Type",      all_model_types, index=0)
    selected_pred      = st.selectbox("Prediction Type", all_pred_types,  index=0)
    selected_data_type = st.selectbox("Data Type", ["Validated", "Predicted"], index=0)
    st.markdown("---")
    show_ci  = st.toggle("Show 90% Confidence Bands", value=False, key="ci_toggle")
    st.markdown("---")
    pct_range = st.slider("Percentile Range", min_value=1, max_value=99,
                          value=(1, 99), step=1)
    st.markdown("---")

    country_names_all = sorted(code_to_name.values())
    cur = st.session_state.selected_country
    if cur not in ["All Countries"] + country_names_all:
        cur = "All Countries"
    opts_c       = ["All Countries"] + country_names_all
    idx_c        = opts_c.index(cur) if cur in opts_c else 0
    selected_name = st.selectbox("Country", opts_c, index=idx_c, key="sidebar_country")
    st.session_state.selected_country = selected_name

    _tmp = fact[
        (fact[COL["model_type"]]      == selected_model) &
        (fact[COL["prediction_type"]] == selected_pred)
    ]
    if selected_data_type == "Validated":
        _tmp = _tmp[_tmp[COL["data_type"]] == "Validated"]
    _tmp = _tmp[
        (_tmp[COL["percentile"]] >= pct_range[0]) &
        (_tmp[COL["percentile"]] <= pct_range[1])
    ]
    if selected_name != "All Countries":
        _tmp = _tmp[_tmp[COL["country_code"]] == name_to_code.get(selected_name, selected_name)]

    avail_years = sorted(_tmp[COL["prediction_year"]].dropna().astype(int).unique())
    cur_y  = st.session_state.selected_year
    opts_y = ["All Years"] + [str(y) for y in avail_years]
    if cur_y not in opts_y:
        cur_y = "All Years"
    idx_y        = opts_y.index(cur_y) if cur_y in opts_y else 0
    selected_year = st.selectbox("Year", opts_y, index=idx_y, key="sidebar_year")
    st.session_state.selected_year = selected_year

    selected_region    = st.selectbox("Region",     ["All"] + all_regions,    index=0, key="sidebar_region")
    selected_subregion = st.selectbox("Sub-Region", ["All"] + all_subregions, index=0, key="sidebar_subregion")
    st.markdown("---")
    st.markdown("**USE** — Update Survey Estimate  \n**WASE** — Without Any Survey Estimate")
    st.caption(f"Con φ {VERSION} · World Food Programme")


# ============================================================
# FILTER HELPERS
# ============================================================
def apply_sidebar(df, force_validated=False):
    out = df[
        (df[COL["model_type"]]      == selected_model) &
        (df[COL["prediction_type"]] == selected_pred)
    ].copy()
    if force_validated:
        out = out[out[COL["data_type"]] == "Validated"]
    elif selected_data_type == "Validated":
        out = out[out[COL["data_type"]] == "Validated"]
    out = out[
        (out[COL["percentile"]] >= pct_range[0]) &
        (out[COL["percentile"]] <= pct_range[1])
    ]
    if selected_name != "All Countries":
        out = out[out[COL["country_code"]] == name_to_code.get(selected_name, selected_name)]
    if selected_region != "All" and COL["region"] in out.columns:
        out = out[out[COL["region"]] == selected_region]
    if selected_subregion != "All" and COL["sub_region"] in out.columns:
        out = out[out[COL["sub_region"]] == selected_subregion]
    return out


def filter_diag_residuals(df):
    if df is None or len(df) == 0:
        return df
    out = df.copy()
    if "pred_type" in out.columns:
        out = out[out["pred_type"] == selected_pred]
    if "percentile" in out.columns:
        out = out[
            (out["percentile"] >= pct_range[0]) &
            (out["percentile"] <= pct_range[1])
        ]
    if selected_name != "All Countries" and "iso" in out.columns:
        iso = name_to_code.get(selected_name, selected_name)
        out = out[out["iso"] == iso]
    if selected_region != "All" and "region" in out.columns:
        out = out[out["region"] == selected_region]
    if selected_subregion != "All":
        for col in ["sub_region", SUB_REGION_COL]:
            if col in out.columns:
                out = out[out[col] == selected_subregion]
                break
    return out


base_df    = apply_sidebar(fact)
base_df_yr = (
    base_df[base_df[COL["prediction_year"]] == int(selected_year)]
    if selected_year != "All Years" else base_df
)


# ============================================================
# SHARED HELPERS
# ============================================================
def _metric_html_table(tbl, group_col, metric_cols):
    show_cols = [group_col] + [c for c in metric_cols if c in tbl.columns]
    tbl       = tbl[show_cols].copy()
    html = '<table class="summary-table"><thead><tr>'
    for c in show_cols:
        html += f"<th>{c}</th>"
    html += "</tr></thead><tbody>"
    for _, row in tbl.iterrows():
        html += "<tr>"
        for c in show_cols:
            v = row[c]
            if c in ("n_obs", "n_countries"):
                html += f"<td>{int(v):,}</td>"
            elif c in ("r2_log", "r2_cons"):
                html += f"<td>{v:.4f}</td>"
            elif c == "mape_pct":
                html += f"<td>{v:.2f}</td>"
            elif isinstance(v, (float, np.floating)):
                html += f"<td>{v:.4f}</td>"
            else:
                html += f"<td>{v}</td>"
        html += "</tr>"
    html += "</tbody></table>"
    return html


def _compute_grouped(df, group_col):
    if df is None or len(df) == 0 or group_col not in df.columns:
        return pd.DataFrame()
    rows = []
    for key, grp in df.groupby(group_col, dropna=False):
        r_l    = grp["resid_log"].values if "resid_log" in grp.columns else np.array([])
        obs_l  = grp["obs_log"].values   if "obs_log"   in grp.columns else np.array([])
        pred_l = grp["pred_log"].values  if "pred_log"  in grp.columns else np.array([])
        valid  = ~(np.isnan(obs_l) | np.isnan(pred_l) | np.isnan(r_l))
        r_l, obs_l, pred_l = r_l[valid], obs_l[valid], pred_l[valid]
        if len(r_l) == 0:
            continue
        obs_c  = safe_exp(obs_l)
        pred_c = safe_exp(pred_l)
        r_c    = pred_c - obs_c
        with np.errstate(divide="ignore", invalid="ignore"):
            ape = np.where(obs_c > 0, np.abs(r_c / obs_c), np.nan)
        ss_res_l = np.sum(r_l ** 2)
        ss_tot_l = np.sum((obs_l - obs_l.mean()) ** 2)
        rows.append({
            group_col:   key,
            "n_obs":     len(r_l),
            "mae_log":   float(np.mean(np.abs(r_l))),
            "rmse_log":  float(np.sqrt(np.mean(r_l ** 2))),
            "bias_log":  float(r_l.mean()),
            "r2_log":    float(1 - ss_res_l / ss_tot_l) if ss_tot_l > 1e-8 else np.nan,
            "mae_cons":  float(np.mean(np.abs(r_c))),
            "rmse_cons": float(np.sqrt(np.mean(r_c ** 2))),
            "mape_pct":  float(np.nanmean(ape) * 100),
        })
    return pd.DataFrame(rows)


def _prep_params(params_df, model: str):
    """
    Normalise a params dataframe so it always has columns:
    param, param_label, pred_type, mean, sd, q05, q95
    and a year column: target_year (USE) or focal_year (WASE).
    Returns None if the dataframe is empty or missing required columns.
    """
    if params_df is None or len(params_df) == 0:
        return None

    df = params_df.copy()

    aliases = {
        "posterior_mean": "mean",
        "posterior_sd":   "sd",
        "posterior_q05":  "q05", "hdi_3%":  "q05",
        "posterior_q95":  "q95", "hdi_97%": "q95",
        "parameter":      "param",
    }
    df = df.rename(columns={k: v for k, v in aliases.items() if k in df.columns})

    if "param" not in df.columns:
        return None

    pmap = PARAM_MAP_USE if model == "USE" else PARAM_MAP_WASE
    if "param_label" not in df.columns:
        df["param_label"] = df["param"].map(pmap).fillna(df["param"])

    if "pred_type" not in df.columns and "mode" in df.columns:
        df["pred_type"] = df["mode"].apply(
            lambda m: "Nowcast" if m == "nowcast" else "Forecast"
        )
    if "pred_type" not in df.columns:
        df["pred_type"] = "Nowcast"

    for col in ["mean", "sd", "q05", "q95"]:
        if col not in df.columns:
            df[col] = np.nan

    return df


# ============================================================
# HEADER
# ============================================================
st.markdown(
    '<div class="main-header">'
    f'<h1>Con φ {VERSION} — Consumption Distribution Explorer</h1>'
    '<p>Predicted and observed consumption per capita by percentile · World Food Programme</p>'
    '</div>',
    unsafe_allow_html=True,
)

(tab_methods, tab_explorer, tab_diagnostics,
 tab_model_perf, tab_predictors) = st.tabs([
    "📖 Overview & Methods",
    "🌍 Results Explorer",
    "📊 Performance Metrics",
    "🔬 Model Evaluation",
    "📐 Predictor Analysis",
])


# ============================================================
# TAB 1 — OVERVIEW & METHODS
# ============================================================
with tab_methods:
    st.markdown("""
<div class="methods-content">

## Con φ — Methodology

The Con φ (pronounced *con fie*) model predicts household consumption in US Dollars
(at 2017 purchasing power parity) for each percentile of the consumption distribution —
from the poorest 1% to the richest 1% — across Low and Middle Income Countries.
It contains two distinct sub-models, each designed for a different data situation:

**Con φ ~ USE (Update Survey Estimate)** asks: *Given a country's most recent survey
estimates of consumption, how do we project forward using GDP growth?* This model
requires at least one prior consumption survey for the country.

**Con φ ~ WASE (Without Any Survey Estimate)** asks: *What would we predict for this
country if we had no consumption data at all, based on its current level of human
and economic development?* This model requires only structural predictor data — no
consumption survey is needed for the target country.

The system can nowcast (estimate the current year) or forecast up to one year ahead.
Estimates are updated each April and October, in line with the IMF's twice-yearly
publication of economic indicators.

## What data does the model use?

The model uses publicly available data only. At the core of the system is the
**World Bank Poverty and Inequality Platform (PIP)** dataset, which contains
household consumption-expenditure estimates for each percentile of the population
across over 100 countries and more than 1,000 surveys conducted since 1977.
These survey estimates are the outcome variable — what the models are trying to
predict or validate against.

The main economic predictors come from the **IMF World Economic Outlook (WEO)**
datasets, published twice yearly in April and October. Because past GDP estimates
are regularly revised using household survey data — including the very surveys in
the PIP dataset — using current revised estimates would introduce a look-ahead bias.
To prevent this, the pipeline uses historical vintage WEO files: the dataset exactly
as it was released at a given point in time, rather than current revised figures.

The USE model uses only these two datasets. The WASE model additionally draws on
structural country indicators sourced from the **Institute for Health Metrics and
Evaluation (IHME)**, the **UN Inter-Agency Group for Child Mortality Estimation
(UN IGME)**, and the **World Bank**: specifically, under-5 mortality rates, mean
years of female education, the share of the population living in rural areas,
government revenue and expenditure as a share of GDP, and natural resource rents
as a share of GDP. These structural covariates are lagged by three years to reflect
the typical publication delays that would apply in a genuine real-time scenario.

## How does the model work?

### Con φ ~ USE Model

The USE model requires at least one prior consumption survey to project forward from.
It builds on the finding that GDP growth passes through to household consumption
with an empirically estimated coefficient, and extends this in two key ways.

First, the model distinguishes between economic **expansions and contractions**.
Contractions tend to hit households more sharply than expansions lift them, so the
model estimates separate passthrough rates for positive and negative GDP growth,
decomposing cumulative GDP growth year-by-year into its expansion and contraction
components.

Second, the passthrough is allowed to **vary across the consumption distribution**,
capturing whether poorer or richer households tend to benefit proportionally more
from growth. A Student-t likelihood provides robustness to outlier survey-to-survey
consumption changes.

### Con φ ~ WASE Model

For countries without a recent consumption survey, the WASE model predicts the entire
consumption distribution from scratch using only widely available structural indicators.

The model represents log-consumption at each percentile as the sum of two components:
a **level** capturing average living standards (driven by GDP per capita, mortality,
rural share, education, fiscal variables), and a **shape** capturing inequality
(varying across a spatial hierarchy of regions and subregions). Flexible RBF spline
corrections along the percentile axis capture departures from the log-logistic baseline.

Both models are estimated using **Stochastic Variational Inference (SVI)** implemented
in NumPyro/JAX on GPU.

## Validation

Both models use a strict rolling Leave-One-Country-Out (LOCO) procedure that prevents
look-ahead bias. IMF vintage files ensure only historically available GDP data is used.
Structural covariates in WASE are lagged by three years.

### Overall performance

The USE model achieves R² ≈ 0.97 and MAE ≈ \$0.79–0.93/day (~9% MAPE) on validated
rows. The WASE model achieves R² ≈ 0.74–0.75 and MAE ≈ \$3/day (~28–29% MAPE),
reflecting the substantially harder task of predicting without any survey anchor.
The two models are designed to complement each other: USE estimates take precedence
where survey data are available, with WASE filling the remainder of the global picture.

</div>
""", unsafe_allow_html=True)


# ============================================================
# TAB 2 — RESULTS EXPLORER
# ============================================================
with tab_explorer:
    available_isos = sorted(base_df[COL["country_code"]].unique())
    if country_dim is not None and "Latitude" in country_dim.columns:
        map_data = pd.DataFrame({"iso": available_isos})
        map_data["name"]       = map_data["iso"].map(code_to_name).fillna(map_data["iso"])
        sel_iso                = name_to_code.get(selected_name, None) if selected_name != "All Countries" else None
        map_data["colour_val"] = np.where(map_data["iso"] == sel_iso, 1.0, 0.3) if sel_iso else 0.5
        mc_col = MODEL_COLOURS.get(selected_model, TERRACOTTA)
        fig_map = go.Figure(go.Choropleth(
            locations=map_data["iso"], locationmode="ISO-3",
            z=map_data["colour_val"], text=map_data["name"],
            hovertemplate="<b>%{text}</b><extra></extra>",
            colorscale=[[0.0, WARM_BEIGE], [0.5, LIGHT_SLATE], [1.0, mc_col]],
            showscale=False, marker_line_color="#ffffff", marker_line_width=0.5,
        ))
        fig_map.update_geos(
            showcoastlines=True, coastlinecolor=LIGHT_GRAY,
            showland=True, landcolor="#f4f3ef",
            showocean=True, oceancolor="#fbfaf7",
            showcountries=True, countrycolor=LIGHT_GRAY,
            showframe=False, projection_type="natural earth",
        )
        fig_map.update_layout(
            height=300, margin=dict(l=0, r=0, t=0, b=0),
            geo=dict(bgcolor="rgba(0,0,0,0)"),
        )
        event = st.plotly_chart(
            fig_map, use_container_width=True, on_select="rerun", key="choropleth_map"
        )
        if event and hasattr(event, "selection") and event.selection:
            pts = event.selection.get("points", [])
            if pts:
                ci = pts[0].get("point_index", None)
                if ci is not None and ci < len(map_data):
                    cn = code_to_name.get(map_data.iloc[ci]["iso"], "")
                    if cn and cn != st.session_state.selected_country:
                        st.session_state.selected_country = cn
                        st.rerun()

    df = base_df_yr
    if len(df) == 0:
        st.warning("No data for the selected filters.")
    else:
        val_df    = df.dropna(subset=[COL["observed_consumption"], COL["predicted_consumption"]])
        n_surveys = count_surveys(df)
        r2_d, mape_d = None, None
        if len(val_df) > 0:
            ov   = val_df[COL["observed_consumption"]].values
            pv   = val_df[COL["predicted_consumption"]].values
            ss_r = np.sum((pv - ov) ** 2)
            ss_t = np.sum((ov - ov.mean()) ** 2)
            if ss_t > 1e-8:
                r2_d = 1 - ss_r / ss_t
            with np.errstate(divide="ignore", invalid="ignore"):
                ap = np.abs((pv - ov) / ov)
                ap = ap[np.isfinite(ap)]
                if len(ap) > 0:
                    mape_d = np.mean(ap) * 100

        st.markdown('<div class="section-header">Consumption Distribution</div>', unsafe_allow_html=True)
        bh = ""
        if r2_d   is not None: bh += f'<span class="metric-badge"><span class="label">R² </span><span class="value">{r2_d:.4f}</span></span>'
        if mape_d is not None: bh += f'<span class="metric-badge"><span class="label">MAPE </span><span class="value">{mape_d:.1f}%</span></span>'
        bh += f'<span class="metric-badge"><span class="label">Surveys </span><span class="value">{n_surveys:,}</span></span>'
        bh += f'<span class="metric-badge"><span class="label">Percentiles </span><span class="value">{pct_range[0]}–{pct_range[1]}</span></span>'
        st.markdown(bh, unsafe_allow_html=True)
        st.markdown("")

        is_single    = (selected_name != "All Countries") and (selected_year != "All Years")
        is_validated = selected_data_type == "Validated"
        fig    = go.Figure()
        mc_col = MODEL_COLOURS.get(selected_model, TERRACOTTA)
        cc_col = CI_COLOURS.get(selected_model, CI_COLOUR)

        if is_single:
            p_df = df.sort_values(COL["percentile"])
            pct  = p_df[COL["percentile"]].values
            pred = p_df[COL["predicted_consumption"]].values
            obs  = p_df[COL["observed_consumption"]].values   if COL["observed_consumption"]   in p_df.columns else np.full(len(p_df), np.nan)
            lo   = p_df[COL["lower_predictive_band"]].values  if COL["lower_predictive_band"]  in p_df.columns else np.full(len(p_df), np.nan)
            hi   = p_df[COL["upper_predictive_band"]].values  if COL["upper_predictive_band"]  in p_df.columns else np.full(len(p_df), np.nan)
            err  = p_df[COL["percentage_error"]].values       if COL["percentage_error"]       in p_df.columns else np.full(len(p_df), np.nan)

            if show_ci:
                ok = ~(np.isnan(lo) | np.isnan(hi))
                if ok.any():
                    fig.add_trace(go.Scatter(
                        x=np.concatenate([pct[ok], pct[ok][::-1]]),
                        y=np.concatenate([hi[ok],  lo[ok][::-1]]),
                        fill="toself", fillcolor=cc_col,
                        line=dict(width=0), name="90% CI", hoverinfo="skip",
                    ))
            fig.add_trace(go.Scatter(
                x=pct, y=pred, mode="lines+markers",
                marker=dict(size=4, color=mc_col),
                line=dict(color=mc_col, width=2.5),
                name=f"Predicted ({selected_model})",
                customdata=np.stack([pct, pred, obs, lo, hi, err], axis=-1),
                hovertemplate=(
                    "<b>Percentile %{customdata[0]:.0f}</b><br>"
                    "Predicted: %{customdata[1]:.3f} $/day<br>"
                    "Observed: %{customdata[2]:.3f} $/day<br>"
                    "Lower CI: %{customdata[3]:.3f}<br>"
                    "Upper CI: %{customdata[4]:.3f}<br>"
                    "Error: %{customdata[5]:.1f}%<extra></extra>"
                ),
            ))
            if is_validated:
                om = ~np.isnan(obs)
                if om.any():
                    fig.add_trace(go.Scatter(
                        x=pct[om], y=obs[om], mode="lines+markers",
                        marker=dict(size=4, color=OBS_COLOUR, symbol="diamond"),
                        line=dict(color=OBS_COLOUR, width=2, dash="dot"),
                        name="Observed",
                        customdata=np.stack([pct[om], obs[om], pred[om], err[om]], axis=-1),
                        hovertemplate=(
                            "<b>Percentile %{customdata[0]:.0f}</b><br>"
                            "Observed: %{customdata[1]:.3f} $/day<br>"
                            "Predicted: %{customdata[2]:.3f} $/day<br>"
                            "Error: %{customdata[3]:.1f}%<extra></extra>"
                        ),
                    ))
        else:
            has_lo = COL["lower_predictive_band"] in df.columns
            has_hi = COL["upper_predictive_band"] in df.columns
            agg_d  = {
                "pred_med": (COL["predicted_consumption"], "median"),
                "n":        (COL["predicted_consumption"], "count"),
            }
            if has_lo: agg_d["ci_lo"] = (COL["lower_predictive_band"], "median")
            if has_hi: agg_d["ci_hi"] = (COL["upper_predictive_band"], "median")
            agg = df.groupby(COL["percentile"]).agg(**agg_d).reset_index()
            if show_ci and has_lo and has_hi:
                ca = agg.dropna(subset=["ci_lo", "ci_hi"])
                if len(ca) > 0:
                    fig.add_trace(go.Scatter(
                        x=np.concatenate([ca[COL["percentile"]].values, ca[COL["percentile"]].values[::-1]]),
                        y=np.concatenate([ca["ci_hi"].values, ca["ci_lo"].values[::-1]]),
                        fill="toself", fillcolor=cc_col, line=dict(width=0),
                        name="90% CI (Median)", hoverinfo="skip",
                    ))
            fig.add_trace(go.Scatter(
                x=agg[COL["percentile"]], y=agg["pred_med"],
                mode="lines+markers",
                marker=dict(size=4, color=mc_col),
                line=dict(color=mc_col, width=2.5),
                name=f"Predicted Median ({selected_model})",
            ))
            if is_validated:
                obs_r = df.dropna(subset=[COL["observed_consumption"]])
                if len(obs_r) > 0:
                    ao = obs_r.groupby(COL["percentile"]).agg(
                        med=(COL["observed_consumption"], "median")
                    ).reset_index()
                    fig.add_trace(go.Scatter(
                        x=ao[COL["percentile"]], y=ao["med"],
                        mode="lines+markers",
                        marker=dict(size=4, color=OBS_COLOUR, symbol="diamond"),
                        line=dict(color=OBS_COLOUR, width=2, dash="dot"),
                        name="Observed Median",
                    ))

        tp = [selected_name if selected_name != "All Countries" else "All Countries"]
        if selected_year != "All Years": tp.append(str(selected_year))
        tp.append(f"{selected_model} · {selected_pred}")
        fig.update_layout(
            template="plotly_white", height=560,
            xaxis_title="Percentile",
            yaxis_title="Consumption per Capita (2017 PPP $/day)",
            legend=dict(orientation="h", y=-0.12, x=0),
            margin=dict(l=60, r=20, t=50, b=70),
            title=dict(text=" — ".join(tp), font=dict(size=14, color=ESPRESSO)),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("View underlying data"):
            dc = [c for c in [
                COL["country_code"], COL["country_name"], COL["prediction_year"],
                COL["percentile"], COL["predicted_consumption"],
                COL["observed_consumption"],
                COL["lower_predictive_band"], COL["upper_predictive_band"],
                COL["percentage_error"], COL["model_type"],
                COL["prediction_type"], COL["data_type"],
            ] if c in df.columns]
            st.dataframe(
                df[dc].sort_values([COL["country_code"], COL["prediction_year"], COL["percentile"]]),
                use_container_width=True, hide_index=True, height=400,
            )


# ============================================================
# TAB 3 — PERFORMANCE METRICS
# ============================================================
with tab_diagnostics:
    fc1, _ = st.columns([1, 3])
    with fc1:
        dy        = sorted(fact[COL["prediction_year"]].dropna().astype(int).unique())
        diag_year = st.selectbox("Year (diagnostics)", ["All"] + [str(y) for y in dy],
                                 index=0, key="diag_year")

    diag_df = apply_sidebar(fact, force_validated=True)
    if diag_year != "All":
        diag_df = diag_df[diag_df[COL["prediction_year"]] == int(diag_year)]
    n_ds = count_surveys(diag_df)

    st.markdown('<div class="section-header">Overall Metrics</div>', unsafe_allow_html=True)
    if len(diag_df) > 0:
        def cr(g):
            o   = g[COL["observed_consumption"]].values
            p   = g[COL["predicted_consumption"]].values
            r   = p - o
            sr  = np.sum(r ** 2)
            st_ = np.sum((o - o.mean()) ** 2)
            with np.errstate(divide="ignore", invalid="ignore"):
                ap = np.abs(r / o)
                ap = np.where(np.isfinite(ap), ap, np.nan)
            return {
                MC["n_obs"]:     len(g),
                "Surveys":       count_surveys(g),
                MC["r2_cons"]:   1 - sr / st_ if st_ > 1e-8 else np.nan,
                MC["rmse_cons"]: np.sqrt(np.mean(r ** 2)),
                MC["mae_cons"]:  np.mean(np.abs(r)),
                MC["mape_pct"]:  np.nanmean(ap) * 100,
                MC["bias_cons"]: np.mean(r),
            }

        rows = []
        gc   = [c for c in [COL["model_type"], COL["prediction_type"]] if c in diag_df.columns]
        for keys, grp in diag_df.groupby(gc, dropna=False):
            if not isinstance(keys, tuple): keys = (keys,)
            row = dict(zip(gc, keys))
            row.update(cr(grp))
            rows.append(row)
        if rows:
            summary = pd.DataFrame(rows)
            dcols   = gc + [MC["n_obs"], "Surveys", MC["r2_cons"], MC["rmse_cons"],
                            MC["mae_cons"], MC["mape_pct"], MC["bias_cons"]]
            dcols   = [c for c in dcols if c in summary.columns]
            html = '<table class="summary-table"><thead><tr>'
            for c in dcols: html += f"<th>{c}</th>"
            html += "</tr></thead><tbody>"
            for _, row in summary.iterrows():
                html += "<tr>"
                for c in dcols:
                    v = row[c]
                    if c in [MC["n_obs"], "Surveys"]: html += f"<td>{int(v):,}</td>"
                    elif c == MC["r2_cons"]:          html += f"<td>{v:.4f}</td>"
                    elif c == MC["mape_pct"]:         html += f"<td>{v:.2f}</td>"
                    elif isinstance(v, float):        html += f"<td>{v:.4f}</td>"
                    else:                             html += f"<td>{v}</td>"
                html += "</tr>"
            html += "</tbody></table>"
            st.markdown(html, unsafe_allow_html=True)
        st.markdown(
            f'<span class="metric-badge"><span class="label">Surveys </span><span class="value">{n_ds:,}</span></span>'
            f'<span class="metric-badge"><span class="label">Percentiles </span><span class="value">{pct_range[0]}–{pct_range[1]}</span></span>',
            unsafe_allow_html=True,
        )
        st.markdown("")
    else:
        st.warning("No validated data for the selected filters.")

    st.markdown('<div class="section-header">Observed vs Predicted</div>', unsafe_allow_html=True)
    if len(diag_df) > 0:
        MP  = 40_000
        ev  = diag_df.sample(min(MP, len(diag_df)), random_state=42) if len(diag_df) > MP else diag_df
        if len(diag_df) > MP: st.caption(f"Showing {MP:,} of {len(diag_df):,} points")
        fig_s  = go.Figure()
        mc_col = MODEL_COLOURS.get(selected_model, TERRACOTTA)
        cn_a   = ev[COL["country_code"]].map(code_to_name).fillna(ev[COL["country_code"]])
        ht = (
            "<b>" + cn_a.astype(str) + "</b> (" + ev[COL["country_code"]].astype(str) + ")<br>"
            + "Year: "       + ev[COL["prediction_year"]].astype(int).astype(str) + "<br>"
            + "Percentile: " + ev[COL["percentile"]].astype(int).astype(str) + "<br>"
            + "Predicted: "  + ev[COL["predicted_consumption"]].map("{:.3f}".format) + "<br>"
            + "Observed: "   + ev[COL["observed_consumption"]].map("{:.3f}".format) + "<br>"
            + "Error: "      + ev[COL["percentage_error"]].map("{:.1f}%".format)
        ).tolist()
        fig_s.add_trace(go.Scattergl(
            x=ev[COL["observed_consumption"]], y=ev[COL["predicted_consumption"]],
            mode="markers", marker=dict(size=3, color=mc_col, opacity=0.3),
            name=selected_model, text=ht, hoverinfo="text",
        ))
        lo_s = ev[[COL["observed_consumption"], COL["predicted_consumption"]]].min().min()
        hi_s = ev[[COL["observed_consumption"], COL["predicted_consumption"]]].max().max()
        pad  = (hi_s - lo_s) * 0.02
        fig_s.add_trace(go.Scatter(
            x=[lo_s - pad, hi_s + pad], y=[lo_s - pad, hi_s + pad],
            mode="lines", line=dict(color=ESPRESSO, width=2),
            name="Perfect fit", hoverinfo="skip",
        ))
        fig_s.update_layout(
            template="plotly_white", height=650,
            xaxis_title="Observed (2017 PPP $/day)",
            yaxis_title="Predicted (2017 PPP $/day)",
            legend=dict(x=0.02, y=0.98),
            margin=dict(l=60, r=20, t=30, b=60),
        )
        fig_s.update_xaxes(constrain="domain")
        fig_s.update_yaxes(scaleanchor="x", scaleratio=1)
        st.plotly_chart(fig_s, use_container_width=True)

    st.markdown('<div class="section-header">Error by Percentile</div>', unsafe_allow_html=True)
    if len(diag_df) > 0:
        pr = []
        for (mt, pt, p), g in diag_df.groupby(
            [COL["model_type"], COL["prediction_type"], COL["percentile"]]
        ):
            r = g[COL["predicted_consumption"]].values - g[COL["observed_consumption"]].values
            pr.append({
                COL["model_type"]:      mt,
                COL["prediction_type"]: pt,
                COL["percentile"]:      p,
                MC["rmse_cons"]:        np.sqrt(np.mean(r ** 2)),
                MC["mae_cons"]:         np.mean(np.abs(r)),
            })
        pm = pd.DataFrame(pr)
        if len(pm) > 0:
            mo = {"RMSE": MC["rmse_cons"], "MAE": MC["mae_cons"]}
            sl = st.selectbox("Metric", list(mo.keys()), index=0, key="dp_m")
            fig_p = px.line(
                pm, x=COL["percentile"], y=mo[sl],
                color=COL["model_type"], line_dash=COL["prediction_type"],
                color_discrete_map=MODEL_COLOURS,
                labels={COL["percentile"]: "Percentile", mo[sl]: sl},
                markers=True,
            )
            fig_p.update_traces(marker=dict(size=4))
            fig_p.update_layout(
                template="plotly_white", height=450,
                legend=dict(orientation="h", y=-0.15),
                margin=dict(l=50, r=20, t=30, b=60),
            )
            st.plotly_chart(fig_p, use_container_width=True)

    st.markdown('<div class="section-header">RMSE by Year</div>', unsafe_allow_html=True)
    if len(diag_df) > 0:
        day = apply_sidebar(fact, force_validated=True)
        yr  = []
        for (mt, pt, y), g in day.groupby(
            [COL["model_type"], COL["prediction_type"], COL["prediction_year"]]
        ):
            r  = g[COL["predicted_consumption"]].values - g[COL["observed_consumption"]].values
            ns = g[[COL["country_code"], COL["prediction_year"]]].drop_duplicates().shape[0]
            yr.append({
                COL["model_type"]:      mt,
                COL["prediction_type"]: pt,
                COL["year"]:            int(y),
                MC["rmse_cons"]:        np.sqrt(np.mean(r ** 2)),
                "Surveys":              ns,
            })
        ym = pd.DataFrame(yr)
        if len(ym) > 0:
            yr_surveys  = ym.groupby(COL["year"])["Surveys"].max().to_dict()
            all_yrs     = sorted(ym[COL["year"]].unique())
            tick_labels = [f"({yr_surveys.get(y, 0)})<br>{y}" for y in all_yrs]
            fig_y = go.Figure()
            for (mt, pt) in ym[[COL["model_type"], COL["prediction_type"]]].drop_duplicates().values:
                s  = ym[(ym[COL["model_type"]] == mt) & (ym[COL["prediction_type"]] == pt)].sort_values(COL["year"])
                if len(s) == 0: continue
                d  = "solid" if pt == "Nowcast" else "dash"
                cl = MODEL_COLOURS.get(mt, TERRACOTTA)
                hv = (
                    "Year: " + s[COL["year"]].astype(str) + "<br>"
                    "RMSE: " + s[MC["rmse_cons"]].map("{:.4f}".format) + "<br>"
                    "Surveys: " + s["Surveys"].astype(str)
                ).tolist()
                fig_y.add_trace(go.Scatter(
                    x=s[COL["year"]], y=s[MC["rmse_cons"]],
                    mode="lines+markers",
                    marker=dict(size=5, color=cl),
                    line=dict(color=cl, dash=d),
                    name=f"{mt} {pt}", text=hv, hoverinfo="text",
                ))
            fig_y.update_layout(
                template="plotly_white", height=420,
                xaxis=dict(
                    title="(Surveys) Year",
                    tickvals=all_yrs, ticktext=tick_labels,
                    tickangle=0, dtick=1,
                ),
                yaxis_title="RMSE (Consumption)",
                legend=dict(orientation="h", y=-0.22),
                margin=dict(l=50, r=20, t=30, b=100),
            )
            st.plotly_chart(fig_y, use_container_width=True)

    if len(diag_df) > 0 and COL["region"] in diag_df.columns:
        st.markdown('<div class="section-header">Error by Region</div>', unsafe_allow_html=True)
        rr = []
        for (mt, pt, rg), g in diag_df.groupby(
            [COL["model_type"], COL["prediction_type"], COL["region"]]
        ):
            r = g[COL["predicted_consumption"]].values - g[COL["observed_consumption"]].values
            rr.append({
                COL["model_type"]:      mt,
                COL["prediction_type"]: pt,
                COL["region"]:          rg,
                MC["rmse_cons"]:        np.sqrt(np.mean(r ** 2)),
            })
        rm = pd.DataFrame(rr)
        if len(rm) > 0:
            fig_r = px.bar(
                rm.sort_values(MC["rmse_cons"], ascending=True),
                y=COL["region"], x=MC["rmse_cons"],
                color=COL["model_type"], barmode="group",
                color_discrete_map=MODEL_COLOURS, orientation="h",
                labels={COL["region"]: "", MC["rmse_cons"]: "RMSE"},
            )
            fig_r.update_layout(
                template="plotly_white", height=max(350, len(rm) * 18),
                legend=dict(orientation="h", y=-0.12),
                margin=dict(l=180, r=20, t=30, b=60),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_r, use_container_width=True)


# ============================================================
# TAB 4 — MODEL EVALUATION
# ============================================================
with tab_model_perf:

    ctrl_col1, ctrl_col2, ctrl_col3, _ = st.columns([1, 1, 1, 2])
    with ctrl_col1:
        eval_model = st.selectbox(
            "Model", ["USE", "WASE"],
            index=0 if selected_model == "USE" else 1,
            key="eval_model",
        )
    with ctrl_col2:
        eval_groupby = st.selectbox(
            "Break down by",
            ["Overall", "Region", "Income Group", "Sub-Region",
             "Year", "Percentile", "Country",
             "dt (USE only)", "Horizon (WASE only)"],
            key="eval_groupby",
        )
    with ctrl_col3:
        eval_metric = st.selectbox(
            "Primary metric",
            ["MAE (Log)", "RMSE (Log)", "Bias (Log)", "R² (Log)",
             "MAPE %", "MAE (Consumption)", "RMSE (Consumption)"],
            key="eval_metric",
        )

    metric_map = {
        "MAE (Log)":          "mae_log",
        "RMSE (Log)":         "rmse_log",
        "Bias (Log)":         "bias_log",
        "R² (Log)":           "r2_log",
        "MAPE %":             "mape_pct",
        "MAE (Consumption)":  "mae_cons",
        "RMSE (Consumption)": "rmse_cons",
    }
    sel_metric_col = metric_map[eval_metric]

    GROUPBY_COL_MAP_EVAL = {
        "Region":              "region",
        "Sub-Region":          "sub_region",
        "Income Group":        "income_group",
        "Year":                "target_year" if eval_model == "USE" else "focal_year",
        "Percentile":          "percentile",
        "Country":             "country_name",
        "dt (USE only)":       "dt",
        "Horizon (WASE only)": "horizon",
    }

    eval_colour = USE_COLOUR if eval_model == "USE" else WASE_COLOUR

    res_df = load_diag_residuals(eval_model)
    res_df = filter_diag_residuals(res_df)

    # ── A. Summary metrics ────────────────────────────────────────
    st.markdown(
        f'<div class="section-header">{eval_model} — Summary Metrics</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="section-sub">Prediction Type = {selected_pred} · '
        f'Percentiles {pct_range[0]}–{pct_range[1]}'
        + (f' · Country = {selected_name}' if selected_name != "All Countries" else '')
        + (f' · Region = {selected_region}' if selected_region != "All" else '')
        + '</div>',
        unsafe_allow_html=True,
    )

    if res_df is not None and len(res_df) > 0:
        yr_col     = "target_year" if eval_model == "USE" else "focal_year"
        badge_n    = len(res_df)
        badge_iso  = res_df["iso"].nunique() if "iso" in res_df.columns else "—"
        badge_yrs  = (
            f"{int(res_df[yr_col].min())}–{int(res_df[yr_col].max())}"
            if yr_col in res_df.columns else "—"
        )
        st.markdown(
            f'<span class="metric-badge"><span class="label">Obs </span><span class="value">{badge_n:,}</span></span>'
            f'<span class="metric-badge"><span class="label">Countries </span><span class="value">{badge_iso}</span></span>'
            f'<span class="metric-badge"><span class="label">Years </span><span class="value">{badge_yrs}</span></span>',
            unsafe_allow_html=True,
        )
        st.markdown("")

        if eval_groupby == "Overall":
            overall_df = _compute_grouped(res_df.assign(_all="All"), "_all")
            if len(overall_df) > 0:
                overall_df = overall_df.rename(columns={"_all": "Scope"})
                overall_df["Scope"] = "All validated rows"
                st.markdown(_metric_html_table(
                    overall_df, "Scope",
                    ["n_obs", "mae_log", "rmse_log", "bias_log", "r2_log",
                     "mae_cons", "rmse_cons", "mape_pct"],
                ), unsafe_allow_html=True)
        else:
            gcol = GROUPBY_COL_MAP_EVAL.get(eval_groupby, "region")
            if gcol not in res_df.columns:
                st.info(f"Column `{gcol}` not available for {eval_model}.")
            else:
                grp_tbl = _compute_grouped(res_df, gcol)
                if len(grp_tbl) > 0:
                    grp_tbl = grp_tbl.sort_values(
                        sel_metric_col,
                        ascending=(sel_metric_col != "r2_log"),
                    )
                    st.markdown(_metric_html_table(
                        grp_tbl, gcol,
                        ["n_obs", "mae_log", "rmse_log", "bias_log", "r2_log",
                         "mae_cons", "rmse_cons", "mape_pct"],
                    ), unsafe_allow_html=True)
    else:
        st.info(
            f"No diagnostic residuals found for {eval_model}. "
            f"Run `conphi_v1_{eval_model.lower()}_diagnostics.py` first."
        )

    # ── B. Metric chart ───────────────────────────────────────────
    st.markdown(
        f'<div class="section-header">{eval_model} — {eval_metric} by {eval_groupby}</div>',
        unsafe_allow_html=True,
    )
    if res_df is not None and len(res_df) > 0 and eval_groupby != "Overall":
        gcol = GROUPBY_COL_MAP_EVAL.get(eval_groupby, "region")
        if gcol in res_df.columns:
            grp_tbl = _compute_grouped(res_df, gcol)
            if len(grp_tbl) > 0 and sel_metric_col in grp_tbl.columns:
                grp_tbl     = grp_tbl.sort_values(
                    sel_metric_col, ascending=(sel_metric_col != "r2_log")
                )
                is_temporal = eval_groupby in (
                    "Year", "dt (USE only)", "Horizon (WASE only)", "Percentile"
                )
                fig_bar = go.Figure()
                if is_temporal:
                    fig_bar.add_trace(go.Scatter(
                        x=grp_tbl[gcol], y=grp_tbl[sel_metric_col],
                        mode="lines+markers",
                        line=dict(color=eval_colour, width=2.5),
                        marker=dict(size=7, color=eval_colour),
                        customdata=grp_tbl[["n_obs"]].values,
                        hovertemplate=(
                            f"{gcol}: %{{x}}<br>{eval_metric}: "
                            f"%{{y:.4f}}<br>n obs: %{{customdata[0]:,}}<extra></extra>"
                        ),
                    ))
                    if sel_metric_col == "bias_log":
                        fig_bar.add_hline(y=0, line=dict(dash="dash", color="grey", width=1))
                else:
                    if eval_groupby == "Region":
                        bar_colours = [WB_PALETTE.get(r, eval_colour) for r in grp_tbl[gcol]]
                    elif eval_groupby == "Income Group":
                        bar_colours = [INCOME_PALETTE.get(r, eval_colour) for r in grp_tbl[gcol]]
                    else:
                        bar_colours = eval_colour
                    fig_bar.add_trace(go.Bar(
                        x=grp_tbl[sel_metric_col], y=grp_tbl[gcol],
                        orientation="h", marker_color=bar_colours,
                        customdata=grp_tbl[["n_obs"]].values,
                        hovertemplate=(
                            f"%{{y}}<br>{eval_metric}: "
                            f"%{{x:.4f}}<br>n obs: %{{customdata[0]:,}}<extra></extra>"
                        ),
                    ))
                    if sel_metric_col == "bias_log":
                        fig_bar.add_vline(x=0, line=dict(dash="dash", color="grey", width=1))
                h = max(400, len(grp_tbl) * 28) if not is_temporal else 420
                fig_bar.update_layout(
                    template="plotly_white", height=h,
                    xaxis_title=eval_metric if is_temporal else "",
                    margin=dict(l=180 if not is_temporal else 60, r=20, t=30, b=60),
                    showlegend=False,
                )
                if not is_temporal:
                    fig_bar.update_layout(yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_bar, use_container_width=True)

    # ── C. Residual violins ───────────────────────────────────────
    st.markdown(
        f'<div class="section-header">{eval_model} — Residual Distribution by Region</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-sub">Log residuals (predicted − observed) on validated rows, '
        'split by World Bank region. Red dashed line = zero bias.</div>',
        unsafe_allow_html=True,
    )
    if res_df is not None and len(res_df) > 0 and "resid_log" in res_df.columns:
        if "region" in res_df.columns:
            fig_vio = go.Figure()
            for grp_val in sorted(res_df["region"].dropna().unique()):
                sub = res_df[res_df["region"] == grp_val]["resid_log"].dropna().values
                if len(sub) == 0: continue
                c = WB_PALETTE.get(str(grp_val), eval_colour)
                fig_vio.add_trace(go.Violin(
                    y=sub, name=str(grp_val),
                    box_visible=True, meanline_visible=True,
                    fillcolor=hex_to_rgba(c, 0.50), line_color=c,
                    hoverinfo="name+y",
                ))
            fig_vio.add_hline(y=0, line=dict(dash="dash", color="red", width=1))
            fig_vio.update_layout(
                template="plotly_white", height=480, showlegend=False,
                yaxis_title="Log Residual (pred − obs)",
                margin=dict(l=60, r=20, t=30, b=80),
            )
            st.plotly_chart(fig_vio, use_container_width=True)
        else:
            st.info("Region column not found in residuals data.")

    # ── D. Coverage calibration ───────────────────────────────────
    st.markdown(
        f'<div class="section-header">{eval_model} — Coverage Calibration</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-sub">Empirical vs nominal coverage, rescaled from the 90% band. '
        'Points above the diagonal = over-coverage (bands too wide); '
        'below = under-coverage (bands too narrow).'
        + (' WASE coverage reflects post-hoc CI shrinkage (factor 0.55).' if eval_model == "WASE" else '')
        + '</div>',
        unsafe_allow_html=True,
    )
    cov_df = load_diag_coverage(eval_model)
    if cov_df is not None and len(cov_df) > 0:
        fig_cov = go.Figure()
        fig_cov.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(dash="dash", color="grey", width=1.5),
            name="Perfect calibration", hoverinfo="skip",
        ))
        cov_colours = {
            "Posterior Mean (μ) 90% CI": SLATE_BLUE,
            "Predictive (ỹ) 90% CI":    eval_colour,
        }
        for bt, grp in cov_df.groupby("band_type"):
            grp = grp.sort_values("nominal")
            c   = cov_colours.get(bt, eval_colour)
            fig_cov.add_trace(go.Scatter(
                x=grp["nominal"], y=grp["empirical"],
                mode="lines+markers",
                line=dict(color=c, width=2.5), marker=dict(size=7),
                name=bt,
                hovertemplate="Nominal: %{x:.0%}<br>Empirical: %{y:.1%}<extra></extra>",
            ))
        fig_cov.update_layout(
            template="plotly_white", height=440,
            xaxis_title="Nominal Coverage", yaxis_title="Empirical Coverage",
            xaxis=dict(tickformat=".0%"), yaxis=dict(tickformat=".0%"),
            legend=dict(x=0.02, y=0.95),
            margin=dict(l=60, r=20, t=30, b=60),
        )
        st.plotly_chart(fig_cov, use_container_width=True)
    else:
        st.info(f"Coverage data not found. Run `conphi_v1_{eval_model.lower()}_diagnostics.py` first.")

    # ── E. Country-level MAE ──────────────────────────────────────
    st.markdown(
        f'<div class="section-header">{eval_model} — Country-Level MAE</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-sub">Mean absolute log-consumption error by country on validated rows. '
        'Colour encodes World Bank region.</div>',
        unsafe_allow_html=True,
    )
    cntry_df = load_diag_country_mae(eval_model)
    if cntry_df is not None and len(cntry_df) > 0:
        if selected_region != "All" and "region" in cntry_df.columns:
            cntry_df = cntry_df[cntry_df["region"] == selected_region]
        if selected_name != "All Countries" and "country_name" in cntry_df.columns:
            cntry_df = cntry_df[cntry_df["country_name"] == selected_name]

        n_top    = st.slider(
            "Show top N countries",
            min_value=10, max_value=min(80, max(10, len(cntry_df))),
            value=min(40, len(cntry_df)), step=5, key="eval_top_n",
        )
        show_df      = cntry_df.sort_values("mae_log", ascending=False).head(n_top)
        bar_colours  = [WB_PALETTE.get(r, eval_colour) for r in show_df["region"].fillna("Unknown")]
        fig_cntry = go.Figure(go.Bar(
            x=show_df["mae_log"], y=show_df["country_name"],
            orientation="h", marker_color=bar_colours,
            customdata=np.stack([
                show_df["n_obs"].values,
                show_df["region"].fillna("Unknown").values,
                (show_df["income_group"].fillna("Unknown").values
                 if "income_group" in show_df.columns else ["—"] * len(show_df)),
                show_df["bias_log"].round(4).values,
            ], axis=1),
            hovertemplate=(
                "<b>%{y}</b><br>MAE (log): %{x:.4f}<br>Bias (log): %{customdata[3]}<br>"
                "n obs: %{customdata[0]}<br>Region: %{customdata[1]}<br>"
                "Income: %{customdata[2]}<extra></extra>"
            ),
        ))
        fig_cntry.update_layout(
            template="plotly_white", height=max(500, n_top * 22),
            xaxis_title="MAE (Log Consumption)",
            yaxis=dict(autorange="reversed"),
            margin=dict(l=180, r=20, t=30, b=60),
        )
        st.plotly_chart(fig_cntry, use_container_width=True)
    else:
        st.info(f"Country MAE data not found. Run `conphi_v1_{eval_model.lower()}_diagnostics.py` first.")

    # ── F. Model-specific temporal panels ─────────────────────────

    # USE — dt breakdown
    if eval_model == "USE" and res_df is not None and len(res_df) > 0 and "dt" in res_df.columns:
        st.markdown(
            '<div class="section-header">USE — Performance by dt (Years Since Survey)</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-sub">How prediction accuracy degrades as the survey anchor ages. '
            'Widening errors at larger dt confirm expected uncertainty accumulation.</div>',
            unsafe_allow_html=True,
        )
        dt_grp = _compute_grouped(res_df, "dt")
        if len(dt_grp) > 0 and sel_metric_col in dt_grp.columns:
            dt_grp = dt_grp.sort_values("dt")
            fig_dt = go.Figure()
            fig_dt.add_trace(go.Scatter(
                x=dt_grp["dt"], y=dt_grp[sel_metric_col],
                mode="lines+markers",
                line=dict(color=USE_COLOUR, width=2.5), marker=dict(size=8),
                customdata=dt_grp[["n_obs"]].values,
                hovertemplate=(
                    "dt=%{x}<br>" + eval_metric +
                    ": %{y:.4f}<br>n=%{customdata[0]:,}<extra></extra>"
                ),
            ))
            if sel_metric_col == "bias_log":
                fig_dt.add_hline(y=0, line=dict(dash="dash", color="grey", width=1))
            fig_dt.update_layout(
                template="plotly_white", height=380,
                xaxis_title="Years Since Survey (dt)", yaxis_title=eval_metric,
                margin=dict(l=60, r=20, t=30, b=60),
            )
            st.plotly_chart(fig_dt, use_container_width=True)

    # WASE — horizon breakdown
    if eval_model == "WASE" and res_df is not None and len(res_df) > 0 and "horizon" in res_df.columns:
        st.markdown(
            '<div class="section-header">WASE — Performance by Forecast Horizon</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-sub">WASE uses structural indicators only, so performance '
            'should degrade less steeply with horizon than USE.</div>',
            unsafe_allow_html=True,
        )
        h_grp = _compute_grouped(res_df, "horizon")
        if len(h_grp) > 0 and sel_metric_col in h_grp.columns:
            h_grp = h_grp.sort_values("horizon")
            fig_hor = go.Figure()
            fig_hor.add_trace(go.Scatter(
                x=h_grp["horizon"], y=h_grp[sel_metric_col],
                mode="lines+markers",
                line=dict(color=WASE_COLOUR, width=2.5), marker=dict(size=8),
                customdata=h_grp[["n_obs"]].values,
                hovertemplate=(
                    "Horizon=%{x}<br>" + eval_metric +
                    ": %{y:.4f}<br>n=%{customdata[0]:,}<extra></extra>"
                ),
            ))
            if sel_metric_col == "bias_log":
                fig_hor.add_hline(y=0, line=dict(dash="dash", color="grey", width=1))
            fig_hor.update_layout(
                template="plotly_white", height=380,
                xaxis_title="Forecast Horizon (0 = Nowcast)", yaxis_title=eval_metric,
                margin=dict(l=60, r=20, t=30, b=60),
            )
            st.plotly_chart(fig_hor, use_container_width=True)

    # ── G. Raw residuals expander ──────────────────────────────────
    with st.expander(f"View {eval_model} residuals data"):
        if res_df is not None and len(res_df) > 0:
            yr_col    = "target_year" if eval_model == "USE" else "focal_year"
            show_cols = [c for c in [
                "iso", "country_name", yr_col, "percentile", "pred_type",
                "region", "income_group",
                "dt" if eval_model == "USE" else "horizon",
                "obs_log", "pred_log", "resid_log",
            ] if c in res_df.columns]
            st.dataframe(
                res_df[show_cols].sort_values(show_cols[:3]),
                use_container_width=True, hide_index=True, height=400,
            )
        else:
            st.info("No residuals data loaded.")


# ============================================================
# TAB 5 — PREDICTOR ANALYSIS
# ============================================================
with tab_predictors:

    pred_ctrl1, pred_ctrl2, _ = st.columns([1, 1, 3])
    with pred_ctrl1:
        pred_model = st.selectbox(
            "Model", ["USE", "WASE"],
            index=0 if selected_model == "USE" else 1,
            key="pred_model",
        )
    with pred_ctrl2:
        pred_pred_type = st.selectbox(
            "Prediction type", all_pred_types, index=0, key="pred_pred_type"
        )

    pred_colour = USE_COLOUR if pred_model == "USE" else WASE_COLOUR
    pmap        = PARAM_MAP_USE if pred_model == "USE" else PARAM_MAP_WASE

    raw_params = load_diag_params(pred_model)
    params_df  = _prep_params(raw_params, pred_model)

    if params_df is None or len(params_df) == 0:
        st.info(
            f"No parameter data found for {pred_model}. "
            f"Run `conphi_v1_{pred_model.lower()}_diagnostics.py` first."
        )
    else:
        if "pred_type" in params_df.columns:
            params_df = params_df[params_df["pred_type"] == pred_pred_type].copy()

        if len(params_df) == 0:
            st.info(f"No rows for pred_type = {pred_pred_type}.")
        else:
            yr_col = "target_year" if pred_model == "USE" else "focal_year"

            with st.expander("📋 Parameter data diagnostics (click to inspect)"):
                st.write("**Columns:**", list(params_df.columns))
                st.write("**Unique params:**", sorted(params_df["param"].unique().tolist()))
                st.write("**Rows:**", len(params_df))
                st.dataframe(params_df.head(20), use_container_width=True)

            # ── USE: passthrough trajectories ──────────────────────
            if pred_model == "USE":
                st.markdown(
                    '<div class="section-header">USE — Asymmetric GDP Passthrough Over Time</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<div class="section-sub">β⁺ (expansion) and β⁻ (contraction) passthrough '
                    'coefficients across rolling target years, with 90% HDI bands. '
                    'Contraction passthrough is consistently larger, confirming asymmetric '
                    'GDP transmission to household consumption.</div>',
                    unsafe_allow_html=True,
                )

                colour_map   = {"beta0_pos": "#2ca02c", "beta0_neg": "#d62728"}
                label_map    = {k: PARAM_MAP_USE.get(k, k) for k in colour_map}
                avail_params = params_df["param"].unique()

                fig_par = go.Figure()
                plotted = False
                for param, c in colour_map.items():
                    if param not in avail_params: continue
                    sub = params_df[params_df["param"] == param].sort_values(yr_col)
                    if sub["mean"].isna().all(): continue
                    plotted = True
                    if not (sub["q05"].isna().all() or sub["q95"].isna().all()):
                        fig_par.add_trace(go.Scatter(
                            x=pd.concat([sub[yr_col], sub[yr_col].iloc[::-1]]),
                            y=pd.concat([sub["q95"], sub["q05"].iloc[::-1]]),
                            fill="toself", fillcolor=hex_to_rgba(c, 0.15),
                            line=dict(width=0), showlegend=False, hoverinfo="skip",
                        ))
                    fig_par.add_trace(go.Scatter(
                        x=sub[yr_col], y=sub["mean"],
                        mode="lines+markers",
                        line=dict(color=c, width=2.5), marker=dict(size=7),
                        name=label_map.get(param, param),
                        hovertemplate=(
                            f"<b>{label_map.get(param, param)}</b><br>"
                            "Year: %{x}<br>Mean: %{y:.4f}<extra></extra>"
                        ),
                    ))

                if plotted:
                    fig_par.add_hline(
                        y=1.0, line=dict(dash="dash", color="grey", width=1.2),
                        annotation_text="Full passthrough (β=1)",
                        annotation_position="top right",
                    )
                    fig_par.update_layout(
                        template="plotly_white", height=420,
                        xaxis_title="Target Year", yaxis_title="Passthrough Coefficient",
                        legend=dict(orientation="h", x=0.25, y=-0.15),
                        margin=dict(l=60, r=20, t=30, b=80),
                    )
                    st.plotly_chart(fig_par, use_container_width=True)
                else:
                    st.warning(
                        "beta0_pos / beta0_neg not found in the parameter data. "
                        "Check the debug expander above for actual param names."
                    )

                # σ and ν
                st.markdown(
                    '<div class="section-header">USE — Noise and Tail Parameters Over Time</div>',
                    unsafe_allow_html=True,
                )
                other_params    = [p for p in ["sigma", "nu"] if p in avail_params]
                param_labels_u  = {"sigma": "σ Noise Scale", "nu": "ν Degrees of Freedom"}
                param_colours_u = {"sigma": SLATE_BLUE, "nu": ESPRESSO}

                if other_params:
                    fig_par2 = go.Figure()
                    for param in other_params:
                        sub = params_df[params_df["param"] == param].sort_values(yr_col)
                        if sub["mean"].isna().all(): continue
                        c = param_colours_u.get(param, pred_colour)
                        if not (sub["q05"].isna().all() or sub["q95"].isna().all()):
                            fig_par2.add_trace(go.Scatter(
                                x=pd.concat([sub[yr_col], sub[yr_col].iloc[::-1]]),
                                y=pd.concat([sub["q95"], sub["q05"].iloc[::-1]]),
                                fill="toself", fillcolor=hex_to_rgba(c, 0.15),
                                line=dict(width=0), showlegend=False, hoverinfo="skip",
                            ))
                        fig_par2.add_trace(go.Scatter(
                            x=sub[yr_col], y=sub["mean"],
                            mode="lines+markers",
                            line=dict(color=c, width=2.5), marker=dict(size=7),
                            name=param_labels_u.get(param, param),
                            hovertemplate=(
                                f"<b>{param_labels_u.get(param, param)}</b><br>"
                                "Year: %{x}<br>Mean: %{y:.4f}<extra></extra>"
                            ),
                        ))
                    fig_par2.update_layout(
                        template="plotly_white", height=380,
                        xaxis_title="Target Year",
                        legend=dict(orientation="h", x=0.3, y=-0.15),
                        margin=dict(l=60, r=20, t=30, b=80),
                    )
                    st.plotly_chart(fig_par2, use_container_width=True)
                else:
                    st.info("sigma / nu not found in parameter data.")

                # All params trajectory grid
                st.markdown(
                    '<div class="section-header">USE — All Parameter Trajectories</div>',
                    unsafe_allow_html=True,
                )
                all_use_params = [p for p in PARAM_MAP_USE if p in avail_params]
                if all_use_params:
                    from plotly.subplots import make_subplots
                    nc = 2
                    nr = int(np.ceil(len(all_use_params) / nc))
                    fig_grid = make_subplots(
                        rows=nr, cols=nc,
                        subplot_titles=[PARAM_MAP_USE.get(p, p) for p in all_use_params],
                        vertical_spacing=0.12, horizontal_spacing=0.10,
                    )
                    for idx, param in enumerate(all_use_params):
                        row, col = idx // nc + 1, idx % nc + 1
                        sub = params_df[params_df["param"] == param].sort_values(yr_col)
                        if sub["mean"].isna().all(): continue
                        c = "#1f77b4"
                        if not (sub["q05"].isna().all() or sub["q95"].isna().all()):
                            fig_grid.add_trace(go.Scatter(
                                x=pd.concat([sub[yr_col], sub[yr_col].iloc[::-1]]),
                                y=pd.concat([sub["q95"], sub["q05"].iloc[::-1]]),
                                fill="toself", fillcolor=hex_to_rgba(c, 0.15),
                                line=dict(width=0), showlegend=False, hoverinfo="skip",
                            ), row=row, col=col)
                        fig_grid.add_trace(go.Scatter(
                            x=sub[yr_col], y=sub["mean"],
                            mode="lines+markers",
                            line=dict(color=c, width=2), marker=dict(size=5),
                            showlegend=False,
                            hovertemplate=(
                                f"<b>{PARAM_MAP_USE.get(param, param)}</b><br>"
                                "Year: %{x}<br>Mean: %{y:.4f}<extra></extra>"
                            ),
                        ), row=row, col=col)
                    fig_grid.update_layout(
                        template="plotly_white", height=nr * 240,
                        margin=dict(t=60, b=40, l=50, r=20),
                    )
                    st.plotly_chart(fig_grid, use_container_width=True)

            # ── WASE: forest plot + trajectories + SNR ─────────────
            if pred_model == "WASE":
                st.markdown(
                    '<div class="section-header">WASE — Coefficient Forest Plot</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<div class="section-sub">Posterior means ± 90% HDI for all structural '
                    'covariates, averaged across LOCO folds. Parameters whose HDI straddles '
                    'zero have little empirical signal in the data.</div>',
                    unsafe_allow_html=True,
                )

                summary = (
                    params_df.groupby("param")[["mean", "q05", "q95"]]
                    .mean()
                    .reset_index()
                )
                summary["param_label"] = summary["param"].map(pmap).fillna(summary["param"])
                summary = summary.sort_values("mean", ascending=False)

                if not summary["mean"].isna().all():
                    fig_forest = go.Figure()
                    fig_forest.add_trace(go.Scatter(
                        x=summary["mean"],
                        y=summary["param_label"],
                        error_x=dict(
                            type="data", symmetric=False,
                            array=(summary["q95"] - summary["mean"]).clip(lower=0).values,
                            arrayminus=(summary["mean"] - summary["q05"]).clip(lower=0).values,
                            color=WASE_COLOUR, thickness=2.5, width=8,
                        ),
                        mode="markers",
                        marker=dict(size=10, color=WASE_COLOUR, symbol="circle"),
                        hovertemplate="<b>%{y}</b><br>Mean: %{x:.4f}<extra></extra>",
                        showlegend=False,
                    ))
                    fig_forest.add_vline(
                        x=0, line=dict(dash="dash", color="grey", width=1.2)
                    )
                    fig_forest.update_layout(
                        template="plotly_white",
                        height=max(400, len(summary) * 45),
                        xaxis_title="Posterior Mean",
                        yaxis=dict(autorange="reversed"),
                        margin=dict(l=260, r=20, t=30, b=60),
                    )
                    st.plotly_chart(fig_forest, use_container_width=True)
                else:
                    st.warning("All mean values are NaN — check the debug expander above.")

                # Coefficient trajectories over focal years
                if yr_col in params_df.columns:
                    st.markdown(
                        '<div class="section-header">WASE — Coefficient Stability Across Folds</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        '<div class="section-sub">How key structural coefficients evolve across '
                        'focal years. Stable bands confirm parameter identifiability.</div>',
                        unsafe_allow_html=True,
                    )
                    key_wase = [p for p in [
                        "gdp_elasticity", "u5_mortality_effect", "rural_share_effect",
                        "gov_rev_effect", "res_rents_effect", "gdp_growth_effect",
                    ] if p in params_df["param"].unique()]

                    if key_wase:
                        from plotly.subplots import make_subplots
                        nc = 2
                        nr = int(np.ceil(len(key_wase) / nc))
                        fig_traj = make_subplots(
                            rows=nr, cols=nc,
                            subplot_titles=[pmap.get(p, p) for p in key_wase],
                            vertical_spacing=0.12, horizontal_spacing=0.10,
                        )
                        for idx, param in enumerate(key_wase):
                            row, col = idx // nc + 1, idx % nc + 1
                            sub = params_df[params_df["param"] == param].sort_values(yr_col)
                            if sub["mean"].isna().all(): continue
                            c = WASE_COLOUR
                            if not (sub["q05"].isna().all() or sub["q95"].isna().all()):
                                fig_traj.add_trace(go.Scatter(
                                    x=pd.concat([sub[yr_col], sub[yr_col].iloc[::-1]]),
                                    y=pd.concat([sub["q95"], sub["q05"].iloc[::-1]]),
                                    fill="toself", fillcolor=hex_to_rgba(c, 0.15),
                                    line=dict(width=0), showlegend=False, hoverinfo="skip",
                                ), row=row, col=col)
                            fig_traj.add_trace(go.Scatter(
                                x=sub[yr_col], y=sub["mean"],
                                mode="lines+markers",
                                line=dict(color=c, width=2), marker=dict(size=5),
                                showlegend=False,
                                hovertemplate=(
                                    f"<b>{pmap.get(param, param)}</b><br>"
                                    "Year: %{x}<br>Mean: %{y:.4f}<extra></extra>"
                                ),
                            ), row=row, col=col)
                        fig_traj.update_layout(
                            template="plotly_white", height=nr * 250,
                            margin=dict(t=60, b=40, l=50, r=20),
                        )
                        st.plotly_chart(fig_traj, use_container_width=True)

                # SNR bar chart
                st.markdown(
                    '<div class="section-header">WASE — Predictor Signal-to-Noise Ratio</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<div class="section-sub">|Posterior mean| / posterior SD averaged across '
                    'folds. SNR > 1 means the signal exceeds the noise; SNR < 1 indicates the '
                    'posterior barely moved from the prior.</div>',
                    unsafe_allow_html=True,
                )

                if "sd" in params_df.columns:
                    snr_df = (
                        params_df.groupby("param")[["mean", "sd"]]
                        .apply(lambda g: pd.Series({
                            "snr":      (np.abs(g["mean"]) /
                                         g["sd"].replace(0, np.nan)).mean(),
                            "abs_mean": np.abs(g["mean"]).mean(),
                        }))
                        .reset_index()
                    )
                    snr_df["param_label"] = snr_df["param"].map(pmap).fillna(snr_df["param"])
                    snr_df = snr_df.dropna(subset=["snr"]).sort_values("snr", ascending=True)

                    if len(snr_df) > 0:
                        bar_colours_snr = [
                            WASE_COLOUR if v >= 1 else LIGHT_GRAY
                            for v in snr_df["snr"]
                        ]
                        fig_snr = go.Figure(go.Bar(
                            x=snr_df["snr"], y=snr_df["param_label"],
                            orientation="h", marker_color=bar_colours_snr,
                            hovertemplate="<b>%{y}</b><br>SNR: %{x:.2f}<extra></extra>",
                        ))
                        fig_snr.add_vline(
                            x=1, line=dict(dash="dot", color="grey", width=1.5),
                            annotation_text="SNR = 1", annotation_position="top right",
                        )
                        fig_snr.update_layout(
                            template="plotly_white",
                            height=max(380, len(snr_df) * 40),
                            xaxis_title="|Mean| / SD (Signal-to-Noise)",
                            yaxis=dict(autorange="reversed"),
                            margin=dict(l=260, r=20, t=30, b=60),
                            showlegend=False,
                        )
                        st.plotly_chart(fig_snr, use_container_width=True)
                else:
                    st.info("No 'sd' column in parameter data — SNR cannot be computed.")