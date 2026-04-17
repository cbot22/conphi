#!/usr/bin/env python3
"""
Con φ v1 — Consumption Distribution Explorer
=============================================
Streamlit dashboard for Con φ v1 model outputs.
Data is read from Google Cloud Storage bucket: conphi

To run locally (with GCS credentials):
  streamlit run app.py
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ============================================================
# PATHS & GCS CONFIGURATION
# ============================================================
VERSION       = "v1"
SCRIPT_DIR    = Path(__file__).resolve().parent
METHODS_FILE  = SCRIPT_DIR / "methods.md"
GUIDE_FILE    = SCRIPT_DIR / "guide.md"
TECH_COMMON_FILE = SCRIPT_DIR / "technical_methods_common.md"
TECH_USE_FILE    = SCRIPT_DIR / "technical_methods_use.md"
TECH_WASE_FILE   = SCRIPT_DIR / "technical_methods_wase.md"

BUCKET     = "conphi"
GCS_PREFIX = f"gs://{BUCKET}/conphi_v1_report"

# ============================================================
# GCS AUTHENTICATION
# ============================================================
import gcsfs
from google.oauth2 import service_account

@st.cache_resource
def _gcs_fs():
    creds = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    return gcsfs.GCSFileSystem(token=creds)

# ============================================================
# DATA LOADERS
# ============================================================
@st.cache_data
def load_fact():
    fs = _gcs_fs()
    with fs.open(f"{GCS_PREFIX}/fact_predictions.parquet") as f:
        return pd.read_parquet(f)

@st.cache_data
def load_country_dim():
    fs = _gcs_fs()
    with fs.open(f"{GCS_PREFIX}/dim_country.parquet") as f:
        return pd.read_parquet(f)

@st.cache_data
def load_diag_parquet(model: str, name: str):
    folder = "diagnostics_use" if model == "USE" else "diagnostics_wase"
    fs     = _gcs_fs()
    path   = f"{GCS_PREFIX}/{folder}/{name}"
    try:
        with fs.open(path) as f:
            return pd.read_parquet(f)
    except Exception:
        return None

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
# COLUMN NAME MAP
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
# PARAMETER MAPS
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

LOG_CLIP       = (-20.0, 20.0)
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

def r2_from_cols(obs, pred):
    mask  = ~(np.isnan(obs) | np.isnan(pred))
    o, p  = obs[mask], pred[mask]
    if len(o) < 2:
        return np.nan
    ss_r  = np.sum((p - o) ** 2)
    ss_t  = np.sum((o - o.mean()) ** 2)
    return float(1 - ss_r / ss_t) if ss_t > 1e-8 else np.nan

@st.cache_data
def _build_persistence(res, fact_df):
    """Build persistence baseline for USE residuals.

    The persistence (naïve) forecast = anchor survey value carried forward.
    Uses the `anchor_log_cons` column directly if available in the residuals
    (requires updated pipeline).  Falls back to a fact-table lookup for
    anchor years that appear as a Prediction Year (typically 2015+).
    """
    res_out = res.copy()

    if "anchor_log_cons" in res_out.columns and res_out["anchor_log_cons"].notna().any():
        # ── Direct column available (full coverage) ───────────
        res_out["persist_resid_log"] = res_out["anchor_log_cons"] - res_out["obs_log"]
    else:
        # ── Fallback: lookup from fact table (partial coverage) ─
        anchor_lookup = (
            fact_df
            .dropna(subset=[COL["observed_log_consumption"]])
            .drop_duplicates(
                subset=[COL["country_code"], COL["prediction_year"], COL["percentile"]],
                keep="first",
            )
            [[COL["country_code"], COL["prediction_year"], COL["percentile"],
              COL["observed_log_consumption"]]]
            .rename(columns={
                COL["country_code"]:              "iso",
                COL["prediction_year"]:           "anchor_year",
                COL["percentile"]:                "percentile",
                COL["observed_log_consumption"]:  "anchor_log_cons",
            })
        )
        anchor_lookup["anchor_year"] = anchor_lookup["anchor_year"].astype(float)
        anchor_lookup["percentile"]  = anchor_lookup["percentile"].astype(float)
        res_out["anchor_year"] = res_out["anchor_year"].astype(float)
        res_out["percentile"]  = res_out["percentile"].astype(float)
        res_out = res_out.merge(
            anchor_lookup,
            on=["iso", "anchor_year", "percentile"],
            how="left",
        )
        res_out["persist_resid_log"] = res_out["anchor_log_cons"] - res_out["obs_log"]

    return res_out

# ============================================================
# APP CONFIG
# ============================================================
st.set_page_config(
    page_title="Con φ — Consumption Explorer",
    page_icon="φ",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Scroll position preservation
st.components.v1.html(
    """
    <script>
        window.parent.document.querySelector('.main').addEventListener('scroll', function() {
            window.parent._streamlit_scroll_pos = this.scrollTop;
        });
        const observer = new MutationObserver(function() {
            const main = window.parent.document.querySelector('.main');
            if (window.parent._streamlit_scroll_pos && main) {
                main.scrollTop = window.parent._streamlit_scroll_pos;
            }
        });
        observer.observe(
            window.parent.document.querySelector('.main'),
            { childList: true, subtree: true }
        );
    </script>
    """,
    height=0,
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

    .perf-summary {{ background: {CREAM}; border-left: 4px solid {LIGHT_SLATE}; border-radius: 4px; padding: 0.9rem 1.2rem; margin: 0.8rem 0 1.2rem 0; font-size: 0.88rem; color: {ESPRESSO}; line-height: 1.65; }}
    .perf-summary strong {{ color: {ESPRESSO}; }}

    .summary-table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; margin: 1rem 0; }}
    .summary-table th {{ background: {ESPRESSO}; color: {CREAM}; padding: 0.6rem 0.8rem; text-align: right; font-weight: 600; font-size: 0.75rem; text-transform: uppercase; }}
    .summary-table th:first-child, .summary-table th:nth-child(2) {{ text-align: left; }}
    .summary-table td {{ padding: 0.5rem 0.8rem; text-align: right; border-bottom: 1px solid {LIGHT_GRAY}; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: {ESPRESSO}; }}
    .summary-table td:first-child, .summary-table td:nth-child(2) {{ text-align: left; font-family: 'Source Sans 3', sans-serif; font-weight: 600; }}
    .summary-table tr:hover {{ background: {CREAM}; }}

    [data-testid="stSidebar"] {{ background: {CREAM}; }}

    /* Compact radio buttons for sidebar controls */
    [data-testid="stSidebar"] [data-testid="stRadio"] label {{ font-size: 0.82rem !important; }}
    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] {{ gap: 0.4rem; }}

    
    #MainMenu {{visibility: hidden;}} footer {{visibility: hidden;}}
</style>
""", unsafe_allow_html=True)

# ── Load core data ─────────────────────────────────────────────
fact        = load_fact()
country_dim = load_country_dim()

if fact is None:
    st.error(
        f"Could not load `fact_predictions.parquet` from GCS bucket `{BUCKET}`.\n\n"
        "Check that the bucket exists, files have been uploaded, and secrets are configured correctly."
    )
    st.stop()

# Derive WFP ISO set from fact table
wfp_isos = set(
    fact.loc[fact[COL["wfp_country"]] == "Yes", COL["country_code"]].unique()
)

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
# SIDEBAR  — Year → dt → Region → Sub-Region → Country (cascade)
#            with reset-on-conflict for country
# ============================================================
with st.sidebar:
    st.markdown("## φ Controls")

    # ── Model / prediction / data type (horizontal radio buttons) ──
    selected_model     = st.radio("Model Type",      all_model_types,              index=0, horizontal=True)
    selected_pred      = st.radio("Prediction Type",  all_pred_types,              index=0, horizontal=True)
    selected_data_type = st.radio("Data Type",       ["Validated", "Predicted"],   index=0, horizontal=True)

    st.markdown("---")

    # ── Display options ────────────────────────────────────────
    show_ci   = st.toggle("Show 90% Confidence Bands", value=True, key="ci_toggle")
    pct_range = st.slider("Percentile Range", min_value=1, max_value=99,
                          value=(1, 99), step=1)

    st.markdown("---")

    # ── Cascading filters: Year → dt → Region → Sub-Region → Country → WFP ──
    # Build a base frame scoped to the current model/pred/data selections
    _cascade_base = fact[
        (fact[COL["model_type"]]      == selected_model) &
        (fact[COL["prediction_type"]] == selected_pred)
    ].copy()
    if selected_data_type == "Validated":
        _cascade_base = _cascade_base[_cascade_base[COL["data_type"]] == "Validated"]

    # ── 1. Year ───────────────────────────────────────────────
    avail_years = sorted(_cascade_base[COL["prediction_year"]].dropna().astype(int).unique())
    cur_y       = st.session_state.selected_year
    opts_y      = ["All Years"] + [str(y) for y in avail_years]
    if cur_y not in opts_y:
        cur_y = "All Years"
    idx_y         = opts_y.index(cur_y)
    selected_year = st.selectbox("Year", opts_y, index=idx_y, key="sidebar_year")
    st.session_state.selected_year = selected_year

    _cascade_yr = _cascade_base.copy()
    if selected_year != "All Years":
        _cascade_yr = _cascade_yr[_cascade_yr[COL["prediction_year"]] == int(selected_year)]

    # ── 1b. Years Since Last Survey (USE only) ────────────────
    if selected_model == "USE" and COL["dt"] in _cascade_yr.columns:
        _dt_vals = _cascade_yr[COL["dt"]].dropna()
        if len(_dt_vals) > 0:
            max_dt_avail = int(_dt_vals.max())
            if max_dt_avail >= 1:
                # Build options: "All", "≤1", "≤2", ... "≤max"
                dt_opts = ["All"] + [f"≤{d}" for d in range(1, max_dt_avail + 1)]
                selected_dt_label = st.selectbox(
                    "Yrs since last survey",
                    dt_opts,
                    index=0,
                    key="sidebar_dt",
                )
                if selected_dt_label == "All":
                    selected_max_dt = None
                else:
                    selected_max_dt = int(selected_dt_label.replace("≤", ""))
            else:
                selected_max_dt = None
        else:
            selected_max_dt = None
    else:
        selected_max_dt = None

    # Apply dt filter to the cascade so downstream filters respect it
    _cascade_dt = _cascade_yr.copy()
    if selected_max_dt is not None and COL["dt"] in _cascade_dt.columns:
        _cascade_dt = _cascade_dt[
            _cascade_dt[COL["dt"]].isna() | (_cascade_dt[COL["dt"]] <= selected_max_dt)
        ]

    # ── 2. Region (cascades from year + dt) ───────────────────
    if COL["region"] in _cascade_dt.columns:
        avail_regions_cascade = sorted(_cascade_dt[COL["region"]].dropna().unique())
    else:
        avail_regions_cascade = all_regions
    selected_region = st.selectbox(
        "Region", ["All"] + avail_regions_cascade, index=0, key="sidebar_region"
    )

    _cascade_region = _cascade_dt.copy()
    if selected_region != "All" and COL["region"] in _cascade_region.columns:
        _cascade_region = _cascade_region[_cascade_region[COL["region"]] == selected_region]

    # ── 3. Sub-Region (cascades from year + dt + region) ──────
    if COL["sub_region"] in _cascade_region.columns:
        avail_subregions = sorted(_cascade_region[COL["sub_region"]].dropna().unique())
    else:
        avail_subregions = all_subregions
    selected_subregion = st.selectbox(
        "Sub-Region", ["All"] + avail_subregions, index=0, key="sidebar_subregion"
    )

    _cascade_sub = _cascade_region.copy()
    if selected_subregion != "All" and COL["sub_region"] in _cascade_sub.columns:
        _cascade_sub = _cascade_sub[_cascade_sub[COL["sub_region"]] == selected_subregion]

    # ── 4. Country (cascades from year + dt + region + sub-region) ─
    # Reset-on-conflict: if the stored country is no longer in the
    # available list given the current region/sub-region, clear it.
    avail_isos_cascade  = sorted(_cascade_sub[COL["country_code"]].unique())
    country_names_avail = sorted([code_to_name.get(c, c) for c in avail_isos_cascade])

    cur_c = st.session_state.selected_country
    if cur_c not in ["All Countries"] + country_names_avail:
        cur_c = "All Countries"
        st.session_state.selected_country = "All Countries"

    opts_c        = ["All Countries"] + country_names_avail
    idx_c         = opts_c.index(cur_c)
    selected_name = st.selectbox("Country", opts_c, index=idx_c, key="sidebar_country")
    st.session_state.selected_country = selected_name

    # ── 5. WFP scope ──────────────────────────────────────────
    selected_wfp = st.selectbox(
        "WFP | All Countries",
        ["All", "WFP Countries"],
        index=0,
        key="sidebar_wfp",
    )

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
    # ── dt filter (USE only) ──────────────────────────────────
    if selected_max_dt is not None and COL["dt"] in out.columns:
        out = out[out[COL["dt"]].isna() | (out[COL["dt"]] <= selected_max_dt)]
    if selected_wfp == "WFP Countries":
        out = out[out[COL["wfp_country"]] == "Yes"]
    if selected_region != "All" and COL["region"] in out.columns:
        out = out[out[COL["region"]] == selected_region]
    if selected_subregion != "All" and COL["sub_region"] in out.columns:
        out = out[out[COL["sub_region"]] == selected_subregion]
    if selected_name != "All Countries":
        out = out[out[COL["country_code"]] == name_to_code.get(selected_name, selected_name)]
    return out


def filter_diag_residuals(df):
    if df is None or len(df) == 0:
        return df
    out = df.copy()
    if selected_wfp == "WFP Countries" and "iso" in out.columns:
        out = out[out["iso"].isin(wfp_isos)]
    if "pred_type" in out.columns:
        out = out[out["pred_type"] == selected_pred]
    if "percentile" in out.columns:
        out = out[
            (out["percentile"] >= pct_range[0]) &
            (out["percentile"] <= pct_range[1])
        ]
    # ── dt filter (USE only) ──────────────────────────────────
    if selected_max_dt is not None and "dt" in out.columns:
        out = out[out["dt"].isna() | (out["dt"] <= selected_max_dt)]
    if selected_region != "All" and "region" in out.columns:
        out = out[out["region"] == selected_region]
    if selected_subregion != "All":
        for col in ["sub_region", SUB_REGION_COL]:
            if col in out.columns:
                out = out[out[col] == selected_subregion]
                break
    if selected_name != "All Countries" and "iso" in out.columns:
        iso = name_to_code.get(selected_name, selected_name)
        out = out[out["iso"] == iso]
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
    if params_df is None or len(params_df) == 0:
        return None
    df = params_df.copy()
    aliases = {
        "posterior_mean": "mean", "posterior_sd": "sd",
        "posterior_q05": "q05", "hdi_3%": "q05",
        "posterior_q95": "q95", "hdi_97%": "q95",
        "parameter": "param",
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

# ── Tab persistence ─────────────────────────────────────────────
if "active_tab" not in st.session_state:
    st.session_state.active_tab = 0

TAB_LABELS = [
    "📖 Overview & Methods",
    "🧭 User Guide",
    "🔬 Technical Methods",
    "🌍 Results Explorer",
    "📊 Model Performance",
    "📐 Predictor Analysis",
]

(tab_methods, tab_guide, tab_tech, tab_explorer, tab_performance, tab_predictors) = st.tabs(TAB_LABELS)

_sentinel_val = st.text_input(
    "active_tab_bridge",
    value=str(st.session_state.active_tab),
    key="_tab_sentinel",
    label_visibility="collapsed",
)
try:
    _incoming = int(_sentinel_val)
    if 0 <= _incoming < len(TAB_LABELS):
        st.session_state.active_tab = _incoming
except (ValueError, TypeError):
    pass

st.components.v1.html(
    f"""
    <script>
    (function() {{
        const WANT = {st.session_state.active_tab};

        function getSentinel() {{
            return Array.from(
                window.parent.document.querySelectorAll('input[type="text"]')
            ).find(el => el.closest('[data-testid="stTextInput"]') !== null
                      && el.value !== null
                      && (el.value === '0' || el.value === '1'
                          || el.value === '2' || el.value === '3'
                          || el.value === '4' || el.value === '5'));
        }}

        function setTab(idx) {{
            const sentinel = getSentinel();
            if (!sentinel) return;
            const setter = Object.getOwnPropertyDescriptor(
                window.HTMLInputElement.prototype, 'value'
            ).set;
            setter.call(sentinel, String(idx));
            sentinel.dispatchEvent(new Event('input', {{ bubbles: true }}));
            sentinel.dispatchEvent(new Event('change', {{ bubbles: true }}));
        }}

        function attachListeners(buttons) {{
            buttons.forEach(function(btn, idx) {{
                btn.addEventListener('click', function() {{ setTab(idx); }}, true);
            }});
        }}

        function restoreAndAttach() {{
            var tabList = window.parent.document.querySelector(
                '[data-baseweb="tab-list"]'
            );
            if (!tabList) {{ setTimeout(restoreAndAttach, 60); return; }}

            var buttons = tabList.querySelectorAll('button[role="tab"]');
            if (buttons.length === 0) {{ setTimeout(restoreAndAttach, 60); return; }}

            if (buttons[WANT] &&
                buttons[WANT].getAttribute('aria-selected') !== 'true') {{
                buttons[WANT].click();
            }}

            attachListeners(buttons);
        }}

        restoreAndAttach();
    }})();
    </script>
    """,
    height=0,
)

# ============================================================
# TAB 1 — OVERVIEW & METHODS
# ============================================================
with tab_methods:
    if METHODS_FILE.exists():
        st.markdown(METHODS_FILE.read_text(encoding="utf-8"))
    else:
        st.warning(
            f"Methods file not found at `{METHODS_FILE}`. "
            "Create `methods.md` in the same folder as `app.py`."
        )

# ============================================================
# TAB 2 — USER GUIDE
# ============================================================
with tab_guide:
    if GUIDE_FILE.exists():
        st.markdown(GUIDE_FILE.read_text(encoding="utf-8"))
    else:
        st.warning(
            f"Guide file not found at `{GUIDE_FILE}`. "
            "Create `guide.md` in the same folder as `app.py`."
        )

# ============================================================
# TAB 3 — TECHNICAL METHODS
# ============================================================
with tab_tech:
    # Common section (tech stack, feature pipeline, GDP growth)
    if TECH_COMMON_FILE.exists():
        st.markdown(TECH_COMMON_FILE.read_text(encoding="utf-8"))
    else:
        st.warning(
            f"Common methods file not found at `{TECH_COMMON_FILE}`. "
            "Create `technical_methods_common.md` in the same folder as `app.py`."
        )

    # Model-specific section (switches on sidebar selection)
    if selected_model == "USE":
        tech_model_file = TECH_USE_FILE
    else:
        tech_model_file = TECH_WASE_FILE

    if tech_model_file.exists():
        st.markdown(tech_model_file.read_text(encoding="utf-8"))
    else:
        st.warning(
            f"Model methods file not found at `{tech_model_file}`. "
            f"Create `technical_methods_{selected_model.lower()}.md` in the same folder as `app.py`."
        )

# ============================================================
# TAB 4 — RESULTS EXPLORER
# ============================================================
with tab_explorer:
    available_isos = sorted(base_df[COL["country_code"]].unique())
    if country_dim is not None and "Latitude" in country_dim.columns:
        map_data = pd.DataFrame({"iso": available_isos})
        map_data["name"] = map_data["iso"].map(code_to_name).fillna(map_data["iso"])

        sel_iso = (
            name_to_code.get(st.session_state.selected_country, None)
            if st.session_state.selected_country != "All Countries" else None
        )
        map_data["colour_val"] = (
            np.where(map_data["iso"] == sel_iso, 1.0, 0.3)
            if sel_iso else 0.5
        )
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
        st.plotly_chart(fig_map, use_container_width=True)
        st.caption("Use the **Country** dropdown in the sidebar to select a country.")

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
            r2_d = r2_from_cols(ov, pv)
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
        if selected_max_dt is not None:
            bh += f'<span class="metric-badge"><span class="label">dt </span><span class="value">≤{selected_max_dt}</span></span>'
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

            # ── Compute % difference between predicted and observed ───
            # err_pct = 100 * (pred - obs) / obs  (signed)
            with np.errstate(divide="ignore", invalid="ignore"):
                err_pct = np.where(
                    (obs != 0) & ~np.isnan(obs) & ~np.isnan(pred),
                    100.0 * (pred - obs) / obs,
                    np.nan,
                )

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
                customdata=np.stack([pct, pred, obs, lo, hi, err_pct], axis=-1),
                hovertemplate=(
                    "<b>Percentile %{customdata[0]:.0f}</b><br>"
                    "Predicted: %{customdata[1]:.3f} $/day<br>"
                    "Observed: %{customdata[2]:.3f} $/day<br>"
                    "Lower CI: %{customdata[3]:.3f}<br>"
                    "Upper CI: %{customdata[4]:.3f}<br>"
                    "Pred − Obs: %{customdata[5]:+.1f}%<extra></extra>"
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
                        customdata=np.stack([pct[om], pred[om], obs[om], err_pct[om]], axis=-1),
                        hovertemplate=(
                            "<b>Percentile %{customdata[0]:.0f}</b><br>"
                            "Observed: %{customdata[2]:.3f} $/day<br>"
                            "Predicted: %{customdata[1]:.3f} $/day<br>"
                            "Pred − Obs: %{customdata[3]:+.1f}%<extra></extra>"
                        ),
                    ))
        else:
            has_lo = COL["lower_predictive_band"] in df.columns
            has_hi = COL["upper_predictive_band"] in df.columns
            agg_d  = {"pred_med": (COL["predicted_consumption"], "median")}
            if has_lo: agg_d["ci_lo"] = (COL["lower_predictive_band"], "median")
            if has_hi: agg_d["ci_hi"] = (COL["upper_predictive_band"], "median")
            agg = df.groupby(COL["percentile"]).agg(**agg_d).reset_index()

            # ── Median observed per percentile (for hover error %) ────
            obs_only = df.dropna(subset=[COL["observed_consumption"]])
            if len(obs_only) > 0:
                obs_med_by_pct = (
                    obs_only.groupby(COL["percentile"])[COL["observed_consumption"]]
                    .median()
                    .rename("obs_med")
                    .reset_index()
                )
                agg = agg.merge(obs_med_by_pct, on=COL["percentile"], how="left")
            else:
                agg["obs_med"] = np.nan

            with np.errstate(divide="ignore", invalid="ignore"):
                agg["err_pct"] = np.where(
                    agg["obs_med"].notna() & (agg["obs_med"] != 0),
                    100.0 * (agg["pred_med"] - agg["obs_med"]) / agg["obs_med"],
                    np.nan,
                )

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
                customdata=np.stack([
                    agg["pred_med"].values,
                    agg["obs_med"].values,
                    agg["err_pct"].values,
                ], axis=-1),
                hovertemplate=(
                    "<b>Percentile %{x}</b><br>"
                    "Predicted (median): %{customdata[0]:.3f} $/day<br>"
                    "Observed (median): %{customdata[1]:.3f} $/day<br>"
                    "Pred − Obs: %{customdata[2]:+.1f}%<extra></extra>"
                ),
            ))
            if is_validated:
                obs_r = df.dropna(subset=[COL["observed_consumption"]])
                if len(obs_r) > 0:
                    ao = obs_r.groupby(COL["percentile"]).agg(
                        med=(COL["observed_consumption"], "median")
                    ).reset_index()
                    # Merge predicted median + err_pct for the hover
                    ao = ao.merge(
                        agg[[COL["percentile"], "pred_med", "err_pct"]],
                        on=COL["percentile"], how="left",
                    )
                    fig.add_trace(go.Scatter(
                        x=ao[COL["percentile"]], y=ao["med"],
                        mode="lines+markers",
                        marker=dict(size=4, color=OBS_COLOUR, symbol="diamond"),
                        line=dict(color=OBS_COLOUR, width=2, dash="dot"),
                        name="Observed Median",
                        customdata=np.stack([
                            ao["pred_med"].values,
                            ao["med"].values,
                            ao["err_pct"].values,
                        ], axis=-1),
                        hovertemplate=(
                            "<b>Percentile %{x}</b><br>"
                            "Observed (median): %{customdata[1]:.3f} $/day<br>"
                            "Predicted (median): %{customdata[0]:.3f} $/day<br>"
                            "Pred − Obs: %{customdata[2]:+.1f}%<extra></extra>"
                        ),
                    ))

        tp = [selected_name if selected_name != "All Countries" else "All Countries"]
        if selected_year != "All Years": tp.append(str(selected_year))
        tp.append(f"{selected_model} · {selected_pred}")
        if selected_max_dt is not None:
            tp.append(f"dt ≤ {selected_max_dt}")
        fig.update_layout(
            template="plotly_white", height=560,
            xaxis_title="Percentile",
            yaxis_title="Consumption per Capita (2021 PPP $/day)",
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
                COL["dt"],
            ] if c in df.columns]
            st.dataframe(
                df[dc].sort_values([COL["country_code"], COL["prediction_year"], COL["percentile"]]),
                use_container_width=True, hide_index=True, height=400,
            )

            
# ============================================================
# TAB 5 — MODEL PERFORMANCE
# ============================================================
with tab_performance:

    eval_model = selected_model
    eval_colour = USE_COLOUR if eval_model == "USE" else WASE_COLOUR

    GROUPBY_COL_MAP = {
        "Region":              "region",
        "Sub-Region":          "sub_region",
        "Income Group":        "income_group",
        "Year":                "target_year" if eval_model == "USE" else "focal_year",
        "Percentile":          "percentile",
        "Country":             "country_name",
        "dt (USE only)":       "dt",
        "Horizon (WASE only)": "horizon",
    }

    ctrl_c1, ctrl_c2, _ = st.columns([1, 1, 3])
    with ctrl_c1:
        eval_groupby = st.selectbox(
            "Break down by",
            ["Overall", "Region", "Income Group", "Sub-Region",
             "Year", "Percentile", "Country",
             "dt (USE only)", "Horizon (WASE only)"],
            key="eval_groupby",
        )

    res_df  = load_diag_residuals(eval_model)
    res_df  = filter_diag_residuals(res_df)
    diag_df = apply_sidebar(fact, force_validated=True)

    METRIC_OPTIONS = ["MAE (Log)", "RMSE (Log)", "Bias (Log)", "R² (Log)",
                      "MAPE %", "MAE (Consumption)", "RMSE (Consumption)"]
    METRIC_MAP = {
        "MAE (Log)":          "mae_log",
        "RMSE (Log)":         "rmse_log",
        "Bias (Log)":         "bias_log",
        "R² (Log)":           "r2_log",
        "MAPE %":             "mape_pct",
        "MAE (Consumption)":  "mae_cons",
        "RMSE (Consumption)": "rmse_cons",
    }

    st.markdown(
        f'<div class="section-header">{eval_model} — Performance Summary</div>',
        unsafe_allow_html=True,
    )

    _use_v  = apply_sidebar(fact, force_validated=True)
    _use_v  = _use_v[_use_v[COL["model_type"]] == "USE"]
    _wase_v = apply_sidebar(fact, force_validated=True)
    _wase_v = _wase_v[_wase_v[COL["model_type"]] == "WASE"]

    def _quick_metrics(df):
        ev = df.dropna(subset=[COL["observed_consumption"], COL["predicted_consumption"]])
        if len(ev) == 0:
            return None
        o = ev[COL["observed_consumption"]].values
        p = ev[COL["predicted_consumption"]].values
        r = p - o
        with np.errstate(divide="ignore", invalid="ignore"):
            ap = np.where(o > 0, np.abs(r / o), np.nan)
        ol = ev[COL["observed_log_consumption"]].values if COL["observed_log_consumption"] in ev.columns else None
        pl = ev[COL["predicted_log_consumption"]].values if COL["predicted_log_consumption"] in ev.columns else None
        r2l = r2_from_cols(ol, pl) if (ol is not None and pl is not None) else np.nan
        return {
            "n":      len(ev),
            "ctries": ev[COL["country_code"]].nunique(),
            "r2":     r2_from_cols(o, p),
            "r2_log": r2l,
            "mae":    float(np.mean(np.abs(r))),
            "mape":   float(np.nanmean(ap) * 100),
        }

    use_m  = _quick_metrics(_use_v)
    wase_m = _quick_metrics(_wase_v)

    # Build dt context string for the summary prose
    _dt_context = f" (dt ≤ {selected_max_dt})" if selected_max_dt is not None else ""

    if eval_model == "USE" and use_m:
        prose = (
            f"Across <strong>{use_m['n']:,} validated observations</strong> in "
            f"<strong>{use_m['ctries']} countries</strong>{_dt_context}, the USE model achieves "
            f"R² = <strong>{use_m['r2']:.3f}</strong> on the consumption scale "
            f"(R² = {use_m['r2_log']:.3f} in log space), "
            f"MAE = <strong>${use_m['mae']:.2f}/day</strong>, "
            f"and MAPE = <strong>{use_m['mape']:.1f}%</strong>. "
            "These metrics reflect strictly out-of-sample LOCO performance — "
            "the model never trained on the country it is evaluated against."
        )
    elif eval_model == "WASE" and wase_m:
        prose = (
            f"Across <strong>{wase_m['n']:,} validated observations</strong> in "
            f"<strong>{wase_m['ctries']} countries</strong>, the WASE model achieves "
            f"R² = <strong>{wase_m['r2']:.3f}</strong> on the consumption scale "
            f"(R² = {wase_m['r2_log']:.3f} in log space), "
            f"MAE = <strong>${wase_m['mae']:.2f}/day</strong>, "
            f"and MAPE = <strong>{wase_m['mape']:.1f}%</strong>. "
            "The substantially harder task of predicting without any survey anchor "
            "explains the wider error margins relative to USE."
        )
    else:
        prose = "No validated rows available for the current filter selection."

    st.markdown(f'<div class="perf-summary">{prose}</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="section-sub">Model = {eval_model} · Prediction Type = {selected_pred} · '
        f'Percentiles {pct_range[0]}–{pct_range[1]}'
        + (f' · dt ≤ {selected_max_dt}' if selected_max_dt is not None else '')
        + (f' · Country = {selected_name}' if selected_name != "All Countries" else '')
        + (f' · Region = {selected_region}' if selected_region != "All" else '')
        + (f' · WFP Countries Only' if selected_wfp == "WFP Countries" else '')
        + '</div>',
        unsafe_allow_html=True,
    )

    if res_df is not None and len(res_df) > 0:
        yr_col    = "target_year" if eval_model == "USE" else "focal_year"
        badge_n   = len(res_df)
        badge_iso = res_df["iso"].nunique() if "iso" in res_df.columns else "—"
        badge_yrs = (
            f"{int(res_df[yr_col].min())}–{int(res_df[yr_col].max())}"
            if yr_col in res_df.columns else "—"
        )
        badge_html = (
            f'<span class="metric-badge"><span class="label">Obs </span><span class="value">{badge_n:,}</span></span>'
            f'<span class="metric-badge"><span class="label">Countries </span><span class="value">{badge_iso}</span></span>'
            f'<span class="metric-badge"><span class="label">Years </span><span class="value">{badge_yrs}</span></span>'
        )
        if selected_max_dt is not None:
            badge_html += f'<span class="metric-badge"><span class="label">dt </span><span class="value">≤{selected_max_dt}</span></span>'
        st.markdown(badge_html, unsafe_allow_html=True)
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
            gcol = GROUPBY_COL_MAP.get(eval_groupby, "region")
            if gcol not in res_df.columns:
                st.info(f"Column `{gcol}` not available for {eval_model}.")
            else:
                tbl_metric = st.selectbox(
                    "Sort table by", METRIC_OPTIONS, index=0, key="perf_tbl_metric"
                )
                tbl_metric_col = METRIC_MAP[tbl_metric]
                grp_tbl = _compute_grouped(res_df, gcol)
                if len(grp_tbl) > 0:
                    grp_tbl = grp_tbl.sort_values(
                        tbl_metric_col, ascending=(tbl_metric_col != "r2_log"),
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

    st.markdown(
        f'<div class="section-header">{eval_model} — Observed vs Predicted</div>',
        unsafe_allow_html=True,
    )
    if len(diag_df) > 0:
        MP  = 40_000
        ev  = diag_df[diag_df[COL["model_type"]] == eval_model]
        ev  = ev.sample(min(MP, len(ev)), random_state=42) if len(ev) > MP else ev
        if len(ev) > MP: st.caption(f"Showing {MP:,} of {len(ev):,} points")
        fig_s  = go.Figure()
        cn_a   = ev[COL["country_code"]].map(code_to_name).fillna(ev[COL["country_code"]])
        ht = (
            "<b>" + cn_a.astype(str) + "</b><br>"
            + "Year: "       + ev[COL["prediction_year"]].astype(int).astype(str) + "<br>"
            + "Percentile: " + ev[COL["percentile"]].astype(int).astype(str) + "<br>"
            + "Predicted: "  + ev[COL["predicted_consumption"]].map("{:.3f}".format) + "<br>"
            + "Observed: "   + ev[COL["observed_consumption"]].map("{:.3f}".format) + "<br>"
            + "Error: "      + ev[COL["percentage_error"]].map("{:.1f}%".format)
        ).tolist()
        fig_s.add_trace(go.Scattergl(
            x=ev[COL["observed_consumption"]], y=ev[COL["predicted_consumption"]],
            mode="markers", marker=dict(size=3, color=eval_colour, opacity=0.3),
            name=eval_model, text=ht, hoverinfo="text",
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
            template="plotly_white", height=600,
            xaxis_title="Observed (2021 PPP $/day)",
            yaxis_title="Predicted (2021 PPP $/day)",
            legend=dict(x=0.02, y=0.98),
            margin=dict(l=60, r=20, t=30, b=60),
        )
        fig_s.update_xaxes(constrain="domain")
        fig_s.update_yaxes(scaleanchor="x", scaleratio=1)
        st.plotly_chart(fig_s, use_container_width=True)

    if res_df is not None and len(res_df) > 0 and eval_groupby != "Overall":
        st.markdown(
            f'<div class="section-header">{eval_model} — Metric by {eval_groupby}</div>',
            unsafe_allow_html=True,
        )
        gcol = GROUPBY_COL_MAP.get(eval_groupby, "region")
        if gcol in res_df.columns:
            chart_metric = st.selectbox(
                "Metric", METRIC_OPTIONS, index=0, key="perf_chart_metric"
            )
            chart_metric_col = METRIC_MAP[chart_metric]
            grp_tbl = _compute_grouped(res_df, gcol)
            if len(grp_tbl) > 0 and chart_metric_col in grp_tbl.columns:
                grp_tbl     = grp_tbl.sort_values(
                    chart_metric_col, ascending=(chart_metric_col != "r2_log")
                )
                is_temporal = eval_groupby in (
                    "Year", "dt (USE only)", "Horizon (WASE only)", "Percentile"
                )
                fig_bar = go.Figure()
                if is_temporal:
                    fig_bar.add_trace(go.Scatter(
                        x=grp_tbl[gcol], y=grp_tbl[chart_metric_col],
                        mode="lines+markers",
                        line=dict(color=eval_colour, width=2.5),
                        marker=dict(size=7, color=eval_colour),
                        customdata=grp_tbl[["n_obs"]].values,
                        hovertemplate=(
                            f"{gcol}: %{{x}}<br>{chart_metric}: "
                            f"%{{y:.4f}}<br>n obs: %{{customdata[0]:,}}<extra></extra>"
                        ),
                    ))
                    if chart_metric_col == "bias_log":
                        fig_bar.add_hline(y=0, line=dict(dash="dash", color="grey", width=1))
                else:
                    if eval_groupby == "Region":
                        bar_colours = [WB_PALETTE.get(r, eval_colour) for r in grp_tbl[gcol]]
                    elif eval_groupby == "Income Group":
                        bar_colours = [INCOME_PALETTE.get(r, eval_colour) for r in grp_tbl[gcol]]
                    else:
                        bar_colours = eval_colour
                    fig_bar.add_trace(go.Bar(
                        x=grp_tbl[chart_metric_col], y=grp_tbl[gcol],
                        orientation="h", marker_color=bar_colours,
                        customdata=grp_tbl[["n_obs"]].values,
                        hovertemplate=(
                            f"%{{y}}<br>{chart_metric}: "
                            f"%{{x:.4f}}<br>n obs: %{{customdata[0]:,}}<extra></extra>"
                        ),
                    ))
                    if chart_metric_col == "bias_log":
                        fig_bar.add_vline(x=0, line=dict(dash="dash", color="grey", width=1))
                h = max(400, len(grp_tbl) * 28) if not is_temporal else 420
                fig_bar.update_layout(
                    template="plotly_white", height=h,
                    xaxis_title=chart_metric if is_temporal else "",
                    margin=dict(l=180 if not is_temporal else 60, r=20, t=30, b=60),
                    showlegend=False,
                )
                if not is_temporal:
                    fig_bar.update_layout(yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown(
        f'<div class="section-header">{eval_model} — Error by Percentile</div>',
        unsafe_allow_html=True,
    )
    _perf_df = diag_df[diag_df[COL["model_type"]] == eval_model]
    if len(_perf_df) > 0:
        pr = []
        for (pt, p), g in _perf_df.groupby([COL["prediction_type"], COL["percentile"]]):
            obs_vals  = g[COL["observed_consumption"]].values
            pred_vals = g[COL["predicted_consumption"]].values
            r = pred_vals - obs_vals
            with np.errstate(divide="ignore", invalid="ignore"):
                ape = np.where(obs_vals > 0, np.abs(r / obs_vals), np.nan)
            ol = g[COL["observed_log_consumption"]].values if COL["observed_log_consumption"] in g.columns else np.array([])
            pl = g[COL["predicted_log_consumption"]].values if COL["predicted_log_consumption"] in g.columns else np.array([])
            rl = pl - ol if len(ol) > 0 and len(pl) > 0 else np.array([])
            pr.append({
                COL["prediction_type"]: pt,
                COL["percentile"]:      p,
                "mae_log":              float(np.mean(np.abs(rl))) if len(rl) > 0 else np.nan,
                "rmse_log":             float(np.sqrt(np.mean(rl ** 2))) if len(rl) > 0 else np.nan,
                "bias_log":             float(rl.mean()) if len(rl) > 0 else np.nan,
                "r2_log":               float(1 - np.sum(rl**2) / np.sum((ol - ol.mean())**2)) if len(rl) > 1 and np.sum((ol - ol.mean())**2) > 1e-8 else np.nan,
                "mape_pct":             float(np.nanmean(ape) * 100),
                "mae_cons":             float(np.mean(np.abs(r))),
                "rmse_cons":            float(np.sqrt(np.mean(r ** 2))),
            })
        pm = pd.DataFrame(pr)
        if len(pm) > 0:
            pct_metric = st.selectbox("Metric", METRIC_OPTIONS, index=0, key="dp_m")
            pct_metric_col = METRIC_MAP[pct_metric]
            fig_p = px.line(
                pm, x=COL["percentile"], y=pct_metric_col,
                color=COL["prediction_type"],
                labels={COL["percentile"]: "Percentile", pct_metric_col: pct_metric},
                markers=True,
                color_discrete_sequence=[eval_colour, SLATE_BLUE],
            )
            fig_p.update_traces(marker=dict(size=4))
            fig_p.update_layout(
                template="plotly_white", height=420,
                legend=dict(orientation="h", y=-0.15),
                margin=dict(l=50, r=20, t=30, b=60),
            )
            st.plotly_chart(fig_p, use_container_width=True)

    st.markdown(
        f'<div class="section-header">{eval_model} — Performance by Year</div>',
        unsafe_allow_html=True,
    )
    _all_val = apply_sidebar(fact, force_validated=True)
    _all_val = _all_val[_all_val[COL["model_type"]] == eval_model]
    if len(_all_val) > 0:
        yr_metric = st.selectbox(
            "Metric", METRIC_OPTIONS, index=1, key="yr_metric"
        )
        yr_metric_col = METRIC_MAP[yr_metric]
        yr = []
        for (pt, y), g in _all_val.groupby([COL["prediction_type"], COL["prediction_year"]]):
            r  = g[COL["predicted_consumption"]].values - g[COL["observed_consumption"]].values
            rl = g[COL["predicted_log_consumption"]].values - g[COL["observed_log_consumption"]].values if COL["predicted_log_consumption"] in g.columns and COL["observed_log_consumption"] in g.columns else np.array([])
            ns = g[[COL["country_code"], COL["prediction_year"]]].drop_duplicates().shape[0]
            with np.errstate(divide="ignore", invalid="ignore"):
                ov  = g[COL["observed_consumption"]].values
                ape = np.where(ov > 0, np.abs(r / ov), np.nan)
            row = {
                COL["prediction_type"]: pt,
                COL["year"]:            int(y),
                "mae_log":   float(np.mean(np.abs(rl))) if len(rl) > 0 else np.nan,
                "rmse_log":  float(np.sqrt(np.mean(rl ** 2))) if len(rl) > 0 else np.nan,
                "bias_log":  float(rl.mean()) if len(rl) > 0 else np.nan,
                "r2_log":    float(1 - np.sum(rl**2) / np.sum((g[COL["observed_log_consumption"]].values - g[COL["observed_log_consumption"]].values.mean())**2)) if len(rl) > 1 and np.sum((g[COL["observed_log_consumption"]].values - g[COL["observed_log_consumption"]].values.mean())**2) > 1e-8 else np.nan,
                "mae_cons":  float(np.mean(np.abs(r))),
                "rmse_cons": float(np.sqrt(np.mean(r ** 2))),
                "mape_pct":  float(np.nanmean(ape) * 100),
                "Surveys":   ns,
            }
            yr.append(row)
        ym = pd.DataFrame(yr)
        if len(ym) > 0 and yr_metric_col in ym.columns:
            yr_surveys  = ym.groupby(COL["year"])["Surveys"].max().to_dict()
            all_yrs     = sorted(ym[COL["year"]].unique())
            tick_labels = [f"({yr_surveys.get(y, 0)})<br>{y}" for y in all_yrs]
            fig_y = go.Figure()
            for pt, grp_y in ym.groupby(COL["prediction_type"]):
                grp_y = grp_y.sort_values(COL["year"])
                fig_y.add_trace(go.Scatter(
                    x=grp_y[COL["year"]], y=grp_y[yr_metric_col],
                    mode="lines+markers",
                    marker=dict(size=5, color=eval_colour),
                    line=dict(color=eval_colour, dash="solid" if pt == "Nowcast" else "dash"),
                    name=f"{eval_model} {pt}",
                ))
            if yr_metric_col == "bias_log":
                fig_y.add_hline(y=0, line=dict(dash="dash", color="grey", width=1))
            fig_y.update_layout(
                template="plotly_white", height=400,
                xaxis=dict(
                    title="(Surveys) Year",
                    tickvals=all_yrs, ticktext=tick_labels,
                    tickangle=0, dtick=1,
                ),
                yaxis_title=yr_metric,
                legend=dict(orientation="h", y=-0.22),
                margin=dict(l=50, r=20, t=30, b=100),
            )
            st.plotly_chart(fig_y, use_container_width=True)

    st.markdown(
        f'<div class="section-header">{eval_model} — Residual Distribution by Region</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-sub">Log residuals (predicted − observed) on validated rows. '
        'Red dashed line = zero bias.</div>',
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

    st.markdown(
        f'<div class="section-header">{eval_model} — Coverage Calibration</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-sub">Empirical vs nominal coverage, rescaled from the 90% band. '
        'Points above the diagonal = over-coverage; below = under-coverage.'
        + (' Post-hoc CI shrinkage (factor 0.55) applied.' if eval_model == "WASE" else '')
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

    st.markdown(
        f'<div class="section-header">{eval_model} — Country-Level Error</div>',
        unsafe_allow_html=True,
    )
    cntry_df = load_diag_country_mae(eval_model)
    if cntry_df is not None and len(cntry_df) > 0:
        if selected_region != "All" and "region" in cntry_df.columns:
            cntry_df = cntry_df[cntry_df["region"] == selected_region]
        if selected_name != "All Countries" and "country_name" in cntry_df.columns:
            cntry_df = cntry_df[cntry_df["country_name"] == selected_name]
        if selected_wfp == "WFP Countries" and "iso" in cntry_df.columns:
            cntry_df = cntry_df[cntry_df["iso"].isin(wfp_isos)]

        cntry_metric = st.selectbox(
            "Metric", ["MAE (Log)", "Bias (Log)", "RMSE (Log)", "MAE (Consumption)", "RMSE (Consumption)"],
            index=0, key="cntry_metric"
        )
        cntry_metric_col = METRIC_MAP.get(cntry_metric, "mae_log")

        n_top   = st.slider(
            "Show top N countries", min_value=10,
            max_value=min(80, max(10, len(cntry_df))),
            value=min(40, len(cntry_df)), step=5, key="eval_top_n",
        )
        sort_asc    = cntry_metric_col == "r2_log"
        show_df     = cntry_df.sort_values(cntry_metric_col if cntry_metric_col in cntry_df.columns else "mae_log",
                                           ascending=sort_asc).head(n_top)
        bar_colours = [eval_colour] * len(show_df)
        x_vals      = show_df[cntry_metric_col].values if cntry_metric_col in show_df.columns else show_df["mae_log"].values
        fig_cntry   = go.Figure(go.Bar(
            x=x_vals, y=show_df["country_name"],
            orientation="h", marker_color=bar_colours,
            customdata=np.stack([
                show_df["n_obs"].values,
                show_df["region"].fillna("Unknown").values,
                (show_df["income_group"].fillna("Unknown").values
                 if "income_group" in show_df.columns else ["—"] * len(show_df)),
                show_df["bias_log"].round(4).values if "bias_log" in show_df.columns else np.zeros(len(show_df)),
            ], axis=1),
            hovertemplate=(
                f"<b>%{{y}}</b><br>{cntry_metric}: %{{x:.4f}}<br>Bias (log): %{{customdata[3]}}<br>"
                "n obs: %{customdata[0]}<br>Region: %{customdata[1]}<br>"
                "Income: %{customdata[2]}<extra></extra>"
            ),
        ))
        fig_cntry.update_layout(
            template="plotly_white", height=max(500, n_top * 22),
            xaxis_title=cntry_metric,
            yaxis=dict(autorange="reversed"),
            margin=dict(l=180, r=20, t=30, b=60),
        )
        st.plotly_chart(fig_cntry, use_container_width=True)
    else:
        st.info(f"Country MAE data not found. Run `conphi_v1_{eval_model.lower()}_diagnostics.py` first.")

    if eval_model == "USE" and res_df is not None and len(res_df) > 0 and "dt" in res_df.columns:
        st.markdown(
            '<div class="section-header">USE — Performance by dt (Years Since Survey)</div>',
            unsafe_allow_html=True,
        )
        dt_metric     = st.selectbox("Metric", METRIC_OPTIONS, index=0, key="dt_metric")
        dt_metric_col = METRIC_MAP[dt_metric]
        dt_grp        = _compute_grouped(res_df, "dt")
        if len(dt_grp) > 0 and dt_metric_col in dt_grp.columns:
            dt_grp = dt_grp.sort_values("dt")
            fig_dt = go.Figure()
            fig_dt.add_trace(go.Scatter(
                x=dt_grp["dt"], y=dt_grp[dt_metric_col],
                mode="lines+markers",
                line=dict(color=USE_COLOUR, width=2.5), marker=dict(size=8),
                customdata=dt_grp[["n_obs"]].values,
                hovertemplate=(
                    "dt=%{x}<br>" + dt_metric +
                    ": %{y:.4f}<br>n=%{customdata[0]:,}<extra></extra>"
                ),
            ))
            if dt_metric_col == "bias_log":
                fig_dt.add_hline(y=0, line=dict(dash="dash", color="grey", width=1))
            fig_dt.update_layout(
                template="plotly_white", height=380,
                xaxis_title="Years Since Survey (dt)", yaxis_title=dt_metric,
                margin=dict(l=60, r=20, t=30, b=60),
            )
            st.plotly_chart(fig_dt, use_container_width=True)

    # ── USE — Value Over Persistence ──────────────────────────
    if eval_model == "USE" and res_df is not None and len(res_df) > 0 and "anchor_year" in res_df.columns:

        persist_raw = _build_persistence(res_df, fact).copy()
        n_total     = len(persist_raw)
        persist_df  = persist_raw.dropna(subset=["persist_resid_log", "resid_log"])
        n_matched   = len(persist_df)
        n_dropped   = n_total - n_matched

        if len(persist_df) > 0:
            st.markdown(
                '<div class="section-header">USE — Value Over Persistence</div>',
                unsafe_allow_html=True,
            )
            _coverage_note = ""
            if n_dropped > 0:
                _coverage_note = (
                    f" <em>({n_dropped:,} of {n_total:,} rows excluded — anchor year "
                    f"predates the validation range and observed anchor consumption "
                    f"is not available; typically older surveys with large dt.)</em>"
                )
            st.markdown(
                '<div class="section-sub">'
                'Persistence (naïve baseline) simply carries forward the last survey estimate '
                'unchanged. Skill score = 1 − MAE<sub>model</sub> / MAE<sub>persistence</sub>: '
                'positive means USE outperforms persistence; zero means no gain; '
                'negative means persistence was better.'
                + _coverage_note
                + '</div>',
                unsafe_allow_html=True,
            )

            # ── Overall skill score badges ────────────────────
            mae_model   = float(np.mean(np.abs(persist_df["resid_log"].values)))
            mae_persist = float(np.mean(np.abs(persist_df["persist_resid_log"].values)))
            skill_overall = 1.0 - mae_model / mae_persist if mae_persist > 1e-12 else np.nan

            badge_persist = (
                f'<span class="metric-badge"><span class="label">MAE Model </span>'
                f'<span class="value">{mae_model:.4f}</span></span>'
                f'<span class="metric-badge"><span class="label">MAE Persistence </span>'
                f'<span class="value">{mae_persist:.4f}</span></span>'
                f'<span class="metric-badge"><span class="label">Skill Score </span>'
                f'<span class="value">{skill_overall:+.3f}</span></span>'
                f'<span class="metric-badge"><span class="label">n </span>'
                f'<span class="value">{len(persist_df):,}</span></span>'
            )
            st.markdown(badge_persist, unsafe_allow_html=True)
            st.markdown("")

            # ── Skill score by dt ─────────────────────────────
            if "dt" in persist_df.columns:
                dt_skill_rows = []
                for dt_val, grp in persist_df.groupby("dt", dropna=False):
                    rl  = grp["resid_log"].values
                    prl = grp["persist_resid_log"].values
                    mae_m = float(np.mean(np.abs(rl)))
                    mae_p = float(np.mean(np.abs(prl)))
                    sk    = 1.0 - mae_m / mae_p if mae_p > 1e-12 else np.nan
                    dt_skill_rows.append({
                        "dt":             dt_val,
                        "mae_model":      mae_m,
                        "mae_persist":    mae_p,
                        "skill":          sk,
                        "n_obs":          len(grp),
                    })
                dt_skill = pd.DataFrame(dt_skill_rows).dropna(subset=["dt"]).sort_values("dt")

                if len(dt_skill) > 0:
                    fig_skill = go.Figure()

                    # Skill score line
                    fig_skill.add_trace(go.Scatter(
                        x=dt_skill["dt"], y=dt_skill["skill"],
                        mode="lines+markers",
                        line=dict(color=USE_COLOUR, width=3),
                        marker=dict(size=10, color=USE_COLOUR, symbol="circle"),
                        name="Skill Score",
                        customdata=np.stack([
                            dt_skill["mae_model"].values,
                            dt_skill["mae_persist"].values,
                            dt_skill["n_obs"].values,
                        ], axis=1),
                        hovertemplate=(
                            "<b>dt = %{x}</b><br>"
                            "Skill: %{y:+.3f}<br>"
                            "MAE model: %{customdata[0]:.4f}<br>"
                            "MAE persistence: %{customdata[1]:.4f}<br>"
                            "n obs: %{customdata[2]:,}<extra></extra>"
                        ),
                    ))

                    # Zero line (no skill)
                    fig_skill.add_hline(
                        y=0, line=dict(dash="dash", color="grey", width=1.2),
                        annotation_text="No gain over persistence",
                        annotation_position="bottom right",
                    )

                    fig_skill.update_layout(
                        template="plotly_white", height=420,
                        xaxis_title="Years Since Survey (dt)",
                        yaxis_title="Skill Score  (1 − MAE_model / MAE_persistence)",
                        margin=dict(l=60, r=20, t=30, b=60),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_skill, use_container_width=True)

            # ── Butterfly: level vs growth error decomposition ─
            st.markdown(
                '<div class="section-header">USE — Level vs Growth Error Decomposition</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="section-sub">'
                'Decomposes USE prediction error into two sources: '
                '<strong>Level error</strong> (error from the anchor survey alone, i.e. persistence error) '
                'and <strong>Growth adjustment</strong> (the incremental change in error from applying '
                'GDP passthrough). Negative growth adjustment = the model corrected the level error; '
                'positive = the model made it worse.'
                '</div>',
                unsafe_allow_html=True,
            )

            persist_view = eval_groupby

            # Decomposition: |USE error| = |level error + growth adjustment|
            # level error = persist_resid_log (anchor − observed)
            # growth adj  = resid_log − persist_resid_log (what the model added)
            persist_df["growth_adj_log"] = persist_df["resid_log"] - persist_df["persist_resid_log"]

            # Map eval_groupby to a residuals-level column name
            _persist_gcol_map = {
                "Region":              "region",
                "Sub-Region":          "sub_region",
                "Income Group":        "income_group",
                "Year":                "target_year",
                "Percentile":          "percentile",
                "Country":             "country_name",
                "dt (USE only)":       "dt",
            }
            if persist_view == "Overall":
                decomp_groups = persist_df.assign(_all="All")
                decomp_gcol   = "_all"
            elif persist_view in _persist_gcol_map and _persist_gcol_map[persist_view] in persist_df.columns:
                decomp_groups = persist_df.copy()
                decomp_gcol   = _persist_gcol_map[persist_view]
            else:
                decomp_groups = persist_df.assign(_all="All")
                decomp_gcol   = "_all"

            decomp_rows = []
            for key, grp in decomp_groups.groupby(decomp_gcol, dropna=False):
                mean_level  = float(grp["persist_resid_log"].mean())
                mean_growth = float(grp["growth_adj_log"].mean())
                mae_level   = float(np.mean(np.abs(grp["persist_resid_log"].values)))
                mae_growth  = float(np.mean(np.abs(grp["growth_adj_log"].values)))
                decomp_rows.append({
                    decomp_gcol:         key,
                    "mean_level_error":  mean_level,
                    "mean_growth_adj":   mean_growth,
                    "mae_level":         mae_level,
                    "mae_growth_adj":    mae_growth,
                    "n_obs":             len(grp),
                })
            decomp_df = pd.DataFrame(decomp_rows)

            if len(decomp_df) > 0:
                if decomp_gcol in ("dt", "target_year", "percentile"):
                    decomp_df = decomp_df.sort_values(decomp_gcol)

                fig_decomp = go.Figure()

                # Level error (persistence) bars
                fig_decomp.add_trace(go.Bar(
                    x=decomp_df[decomp_gcol].astype(str),
                    y=decomp_df["mean_level_error"],
                    name="Level Error (persistence)",
                    marker_color=SLATE_BLUE,
                    customdata=decomp_df[["mae_level", "n_obs"]].values,
                    hovertemplate=(
                        "<b>%{x}</b><br>"
                        "Mean level error: %{y:+.4f}<br>"
                        "MAE (level): %{customdata[0]:.4f}<br>"
                        "n: %{customdata[1]:,}<extra></extra>"
                    ),
                ))

                # Growth adjustment bars (stacked on top)
                fig_decomp.add_trace(go.Bar(
                    x=decomp_df[decomp_gcol].astype(str),
                    y=decomp_df["mean_growth_adj"],
                    name="Growth Adjustment (GDP passthrough)",
                    marker_color=USE_COLOUR,
                    customdata=decomp_df[["mae_growth_adj", "n_obs"]].values,
                    hovertemplate=(
                        "<b>%{x}</b><br>"
                        "Mean growth adj: %{y:+.4f}<br>"
                        "MAE (growth adj): %{customdata[0]:.4f}<br>"
                        "n: %{customdata[1]:,}<extra></extra>"
                    ),
                ))

                fig_decomp.add_hline(
                    y=0, line=dict(dash="dash", color="grey", width=1),
                )

                fig_decomp.update_layout(
                    template="plotly_white", height=440,
                    barmode="relative",
                    xaxis_title=persist_view if persist_view != "Overall" else "",
                    yaxis_title="Mean Log Residual",
                    legend=dict(orientation="h", y=-0.18),
                    margin=dict(l=60, r=20, t=30, b=80),
                )
                st.plotly_chart(fig_decomp, use_container_width=True)

            # ── MAE comparison table ──────────────────────────
            with st.expander("View persistence comparison table"):
                if persist_view != "Overall" and len(decomp_df) > 0:
                    # Build a richer table with model vs persistence MAE
                    tbl_rows = []
                    for key, grp in decomp_groups.groupby(decomp_gcol, dropna=False):
                        rl  = grp["resid_log"].values
                        prl = grp["persist_resid_log"].values
                        mae_m = float(np.mean(np.abs(rl)))
                        mae_p = float(np.mean(np.abs(prl)))
                        sk    = 1.0 - mae_m / mae_p if mae_p > 1e-12 else np.nan
                        tbl_rows.append({
                            decomp_gcol:    key,
                            "n_obs":        len(grp),
                            "mae_model":    mae_m,
                            "mae_persist":  mae_p,
                            "skill":        sk,
                        })
                    tbl_df = pd.DataFrame(tbl_rows)
                    if decomp_gcol in ("dt", "target_year", "percentile"):
                        tbl_df = tbl_df.sort_values(decomp_gcol)
                    else:
                        tbl_df = tbl_df.sort_values("skill", ascending=False)

                    # Render as HTML table
                    tbl_html = '<table class="summary-table"><thead><tr>'
                    for c in [decomp_gcol, "n_obs", "mae_model", "mae_persist", "skill"]:
                        tbl_html += f"<th>{c}</th>"
                    tbl_html += "</tr></thead><tbody>"
                    for _, row in tbl_df.iterrows():
                        tbl_html += "<tr>"
                        for c in [decomp_gcol, "n_obs", "mae_model", "mae_persist", "skill"]:
                            v = row[c]
                            if c == "n_obs":
                                tbl_html += f"<td>{int(v):,}</td>"
                            elif c == "skill":
                                tbl_html += f"<td>{v:+.3f}</td>"
                            elif isinstance(v, (float, np.floating)):
                                tbl_html += f"<td>{v:.4f}</td>"
                            else:
                                tbl_html += f"<td>{v}</td>"
                        tbl_html += "</tr>"
                    tbl_html += "</tbody></table>"
                    st.markdown(tbl_html, unsafe_allow_html=True)
                else:
                    st.write(
                        f"Overall skill score: **{skill_overall:+.3f}** "
                        f"(MAE model = {mae_model:.4f}, MAE persistence = {mae_persist:.4f})"
                    )

    if eval_model == "WASE" and res_df is not None and len(res_df) > 0 and "horizon" in res_df.columns:
        st.markdown(
            '<div class="section-header">WASE — Performance by Forecast Horizon</div>',
            unsafe_allow_html=True,
        )
        hor_metric     = st.selectbox("Metric", METRIC_OPTIONS, index=0, key="hor_metric")
        hor_metric_col = METRIC_MAP[hor_metric]
        h_grp          = _compute_grouped(res_df, "horizon")
        if len(h_grp) > 0 and hor_metric_col in h_grp.columns:
            h_grp = h_grp.sort_values("horizon")
            fig_hor = go.Figure()
            fig_hor.add_trace(go.Scatter(
                x=h_grp["horizon"], y=h_grp[hor_metric_col],
                mode="lines+markers",
                line=dict(color=WASE_COLOUR, width=2.5), marker=dict(size=8),
                customdata=h_grp[["n_obs"]].values,
                hovertemplate=(
                    "Horizon=%{x}<br>" + hor_metric +
                    ": %{y:.4f}<br>n=%{customdata[0]:,}<extra></extra>"
                ),
            ))
            if hor_metric_col == "bias_log":
                fig_hor.add_hline(y=0, line=dict(dash="dash", color="grey", width=1))
            fig_hor.update_layout(
                template="plotly_white", height=380,
                xaxis_title="Forecast Horizon (0 = Nowcast)", yaxis_title=hor_metric,
                margin=dict(l=60, r=20, t=30, b=60),
            )
            st.plotly_chart(fig_hor, use_container_width=True)

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
# TAB 6 — PREDICTOR ANALYSIS
# ============================================================
with tab_predictors:

    pred_model      = selected_model
    pred_pred_type  = selected_pred
    pred_colour     = USE_COLOUR if pred_model == "USE" else WASE_COLOUR
    pmap            = PARAM_MAP_USE if pred_model == "USE" else PARAM_MAP_WASE

    raw_params = load_diag_params(pred_model)
    params_df  = _prep_params(raw_params, pred_model)

    if params_df is None or len(params_df) == 0:
        st.info(
            f"No parameter data found for {pred_model}. "
            f"Run `conphi_v1_{pred_model.lower()}_diagnostics.py` first."
        )
    else:
        if "pred_type" in params_df.columns:
            available_pred_types = params_df["pred_type"].unique().tolist()
            if pred_pred_type in available_pred_types:
                params_df = params_df[params_df["pred_type"] == pred_pred_type].copy()
            else:
                fallback = available_pred_types[0]
                params_df = params_df[params_df["pred_type"] == fallback].copy()
                st.caption(
                    f"Prediction type '{pred_pred_type}' not available for {pred_model} "
                    f"parameters — showing '{fallback}' instead."
                )

        if len(params_df) == 0:
            st.info("No parameter rows available for the current selection.")
        else:
            yr_col = "target_year" if pred_model == "USE" else "focal_year"

            st.markdown(
                f'<div class="section-header">{pred_model} — Predictor Summary</div>',
                unsafe_allow_html=True,
            )

            if pred_model == "USE":
                pos_rows = params_df[params_df["param"] == "beta0_pos"]
                neg_rows = params_df[params_df["param"] == "beta0_neg"]
                if len(pos_rows) > 0 and len(neg_rows) > 0:
                    b_pos = float(pos_rows["mean"].mean())
                    b_neg = float(neg_rows["mean"].mean())
                    prose = (
                        f"Across rolling target years, the average expansion passthrough is "
                        f"<strong>β⁺ = {b_pos:.3f}</strong> and contraction passthrough is "
                        f"<strong>β⁻ = {b_neg:.3f}</strong>. "
                        f"{'Contraction passthrough exceeds expansion passthrough' if abs(b_neg) > abs(b_pos) else 'Expansion and contraction passthrough are broadly similar'}, "
                        f"consistent with the asymmetric transmission of GDP shocks to household consumption. "
                        f"Neither coefficient reaches full passthrough (β = 1), indicating that households "
                        f"do not absorb the full magnitude of GDP fluctuations in either direction."
                    )
                else:
                    prose = "Parameter trajectories are shown below."

            else:
                if "sd" in params_df.columns and len(params_df) > 0:
                    snr = (
                        params_df.groupby("param")
                        .apply(lambda g: (np.abs(g["mean"]) / g["sd"].replace(0, np.nan)).mean())
                        .dropna()
                        .sort_values(ascending=False)
                    )
                    if len(snr) > 0:
                        top3       = snr.head(3)
                        top_labels = [pmap.get(p, p) for p in top3.index]
                        top_snrs   = top3.values
                        low_params = snr[snr < 1].index.tolist()
                        low_labels = [pmap.get(p, p) for p in low_params]

                        mean_by_param = params_df.groupby("param")["mean"].mean()
                        gdp_el    = mean_by_param.get("gdp_elasticity", None)
                        ineq      = mean_by_param.get("baseline_inequality", None)
                        u5        = mean_by_param.get("u5_mortality_effect", None)
                        rural     = mean_by_param.get("rural_share_effect", None)
                        gov_rev   = mean_by_param.get("gov_rev_effect", None)
                        res_rents = mean_by_param.get("res_rents_effect", None)

                        prose = (
                            f"The WASE model predicts consumption distributions from structural "
                            f"country indicators. The strongest predictors by signal-to-noise ratio are: "
                            f"<strong>{top_labels[0]}</strong> (SNR = {top_snrs[0]:.1f})"
                        )
                        if len(top_labels) > 1:
                            prose += f", <strong>{top_labels[1]}</strong> (SNR = {top_snrs[1]:.1f})"
                        if len(top_labels) > 2:
                            prose += f", and <strong>{top_labels[2]}</strong> (SNR = {top_snrs[2]:.1f})"
                        prose += ". "

                        if gdp_el is not None:
                            prose += (
                                f"GDP per capita elasticity averages <strong>{gdp_el:.3f}</strong>, "
                                f"meaning a 1% increase in GDP per capita is associated with approximately "
                                f"a {gdp_el * 100:.1f}% increase in log-consumption at the distribution centre. "
                            )
                        if ineq is not None:
                            prose += (
                                f"The baseline inequality parameter (log-logistic shape) averages "
                                f"<strong>{ineq:.3f}</strong> — higher values indicate a more compressed "
                                f"distribution with lower inequality. "
                            )
                        if u5 is not None:
                            direction = "negative" if u5 < 0 else "positive"
                            prose += (
                                f"Under-5 mortality has a {direction} association with consumption "
                                f"(coefficient = {u5:.4f}), consistent with its role as a proxy for "
                                f"broader deprivation. "
                            )
                        if gov_rev is not None or res_rents is not None:
                            prose += "Fiscal variables (government revenue"
                            if res_rents is not None:
                                prose += f", resource rents [{res_rents:.4f}]"
                            prose += ") contribute modest additional signal beyond GDP. "
                        if low_labels:
                            prose += (
                                f"Parameters with SNR below 1 — where the posterior is barely "
                                f"distinguishable from the prior — include "
                                f"{', '.join(f'<em>{l}</em>' for l in low_labels[:5])}. "
                                f"These predictors have little empirical support in the current "
                                f"training data and could be candidates for removal or tighter priors "
                                f"in future model iterations."
                            )
                    else:
                        prose = "Predictor importance is shown below."
                else:
                    prose = "Predictor importance is shown below."

            st.markdown(f'<div class="perf-summary">{prose}</div>', unsafe_allow_html=True)

            with st.expander("📋 Parameter data diagnostics"):
                st.write("**Columns:**", list(params_df.columns))
                st.write("**Unique params:**", sorted(params_df["param"].unique().tolist()))
                st.write("**Rows:**", len(params_df))
                st.dataframe(params_df.head(20), use_container_width=True)

            if pred_model == "USE":
                avail_params = params_df["param"].unique()

                st.markdown(
                    '<div class="section-header">USE — Asymmetric GDP Passthrough Over Time</div>',
                    unsafe_allow_html=True,
                )
                colour_map = {"beta0_pos": "#2ca02c", "beta0_neg": "#d62728"}
                label_map  = {k: PARAM_MAP_USE.get(k, k) for k in colour_map}
                fig_par    = go.Figure()
                plotted    = False
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

                st.markdown(
                    '<div class="section-header">USE — Effective Passthrough Across Percentiles</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<div class="section-sub">'
                    'Effective passthrough at each percentile: β_eff(p) = β₀ ± β_p · bell(p), '
                    'where bell(p) = 4·p·(1−p). Shaded band shows 90% CI envelope averaged '
                    'across all available target years.'
                    '</div>',
                    unsafe_allow_html=True,
                )
                pct_grid = np.linspace(0.01, 0.99, 99)
                bell     = 4 * pct_grid * (1 - pct_grid)

                has_spline_params = any(
                    p in avail_params for p in ["beta0_pos", "beta0_neg", "beta_p_pos", "beta_p_neg"]
                )
                if has_spline_params:
                    fig_spline = go.Figure()
                    spline_cases = [
                        ("beta0_pos", "beta_p_pos", "#2ca02c", "Expansion (β⁺ + β_p⁺·bell(p))"),
                        ("beta0_neg", "beta_p_neg", "#d62728", "Contraction (β⁻ + β_p⁻·bell(p))"),
                    ]
                    for b0_param, bp_param, c, label in spline_cases:
                        if b0_param not in avail_params:
                            continue
                        b0_rows = params_df[params_df["param"] == b0_param]
                        bp_rows = params_df[params_df["param"] == bp_param] if bp_param in avail_params else pd.DataFrame()

                        b0_mean = float(b0_rows["mean"].mean()) if len(b0_rows) > 0 else 0.0
                        b0_q05  = float(b0_rows["q05"].mean())  if "q05" in b0_rows.columns and len(b0_rows) > 0 else b0_mean
                        b0_q95  = float(b0_rows["q95"].mean())  if "q95" in b0_rows.columns and len(b0_rows) > 0 else b0_mean

                        bp_mean = float(bp_rows["mean"].mean()) if len(bp_rows) > 0 else 0.0
                        bp_q05  = float(bp_rows["q05"].mean())  if "q05" in bp_rows.columns and len(bp_rows) > 0 else bp_mean
                        bp_q95  = float(bp_rows["q95"].mean())  if "q95" in bp_rows.columns and len(bp_rows) > 0 else bp_mean

                        eff_mean = b0_mean + bp_mean * bell
                        eff_lo   = b0_q05  + bp_q05  * bell
                        eff_hi   = b0_q95  + bp_q95  * bell

                        fig_spline.add_trace(go.Scatter(
                            x=np.concatenate([pct_grid * 100, pct_grid[::-1] * 100]),
                            y=np.concatenate([eff_hi, eff_lo[::-1]]),
                            fill="toself", fillcolor=hex_to_rgba(c, 0.15),
                            line=dict(width=0), showlegend=False, hoverinfo="skip",
                        ))
                        fig_spline.add_trace(go.Scatter(
                            x=pct_grid * 100, y=eff_mean,
                            mode="lines",
                            line=dict(color=c, width=2.5),
                            name=label,
                            hovertemplate=(
                                "Percentile: %{x:.0f}<br>"
                                f"{label}: %{{y:.4f}}<extra></extra>"
                            ),
                        ))

                    fig_spline.add_hline(
                        y=1.0, line=dict(dash="dash", color="grey", width=1.2),
                        annotation_text="Full passthrough (β=1)",
                        annotation_position="top right",
                    )
                    fig_spline.add_hline(
                        y=0.0, line=dict(dash="dot", color="lightgrey", width=1),
                        annotation_text="Zero passthrough",
                        annotation_position="bottom right",
                    )
                    fig_spline.update_layout(
                        template="plotly_white", height=420,
                        xaxis_title="Percentile",
                        yaxis_title="Effective Passthrough Coefficient",
                        legend=dict(orientation="h", x=0.15, y=-0.15),
                        margin=dict(l=60, r=20, t=30, b=80),
                    )
                    st.plotly_chart(fig_spline, use_container_width=True)
                else:
                    st.info("Spline tilt parameters (beta_p_pos / beta_p_neg) not found in parameter data.")

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
                        ))
                    fig_par2.update_layout(
                        template="plotly_white", height=380,
                        xaxis_title="Target Year",
                        legend=dict(orientation="h", x=0.3, y=-0.15),
                        margin=dict(l=60, r=20, t=30, b=80),
                    )
                    st.plotly_chart(fig_par2, use_container_width=True)

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
                        ), row=row, col=col)
                    fig_grid.update_layout(
                        template="plotly_white", height=nr * 240,
                        margin=dict(t=60, b=40, l=50, r=20),
                    )
                    st.plotly_chart(fig_grid, use_container_width=True)

            if pred_model == "WASE":

                st.markdown(
                    '<div class="section-header">WASE — Coefficient Forest Plot</div>',
                    unsafe_allow_html=True,
                )
                summary = (
                    params_df.groupby("param")[["mean", "q05", "q95"]]
                    .mean().reset_index()
                )
                summary["param_label"] = summary["param"].map(pmap).fillna(summary["param"])
                summary = summary.sort_values("mean", ascending=False)

                if not summary["mean"].isna().all():
                    fig_forest = go.Figure()
                    fig_forest.add_trace(go.Scatter(
                        x=summary["mean"], y=summary["param_label"],
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
                    fig_forest.add_vline(x=0, line=dict(dash="dash", color="grey", width=1.2))
                    fig_forest.update_layout(
                        template="plotly_white",
                        height=max(400, len(summary) * 45),
                        xaxis_title="Posterior Mean",
                        yaxis=dict(autorange="reversed"),
                        margin=dict(l=260, r=20, t=30, b=60),
                    )
                    st.plotly_chart(fig_forest, use_container_width=True)

                if yr_col in params_df.columns:
                    st.markdown(
                        '<div class="section-header">WASE — Coefficient Stability Across Folds</div>',
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
                            ), row=row, col=col)
                        fig_traj.update_layout(
                            template="plotly_white", height=nr * 250,
                            margin=dict(t=60, b=40, l=50, r=20),
                        )
                        st.plotly_chart(fig_traj, use_container_width=True)

                rbf_params = [p for p in ["rbf_weight_1", "rbf_weight_2", "rbf_weight_3"]
                              if p in params_df["param"].unique()]
                if rbf_params:
                    st.markdown(
                        '<div class="section-header">WASE — RBF Spline Weights Across Percentiles</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        '<div class="section-sub">'
                        'The WASE model uses 5 RBF (radial basis function) knots evenly spaced '
                        'across the percentile range. The chart below shows the posterior mean '
                        'weight at each knot, indicating where the spline contributes most to '
                        'shaping the predicted distribution.'
                        '</div>',
                        unsafe_allow_html=True,
                    )
                    n_knots    = 5
                    knot_pcts  = np.linspace(0.1, 0.9, n_knots) * 100
                    all_rbf    = [p for p in params_df["param"].unique() if p.startswith("rbf_weight")]
                    all_rbf    = sorted(all_rbf, key=lambda x: int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else 99)
                    rbf_knot_x = knot_pcts[:len(all_rbf)]
                    rbf_means  = []
                    rbf_lo     = []
                    rbf_hi     = []
                    for rp in all_rbf:
                        rows = params_df[params_df["param"] == rp]
                        rbf_means.append(float(rows["mean"].mean()))
                        rbf_lo.append(float(rows["q05"].mean()) if "q05" in rows.columns else float(rows["mean"].mean()))
                        rbf_hi.append(float(rows["q95"].mean()) if "q95" in rows.columns else float(rows["mean"].mean()))

                    fig_rbf = go.Figure()
                    fig_rbf.add_trace(go.Scatter(
                        x=np.concatenate([rbf_knot_x, rbf_knot_x[::-1]]),
                        y=np.concatenate([rbf_hi, rbf_lo[::-1]]),
                        fill="toself", fillcolor=hex_to_rgba(WASE_COLOUR, 0.20),
                        line=dict(width=0), showlegend=False, hoverinfo="skip",
                    ))
                    fig_rbf.add_trace(go.Scatter(
                        x=rbf_knot_x, y=rbf_means,
                        mode="lines+markers",
                        line=dict(color=WASE_COLOUR, width=2.5),
                        marker=dict(size=10, color=WASE_COLOUR, symbol="diamond"),
                        name="RBF Weight (posterior mean)",
                        hovertemplate=(
                            "Knot at p=%{x:.0f}th pct<br>"
                            "Weight: %{y:.4f}<extra></extra>"
                        ),
                    ))
                    fig_rbf.add_hline(y=0, line=dict(dash="dot", color="grey", width=1))
                    fig_rbf.update_layout(
                        template="plotly_white", height=380,
                        xaxis_title="Percentile (knot location)",
                        yaxis_title="Posterior Mean RBF Weight",
                        legend=dict(orientation="h", y=-0.15),
                        margin=dict(l=60, r=20, t=30, b=80),
                    )
                    st.plotly_chart(fig_rbf, use_container_width=True)

                st.markdown(
                    '<div class="section-header">WASE — Predictor Signal-to-Noise Ratio</div>',
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