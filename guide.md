# 🧭 User Guide — Con φ Consumption Distribution Explorer

Con φ (Conphi) is a Bayesian consumption distribution nowcasting and forecasting system. This guide explains how to use the dashboard and use the controls effectively.

---

## Tabs

### 📖 Overview & Methods
A plain-language description of the Con φ methodology, covering how the two sub-models work, what data they use, and how outputs should be interpreted. Start here if you are new to the system.

### 🌍 Results Explorer
Shows predicted consumption distributions (and observed survey distributions where available) as a curve across percentiles for the selected country, year, and model. Use this tab to inspect individual country trajectories or browse aggregate patterns across the full portfolio.

### 📊 Model Performance
Diagnostic summaries of how well each model performs against held-out survey data. Includes observed vs predicted scatterplots, error breakdowns by region, income group, year, and percentile, residual distributions, and coverage calibration charts. Use this tab to understand model reliability and identify where predictions should be treated with more caution.

### 📐 Predictor Analysis
Detailed inspection of the model's learned parameters. For USE, this shows how GDP passthrough coefficients evolve across rolling target years. For WASE, this shows the coefficient forest plot, predictor signal-to-noise ratios, and RBF spline weights. Use this tab to understand what is driving predictions and which predictors have strong vs weak empirical support.

---

## Sidebar Controls

### Model Controls

**Model Type**
Selects between the two Con φ sub-models:
- **USE** (Update Survey Estimate) — projects consumption forward from a known survey anchor using asymmetric GDP passthrough. Use this when a recent survey exists for the country.
- **WASE** (Without Any Survey Estimate) — predicts the full consumption distribution from structural country indicators alone, without any survey anchor. Use this for countries with no recent survey data.

**Prediction Type**
- **Nowcast** — estimate for the current period.
- **Forecast** — projection into future periods beyond the last available data.

**Data Type**
- **Validated** — rows where an observed survey value exists, allowing direct comparison of predicted vs observed. Use this for performance assessment.
- **Predicted** — all model output rows, including periods with no observed counterpart. Use this for the full coverage picture.

---

### Display Options

**Show 90% Confidence Bands**
Toggles the shaded uncertainty envelope on the Results Explorer chart. The band reflects posterior uncertainty around the predicted consumption curve. Wider bands indicate lower confidence, typically at the tails of the distribution or for countries with sparse data.

**Percentile Range**
Restricts the chart and all downstream calculations to a slice of the consumption distribution. For example, setting 10–90 focuses on the middle of the distribution and excludes the most uncertain tail estimates.

---

### Geographic Filters

The four geographic filters — **Year**, **Region**, **Sub-Region**, and **Country** — work together as a cascading system. Selections at a broader level restrict the options available at a narrower level.

**How the cascade works:**
1. **Year** is applied first, limiting all downstream options to years with available data.
2. **Region** filters the available Sub-Regions and Countries to those within the selected region.
3. **Sub-Region** further filters the available Countries.
4. **Country** selects a single country for detailed inspection.

**Important:** If you select a Region or Sub-Region that is incompatible with a Country you previously selected, the Country filter will automatically reset to *All Countries*. This prevents silent mismatches where the geographic filters contradict each other.

**WFP \| All Countries**
Restricts the portfolio to countries where WFP has an active operational presence. Useful for focusing performance assessment and results on the core operational scope.

---

## Tips

- The **Results Explorer map** is clickable — selecting a country on the choropleth will update the Country filter in the sidebar.
- The **Model Performance** and **Predictor Analysis** tabs both respond to the Model Type and Prediction Type selected in the sidebar, so switch between USE and WASE in the sidebar to compare.
- In the **Model Performance** tab, the *Break down by* dropdown lets you slice error metrics by region, income group, year, country, or forecast horizon — useful for identifying systematic weaknesses.
- Diagnostic charts in the Performance and Predictor tabs are only populated after the relevant diagnostic scripts (`conphi_v1_use_diagnostics.py` and `conphi_v1_wase_diagnostics.py`) have been run and their outputs uploaded to GCS.
