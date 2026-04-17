# User Guide — Con φ Consumption Distribution Explorer

Con φ (Conphi) is a Bayesian consumption distribution nowcasting and forecasting system. This guide explains how to use the dashboard and use the controls effectively.

---

## Tabs

### Results Explorer
Shows predicted consumption distributions (and observed survey distributions where available) as a curve across percentiles for the selected country, year, and model. Use this tab to inspect individual country trajectories or browse aggregate patterns across the full portfolio.

### Model Performance
Diagnostic summaries of how well each model performs against held-out survey data. Includes observed vs predicted scatterplots, error breakdowns by region, income group, year, and percentile, residual distributions, and coverage calibration charts. Use this tab to understand model reliability and identify where predictions should be treated with more caution.

### Predictor Analysis
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

### Filters

The filters — Year, Yrs since last survey, Region, Sub-Region, and Country — work together as a cascading system. Selections at a broader level restrict the options available at a narrower level.
How the cascade works:

- Year is applied first, limiting all downstream options to years with available data.
- Yrs since last survey (USE only) filters to observations where the survey anchor is within the selected recency window (e.g. ≤2 means the prediction is based on a survey no more than 2 years old). This further restricts the available Regions, Sub-Regions, and Countries.
- Region filters the available Sub-Regions and Countries to those within the selected region.
- Sub-Region further filters the available Countries.
- Country selects a single country for detailed inspection.

**Important:** If you select a Region or Sub-Region that is incompatible with a Country you previously selected, the Country filter will automatically reset to *All Countries*. This prevents silent mismatches where the geographic filters contradict each other.

**WFP \| All Countries**
Restricts the portfolio to countries where WFP has an active operational presence. Useful for focusing performance assessment and results on the core operational scope.


### 8.2 Metrics reported

For each vintage year and horizon:

- **MAE (Consumption)**: mean absolute error in $/day
- **RMSE (Consumption)**: root mean squared error in $/day
- **MAPE %**: mean absolute percentage error
- **MAE (Log)**: mean absolute error in log space
- **RMSE (Log)**: root mean squared error in log space
- **Bias (Log)**: mean signed error in log space (positive = over-prediction)
- **R² (Log)**: coefficient of determination in log space
- **Coverage 90%**: fraction of observations falling within the 90% predictive interval

---

## USE — Value Over Persistence

This section of the Model Performance tab answers a simple question: is the USE model actually adding value over just carrying the last survey forward unchanged? "Persistence" is the naïve baseline — assume nothing has changed since the last survey was conducted and use that anchor value as the prediction. It's the bar any forecasting model has to clear to justify its existence.

The headline metric is the **skill score**: 1 − MAE_model / MAE_persistence. Positive means USE beats persistence, zero means no improvement, negative means persistence was better and you'd be better off ignoring the model. As a rule of thumb, a skill score around 0.05–0.10 is a meaningful gain at the country-year level given the noise in survey data; values above 0.20 are strong.

### Skill score by dt

This line chart shows the skill score on the y-axis against years since the last survey on the x-axis. Read it as the model's added value as the survey anchor ages. Expect the line to start near zero at dt=1 (when the anchor is fresh, persistence is hard to beat — there's not much GDP movement to translate into consumption change yet) and rise as dt grows (when the anchor is stale, persistence becomes increasingly wrong and any sensible GDP-passthrough adjustment helps). A flat or declining line at high dt would be a warning sign — it would suggest the passthrough mechanism is not extracting useful signal from cumulative GDP growth, and that the model is essentially treading water.

The dashed grey line at zero is the no-skill threshold. Hover any point to see the underlying MAE for both the model and the persistence baseline alongside the sample size.

### Level vs Growth Error Decomposition

This stacked bar chart decomposes the average USE prediction error into two additive components for each group (region, year, dt, etc., depending on what you've selected in *Break down by*):

- **Level error** (blue bars) — the error you'd have made using persistence alone. This is structural: it reflects how much consumption has actually drifted from the anchor over time, regardless of what the model does. Large blue bars mean the anchor was a poor approximation of the truth.
- **Growth adjustment** (terracotta bars) — what USE added on top by applying GDP passthrough. The sign matters here: a *negative* growth adjustment that partially cancels a positive level error means the model successfully pulled the prediction closer to the truth. A *positive* growth adjustment that compounds with a positive level error means the model made things worse.

The visual cue is whether the two bars push in opposite directions (model is correcting the anchor's drift, good) or the same direction (model is amplifying error, bad). The total bar height is the net mean residual; grouped breakdowns let you see whether the model's value-add is concentrated in particular regions, years, or dt windows. This is often more diagnostic than the skill score alone, because it tells you *why* the model is or isn't beating persistence rather than just whether it is.

The expander below the chart provides a comparison table with model MAE, persistence MAE, and the per-group skill score, which is useful for spotting groups where the model underperforms even when the overall skill score is positive.

---

## Tips

- The **Results Explorer map** is clickable — selecting a country on the choropleth will update the Country filter in the sidebar.
- The **Model Performance** and **Predictor Analysis** tabs both respond to the Model Type and Prediction Type selected in the sidebar, so switch between USE and WASE in the sidebar to compare.
- In the **Model Performance** tab, the *Break down by* dropdown lets you slice error metrics by region, income group, year, country, or forecast horizon — useful for identifying systematic weaknesses.
- Diagnostic charts in the Performance and Predictor tabs are only populated after the relevant diagnostic scripts (`conphi_v1_use_diagnostics.py` and `conphi_v1_wase_diagnostics.py`) have been run and their outputs uploaded to GCS.