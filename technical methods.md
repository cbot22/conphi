# Con φ - Codebase.**
All model code is available on [GitHub](https://github.com/cbot22/conphi).

# Con φ v1 - USE (Update Survey Estimate): Technical Methods

## 1. Purpose and Role Within Con φ

USE is one of two sub-models in the Con φ system. It is deployed when a country has an existing consumption survey: USE projects that survey's consumption distribution forward (or backward) to a target year using observed and forecast GDP growth. The key question USE answers is: *given what we knew about this country's consumption distribution at the time of its last survey, and given what happened to GDP between then and now, what does the consumption distribution look like today?*

USE is the "survey-anchored" arm of Con φ. It complements WASE (Without Any Survey Estimate), which predicts consumption distributions purely from structural country indicators without any survey anchor.

---

## 2. Core Model: Asymmetric GDP Passthrough

### 2.1 The Passthrough Concept

The central premise is that changes in GDP per capita translate into changes in household consumption, but not one-for-one. The "passthrough rate" is the fraction of GDP growth that reaches households. USE estimates this rate from historical pairs of consumption surveys within the same country where the survey methodology remained consistent (a "comparable spell").

### 2.2 Why Asymmetric?

Economic contractions do not simply reverse the gains of expansions. During downturns, consumption tends to fall more sharply than it rises during booms - households buffer against gains but are forced to absorb losses. USE therefore estimates separate passthrough rates for positive and negative GDP growth years. This is not imposed as an assumption; the model learns the asymmetry from data. Empirically, expansion passthrough averages approximately 0.44–0.47 and contraction passthrough approximately 0.65–0.85, confirming that losses transmit more strongly than gains.

### 2.3 Mathematical Specification

For consecutive surveys within the same comparable spell, the model specifies:

```
Δlog(cons_p) = β⁺(p) × Σ⁺ Δlog(GDP) + β⁻(p) × Σ⁻ Δlog(GDP) + ε
```

Where:

- **Δlog(cons_p)**: Change in log consumption at percentile p between two survey years
- **Σ⁺ Δlog(GDP)**: Cumulative positive GDP growth between the two survey years (expansion component)
- **Σ⁻ Δlog(GDP)**: Cumulative negative GDP growth between the two survey years (contraction component)
- **β⁺(p)**, **β⁻(p)**: Percentile-varying passthrough coefficients for expansions and contractions
- **ε**: Observation noise

The GDP growth decomposition is performed year-by-year. For a pair of surveys at years y₀ and y₁:

```
Σ⁺ = Σ_{t=y₀+1}^{y₁}  max(g_t, 0)     (sum of positive growth years)
Σ⁻ = Σ_{t=y₀+1}^{y₁}  min(g_t, 0)     (sum of negative growth years)
```

This year-by-year decomposition - rather than simply taking total growth and splitting at zero - captures the realistic case where a country experiences both expansion and contraction years within a single survey gap.

### 2.4 Distributional Tilt

Passthrough varies by percentile via a logit transformation:

```
β⁺(p) = β₀⁺ + β_p⁺ × logit_p_std
β⁻(p) = β₀⁻ + β_p⁻ × logit_p_std
```

Where `logit_p_std` is the standardised logit of the percentile (mean 0, sd 1, computed from training data). The logit transform maps percentiles from (0,1) to (-∞, +∞) and naturally stretches the tails relative to the centre. Standardisation ensures the tilt coefficients β_p are on a comparable scale to the base passthrough coefficients β₀.

A negative β_p means poorer households benefit proportionally more from growth (pro-poor passthrough); a positive value means richer households gain more. This allows the model to capture the empirical reality that GDP growth does not shift the entire distribution uniformly.

### 2.5 Likelihood

```
ε ~ StudentT(ν, 0, σ)
```

A Student-t likelihood is used rather than Gaussian. The Student-t has heavier tails controlled by the degrees of freedom parameter ν, which is estimated from data. This provides two advantages:

1. **Robustness**: consumption data contains measurement error and genuine outliers from structural breaks; the Student-t downweights these naturally rather than letting them dominate the fit
2. **Calibrated uncertainty**: the estimated ν (typically 3–5 in practice) produces well-calibrated 90% predictive intervals without ad-hoc adjustments

When ν is large (>30), the Student-t converges to the Normal; when small, it accommodates heavy tails. The data determines which regime is appropriate.

---

## 3. Prior Specification

### 3.1 Expansion Passthrough β₀⁺

```python
beta0_pos ~ Normal(0.70, 0.30)
```

Prior mean of 0.70 reflects the empirical expectation that roughly 70% of GDP growth reaches households on average. The standard deviation of 0.30 allows the posterior to range from near-zero to full passthrough (β=1) while gently penalising extreme values. This prior is weakly informative - it expresses the belief that passthrough is positive and less than full, but allows substantial deviation.

### 3.2 Contraction Passthrough β₀⁻

```python
beta0_neg ~ Normal(0.70, 0.50)
```

The prior mean is identical to expansion (0.70), expressing no prior asymmetry - any asymmetry must be learned from data. The wider standard deviation (0.50 vs 0.30) reflects that contraction episodes are rarer in the data, so there is genuinely more prior uncertainty about this parameter. The wider prior gives the data more room to pull the posterior away from the mean.

### 3.3 Expansion Distributional Tilt β_p⁺

```python
beta_p_pos ~ Normal(0.00, 0.10)
```

Zero-centred prior: no prior expectation that growth is pro-poor or pro-rich. The tight standard deviation (0.10) reflects the belief that distributional tilt effects are small relative to the base passthrough. This prevents the tilt from dominating the fit when data is sparse.

### 3.4 Contraction Distributional Tilt β_p⁻

```python
beta_p_neg ~ Normal(0.00, 0.30)
```

Also zero-centred but with a wider prior (0.30 vs 0.10), again because contraction data is sparse and we need more prior latitude. This allows the model to learn that contractions may affect different parts of the distribution differently - for example, that the poorest households may be disproportionately affected by economic downturns.

### 3.5 Observation Noise σ

```python
sigma ~ HalfNormal(0.10)
```

HalfNormal(0.10) constrains noise to be positive and centres mass near small values. The 0.10 scale reflects that log-consumption residuals are typically modest (a noise of 0.10 in log space corresponds to roughly ±10% in consumption). The half-normal has a long right tail, so if the data requires larger noise, the posterior can accommodate it.

### 3.6 Degrees of Freedom ν

```python
nu ~ Gamma(2.0, 0.1)
```

Gamma(2.0, 0.1) has mean 20 and concentrates mass above 2 (ensuring the Student-t variance exists) while allowing values as low as 3–4 (very heavy tails) or as high as 50+ (approximately Normal). This prior is deliberately permissive - it lets the data determine the tail behaviour. In practice, the posterior typically settles around ν ≈ 3–5, indicating genuinely heavy tails in consumption growth residuals.

---

## 4. Data Preparation

### 4.1 Feature Files and Vintage Control

Each run is anchored to a specific vintage year. The input is a parquet file (`conphi_v1_features_{year}.parquet`) that represents the world as it was known at the start of that year:

- All consumption surveys published before that date
- GDP estimates and forecasts from the corresponding IMF WEO release

Consumption outcomes at or after the vintage year are explicitly masked to NaN before training. This ensures the model never trains on data it would not have had in real time - a strict no-leakage protocol.

### 4.2 GDP Lookup Construction

A GDP lookup dictionary `{(iso, year): growth_rate}` is built from all rows in the feature file before any filtering. This is important because GDP growth is needed for every country-year in the projection path, including years and countries that may not have consumption surveys.

### 4.3 Consumption Survey Filtering

Only rows where `welfare_type == "consumption"` are retained. This excludes income-based surveys, which measure a different welfare concept and are not comparable. Countries in the `DROP_ISOS` list (currently just Kosovo/XKX due to data quality issues) are excluded.

### 4.4 Logit Percentile Transformation

Percentiles (1–99) are transformed to logit space:

```python
p = percentile / 100.0
p = clip(p, 1e-6, 1 - 1e-6)  # avoid log(0)
logit_p = log(p / (1 - p))
```

The clipping at 1e-6 prevents numerical issues at the boundaries. The logit transform is natural for percentile data: if consumption follows a log-logistic (Fisk) distribution, log-consumption is linear in logit(p). The transform also stretches the tails, giving the model more resolution where distributional differences matter most.

### 4.5 Comparable Spell Identification

The `comparable_spell_raw` column identifies periods within which a country's survey methodology remained consistent enough that changes in measured consumption reflect genuine changes in living standards, not methodological artefacts. Only survey pairs within the same comparable spell are used for training.

### 4.6 Duplicate Handling

A check for duplicated (iso, year, percentile) rows is performed. If duplicates exist, only the first is kept and a warning is printed. This is a defensive measure against data quality issues in the feature pipeline.

---

## 5. Training Pair Construction

### 5.1 Pair Definition

Each training observation is a pair of consecutive consumption surveys from the same country within the same comparable spell. For each pair, the script records:

- The change in log consumption at every percentile: `Δlog(cons) = log(cons_end) - log(cons_start)`
- The cumulative positive and negative GDP growth between the two survey years

### 5.2 Maximum Survey Gap Filter

Pairs where the survey gap exceeds `MAX_DT` years (currently 50) are dropped. Very long gaps accumulate forecast errors and structural noise that make the GDP-consumption relationship unreliable. The threshold of 50 is permissive - it essentially only excludes implausible gaps.

### 5.3 GDP Decomposition

For each pair, the `cumulative_growth_asymmetric()` function walks through each year from y_start+1 to y_end and splits annual GDP growth into positive and negative components:

```python
for y in range(y_start + 1, y_end + 1):
    g = gdp_dict.get((iso, y), NaN)
    if g > 0:
        cum_g_pos += g
    else:
        cum_g_neg += g
```

If any year in the range is missing GDP data, the entire pair is flagged as invalid and excluded. This ensures the model never trains on pairs where the GDP path is incomplete.

### 5.4 Expansion to Percentile Level

The pair-level data (one row per survey pair) is expanded to percentile level by merging with the consumption data at each percentile for both the start and end surveys. The final training observation has:

- `delta_log_cons`: change in log consumption at this percentile
- `delta_log_gdp_pos`: cumulative positive GDP growth
- `delta_log_gdp_neg`: cumulative negative GDP growth
- `logit_p`: logit-transformed percentile

### 5.5 Logit(p) Standardisation

The logit(p) values from the training set are standardised to mean 0, sd 1:

```python
logit_p_std = (logit_p - center) / scale
```

The center and scale are saved and applied identically to test data during prediction. This standardisation ensures the tilt coefficients β_p are on a numerically stable and interpretable scale.

---

## 6. Estimation: Stochastic Variational Inference

### 6.1 Why SVI?

Stochastic Variational Inference is used rather than MCMC (e.g. NUTS) for computational efficiency. The model is fitted independently for each of 11 vintage years (2015–2025), and SVI converges in seconds per fit versus minutes for MCMC. The trade-off is that SVI provides an approximate posterior rather than exact samples, but for this model with 6 parameters the approximation is adequate.

### 6.2 Guide Architecture

```python
guide = AutoBNAFNormal(consumption_growth_model, num_flows=1, hidden_factors=[8, 8])
```

A Block Neural Autoregressive Flow (BNAF) guide is used. This is a normalising flow that transforms a simple base distribution (multivariate Normal) through a neural network to approximate the true posterior. The flow can capture correlations and non-Gaussianity in the posterior that a simple mean-field guide would miss. The architecture uses 1 flow with hidden layers of size 8×8 - small but sufficient for 6 parameters.

### 6.3 Optimiser and Training

```python
optimizer = optax.adam(learning_rate=0.005)
svi_steps = 10_000
```

Adam optimiser with learning rate 0.005 for 10,000 steps. The ELBO (Evidence Lower Bound) loss is minimised. No learning rate scheduling is used for USE (unlike WASE which uses cosine decay) - the simpler model converges reliably with a constant rate.

### 6.4 Posterior Sampling

After SVI convergence, posterior samples are drawn in two stages:

1. **Guide samples**: 4,000 samples (1,000 × 4 groups) from the fitted guide distribution
2. **Model samples**: these guide samples are propagated through the generative model to obtain deterministic quantities (e.g. predicted means)

This two-stage approach separates the approximate posterior over parameters from the model's forward predictions.

---

## 7. Prediction

### 7.1 Anchor Selection

For each country, the most recent survey within a comparable spell is selected as the anchor. The anchor provides the baseline consumption distribution from which forward projections are made.

### 7.2 Projection Mechanism

For each anchor country, prediction year, and percentile:

1. Compute cumulative positive and negative GDP growth from the anchor year to the prediction year
2. For each posterior draw i (500 draws subsampled from the 4,000 posterior samples):
   - Compute percentile-varying passthrough: `β_total_pos[i] = β₀⁺[i] + β_p⁺[i] × logit_p_std`
   - Compute predicted growth: `δ[i] = β_total_pos[i] × g_pos + β_total_neg[i] × g_neg`
   - Compute latent prediction: `log_cons_pred[i] = log_cons_anchor + δ[i]`
   - Add observation noise: `log_cons_obs[i] = log_cons_pred[i] + StudentT(ν[i], 0, σ[i])`

### 7.3 Uncertainty Quantification

The prediction produces two types of intervals:

- **Posterior mean interval** (μ): 5th and 95th percentiles of `log_cons_pred` across draws - captures parameter uncertainty only
- **Predictive interval** (ỹ): 5th and 95th percentiles of `log_cons_obs` across draws - captures parameter uncertainty plus observation noise

The predictive interval is always wider than the posterior mean interval. Both are reported as 90% intervals.

### 7.4 Prediction Modes

- **Nowcast** (`PREDICTION_MODE = "nowcast"`): projects to the vintage year itself (horizon 0)
- **Forecast** (`PREDICTION_MODE = "forecast"`): projects beyond the vintage year by `FORECAST_HORIZON` years (1–5), using IMF WEO GDP forecasts embedded in the feature file

The model is fitted identically in both modes - only the prediction horizon differs. The posterior parameters are mode-invariant.

---

## 8. Evaluation

### 8.1 Out-of-Sample Design

Evaluation is strictly out-of-sample via vintage control. The model trained on vintage year t never sees consumption data at or after year t. When predictions for year t are compared to actual surveys at year t, this constitutes a genuine out-of-sample test.

### 8.3 Actuals Merging

After prediction, actual consumption values from the feature file are merged back into the prediction dataframe. Rows where actuals exist become "Validated" rows; rows without actuals are "Prediction Only". Only validated rows contribute to evaluation metrics.

---

## 9. Output Structure

Each vintage year produces:

- `posterior_samples_{year}.npz`: raw NumPyro posterior samples
- `inference_data_{year}.nc`: ArviZ InferenceData object for posterior diagnostics
- `posterior_summary_{year}.csv`: parameter means, SDs, and 90% HDI
- `train_pairs_{year}.parquet`: the training data used for this vintage
- `logit_p_spec_{year}.json`: the logit(p) standardisation parameters (center, scale)
- `predictions_{mode}_{year}.parquet`: predictions with uncertainty intervals
- `metrics_{mode}_{year}.parquet`: evaluation metrics

Consolidated master files across all vintages are written incrementally (after each vintage year) so partial results are always available if the run is interrupted.

---

## 10. Hyperparameter Summary

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| SVI steps | 10,000 | Sufficient for 6-parameter model convergence |
| Learning rate | 0.005 | Standard for Adam with BNAF guide |
| Posterior samples | 4,000 | 1,000 × 4 groups, adequate for summary statistics |
| Prediction draws | 500 | Subsampled from 4,000 for efficiency |
| Max survey gap (dt) | 50 years | Permissive; only excludes implausible gaps |
| Random seed | 42 | Reproducibility |
| BNAF hidden factors | [8, 8] | Small but sufficient for 6 parameters |
| Likelihood | Student-t | Heavy tails for robustness |
| Vintage years | 2015–2025 | Rolling evaluation window |
| Excluded ISOs | XKX (Kosovo) | Data quality |

---

## 11. Known Limitations

1. **No country-specific parameters**: the passthrough rates are global (shared across all countries). Country-level heterogeneity in passthrough is absorbed into the observation noise, contributing to wider predictive intervals.

2. **GDP forecast uncertainty not propagated**: when projecting beyond the vintage year, IMF WEO GDP forecasts are treated as known. In reality, GDP forecasts have their own uncertainty that compounds with passthrough uncertainty.

3. **Comparable spell dependency**: the model requires comparable spell metadata. Countries with frequent methodology changes have fewer usable training pairs.

4. **Linear passthrough**: the model assumes a linear relationship between GDP growth and consumption growth (conditional on percentile). Non-linear effects (e.g. diminishing returns to growth) are not captured.

5. **SVI approximation**: the BNAF normalising flow provides an approximate posterior. For this 6-parameter model, the approximation error is expected to be small, but it is not zero.

## Technical Implementation

Con φ is implemented in Python and runs on GPU in Google Colab. The core
computational stack is as follows.

**Probabilistic programming and inference** is handled by
[NumPyro](https://num.pyro.ai/en/stable/), a lightweight probabilistic programming
library built on JAX. Stochastic Variational Inference is performed using NumPyro's
`SVI` module with a `Trace_ELBO` objective and an
[AutoBNAFNormal](https://num.pyro.ai/en/stable/autoguide.html#numpyro.infer.autoguide.AutoBNAFNormal)
guide - a Block Neural Autoregressive Flow that learns a flexible, non-Gaussian
approximation to the posterior. Learning rate scheduling uses
[Optax](https://optax.readthedocs.io/en/latest/), a gradient processing library
for JAX. USE uses a flat Adam learning rate, while WASE uses a cosine decay schedule
that anneals from the initial learning rate to 1% of its value over the course of
training, which helps the optimiser settle into sharper posterior modes in the more
complex hierarchical model.

**Variational inference and normalising flows.** Rather than full Markov Chain Monte
Carlo (MCMC), Con φ uses Stochastic Variational Inference (SVI), which reframes
posterior inference as an optimisation problem. SVI minimises the Kullback-Leibler
divergence KL(q‖p) between a variational approximation q and the true posterior p.
The variational family used here is a
[Block Neural Autoregressive Flow (BNAF)](https://arxiv.org/abs/1904.04676) - a
normalising flow that learns an invertible transformation from a simple base
distribution to a flexible approximate posterior, capable of capturing non-Gaussian
geometry and posterior correlations that mean-field approximations cannot represent.

This approach offers significant computational advantages: SVI converges in minutes
on GPU where MCMC might require hours, making the rolling LOCO cross-validation
procedure - which requires fitting hundreds of separate models - practically feasible.

However, SVI with KL(q‖p) minimisation carries well-documented drawbacks. The
objective tends to produce approximations that are overconfident - the variational
posterior typically underestimates posterior variance, particularly in the tails.
This manifests in Con φ as systematically narrow credible intervals, which is why a
post-hoc variance inflation factor of 1.5× is applied to WASE predictive intervals
to achieve empirical coverage closer to nominal levels. A second limitation is that
SVI provides no convergence guarantees analogous to MCMC's ergodic theorem - the ELBO
may converge to a local optimum, and there is no direct equivalent of the Gelman-Rubin
R̂ diagnostic to assess whether inference has succeeded.

**Array computation and automatic differentiation** are provided by
[JAX](https://jax.readthedocs.io/en/latest/), which compiles and executes numerical
code on GPU via XLA. All WASE training arrays are padded to a fixed size so that JAX
compiles the SVI step function exactly once across all cross-validation folds,
avoiding repeated recompilation overhead.

**Posterior diagnostics** use [ArviZ](https://python.arviz.org/en/stable/), a library
for exploratory analysis of Bayesian models, used to compute posterior summaries and
store inference data in NetCDF format.

**Data processing** uses [pandas](https://pandas.pydata.org/docs/) and
[NumPy](https://numpy.org/doc/stable/). Model outputs are serialised as
[Apache Parquet](https://parquet.apache.org/) files via
[PyArrow](https://arrow.apache.org/docs/python/index.html) for efficient storage and
retrieval.

**Visualisation and dashboarding** use [Plotly](https://plotly.com/python/) for
interactive figures and [Streamlit](https://docs.streamlit.io/) for the web application.

**Hardware.** Model training runs on an NVIDIA GPU (typically an A100 or T4 in Google
Colab). GPU acceleration is essential for the SVI fitting loop; typical runtimes are
30–90 seconds per LOCO fold depending on training set size and the number of SVI steps.
