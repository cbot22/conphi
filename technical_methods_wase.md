
---

## WASE: Without Any Survey Estimate

### Purpose and Role Within Con φ

WASE is deployed when a country has no usable consumption survey, or when a
structural prediction is preferred over a survey-anchored projection. It predicts the
full household consumption distribution purely from country-level structural
indicators. WASE learns this mapping from countries that do have surveys, then applies
it to countries that do not — making it the "never-seen countries" arm of Con φ.

---

### Core Model: Hierarchical Log-Logistic Regression

Each observation is a single percentile from a single country-year survey. WASE
predicts log-consumption at percentile p as:

```
log_cons = LEVEL + SHAPE × logit(p) + SPLINE(logit(p)) + ε
```

Where:

- **LEVEL** — average consumption, driven by GDP per capita, under-5 mortality,
  female education, rural population share, GDP growth, government revenue, resource
  rents, plus region and sub-region random effects.

- **SHAPE** — inequality (the log-logistic shape parameter γ), varying by region
  and sub-region. Higher γ indicates a more compressed distribution with lower
  inequality.

- **SPLINE** — nonlinear corrections via radial basis functions, capturing departures
  from the log-logistic shape (e.g. heavier tails, subsistence floors).

The `logit(p)` transformation reflects the log-logistic (Fisk) distribution: if
consumption is Fisk-distributed, log-consumption is linear in logit(p). The spline
term relaxes this assumption for the real-world distributions we observe.

#### Why Log-Logistic?

The log-logistic distribution is a natural choice for modelling consumption
distributions. It has a closed-form CDF and quantile function, its shape parameter
directly controls inequality (analogous to the Gini coefficient), and it nests the
log-normal as a limiting case. The Fisk distribution — the name used in the poverty
measurement literature — is equivalent to the log-logistic. The `logit(p)`
transformation means that if consumption were perfectly Fisk-distributed, the model
would reduce to a simple linear regression of log-consumption on logit(p). The RBF
spline term captures the departures from this idealised shape that real consumption
distributions exhibit.

---

### Hierarchical Structure

The model uses a three-level Bayesian hierarchy:

```
Global → Region (World Bank regions) → Sub-region (region23)
```

This gives partial pooling: data-sparse sub-regions borrow strength from their parent
region and the global mean, while data-rich sub-regions can deviate as the data
warrants. Both the level (intercept) and shape (inequality) components have their own
hierarchy, as does the RBF spline correction. Sum-to-zero constraints at each level
ensure identifiability.

---

### Covariates

WASE uses the following centred, weighted covariates to predict the level component:

| Covariate | Source | Role |
|-----------|--------|------|
| `log_gdp_pp` (centred) | IMF WEO (PPPPC) | Primary determinant of consumption level |
| `log_gdp_pp²` (centred) | Derived | Captures diminishing returns to GDP |
| `gdp_growth` (centred) | diff(log_gdp_pp) | Short-run deviation from structural level |
| `|gdp_growth|` (centred) | Derived | Growth volatility effect |
| `u5mort_lag3` (centred) | World Bank | Proxy for broader deprivation; lagged 3 years |
| `rural_pct_lag3` (centred) | World Bank | Rural population share; lagged 3 years |
| `edu_mean_fem_lag3` (centred) | IHME | Female education; lagged 3 years |
| `gov_rev_lag3` (centred) | IMF WEO | Government revenue as % of GDP; lagged 3 years |
| `res_rents_lag3` (centred) | World Bank | Natural resource rents as % of GDP; lagged 3 years |

All covariates are centred using weighted means from the training set, where the
weights are the survey-count weights (see below). This ensures the intercept β₀
represents the expected log-consumption at the weighted-average covariate profile.

---

### Survey-Count Weighting

Countries with many surveys would otherwise dominate the training set. WASE
downweights them using `1/√(n_surveys)`, normalised to mean 1. This ensures fragile
states with one or two surveys get proportional influence — exactly the countries the
model most needs to work for. The weighting is applied in covariate centering, the
empirical Bayes OLS, and the likelihood.

---

### RBF Percentile Basis

Five Gaussian radial basis functions are placed at evenly spaced points on the
logit(p) axis (at logit-percentile values −3.0, −1.5, 0.0, 1.5, 3.0) with width
parameter 1.5. Each has a global coefficient plus a region-level deviation, allowing
different regions to have different tail shapes. The RBF basis captures nonlinear
departures from the log-logistic shape — for example, subsistence floors that compress
the left tail, or top-coding effects that truncate the right tail.

---

### Empirical Bayes Priors

Before fitting the full Bayesian model, a weighted OLS regression is run on the
training data to obtain sensible starting points for the regression coefficients.
These OLS estimates become the prior means for the Bayesian model, with pre-specified
prior SDs controlling how far the posterior can deviate. This is fold-specific — no
data leakage. The OLS includes region-specific slopes on logit(p) to capture regional
inequality differences.

---

### Prior Specification

| Parameter | Prior | Rationale |
|-----------|-------|-----------|
| β₀ (Intercept) | Normal(OLS estimate, 1.0) | Centred on empirical Bayes; wide SD |
| β_gdp | Normal(OLS, 0.10) | GDP elasticity tightly centred on OLS |
| β_gdp² | Normal(0, 0.05) | Zero-centred; allows modest curvature |
| β_u5m | Normal(OLS, 0.10) | Under-5 mortality effect |
| β_rural | Normal(OLS, 0.003) | Rural share effect; very tight |
| β_edu_fem | Normal(OLS, 0.05) | Female education effect |
| β_gdp_growth | Normal(0, 0.05) | Zero-centred; GDP growth short-run effect |
| β_gov_rev | Normal(0, 0.004) | Zero-centred; very tight exploratory prior |
| β_res_rents | Normal(0, 0.004) | Zero-centred; very tight exploratory prior |
| γ₀ (Shape, raw) | Normal(inv_transform(OLS), 0.35) | Log-logistic shape, centred on OLS |
| σ_base (Noise baseline) | HalfNormal(0.7) | Observation noise floor |
| σ_tail (Noise tail widening) | HalfNormal(0.5) | Extra noise at distribution tails |
| σ_region_level | HalfNormal(0.40) | Region-level intercept pooling |
| σ_r23_level | HalfNormal(0.20) | Sub-region-level intercept pooling |
| σ_region_shape | HalfNormal(0.12) | Region-level shape pooling |
| σ_r23_shape | HalfNormal(0.08) | Sub-region-level shape pooling |
| α_spline (RBF weights) | Normal(0, 0.04) | Global spline coefficients |
| σ_region_spline | HalfNormal(0.02) | Region-level spline deviations |

---

### Likelihood

```
ε ~ Laplace(0, σ_base + σ_tail × |logit(p)|)
```

WASE uses a Laplace (double-exponential) likelihood rather than Gaussian. The Laplace
distribution is more robust to outliers than the Gaussian — appropriate for
consumption data with measurement error and extreme values. The heteroscedastic noise
structure (`σ_base + σ_tail × |logit(p)|`) allows wider noise at the distribution
tails where measurement is less precise.

---

### JAX Compilation Strategy

All training arrays are padded to a fixed maximum size (`PAD_N`) with a boolean
`obs_mask`. Masked (padded) entries contribute nothing to the ELBO and do not affect
inference. This means JAX compiles the SVI step function exactly once across all LOCO
folds, rather than recompiling for each fold's different training set size — a
critical efficiency optimisation when running hundreds of folds.

---

### Estimation

Stochastic Variational Inference with a Block Neural Autoregressive Flow (BNAF) guide,
using a cosine decay learning rate schedule that anneals from 0.005 to 1% of that
value over 12,000 steps. After convergence, 4,000 posterior samples (1,000 × 4 groups)
are drawn. A post-hoc variance inflation factor of 1.5× is applied to observation
noise to correct for SVI's known tendency to underestimate posterior variance.

---

### Prediction

For countries whose region or sub-region was not seen in training, WASE falls back to
the global mean (for the level component) or the global shape parameter (for
inequality). Predictions include both posterior mean intervals (parameter uncertainty)
and predictive intervals (parameter + observation noise, with 1.5× inflation).

---

### Validation

WASE uses two cross-validation phases:

1. **Rolling LOCO** — for each focal year, train on all data before that year, hold
   out one country at a time, and predict its distribution.

2. **All-time LOCO** — train on all years, hold out one country entirely. This
   assesses the model's ability to predict countries it has never seen.

A third phase, **Production**, trains on all data and predicts every LMIC country-year
in the target range (2015–2026), whether or not a survey exists. This is the
operational output used for WFP's expenditure monitoring.

---

### Hyperparameter Summary

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| SVI steps | 12,000 | Higher than USE due to more complex model |
| Learning rate | 0.005 (cosine decay to 1%) | Cosine annealing helps settle into sharper modes |
| Posterior samples | 4,000 | 1,000 × 4 groups |
| Variance inflation | 1.5× | Corrects SVI under-dispersion |
| Likelihood | Laplace | Robust to outliers |
| BNAF hidden factors | [8, 8] | Adequate for ~20+ parameter model |
| RBF knots | 5 (at −3.0, −1.5, 0.0, 1.5, 3.0) | Covers full logit(p) range |
| RBF width | 1.5 | Smooth, overlapping basis functions |
| Random seed | 42 | Reproducibility |
| Focal years | 2015–2026 | Rolling evaluation window |

---

### Known Limitations

1. **No temporal dynamics**: WASE is a cross-sectional model — it predicts what a
   country's distribution "should" look like given its structural indicators, but
   does not model how the distribution evolves over time. GDP growth enters as a
   covariate but does not accumulate.

2. **Hierarchical structure fixed**: the region → sub-region hierarchy is fixed by
   World Bank and UN classifications. Countries that fit poorly in their assigned
   group (e.g. small island states) may be systematically misestimated.

3. **SVI approximation with inflation**: the 1.5× variance inflation is a crude
   correction. In principle, better-calibrated intervals could be obtained through
   MCMC or more sophisticated variational families, at higher computational cost.

4. **Covariate imputation**: several covariates (government revenue, resource rents)
   require cascade imputation for countries with missing data. The imputed values
   are treated as known in the likelihood, which understates the true uncertainty
   for those countries.

5. **Survey-count weighting**: the `1/√(n)` weighting is a heuristic. The optimal
   weighting scheme depends on the relative informativeness of each survey, which
   is not straightforward to estimate.
