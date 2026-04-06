
---

## USE: Update Survey Estimate

### Purpose and Role Within Con φ

USE is deployed when a country has an existing consumption survey: it projects that
survey's consumption distribution forward (or backward) to a target year using
observed and forecast GDP growth. The key question USE answers is: *given what we
knew about this country's consumption distribution at the time of its last survey,
and given what happened to GDP between then and now, what does the consumption
distribution look like today?*

USE is the "survey-anchored" arm of Con φ. It complements WASE (Without Any Survey
Estimate), which predicts consumption distributions purely from structural country
indicators without any survey anchor.

---

### Core Model: Asymmetric GDP Passthrough

#### The Passthrough Concept

The central premise is that changes in GDP per capita translate into changes in
household consumption, but not one-for-one. The "passthrough rate" is the fraction
of GDP growth that reaches households. USE estimates this rate from historical pairs
of consumption surveys within the same country where the survey methodology remained
consistent (a "comparable spell").

#### Why Asymmetric?

Economic contractions do not simply reverse the gains of expansions. During downturns,
consumption tends to fall more sharply than it rises during booms — households buffer
against gains but are forced to absorb losses. USE therefore estimates separate
passthrough rates for positive and negative GDP growth years. This is not imposed as
an assumption; the model learns the asymmetry from data. Empirically, expansion
passthrough averages approximately 0.44–0.47 and contraction passthrough approximately
0.65–0.85, confirming that losses transmit more strongly than gains.

#### Mathematical Specification

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

This year-by-year decomposition — rather than simply taking total growth and splitting
at zero — captures the realistic case where a country experiences both expansion and
contraction years within a single survey gap. Because GDP growth is computed as
`diff(log_gdp_pp)` (see Feature Pipeline above), summing these year-by-year
differences gives the exact log growth ratio over the period — there is no
approximation error from the decomposition.

#### Distributional Tilt

Passthrough varies by percentile via a logit transformation:

```
β⁺(p) = β₀⁺ + β_p⁺ × logit_p_std
β⁻(p) = β₀⁻ + β_p⁻ × logit_p_std
```

Where `logit_p_std` is the standardised logit of the percentile (mean 0, sd 1,
computed from training data). The logit transform maps percentiles from (0,1) to
(−∞, +∞) and naturally stretches the tails relative to the centre. Standardisation
ensures the tilt coefficients β_p are on a comparable scale to the base passthrough
coefficients β₀.

A negative β_p means poorer households benefit proportionally more from growth
(pro-poor passthrough); a positive value means richer households gain more. This
allows the model to capture the empirical reality that GDP growth does not shift the
entire distribution uniformly.

#### Likelihood

```
ε ~ StudentT(ν, 0, σ)
```

A Student-t likelihood is used rather than Gaussian. The Student-t has heavier tails
controlled by the degrees of freedom parameter ν, which is estimated from data. This
provides two advantages:

1. **Robustness**: consumption data contains measurement error and genuine outliers
   from structural breaks; the Student-t downweights these naturally rather than
   letting them dominate the fit.
2. **Calibrated uncertainty**: the estimated ν (typically 3–5 in practice) produces
   well-calibrated 90% predictive intervals without ad-hoc adjustments.

When ν is large (>30), the Student-t converges to the Normal; when small, it
accommodates heavy tails. The data determines which regime is appropriate.

---

### Prior Specification

| Parameter | Prior | Rationale |
|-----------|-------|-----------|
| β₀⁺ (Expansion passthrough) | Normal(0.70, 0.30) | Centred on the World Bank's canonical 0.70 estimate; SD allows range from near-zero to full passthrough |
| β₀⁻ (Contraction passthrough) | Normal(0.70, 0.50) | Same centre but wider SD — contraction episodes are rarer, so more prior uncertainty is warranted |
| β_p⁺ (Expansion tilt) | Normal(0.00, 0.10) | Zero-centred (no prior pro-poor or pro-rich expectation); tight SD keeps tilt small relative to base |
| β_p⁻ (Contraction tilt) | Normal(0.00, 0.30) | Zero-centred but wider — sparse contraction data requires more prior latitude |
| σ (Observation noise) | HalfNormal(0.10) | Log-consumption residuals are typically modest (~±10%); half-normal has a long right tail for larger noise |
| ν (Degrees of freedom) | Gamma(2.0, 0.1) | Mean 20, concentrates mass above 2 (variance exists), allows 3–4 (heavy tails) through 50+ (≈Normal) |

---

### Data Preparation

#### Training Pair Construction

Each training observation is a pair of consecutive consumption surveys from the same
country within the same comparable spell. For each pair, the script records the change
in log consumption at every percentile and the cumulative positive and negative GDP
growth between the two survey years.

The `cumulative_growth_asymmetric()` function walks through each year from y_start+1
to y_end and splits annual GDP growth (i.e. each year's `diff(log_gdp_pp)`) into
positive and negative components. If any year in the range is missing GDP data, the
entire pair is excluded — the model never trains on pairs where the GDP path is
incomplete.

The pair-level data is expanded to percentile level, and logit(p) values are standardised
to mean 0, sd 1 from the training set (with centre and scale saved for application to test 
data).

Con φ USE trains exclusively on comparable survey pairs — consecutive surveys within 
the same comparable spell — rather than attempting to bridge non-comparable sequences. 
This contrasts with the World Bank's PIP, which develops three approaches (A, B, C) 
that progressively incorporate more information from older and non-comparable surveys to 
reconstruct complete historical poverty series back to 1981. 

The difference reflects the distinct objectives: PIP requires a poverty estimate for 
every country-year over four decades, necessitating assumptions about how much to trust
old data; Con φ is a forecasting system that projects forward from recent survey anchors, 
so the historical reconstruction problem does not arise. Training only on comparable pairs
ensures that the learned passthrough rates are not contaminated by measurement artefacts 
from changes in survey design.

#### Logit Percentile Transformation

Percentiles (1–99) are transformed to logit space:

```
logit(p) = log(p / (1 − p))
```

The logit transform is natural for percentile data: if consumption follows a
log-logistic (Fisk) distribution, log-consumption is linear in logit(p). The
transform also stretches the tails, giving the model more resolution where
distributional differences matter most.

---

### Estimation

Stochastic Variational Inference with a Block Neural Autoregressive Flow (BNAF) guide.
Adam optimiser with learning rate 0.005 for 10,000 steps. After convergence, 4,000
posterior samples (1,000 × 4 groups) are drawn from the fitted guide and propagated
through the generative model.

---

### Prediction

For each anchor country, prediction year, and percentile, the model:

1. Computes cumulative positive and negative GDP growth from the anchor year to the
   prediction year (using vintage-appropriate IMF WEO data).
2. For each of 500 posterior draws (subsampled from 4,000):
   - Computes percentile-varying passthrough
   - Projects log-consumption forward
   - Adds Student-t observation noise

This produces two types of uncertainty intervals: posterior mean intervals (parameter
uncertainty only) and predictive intervals (parameter + observation noise).

---

### Evaluation

Evaluation is strictly out-of-sample via vintage control. The model trained on
vintage year *t* never sees consumption data at or after year *t*. When predictions
for year *t* are compared to actual surveys at year *t*, this constitutes a genuine
out-of-sample test.

---

### Hyperparameter Summary

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

### Known Limitations

1. **No country-specific parameters**: the passthrough rates are global (shared across
   all countries). Country-level heterogeneity is absorbed into observation noise.

2. **GDP forecast uncertainty not propagated**: when projecting beyond the vintage
   year, IMF WEO GDP forecasts are treated as known. In reality, GDP forecasts have
   their own uncertainty that compounds with passthrough uncertainty.

3. **Comparable spell dependency**: the model requires comparable spell metadata.
   Countries with frequent methodology changes have fewer usable training pairs.

4. **Linear passthrough**: the model assumes a linear relationship between GDP growth
   and consumption growth (conditional on percentile). Non-linear effects are not
   captured.

5. **SVI approximation**: the BNAF normalising flow provides an approximate posterior.
   For this 6-parameter model, the approximation error is expected to be small, but
   it is not zero.
