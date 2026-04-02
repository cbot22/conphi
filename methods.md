# Con φ — Methodology

The Con φ (pronounced *con fie*) model predicts household consumption in US Dollars
(at 2021 purchasing power parity) for each percentile of the consumption distribution —
from the poorest 1% to the richest 1% — across Low and Middle Income Countries.
It contains two distinct sub-models, each designed for a different data situation:

**Con φ ~ USE (Update Survey Estimate)** asks: *Given a country's most recent survey
estimates of consumption, how do we project forward using GDP growth?* This model
requires at least one prior consumption survey for the country.

**Con φ ~ WASE (Without Any Survey Estimate)** asks: *What would we predict for this
country if we had no consumption data at all, based on its current level of human
and economic development?* This model requires only structural predictor data — no
consumption survey is needed for the target country.

The system can nowcast (estimate the current year) or forecast up to five years ahead
(currently configured for one-year-ahead forecasting). Estimates are updated each
April and October, in line with the IMF's twice-yearly publication of economic
indicators.

---
## Data Sources
The model uses publicly available data only.
### Consumption Surveys
At the core of the system is the **World Bank Poverty and Inequality Platform (PIP)**
dataset, which contains household consumption-expenditure estimates — expressed in
2021 PPP US dollars per person per day — for each percentile of the population across
over 100 countries and more than 1,000 surveys conducted since 1977. These survey
estimates are the outcome variable — what the models are trying to predict or validate
against. Only national-level, consumption welfare type surveys are used in training.

The table below shows the number of distinct PIP surveys available in training data
up to each target year:

| Target Year | Surveys in Training | Countries Covered |
|:-----------:|:-------------------:|:-----------------:|
| 2015        | 687                 | 113               |
| 2016        | 723                 | 115               |
| 2017        | 754                 | 118               |
| 2018        | 779                 | 118               |
| 2019        | 818                 | 119               |
| 2020        | 855                 | 121               |
| 2021        | 873                 | 121               |
| 2022        | 900                 | 121               |
| 2023        | 919                 | 123               |
| 2024        | 927                 | 123               |
| 2025        | 928                 | 123               |

Surveys are counted as distinct national-level country-year combinations with observed
consumption outcomes within a comparable spell.

### Economic Indicators

The main economic predictors come from the **IMF World Economic Outlook (WEO)**
datasets, published twice yearly in April and October. Because past GDP estimates
are regularly revised using household survey data — including the very surveys in
the PIP dataset — using current revised estimates would introduce a look-ahead bias.
To prevent this, the pipeline uses historical vintage WEO files: the dataset exactly
as it was released at a given point in time, rather than current revised figures.

The USE model uses only PIP and WEO data. The WASE model additionally draws on
structural country indicators from three further sources:

- **Institute for Health Metrics and Evaluation (IHME)** — under-5 mortality rates
- **UN Inter-Agency Group for Child Mortality Estimation (UN IGME)** — child mortality
- **World Bank** — mean years of female education, rural population share, government
  revenue and expenditure as a share of GDP, and natural resource rents as a share of GDP

All structural covariates are lagged by three years to reflect the typical publication
delays that would apply in a genuine real-time scenario.

---

## How the Models Work

### Con φ ~ USE Model

The USE model requires at least one prior consumption survey to project forward from.
It builds on the finding — documented in the 2025 World Bank research paper
[Constructing Comparable Global Poverty Trends](https://openknowledge.worldbank.org/entities/publication/293e54f3-b0f4-42e0-baec-e10edcf7cca6)
— that when two surveys are methodologically comparable, changes in household
consumption between them can be estimated from GDP growth using a passthrough
coefficient: GDP growth of 1% translates into roughly 0.7% growth in household
consumption across the distribution.

The requirement for methodological comparability is important. Household surveys
differ in their welfare aggregate (consumption vs. income), coverage (national vs.
urban-only), recall periods, and questionnaire design. When the methodology changes
between two surveys, observed changes in measured consumption may reflect the
methodological shift rather than genuine changes in living standards. Con φ therefore
restricts training to survey pairs within the same *comparable spell* — a period
during which the underlying methodology is consistent enough that changes in measured
consumption can be attributed to real economic change rather than measurement artefact.

USE is a flat global model — it does not include any hierarchical or regional
structure. All countries share the same passthrough coefficients, estimated from
the pooled global training set of comparable survey pairs. This is appropriate
because the model estimates only a small number of parameters (six in total) and the
relationship between GDP growth and consumption growth has been shown to be reasonably
stable across country contexts.

Con φ ~ USE extends the passthrough framework within a Bayesian framework in three
key ways.

**Asymmetric passthrough.** The model distinguishes between economic expansions and
contractions. Empirically, contraction passthrough (~0.65–0.85) is consistently
larger in magnitude than expansion passthrough (~0.44–0.47): households absorb income
losses more fully than they capture equivalent gains. This asymmetry is consistent
with consumption smoothing theory — households dissave or borrow during downturns but
may not fully adjust upward during expansions — and with the empirical literature on
the non-linearity of poverty responses to growth. The model therefore estimates
separate passthrough rates for positive and negative GDP growth years.

**Year-by-year decomposition.** Rather than using the total cumulative GDP change
between two surveys, the model decomposes annual growth year-by-year into its positive
and negative components. This matters because the sequence of growth matters, not just
the total. A country that experienced three years of contraction followed by two years
of recovery should not be treated the same as a country with five years of moderate
positive growth, even if the cumulative totals are similar — the former will have
depleted household buffers in ways that the latter will not.

**Distributional variation.** The passthrough is allowed to vary across the consumption
distribution via the logit(p) term, capturing whether poorer or richer households
benefit proportionally more from growth. A negative β_p coefficient implies pro-poor
growth transmission; a positive coefficient implies richer households gain
proportionally more.

The model is specified as:

```
Δlog(cons_p) = β⁺(p) × Σ⁺ Δlog(GDP) + β⁻(p) × Σ⁻ Δlog(GDP) + ε

β⁺(p) = β₀⁺ + β_p⁺ × logit(p)     [expansion passthrough, varying by percentile]
β⁻(p) = β₀⁻ + β_p⁻ × logit(p)     [contraction passthrough, varying by percentile]
ε ~ StudentT(ν, 0, σ)
```

where Σ⁺ and Σ⁻ are the cumulative positive and negative components of GDP growth
between the anchor survey year and the prediction year, decomposed year-by-year.

A Student-t likelihood is used rather than a Gaussian because the distribution of
survey-to-survey consumption changes exhibits heavy tails — occasional large shocks
that a Normal distribution would treat as implausibly extreme. The Student-t
accommodates these through its degrees-of-freedom parameter ν, which is estimated
from the data. Small ν (≈3–5) indicates heavy tails; as ν grows large, the
distribution converges to the Normal. Estimating ν rather than fixing it lets the
data determine how much tail weight is appropriate, yielding well-calibrated 90%
predictive intervals without manual tuning.

**Priors.** The expansion passthrough coefficients receive a prior of Normal(0.70, 0.30),
centred on the empirical consensus that roughly 70% of GDP growth passes through to
household consumption. The contraction passthrough prior is wider — Normal(0.70, 0.50)
— reflecting the greater uncertainty arising from fewer contraction episodes in the
training data. The distributional tilt parameters β_p⁺ and β_p⁻ are centred at zero
with narrow priors, encoding an initial expectation that passthrough is uniform across
the distribution but allowing the data to reveal systematic variation. Observation
noise σ has a HalfNormal(0.10) prior, and the degrees of freedom ν has a
Gamma(2, 0.1) prior with a mean of approximately 20.

**Prediction.** To generate predictions, USE identifies the most recent survey anchor
for each country within a comparable spell, then projects forward by applying the
learned passthrough rates to cumulative GDP growth between the anchor year and the
prediction year. Predictive uncertainty comes from two sources: parameter uncertainty
(different posterior draws yield different passthrough rates) and observation noise
(Student-t draws with per-sample ν and σ). The model generates 500 posterior predictive
draws per country-percentile, from which point estimates and credible intervals are
computed.

---

### Con φ ~ WASE Model

For countries without a recent consumption survey, the WASE model predicts the entire
consumption distribution from scratch using only widely available structural indicators.
It builds on the 2025 World Bank research paper
[Predicting Income Distributions from Almost Nothing](https://openknowledge.worldbank.org/entities/publication/91bde060-4b65-463e-9152-c69bb39391d4),
which showed that the full consumption distribution can be well approximated by a
log-logistic (Fisk) distribution whose parameters are simple functions of a small set
of country-level covariates.

The log-logistic distribution is a natural choice for consumption. Household
consumption is strictly positive and right-skewed, and the log-logistic has the
convenient property that if consumption is Fisk-distributed, log-consumption is
linear in logit(p) — meaning the entire distributional curve can be characterised by
just two parameters: a location (average living standards) and a shape (inequality).
WASE models these two parameters separately, using different covariate sets and
allowing them to vary independently across the country hierarchy.

Con φ ~ WASE extends the underlying framework within a full Bayesian model,
representing log-consumption at each percentile as:

```
log_cons = LEVEL + SHAPE × logit(p) + SPLINE(logit(p)) + ε
ε ~ Laplace(0, σ_base + σ_tail × |logit(p)|)
```

- **LEVEL** captures average living standards, driven by GDP per capita (and its
  square, allowing diminishing returns), under-5 mortality, rural population share,
  female education, GDP growth, the absolute value of GDP growth (capturing
  volatility effects), government revenue, and resource rents — all with hierarchical
  random effects at the World Bank region and sub-region level.
- **SHAPE** captures inequality via the log-logistic gamma parameter, constrained to
  the interval (ε, 0.95) via a sigmoid transform to ensure identifiability. The
  shape parameter also varies hierarchically across regions and sub-regions, allowing
  different parts of the world to have systematically different inequality structures.
- **SPLINE** captures nonlinear distributional corrections via five Gaussian radial
  basis functions (RBFs) placed at evenly spaced points along the logit(p) axis
  (at −3.0, −1.5, 0.0, 1.5, 3.0, with width 1.5). Each RBF has a global coefficient
  plus region-level deviations, allowing different regions to have different tail
  shapes and departures from the log-logistic baseline — such as heavier tails or
  subsistence consumption floors that the two-parameter log-logistic cannot represent.

**Laplace likelihood.** WASE uses a Laplace (double-exponential) likelihood rather
than a Gaussian. The Laplace distribution has heavier tails than the Normal,
providing greater robustness to outliers — appropriate for consumption data where
measurement error and extreme values are common, particularly in conflict-affected
or fragile contexts. While the Student-t used by USE achieves a similar effect
through its ν parameter, the Laplace provides a fixed level of tail robustness
without an additional parameter to estimate, which is advantageous in the more
complex WASE model.

**Heteroscedastic noise.** The observation noise is not constant across the
distribution. The noise scale is modelled as σ_base + σ_tail × |logit(p)|, meaning
that uncertainty increases at the extremes of the distribution — the very poorest
and very richest percentiles — where consumption is harder to measure and more
variable. This heteroscedastic structure allows the model to be appropriately less
confident at the tails, where survey estimates are noisiest, and more confident in
the middle of the distribution where data is densest and most reliable.

**Survey-count weighting.** Without adjustment, countries with many surveys — such as
India, China, or Brazil — would dominate the training data and the model would be
implicitly optimised for data-rich contexts. This would be counterproductive: WASE is
most operationally important precisely for fragile and conflict-affected states with
sparse survey histories. The model therefore downweights observations from data-rich
countries using inverse-square-root survey counts (1/√n), normalised to mean 1,
ensuring that countries with a single available survey receive proportional influence
in training. The survey-count weighting is applied consistently across three stages:
covariate centering (so that weighted means reflect the effective training
distribution), the empirical Bayes OLS (so that prior means are not dominated by
data-rich countries), and the likelihood (via inflation of the observation noise for
heavily-surveyed countries).

**Empirical Bayes priors.** Rather than specifying fixed prior means for the regression
coefficients, WASE runs a fold-specific weighted OLS regression on the training data
to obtain data-informed prior means before fitting the full Bayesian model. This means
the priors adapt to the available training data for each focal year rather than being
fixed constants — a form of empirical Bayes that improves convergence and reduces
sensitivity to prior specification, while maintaining the regularisation benefits of
the Bayesian framework. The OLS includes region-specific slopes on logit(p) to capture
regional inequality differences, and is weighted by the survey-count weights. Covariates
beyond the core set (GDP growth, its absolute value, government revenue, and resource
rents) receive tight zero-centred priors rather than empirical Bayes means, encoding a
sceptical default that these additional predictors contribute nothing until the data
provides evidence otherwise.

**Partial pooling via hierarchy.** WASE uses a three-level Bayesian hierarchy
(Global → World Bank Region → Sub-Region) with partial pooling. In practice this means
that countries in data-sparse regions borrow strength from neighbouring countries and
from the global mean, rather than being estimated in isolation. A country with only
one or two surveys will have its estimates pulled toward the regional average; a
country with many surveys can deviate from that average as the data warrants. This is
particularly important for Sub-Saharan Africa and fragile states, where survey
coverage is thinnest and the need for reliable estimates is greatest.

Both the level (intercept) and shape (inequality) components have their own independent
hierarchical structure, as do the spline coefficients (which have region-level
deviations). Sum-to-zero constraints are applied at each level of the hierarchy to
ensure identifiability: region-level random effects are centred globally, and
sub-region-level random effects are centred within their parent region. Without these
constraints, the model could shift probability mass freely between the global intercept
and the random effects without changing predictions, leading to non-identifiability
and poor convergence.

USE, by contrast, is a flat global model with no hierarchical structure — all countries
share the same six parameters.

**Prediction and fallback for unseen regions.** When predicting for a country whose
region or sub-region was not observed in training, WASE falls back gracefully through
the hierarchy: an unseen sub-region uses its parent region's random effects, and an
unseen region uses the global mean (zero random effect). This ensures the model can
always generate predictions for any country with available covariates, even if no
country from its region has ever appeared in the training data — a scenario that is
rare but operationally important.

**SVI variance inflation.** SVI with KL(q‖p) minimisation is known to underestimate
posterior variance. To compensate, WASE applies a post-hoc variance inflation factor
of 1.5× to the observation noise parameters (σ_base and σ_tail) during prediction —
not during fitting. This inflates the predictive intervals to achieve empirical
coverage closer to the nominal 90% level without distorting the point estimates or
the posterior parameter summaries.

**JAX compilation strategy.** All training arrays are padded to a fixed maximum size
with a boolean observation mask. Padded entries contribute nothing to the ELBO and
do not affect inference. This ensures JAX compiles the SVI step function exactly once
across all LOCO folds, rather than recompiling for each fold's different training set
size — a practical necessity when fitting hundreds of separate models in the
cross-validation loop.

**Three-phase execution.** WASE runs in three phases. Phase 1 (Rolling LOCO) performs
the rolling leave-one-country-out cross-validation described in the Validation section,
producing out-of-sample predictions for every country at every focal year. Phase 2
(All-Time LOCO) trains on all available years and holds out one country at a time,
providing an assessment of each country's predictability that is not conditioned on a
specific focal year. Phase 3 (Production) trains on all data and generates predictions
for every LMIC country-year in the target range — including countries with no
consumption survey at all. The production predictions are the operational outputs used
in the dashboard and downstream analyses.

Both models are estimated using Stochastic Variational Inference (SVI) with a Block
Neural Autoregressive Flow guide implemented in NumPyro/JAX on GPU, as described in
the Technical Implementation section below.

---

## Vintage Control and Leakage Prevention

Each model run is anchored to a specific vintage year. Consumption outcomes at or
after the vintage year are masked before training, and GDP growth data comes from
the IMF WEO release corresponding to that vintage — not current revised estimates.
This ensures the model only learns from data that would have genuinely been available
at that point in time, and that validation metrics reflect true out-of-sample
performance.

For USE, this means the model trains on survey pairs where both surveys predate the
vintage year, and projects forward using GDP growth that includes IMF forecasts
(rather than actuals) for years at or beyond the vintage. For WASE, structural
covariates are additionally lagged by three years — so a model anchored to vintage
2024 uses covariate values from 2021 or earlier — reflecting the real-world delays
in publication of mortality, education, and fiscal indicators.

---

## Validation

The two models use different cross-validation strategies, each appropriate to the
model's structure.

**USE** uses a rolling temporal cross-validation. For each vintage year, the model is
trained on all comparable survey pairs where both surveys predate that year, and
predictions are generated for the vintage year (nowcast) or up to one year ahead
(forecast). Because USE is a flat global model with no country-level parameters,
there is no need to hold out individual countries — the model never "memorises"
country-specific effects. Evaluation compares predictions against held-out survey
outcomes that were masked during training.

**WASE** uses a strict rolling **Leave-One-Country-Out (LOCO)** procedure. For each
focal year, the model is trained on all data from prior years excluding the holdout
country, and predictions are generated for that country. This is repeated for every
country in the dataset, rolling through time so each focal year uses only data that
would have been available at that point. Because WASE includes region and sub-region
random effects, holding out individual countries tests whether the hierarchical
structure can generalise to unseen members of a region — the operationally relevant
scenario.

This cross-validation framework is substantially more demanding than standard
train-test splits, because it simultaneously tests temporal generalisation (the model
must predict forward in time) and geographic generalisation (the model must predict
countries it has never seen during training for that fold). IMF vintage files ensure
only historically available GDP data is used. Structural covariates in WASE are lagged
by three years to reflect real-world publication delays.

---

## Blending USE and WASE

The two models are designed to complement each other. USE estimates take precedence
where survey data are available, as anchoring to an observed consumption distribution
substantially reduces prediction error. WASE fills the remainder of the global picture —
covering countries with no usable surveys, or providing a structural baseline for
plausibility checking of USE projections.

In the results explorer, both model outputs are available for any country. For
countries with survey data, comparing USE and WASE estimates is informative: large
divergence between the two models can indicate either a structural break in the
country (USE may be anchored to a pre-crisis survey) or a data quality concern in
the WASE structural covariates.

In future implementations, the two models could be formally combined using a
sigmoid weighting function calibrated on the survey gap (dt). Results show that USE
predictions degrade substantially when the anchor survey is more than 15 years old,
at which point WASE — which does not depend on survey recency — becomes the more
reliable estimate. A blending architecture would allow the system to transition
smoothly between models as the anchor ages.

---

## Limitations

Con φ is designed to produce nationally representative consumption estimates at
the percentile level. Several important dimensions of household welfare are not
currently captured by the model.

**Seasonality.** The model produces a single annual estimate of the consumption
distribution for each country-year. Intra-year variation in consumption — driven
by agricultural cycles, weather shocks, price seasonality, and seasonal employment
— is not modelled. For countries with pronounced seasonal food insecurity, the
annual estimate may mask substantial within-year hardship that is not visible in
the headline figures. Future implementations could incorporate seasonal adjustment
factors derived from high-frequency price or remote sensing data.

**Food expenditure vs. total consumption.** Con φ predicts total household
consumption expenditure per capita, which is the standard welfare aggregate used
in the World Bank PIP dataset. This is not equivalent to food expenditure or dietary
adequacy. In lower-income contexts, food may account for 50–70% of total consumption,
but the relationship varies substantially across countries, income levels, and over
time. Users should be cautious about interpreting Con φ outputs as direct proxies
for food security without accounting for food expenditure shares.

**Subnational estimates.** All estimates are at the national level. The model does
not currently produce subnational estimates — by region, urban/rural, or administrative
unit — which limits its utility for targeting interventions in geographically
heterogeneous countries. Subnational prediction is a recognised priority for future
development; the hierarchical model architecture is in principle extensible to
sub-national units if appropriate covariate and survey data are available.

---

## Technical Implementation

Con φ is implemented in Python and runs on GPU in Google Colab. The core
computational stack is as follows.

**Probabilistic programming and inference** is handled by
[NumPyro](https://num.pyro.ai/en/stable/), a lightweight probabilistic programming
library built on JAX. Stochastic Variational Inference is performed using NumPyro's
`SVI` module with a `Trace_ELBO` objective and an
[AutoBNAFNormal](https://num.pyro.ai/en/stable/autoguide.html#numpyro.infer.autoguide.AutoBNAFNormal)
guide — a Block Neural Autoregressive Flow that learns a flexible, non-Gaussian
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
[Block Neural Autoregressive Flow (BNAF)](https://arxiv.org/abs/1904.04676) — a
normalising flow that learns an invertible transformation from a simple base
distribution to a flexible approximate posterior, capable of capturing non-Gaussian
geometry and posterior correlations that mean-field approximations cannot represent.

This approach offers significant computational advantages: SVI converges in minutes
on GPU where MCMC might require hours, making the rolling LOCO cross-validation
procedure — which requires fitting hundreds of separate models — practically feasible.

However, SVI with KL(q‖p) minimisation carries well-documented drawbacks. The
objective tends to produce approximations that are overconfident — the variational
posterior typically underestimates posterior variance, particularly in the tails.
This manifests in Con φ as systematically narrow credible intervals, which is why a
post-hoc variance inflation factor of 1.5× is applied to WASE predictive intervals
to achieve empirical coverage closer to nominal levels. A second limitation is that
SVI provides no convergence guarantees analogous to MCMC's ergodic theorem — the ELBO
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

**Codebase.**
All model code is available on [GitHub](https://github.com/cbot22/conphi).

---

## Future Work

**MCMC sampling.** A natural extension of Con φ would replace SVI with full MCMC
sampling using [NumPyro's NUTS sampler](https://num.pyro.ai/en/stable/mcmc.html)
(No-U-Turn Sampler), which would provide asymptotically exact posterior inference,
proper uncertainty quantification without variance inflation corrections, and access
to standard MCMC diagnostics including R̂, effective sample size, and divergence counts.
The primary constraint is computational: a single NUTS run for the WASE model would
likely require several hours on GPU, making the full LOCO cross-validation procedure
prohibitively expensive at present. This remains a priority for future iterations as
GPU availability and model efficiency improve.

**Integration of additional survey sources.** Con φ currently trains exclusively on
World Bank PIP surveys. A significant extension would incorporate additional survey
sources — including WFP's own consumption and food security surveys (such as the
Emergency Food Security Assessment, EFSA, and the Comprehensive Food Security and
Vulnerability Analysis, CFSVA) — which provide more recent and more granular coverage
in exactly the fragile and conflict-affected contexts where PIP data is most sparse.

The Bayesian hierarchical framework is well-suited to this extension. Different survey
instruments can be assigned separate likelihood components or survey-specific
measurement error parameters, allowing the model to pool information across sources
while explicitly accounting for the fact that WFP rapid assessments and full household
budget surveys measure consumption with different precision and scope. In practice, this
would mean treating survey source as a factor in the likelihood — with its own noise
parameter estimated from the data — rather than pooling all surveys under a single
observation model. This approach is analogous to meta-analytic random effects models
and is a natural fit for the existing SVI inference framework.

**Subnational estimates.** Extending the hierarchical model to sub-national units would
substantially increase operational utility for country-level targeting and programme
planning. The three-level hierarchy (Global → Region → Sub-Region) is in principle
extensible to a fourth level (Country → Admin-1), provided that sub-national covariate
data and survey estimates are available. WFP's own subnational data holdings — including
area-level food security indicators and population estimates — could serve as the
covariate backbone for such an extension. As with additional survey sources, the
Bayesian framework would allow subnational estimates to borrow strength from national-
level parameters in data-sparse areas, rather than requiring each administrative unit
to be estimated independently.

**Model blending.** A formal sigmoid blending architecture combining USE and WASE
estimates as a function of survey age (dt) would improve estimates for countries with
ageing survey anchors, allowing the system to transition smoothly from survey-anchored
to structural predictions as the most recent survey becomes increasingly stale.

**Seasonality.** Incorporating high-frequency price or remote sensing signals to
produce seasonal adjustments would improve the model's relevance for acute food
security monitoring, particularly in contexts with strong agricultural cycles.