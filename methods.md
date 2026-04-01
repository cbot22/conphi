# Con φ — Methodology

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

---
## Data Sources
The model uses publicly available data only.
### Consumption Surveys
At the core of the system is the **World Bank Poverty and Inequality Platform (PIP)**
dataset, which contains household consumption-expenditure estimates for each percentile
of the population across over 100 countries and more than 1,000 surveys conducted since
1977. These survey estimates are the outcome variable — what the models are trying to
predict or validate against. Only national-level, consumption welfare type surveys are
used in training.

The table below shows the number of distinct PIP surveys available in training data
up to each target year:

<div style="font-size: 0.82rem; margin: 0.8rem 0 1.2rem 0;">

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

</div>
<div style="font-size: 0.78rem; color: #56798D; margin: -0.8rem 0 1.2rem 0;"><em>Surveys are counted as distinct national-level country-year combinations with observed consumption outcomes within a comparable spell.</em></div>

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
proportionally more. A Student-t likelihood with estimated degrees of freedom ν
provides robustness to the heavy tails that characterise survey-to-survey consumption
changes, particularly in volatile economies.

The model is specified as:

```
Δlog(cons_p) = β⁺(p) × Σ⁺ Δlog(GDP) + β⁻(p) × Σ⁻ Δlog(GDP) + ε

β⁺(p) = β₀⁺ + β_p⁺ × logit(p)     [expansion passthrough, varying by percentile]
β⁻(p) = β₀⁻ + β_p⁻ × logit(p)     [contraction passthrough, varying by percentile]
ε ~ StudentT(ν, 0, σ)
```

where Σ⁺ and Σ⁻ are the cumulative positive and negative components of GDP growth
between the anchor survey year and the prediction year, decomposed year-by-year.

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
```

- **LEVEL** captures average living standards, driven by GDP per capita, under-5
  mortality, rural population share, female education, government revenue, and resource
  rents — all with hierarchical random effects at the World Bank region and sub-region level.
- **SHAPE** captures inequality via the log-logistic gamma parameter, also varying
  hierarchically across regions and sub-regions.
- **SPLINE** captures nonlinear distributional corrections via five Gaussian radial
  basis functions placed along the logit(p) axis, allowing departures from the
  log-logistic baseline such as heavier tails or subsistence consumption floors
  that the two-parameter log-logistic cannot represent.

**Survey-count weighting.** Without adjustment, countries with many surveys — such as
India, China, or Brazil — would dominate the training data and the model would be
implicitly optimised for data-rich contexts. This would be counterproductive: WASE is
most operationally important precisely for fragile and conflict-affected states with
sparse survey histories. The model therefore downweights observations from data-rich
countries using inverse-square-root survey counts (1/√n), normalised to mean 1,
ensuring that countries with a single available survey receive proportional influence
in training.

**Empirical Bayes priors.** Rather than specifying fixed prior means for the regression
coefficients, WASE runs a fold-specific weighted OLS regression on the training data
to obtain data-informed prior means before fitting the full Bayesian model. This means
the priors adapt to the available training data for each focal year rather than being
fixed constants — a form of empirical Bayes that improves convergence and reduces
sensitivity to prior specification, while maintaining the regularisation benefits of
the Bayesian framework.

**Partial pooling via hierarchy.** Both models use a three-level Bayesian hierarchy
(Global → World Bank Region → Sub-Region) with partial pooling. In practice this means
that countries in data-sparse regions borrow strength from neighbouring countries and
from the global mean, rather than being estimated in isolation. A country with only
one or two surveys will have its estimates pulled toward the regional average; a
country with many surveys can deviate from that average as the data warrants. This is
particularly important for Sub-Saharan Africa and fragile states, where survey
coverage is thinnest and the need for reliable estimates is greatest.

Both models are estimated using Stochastic Variational Inference (SVI) with a Block
Neural Autoregressive Flow guide implemented in NumPyro/JAX on GPU, as described in
the Technical Implementation section below.

---

## Vintage Control and Leakage Prevention

Each model run is anchored to a specific vintage year. Consumption outcomes at or
after the vintage year are masked before training, and GDP growth data comes from
the IMF WEO release corresponding to that vintage — not current revised estimates.
This ensures the model only learns from data that would have genuinely been available
at that point in time, and that validation metrics reflect true out-of-sample performance.

---

## Validation

Both models use a strict rolling **Leave-One-Country-Out (LOCO)** procedure. For each
focal year, the model is trained on all data from prior years excluding the holdout
country, and predictions are generated for that country. This is repeated for every
country in the dataset, rolling through time so each focal year uses only data that
would have been available at that point.

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
[Optax](https://optax.readthedocs.io/en/latest/), a gradient processing library for JAX.

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
post-hoc variance inflation factor is applied to WASE predictive intervals to achieve
empirical coverage closer to nominal levels. A second limitation is that SVI provides
no convergence guarantees analogous to MCMC's ergodic theorem — the ELBO may converge
to a local optimum, and there is no direct equivalent of the Gelman-Rubin R̂ diagnostic
to assess whether inference has succeeded.

**Array computation and automatic differentiation** are provided by
[JAX](https://jax.readthedocs.io/en/latest/), which compiles and executes numerical
code on GPU via XLA. All training arrays are padded to a fixed size so that JAX
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