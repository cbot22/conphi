# Con φ - Methodology

All model code is available on [GitHub](https://github.com/cbot22/conphi).

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
[Constructing Comparable Global Poverty Trends](https://openknowledge.worldbank.org/entities/publication/293e54f3-b0f4-42e0-baec-e10edcf7cca6) —
that when two surveys are methodologically comparable, changes in household
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
the pooled global training set of comparable survey pairs. The model estimates
six parameters in total.

Con φ ~ USE extends the passthrough framework within a Bayesian setting in three
key ways:

**Asymmetric passthrough.** The model distinguishes between economic expansions and
contractions. Empirically, contraction passthrough (~0.65–0.85) is consistently
larger in magnitude than expansion passthrough (~0.44–0.47): households absorb income
losses more fully than they capture equivalent gains. This asymmetry is consistent
with consumption smoothing theory and with the empirical literature on the
non-linearity of poverty responses to growth. The model estimates separate passthrough
rates for positive and negative GDP growth years.

**Year-by-year decomposition.** Rather than using the total cumulative GDP change
between two surveys, the model decomposes annual growth year-by-year into its positive
and negative components. A country that experienced three years of contraction followed
by two years of recovery should not be treated the same as a country with five years
of moderate positive growth, even if the cumulative totals are similar.

**Distributional variation.** The passthrough is allowed to vary across the consumption
distribution via a logit(p) term, capturing whether poorer or richer households
benefit proportionally more from growth.

The model is specified as:

```
Δlog(cons_p) = β⁺(p) × Σ⁺ Δlog(GDP) + β⁻(p) × Σ⁻ Δlog(GDP) + ε

β⁺(p) = β₀⁺ + β_p⁺ × logit(p)     [expansion passthrough, varying by percentile]
β⁻(p) = β₀⁻ + β_p⁻ × logit(p)     [contraction passthrough, varying by percentile]
ε ~ StudentT(ν, 0, σ)
```

where Σ⁺ and Σ⁻ are the cumulative positive and negative components of GDP growth
between the anchor survey year and the prediction year, decomposed year-by-year.

A Student-t likelihood accommodates the heavy-tailed distribution of survey-to-survey
consumption changes, with degrees of freedom ν estimated from the data. Weakly
informative priors centre passthrough at 0.70 with wider uncertainty on the
contraction side; distributional tilt priors are centred at zero. Full prior
specifications and estimation details are provided in the companion
**USE Technical Methods** document.

**Prediction.** USE identifies the most recent survey anchor for each country within
a comparable spell, then projects forward by applying the learned passthrough rates
to cumulative GDP growth between the anchor year and the prediction year. The model
generates 500 posterior predictive draws per country-percentile, from which point
estimates and 90% credible intervals are computed. Predictive uncertainty reflects
both parameter uncertainty and observation noise.

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

Con φ ~ WASE represents log-consumption at each percentile as:

```
log_cons = LEVEL + SHAPE × logit(p) + SPLINE(logit(p)) + ε
ε ~ Laplace(0, σ_base + σ_tail × |logit(p)|)
```

- **LEVEL** captures average living standards, driven by GDP per capita (and its
  square), under-5 mortality, rural population share, female education, GDP growth,
  growth volatility, government revenue, and resource rents — all with hierarchical
  random effects at the World Bank region and sub-region level.
- **SHAPE** captures inequality via the log-logistic gamma parameter, also varying
  hierarchically across regions and sub-regions.
- **SPLINE** captures nonlinear distributional corrections via five Gaussian radial
  basis functions placed along the logit(p) axis, with global coefficients plus
  region-level deviations to allow different tail shapes and departures from the
  log-logistic baseline.

**Laplace likelihood with heteroscedastic noise.** WASE uses a Laplace likelihood
for robustness to outliers, with observation noise that increases at the extremes
of the distribution: σ_base + σ_tail × |logit(p)|. This allows the model to be
appropriately less confident at the tails, where survey estimates are noisiest.

**Survey-count weighting.** Countries with many surveys are downweighted using
inverse-square-root survey counts (1/√n), normalised to mean 1, ensuring that
data-sparse countries — typically the fragile and conflict-affected states where
WASE is most operationally important — receive proportional influence in training.

**Empirical Bayes priors.** Rather than fixed prior means, WASE runs a
fold-specific weighted OLS regression to obtain data-informed prior means before
fitting the full Bayesian model. Covariates beyond the core set receive tight
zero-centred priors, encoding a sceptical default that they contribute nothing until
the data provides evidence otherwise.

**Partial pooling via hierarchy.** WASE uses a three-level Bayesian hierarchy
(Global → World Bank Region → Sub-Region) with partial pooling. Countries in
data-sparse regions borrow strength from neighbouring countries and from the global
mean. Sum-to-zero constraints are applied at each level to ensure identifiability.

**Prediction and fallback for unseen regions.** When predicting for a country whose
region or sub-region was not observed in training, WASE falls back gracefully through
the hierarchy: an unseen sub-region uses its parent region's random effects, and an
unseen region uses the global mean.

**SVI variance inflation.** To compensate for the known tendency of SVI to
underestimate posterior variance, WASE applies a post-hoc variance inflation factor
of 1.5× to the observation noise parameters during prediction — not during fitting —
to achieve empirical coverage closer to the nominal 90% level.

**Three-phase execution.** WASE runs in three phases: Phase 1 (Rolling LOCO) performs
rolling leave-one-country-out cross-validation; Phase 2 (All-Time LOCO) trains on
all available years holding out one country at a time; Phase 3 (Production) trains on
all data and generates predictions for every LMIC country-year, including countries
with no consumption survey at all.

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
for years at or beyond the vintage. For WASE, structural covariates are additionally
lagged by three years — so a model anchored to vintage 2024 uses covariate values
from 2021 or earlier.

---

## Validation

The two models use different cross-validation strategies, each appropriate to the
model's structure.

**USE** uses a rolling temporal cross-validation. For each vintage year, the model is
trained on all comparable survey pairs where both surveys predate that year, and
predictions are generated for the vintage year (nowcast) or up to one year ahead
(forecast). Because USE is a flat global model with no country-level parameters,
there is no need to hold out individual countries. Evaluation compares predictions
against held-out survey outcomes that were masked during training.

**WASE** uses a strict rolling **Leave-One-Country-Out (LOCO)** procedure. For each
focal year, the model is trained on all data from prior years excluding the holdout
country, and predictions are generated for that country. This is repeated for every
country in the dataset, rolling through time. Because WASE includes region and
sub-region random effects, holding out individual countries tests whether the
hierarchical structure can generalise to unseen members of a region.

This cross-validation framework simultaneously tests temporal generalisation (the
model must predict forward in time) and geographic generalisation (the model must
predict countries it has never seen during training for that fold).

---

## Blending USE and WASE

The two models are designed to complement each other. USE estimates take precedence
where survey data are available, as anchoring to an observed consumption distribution
substantially reduces prediction error. WASE fills the remainder of the global
picture — covering countries with no usable surveys, or providing a structural
baseline for plausibility checking of USE projections.

In the results explorer, both model outputs are available for any country. Large
divergence between the two models can indicate either a structural break in the
country (USE may be anchored to a pre-crisis survey) or a data quality concern in
the WASE structural covariates.

Results show that USE predictions degrade substantially when the anchor survey is
more than 15 years old, at which point WASE becomes the more reliable estimate. A
future blending architecture using a sigmoid weighting function calibrated on the
survey gap would allow the system to transition smoothly between models as the
anchor ages.

---

## Technical Implementation

Con φ is implemented in Python and runs on GPU in Google Colab. The core stack
comprises NumPyro/JAX for probabilistic programming and Stochastic Variational
Inference (SVI), with a Block Neural Autoregressive Flow (BNAF) guide that learns
a flexible approximate posterior. USE uses a flat Adam learning rate; WASE uses
cosine decay scheduling. ArviZ is used for posterior diagnostics and NetCDF storage.
Data processing uses pandas and NumPy with Apache Parquet serialisation; visualisation
uses Plotly and Streamlit. Model training runs on NVIDIA GPU (typically A100 or T4),
with typical runtimes of 30–90 seconds per LOCO fold.

Further technical detail on inference, prior specifications, data preparation, and
evaluation procedures is provided in the companion **USE Technical Methods** and
**WASE Technical Methods** documents.

---

## Limitations

**Seasonality.** The model produces a single annual estimate per country-year.
Intra-year variation driven by agricultural cycles, weather shocks, and seasonal
employment is not captured.

**Food expenditure vs. total consumption.** Con φ predicts total household
consumption expenditure per capita, not food expenditure or dietary adequacy. In
lower-income contexts food may account for 50–70% of total consumption, but the
relationship varies substantially.

**Subnational estimates.** All estimates are at the national level. The hierarchical
architecture is in principle extensible to sub-national units if appropriate data
are available.

---

## Future Work

**MCMC sampling.** Replacing SVI with full MCMC via NumPyro's NUTS sampler would
provide exact posterior inference and standard convergence diagnostics, removing the
need for variance inflation corrections. Computational cost currently makes this
impractical for the full LOCO cross-validation loop.

**Integration of additional survey sources.** Incorporating WFP consumption-expenditure
module data would improve coverage in fragile and conflict-affected contexts. The
Bayesian framework supports survey-specific measurement error parameters to pool
information across sources while accounting for differences in precision and scope.

**Subnational estimates.** Extending the hierarchy to a fourth level
(Country → Admin-1) using WFP's subnational data holdings as a covariate backbone.

**Model blending.** A formal sigmoid blending architecture combining USE and WASE
estimates as a function of survey age.

**Seasonality.** Incorporating high-frequency price or remote sensing signals for
seasonal adjustments.