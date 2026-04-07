# Con φ — Technical Methods

All model code is available on [GitHub](https://github.com/cbot22/conphi).

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
posterior inference as an optimisation problem. SVI minimises the Kullback–Leibler
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
may converge to a local optimum, and there is no direct equivalent of the Gelman–Rubin
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

---

## Feature Pipeline: Vintage-Controlled Inputs

Both USE and WASE are fed by a shared feature pipeline that produces one parquet file
per vintage year. Each file contains a full ISO × year × percentile panel, but with
strict controls on what information is included — this is the foundational
leakage-prevention mechanism for the entire Con φ system.

### Vintage Control Architecture

For each vintage year *t*, the pipeline enforces that the model input file contains
only information that was available as of year *t*:

- **IMF macro data** is loaded from the WEO vintage published in October of year *t*
  (file named `{t}_10.csv`). GDP forecasts beyond *t* are WEO projections, not
  realised values. This means that when the model predicts for year *t*, it uses
  the GDP forecasts that an analyst would have had access to in October of year *t*
  — not the actual GDP that was later observed.

- **Non-IMF structural covariates** (rural share, education, mortality, remittances,
  resource rents, exchange rates, rainfall anomalies) are frozen at their last
  observed value ≤ *t*. This prevents future covariate values from leaking into
  predictions.

- **Lagged covariates** (under-5 mortality, rural share, female education,
  remittances, resource rents, government revenue, government expenditure) are
  lagged 3 years AND frozen after the vintage year. The lag prevents simultaneity
  bias; the freeze prevents look-ahead.

- **Survey outcomes** (consumption percentiles from the World Bank's PIP database)
  are included as-is in the feature file — the pipeline provides all available data.
  Temporal masking of outcomes is performed downstream in the model scripts'
  train/test splitting logic, not in the pipeline itself.

### GDP Growth Construction

GDP Growth Construction
GDP growth (gdp_growth) is a critical input to both models and is computed as the
first difference of log GDP per capita in PPP terms:
gdp_growth_t = log(GDP_pp_t) − log(GDP_pp_{t−1}) = diff(log_gdp_pp)
This is preferred over percentage change of real local-currency GDP per capita for
several reasons:

Coverage: log_gdp_pp (from the IMF WEO indicator PPPPC) covers ~122
countries versus ~85 for real local-currency GDP per capita (NGDPRPC). The
coverage gap arises because PPP conversion relies on externally anchored ICP
benchmarks that the IMF can extrapolate for nearly all member countries, whereas
real local-currency GDP requires each country's statistical office to maintain a
consistent deflator series — something many LICs lack.
Exact additivity: summing diff(log) over a time interval gives the exact
log growth ratio: Σ diff(log GDP) = log(GDP_T / GDP_0). This property is
exploited directly by USE when it decomposes cumulative GDP growth into
positive and negative components.
Unit consistency: PPP units match the consumption target (2021 PPP USD/day),
ensuring the GDP-consumption relationship is estimated in commensurate units.
Internal consistency: since log_gdp_pp is also used as the level covariate
in WASE, growth is mechanically the first difference of the same variable —
avoiding any ambiguity about which GDP concept is being used.

A known trade-off is that diff(log(PPPPC)) reflects both real output growth and
shifts in relative price levels across countries. However, since the consumption
target (PIP percentiles) is expressed in the same PPP unit system, relative price
movements appear on both sides of the regression and are largely absorbed by the
learned passthrough coefficient. The residual concern is limited to the wedge
between economy-wide PPP conversion factors (which cover investment, government
spending, and trade) and household-consumption-specific PPPs — a second-order
effect for most countries in most years.


This construction differs from the World Bank's Poverty and Inequality Platform
(PIP), which uses real GDP per capita for low-income and lower-middle-income
countries but switches to Household Final Consumption Expenditure (HFCE) per capita
for upper-middle and high-income countries (Mahler et al. 2025). 

Con φ uses a single GDP-based series throughout. PIP also applies a fixed symmetric passthrough discount
of 0.7 to consumption vectors, reflecting an assumption that 30% of GDP growth is
saved rather than consumed. Con φ instead learns the passthrough from data, finding
it to be asymmetric: contraction passthrough (~0.65–0.85) substantially exceeds
expansion passthrough (~0.44–0.47), consistent with the empirical regularity that
household consumption falls more sharply during downturns than it rises during
booms. A further difference is that PIP's national accounts series are not
vintage-controlled — they use the latest revised GDP figures retrospectively —
whereas Con φ loads the WEO file published in October of each vintage year, so GDP
values beyond that date are IMF forecasts rather than realised outturns.


### Imputation Strategy

Missing covariate values are filled using the mildest defensible method, applied
within each vintage-specific pipeline so that each vintage file is self-consistent:

- **Remittances**: leading zeros → linear interpolation → trailing forward-fill.
  Zero remittances before the first observation is plausible for many countries.

- **Resource rents**: backward-fill → linear interpolation → trailing forward-fill.
  A country with oil in 2010 likely had oil in 2005 — leading zeros are not
  appropriate here.

- **Government revenue**: backward-fill → interpolation →
  forward-fill within country, then a GDP-conditioned cascade imputation:
  region23 × GDP-quartile × year median → region × GDP-quartile × year median →
  GDP-quartile × year median → year median → global median. The GDP conditioning
  prevents high-income country fiscal ratios from contaminating low-income country
  fills (and vice versa). GDP quartile bins are anchored at the vintage year to
  prevent look-ahead.

### Region Hierarchy

Countries are assigned to a three-level geographic hierarchy: Global → Region
(World Bank regions, e.g. "Sub-Saharan Africa") → Sub-region (UN M.49 sub-regions,
referred to as `region23` in the codebase, e.g. "Eastern Africa", "Western Africa").
The pipeline validates that this nesting is clean (each sub-region belongs to
exactly one region) and applies manual overrides where the UN classification is
ambiguous or contested (e.g. Kosovo → Southern Europe, Sudan/South Sudan → Eastern
Africa).
