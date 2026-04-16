# bayesiansurprise

`bayesiansurprise` is a Python port of the Bayesian Surprise implementation in
the R package `bayesiansurpriser`.

The core calculation measures how much each observation updates prior beliefs
over an explicit model space:

```text
Surprise = KL(P(M | D_i) || P(M))
```

Posterior model probabilities are normalized by default. The unnormalized
per-region score used by the original Correll and Heer JavaScript demo is
available only through an explicit legacy mode.

## Install for Development

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
```

Optional census workflow dependencies are split out:

```bash
python -m pip install ".[geo]"     # GeoPandas plotting workflows
python -m pip install ".[canada]"  # pycancensus + GeoPandas
python -m pip install ".[us]"      # census + us + GeoPandas
```

## Quick Start

```python
import pandas as pd
import bayesiansurprise as bs

df = pd.DataFrame({
    "region": ["a", "b", "c", "d"],
    "events": [50, 100, 150, 200],
    "population": [10_000, 50_000, 100_000, 25_000],
})

result = bs.surprise(df, observed="events", expected="population")

print(result[["region", "surprise", "signed_surprise"]])
```

## Scope

This initial scaffold focuses on the validated mathematical core:

* uniform, base-rate, Gaussian, sampled/KDE, and de Moivre funnel models
* normalized per-observation Bayesian Surprise
* explicit legacy unnormalized scoring mode for comparison
* pandas-friendly tabular workflows
* GeoPandas-preserving surprise workflows
* rate-scale funnel diagnostics for sample-size context

```python
funnel = bs.compute_funnel_data(
    observed=df["events"],
    sample_size=df["population"],
)
bs.plot_funnel(funnel)
```

## Census Examples

The `examples/` directory includes API-backed workflows that are not run during
tests:

* `canadian_census_pycancensus.py` mirrors the corrected cancensus examples with
  `pycancensus`, modeling rates directly with `uniform`, `gaussian`, and
  `sampled` models.
* `us_census_census.py` shows the same rate-distribution pattern with the
  Python `census` package. `cenpy` and `censusdis` are also viable U.S. Census
  options, but they are not core dependencies here.
