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

Geospatial plotting and richer visualization helpers will be layered on top of
this core.

