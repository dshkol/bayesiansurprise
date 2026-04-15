"""U.S. Census workflows using the ``census`` package.

There is no exact Python clone of tidycensus with the same API surface. This
example uses the actively maintained, BSD-licensed ``census`` package for data
access and keeps geometry joining out of the core example. For richer Python GIS
workflows, ``cenpy`` and ``censusdis`` are also worth evaluating.

Run with:
    CENSUS_API_KEY=... PYTHONPATH=src python examples/us_census_census.py
"""

from __future__ import annotations

import os

import pandas as pd

import bayesiansurprise as bs


def _require_census_key() -> str:
    key = os.getenv("CENSUS_API_KEY")
    if not key:
        raise RuntimeError(
            "Set CENSUS_API_KEY to run U.S. Census examples. "
            "Get a key at https://api.census.gov/data/key_signup.html"
        )
    return key


def state_poverty_rates(*, year: int = 2022) -> pd.DataFrame:
    """Find states with atypical ACS poverty rates."""

    from census import Census

    c = Census(_require_census_key())
    rows = c.acs5.get(
        ("NAME", "B17001_001E", "B17001_002E"),
        {"for": "state:*"},
        year=year,
    )
    data = pd.DataFrame(rows)
    data["total_pop"] = pd.to_numeric(data["B17001_001E"], errors="coerce")
    data["poverty"] = pd.to_numeric(data["B17001_002E"], errors="coerce")
    data = data.dropna(subset=["total_pop", "poverty"])
    data = data[data["total_pop"] > 0].copy()
    data["poverty_rate"] = data["poverty"] / data["total_pop"]

    return bs.surprise(
        data,
        observed="poverty_rate",
        models=("uniform", "gaussian", "sampled"),
    )


def cook_county_tract_poverty_rates(*, year: int = 2022) -> pd.DataFrame:
    """Find atypical tract-level ACS poverty rates in Cook County, Illinois."""

    from census import Census

    c = Census(_require_census_key())
    rows = c.acs5.get(
        ("NAME", "B17001_001E", "B17001_002E"),
        {"for": "tract:*", "in": "state:17 county:031"},
        year=year,
    )
    data = pd.DataFrame(rows)
    data["total_pop"] = pd.to_numeric(data["B17001_001E"], errors="coerce")
    data["poverty"] = pd.to_numeric(data["B17001_002E"], errors="coerce")
    data = data.dropna(subset=["total_pop", "poverty"])
    data = data[data["total_pop"] > 100].copy()
    data["poverty_rate"] = data["poverty"] / data["total_pop"]

    return bs.surprise(
        data,
        observed="poverty_rate",
        models=("uniform", "gaussian", "sampled"),
    )


def main() -> None:
    result = state_poverty_rates()
    print(
        result.assign(abs_surprise=result["signed_surprise"].abs())
        .sort_values("abs_surprise", ascending=False)
        [["NAME", "poverty_rate", "surprise", "signed_surprise"]]
        .head(10)
    )


if __name__ == "__main__":
    main()
