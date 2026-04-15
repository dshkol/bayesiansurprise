"""Canadian Census workflows using pycancensus.

These examples mirror the corrected R cancensus vignette pattern:
model rates directly with distributional models and keep funnel plots as a
separate diagnostic for reliability. They require a CensusMapper API key.

Run with:
    CANCENSUS_API_KEY=... PYTHONPATH=src python examples/canadian_census_pycancensus.py
"""

from __future__ import annotations

import os

import bayesiansurprise as bs


def _require_cancensus_key() -> str:
    key = os.getenv("CANCENSUS_API_KEY") or os.getenv("CM_API_KEY")
    if not key:
        raise RuntimeError(
            "Set CANCENSUS_API_KEY or CM_API_KEY to run pycancensus examples. "
            "Get a key at https://censusmapper.ca/users/sign_up"
        )
    return key


def low_income_divisions(*, geo_format: str | None = None):
    """Find atypical low-income rates across Canadian Census Divisions."""

    import pycancensus as pc

    data = pc.get_census(
        dataset="CA21",
        regions={"C": "01"},
        vectors=["v_CA21_1085", "v_CA21_1090"],
        level="CD",
        geo_format=geo_format,
        labels="short",
        quiet=True,
        api_key=_require_cancensus_key(),
    )
    data = data.dropna(subset=["v_CA21_1085", "v_CA21_1090"]).copy()
    data = data[data["v_CA21_1085"] > 0]
    data["low_income_rate"] = data["v_CA21_1090"] / data["v_CA21_1085"]

    return bs.surprise(
        data,
        observed="low_income_rate",
        models=("uniform", "gaussian", "sampled"),
    )


def vancouver_home_ownership(*, geo_format: str | None = None):
    """Find atypical tract-level owner-occupied household rates in Vancouver."""

    import pycancensus as pc

    data = pc.get_census(
        dataset="CA21",
        regions={"CMA": "59933"},
        vectors=["v_CA21_4237", "v_CA21_4238"],
        level="CT",
        geo_format=geo_format,
        labels="short",
        quiet=True,
        api_key=_require_cancensus_key(),
    )
    data = data.dropna(subset=["v_CA21_4237", "v_CA21_4238"]).copy()
    data = data[data["v_CA21_4237"] > 50]
    data["owner_rate"] = data["v_CA21_4238"] / data["v_CA21_4237"]

    return bs.surprise(
        data,
        observed="owner_rate",
        models=("uniform", "gaussian", "sampled"),
    )


def toronto_non_official_language(*, geo_format: str | None = None):
    """Find atypical tract-level non-official-language concentrations in Toronto."""

    import pycancensus as pc

    data = pc.get_census(
        dataset="CA21",
        regions={"CMA": "35535"},
        vectors=["v_CA21_1144", "v_CA21_1153"],
        level="CT",
        geo_format=geo_format,
        labels="short",
        quiet=True,
        api_key=_require_cancensus_key(),
    )
    data = data.dropna(subset=["v_CA21_1144", "v_CA21_1153"]).copy()
    data = data[data["v_CA21_1144"] > 100]
    data["no_official_rate"] = data["v_CA21_1153"] / data["v_CA21_1144"]

    return bs.surprise(
        data,
        observed="no_official_rate",
        models=("uniform", "gaussian", "sampled"),
    )


def main() -> None:
    result = low_income_divisions()
    display_cols = ["Region Name", "low_income_rate", "surprise", "signed_surprise"]
    print(
        result.assign(abs_surprise=result["signed_surprise"].abs())
        .sort_values("abs_surprise", ascending=False)
        [display_cols]
        .head(10)
    )


if __name__ == "__main__":
    main()

