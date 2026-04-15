import json
from pathlib import Path

import numpy as np
import pandas as pd

import bayesiansurprise as bs


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "r_reference.json"


def load_reference():
    with FIXTURE_PATH.open() as f:
        return json.load(f)["cases"]


def assert_reference_result(result, reference, *, rtol=1e-12, atol=1e-12):
    np.testing.assert_allclose(result.surprise, reference["surprise"], rtol=rtol, atol=atol)
    np.testing.assert_allclose(result.signed_surprise, reference["signed_surprise"], rtol=rtol, atol=atol)
    np.testing.assert_allclose(result.model_space.prior, reference["prior"], rtol=rtol, atol=atol)
    assert result.model_space.names == tuple(reference["model_names"])

    if "posteriors" in reference:
        np.testing.assert_allclose(result.posteriors, reference["posteriors"], rtol=rtol, atol=atol)
    if "model_contributions" in reference:
        np.testing.assert_allclose(
            result.model_contributions,
            reference["model_contributions"],
            rtol=rtol,
            atol=atol,
        )


def test_count_default_matches_r_reference():
    ref = load_reference()["count_default"]
    observed = np.array(ref["observed"])
    expected = np.array(ref["expected"])
    result = bs.compute_surprise(
        bs.model_space(
            bs.UniformModel(),
            bs.BaseRateModel(expected),
            bs.FunnelModel(expected),
        ),
        observed,
        expected=expected,
        return_posteriors=True,
        return_contributions=True,
    )

    assert_reference_result(result, ref)


def test_gaussian_rates_match_r_reference():
    ref = load_reference()["gaussian_rates"]
    observed = np.array(ref["observed"])
    result = bs.compute_surprise(
        bs.model_space(bs.UniformModel(), bs.GaussianModel()),
        observed,
        return_posteriors=True,
        return_contributions=True,
    )

    assert_reference_result(result, ref)


def test_legacy_unnormalized_mode_matches_r_reference():
    ref = load_reference()["legacy_unnormalized"]
    observed = np.array(ref["observed"])
    expected = np.array(ref["expected"])
    result = bs.compute_surprise(
        bs.model_space(
            bs.UniformModel(),
            bs.FunnelModel(expected, formula="paper"),
            prior=[0.5, 0.5],
        ),
        observed,
        expected=expected,
        return_posteriors=True,
        return_contributions=True,
        normalize_posterior=False,
    )

    assert_reference_result(result, ref)


def test_dataframe_surprise_matches_r_reference():
    ref = load_reference()["dataframe_surprise"]
    data = pd.DataFrame(
        {
            "region": ["a", "b", "c", "d"],
            "events": ref["observed"],
            "population": ref["expected"],
        }
    )

    result = bs.surprise(
        data,
        observed="events",
        expected="population",
        models=("uniform", "baserate", "funnel"),
    )

    np.testing.assert_allclose(result["surprise"], ref["surprise"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        result["signed_surprise"],
        ref["signed_surprise"],
        rtol=1e-12,
        atol=1e-12,
    )

