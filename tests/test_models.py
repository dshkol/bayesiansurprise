import numpy as np

import bayesiansurprise as bs


def test_uniform_model_returns_constant_per_region_likelihood():
    model = bs.UniformModel()
    observed = [10, 20, 30, 40, 50]

    assert model.log_likelihood(observed, region_idx=0) == 0
    assert model.log_likelihood(observed, region_idx=2) == 0
    assert model.log_likelihood(observed, region_idx=4) == 0


def test_baserate_model_normalizes_expected_values():
    model = bs.BaseRateModel([100, 200, 300, 400])
    assert model.kind == "baserate"
    assert model.expected_prop.sum() == pytest_approx(1)


def test_gaussian_model_can_be_fixed_or_fit_from_data():
    model = bs.GaussianModel()
    assert model.fit_from_data

    fixed = bs.GaussianModel(mu=50, sigma=10, fit_from_data=False)
    assert fixed.mu == 50
    assert fixed.sigma == 10


def test_sampled_model_accepts_sample_fraction():
    model = bs.SampledModel(sample_frac=0.2)
    assert model.sample_frac == 0.2
    assert np.isfinite(model.log_likelihood([1, 2, 3, 4, 5], region_idx=0))


def test_funnel_model_uses_paper_name_by_default():
    model = bs.FunnelModel([10000, 50000, 100000, 25000])
    assert model.kind == "funnel"
    assert model.name == "de Moivre Funnel (paper)"


def test_model_space_combines_models_with_uniform_prior():
    space = bs.model_space(bs.UniformModel(), bs.GaussianModel())

    assert space.n_models == 2
    np.testing.assert_allclose(space.prior, [0.5, 0.5])


def test_model_space_accepts_custom_prior():
    space = bs.model_space(bs.UniformModel(), bs.GaussianModel(), prior=[0.3, 0.7])
    np.testing.assert_allclose(space.prior, [0.3, 0.7])


def test_default_model_space_matches_r_package_shape():
    space = bs.default_model_space([100, 200, 300, 400])
    assert space.n_models == 3
    assert space.names == ("Uniform", "Base Rate", "de Moivre Funnel (paper)")


def pytest_approx(value):
    import pytest

    return pytest.approx(value)

