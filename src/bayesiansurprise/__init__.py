"""Bayesian Surprise for model-based thematic map diagnostics."""

from ._core import (
    SurpriseResult,
    auto_surprise,
    bayesian_update,
    compute_surprise,
    get_signed_surprise,
    get_surprise,
    get_surprise_result,
    kl_divergence,
    log_sum_exp,
    surprise,
)
from ._models import (
    BaseRateModel,
    FunnelModel,
    GaussianModel,
    Model,
    ModelSpace,
    SampledModel,
    UniformModel,
    default_model_space,
    model_space,
)
from ._plotting import plot_funnel, plot_signed_surprise, plot_surprise
from ._utils import compute_funnel_data, funnel_pvalue, funnel_zscore, normalize_prob, normalize_rate

__all__ = [
    "BaseRateModel",
    "FunnelModel",
    "GaussianModel",
    "Model",
    "ModelSpace",
    "SampledModel",
    "SurpriseResult",
    "UniformModel",
    "auto_surprise",
    "bayesian_update",
    "compute_surprise",
    "compute_funnel_data",
    "default_model_space",
    "funnel_pvalue",
    "funnel_zscore",
    "get_signed_surprise",
    "get_surprise",
    "get_surprise_result",
    "kl_divergence",
    "log_sum_exp",
    "model_space",
    "normalize_prob",
    "normalize_rate",
    "plot_funnel",
    "plot_signed_surprise",
    "plot_surprise",
    "surprise",
]

__version__ = "0.1.0"
