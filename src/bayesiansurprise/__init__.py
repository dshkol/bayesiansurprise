"""Bayesian Surprise for model-based thematic map diagnostics."""

from ._core import (
    SurpriseResult,
    auto_surprise,
    bayesian_update,
    compute_surprise,
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
from ._utils import funnel_pvalue, funnel_zscore, normalize_prob, normalize_rate

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
    "default_model_space",
    "funnel_pvalue",
    "funnel_zscore",
    "kl_divergence",
    "log_sum_exp",
    "model_space",
    "normalize_prob",
    "normalize_rate",
    "surprise",
]

__version__ = "0.1.0"

