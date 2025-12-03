"""Quantile estimators"""

import numpy as np
from typing import Tuple
from scipy.stats import binom
from scipy.stats import norm


class QuantileEstimator:
    """Prototype class to specify the interface"""

    def __call__(self, level: float) -> float:
        return 0.0


class NormalQuantile(QuantileEstimator):
    """Get the exact quantile of a normally distributed scalar RV"""

    def __init__(self, μ, var):
        self.loc = (μ,)
        self.scale = np.sqrt(var)

    def __call__(self, level: float) -> float:
        return norm.ppf(level, loc=self.loc, scale=self.scale)[0][0]


class DistributionFreeQuantile:
    """Get a lowerbound on the quantile of any scalar RV with confidence β"""

    def __init__(self, samples):
        self.samples = samples

        self.n = len(samples)

    def __call__(self, level: float, β: float) -> float | None:
        n = self.n
        k_ = int(binom.ppf(β, n, level))
        if k_ == 0:
            return None
        return np.partition(self.samples, k_ - 1)[k_ - 1]


def transform_gaussian(
    μ: np.ndarray, Σ: np.ndarray, A: np.ndarray, b: np.ndarray | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Get the mean and covariance of A ξ + b, where ξ ~ N(μ, Σ)"""
    out_size = A.shape[0] if A.ndim == 2 else 1
    b = np.zeros(out_size) if b is None else b
    μ = np.atleast_1d(μ)
    Σ = np.atleast_1d(Σ)
    return A @ μ + b, A @ Σ @ A.T

