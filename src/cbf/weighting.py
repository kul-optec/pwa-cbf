"weighting schemes"

from typing import List, Tuple, Callable
import geomin as gp
from .simulation import State
from scipy.stats import norm

import numpy as np


class Weighting:
    """Prototype class to specify the interface"""

    def __call__(
        self,
        conf: float,
        obstacles: List[gp.Polyhedron],
        idx_combination: Tuple[int],
        x: State,
        t: int,
    ) -> float:
        pass


class CDFWeighting(Weighting):
    def __init__(self):
        pass

    def __call__(
        self,
        conf: float,
        obstacles: List[gp.Polyhedron],
        idx_combination: Tuple[int],
        x: State,
        t: int,
    ):
        pass


class NormalCDFWeighting(CDFWeighting):
    def __init__(self, μ: np.ndarray, Σ: np.ndarray, g: Callable):
        self.μ = μ
        self.Σ = Σ
        self.g = g

    def __call__(
        self,
        conf: float,
        obstacles: List[gp.Polyhedron],
        idx_combination: Tuple[int],
        x: State,
        t: int,
    ):
        weights = np.empty(len(obstacles))
        T = 1
        qs = np.empty(len(obstacles))
        if obstacles is not None:
            for i, obstacle in enumerate(obstacles):
                ci = obstacle.H[idx_combination[i]]
                bi = obstacle.h[idx_combination[i]]
                trans = trans = np.atleast_1d(ci @ self.g(x))
                mu, var = transform_gaussian(self.μ, self.Σ, trans)
                qs[i] = bi - ci.T @ x
            for i, obstacle in enumerate(obstacles):
                qs[i] = -qs[i] / np.sum(qs)
                logw = norm.logcdf(qs[i], loc=mu, scale=np.sqrt(var))
                weights[i] = logw / T
        weights = np.exp(weights)
        # if np.sum(weights)>(1-conf):
        #     print(np.sum(weights))
        return weights / np.sum(weights) * (1 - conf)


def transform_gaussian(
    μ: np.ndarray, Σ: np.ndarray, A: np.ndarray, b: np.ndarray | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Get the mean and covariance of A ξ + b, where ξ ~ N(μ, Σ)"""
    out_size = A.shape[0] if A.ndim == 2 else 1
    b = np.zeros(out_size) if b is None else b
    μ = np.atleast_1d(μ)
    Σ = np.atleast_1d(Σ)
    return A @ μ + b, A @ Σ @ A.T
