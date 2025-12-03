"""Dynamics models"""

from dataclasses import dataclass
import numpy as np
from typing import Callable

State = np.ndarray
InputMat = np.ndarray  # ns x nu
NoiseTrans = np.ndarray  # ns x nnoise
Input = np.ndarray
Noise = np.ndarray
StateToState = Callable[[State], State]
StateToTransfer = Callable[[State], InputMat]
StateToNoiseTransfer = Callable[[State], NoiseTrans]


class Dynamics:
    """Prototype class to specify the interface"""

    def f(self, x: State, u: Input) -> State:
        return x

    def g(self, x: State) -> np.ndarray | float:
        return 1.0

    @property
    def ns(self) -> int:
        return 0

    @property
    def nu(self) -> int:
        return 0


@dataclass
class CtrlAffine(Dynamics):
    """f_c(x) + F_l(x) @ u + G(x) ξ"""

    f_c: StateToState
    f_l: StateToTransfer
    G: StateToNoiseTransfer
    state_dim: int
    input_dim: int

    def f(self, x: State, u: Input) -> State:
        return self.f_c(x) + self.f_l(x) @ u

    def g(self, x: State) -> np.ndarray:
        return self.G(x)

    @property
    def ns(self) -> int:
        return self.state_dim

    @property
    def nu(self) -> int:
        return self.input_dim


@dataclass
class LtiSys(Dynamics):
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    c: np.ndarray

    def f(self, x: State, u: Input) -> State:
        return self.A @ x + self.B @ u

    def g(self, x: State) -> np.ndarray:
        return (self.C @ x).squeeze() + self.c

    @property
    def ns(self) -> int:
        return self.A.shape[0]

    @property
    def nu(self) -> int:
        return self.B.shape[1]


def double_integrator(Ts: float) -> LtiSys:
    A = np.array([[1, Ts], [0, 1]])
    B = np.array([[0], [Ts]])

    C = np.zeros((2, 2))
    c = np.eye(2)

    return LtiSys(A, B, C, c)


def single_integrator(Ts: float) -> LtiSys:
    A = np.eye(2)
    B = np.eye(2) * Ts

    C = np.zeros((2, 2))
    c = np.eye(2)

    return LtiSys(A, B, C, c)


def quadruped(ts: float, noise_trans: StateToNoiseTransfer | None = None) -> CtrlAffine:
    def f_const(x: State) -> State:
        return x

    def f_lin(x: State) -> InputMat:
        cos = np.cos(x[2])
        sin = np.sin(x[2])
        # Rotation matrix around z
        B = ts * np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])
        return B

    if noise_trans is not None:
        g = noise_trans
    else:

        def g(x: State) -> NoiseTrans:
            return np.eye(3)

    return CtrlAffine(f_const, f_lin, g, state_dim=3, input_dim=3)


