"""Simulation module"""

from dataclasses import dataclass
import numpy as np
from typing import Callable
from .dynamics import State, Input, Noise
import time

NomDynamics = Callable[[State, Input, Noise], State]
Policy = Callable[[State, int], Input]
# optionally return timings in s
FalliblePolicy = Callable[[State, int], Input | tuple[Input, float] | None]


def do_nothing():
    pass


@dataclass
class SimulationRecord:
    x: np.ndarray
    u: np.ndarray
    noise: np.ndarray
    time_ns: np.ndarray
    label: str
    fail_idx: int | float
    stats: dict

    def __post_init__(self):
        self.x = np.asarray(self.x)
        self.u = np.asarray(self.u)
        self.noise = np.asarray(self.noise)
        self.time_ns = np.asarray(self.time_ns)

    def to_dict(self) -> dict:
        return dict(
            x=self.x.tolist(),
            u=self.u.tolist(),
            noise=self.noise.tolist(),
            time_ns=self.time_ns.tolist(),
            label=self.label,
            fail_idx=self.fail_idx,
            stats={k: np.array(v).tolist() for k, v in self.stats.items()},
        )

    @property
    def n_steps(self) -> int:
        return len(self.x)

    @property
    def j_selection(self) -> list[list[int]]:
        return self.stats.get("j_selection", [])


class Simulation:
    def __init__(
        self,
        x0: State,
        f: NomDynamics,
        κ: FalliblePolicy,
        horizon: int,
        sample: Callable[[], Noise],
        get_stats: Callable[[], dict] | None = None,
        initialize: Callable[[], None] | None = None,
        label: str = "",
        run: bool = True,
    ):
        """initialize: function to get called at the beginning of the simulation."""
        self.x = np.array([[]])
        self.u = np.array([[]])
        self.noise = np.array([[]])
        self.fail_index = np.inf
        self.stats = dict()
        self.get_stats = get_stats if get_stats is not None else dict
        self.initialize = initialize if initialize is not None else do_nothing
        self.time = np.empty(horizon)
        self.label = label
        if κ is not None:
            if run:
                self.run(x0, f, κ, horizon, sample)
        else:
            self.fail_index=0
            self.x=np.array([x0])

    def run(
        self,
        x0: State,
        f: NomDynamics,
        κ: FalliblePolicy,
        horizon: int,
        sample: Callable[[], Noise],
    ):
        x = [x0]
        u = []
        ξ = []
        xk = x0
        self.initialize()
        for k in range(horizon):
            ξk = sample()
            ξ.append(ξk)
            t_1 = time.perf_counter_ns()
            # handle optional timing return
            result = κ(xk, k)
            t_2 = time.perf_counter_ns()
            # Only uk returned
            if not isinstance(result, tuple):
                uk = result
                # use timing from perf_counter
                self.time[k] = t_2 - t_1

            # uk and t returned
            else:
                uk, t = result
                # use timing of controller
                self.time[k] = t * 10**9  # s to ns
            if uk is None:
                self.fail_index = k
                break
            xk = f(xk, uk, ξk)
            u.append(uk)
            x.append(xk)
        self.x = np.array(x)
        self.u = np.array(u)
        self.noise = np.array(ξ)
        self.stats = self.get_stats()

    @property
    def failed(self) -> bool:
        return not np.isinf(self.fail_index)

    def export(self) -> SimulationRecord:
        return SimulationRecord(
            self.x,
            self.u,
            self.noise,
            self.time,
            self.label,
            self.fail_index,
            self.stats,
        )

    def to_dict(self) -> dict:
        return self.export().to_dict()

    def plot_state_trajectory(self, **kwargs):
        import matplotlib.pyplot as plt

        opts = dict(color="k", marker=".")
        opts.update(kwargs)
        plt.plot(*self.x[:, :2].T, **opts)
        if self.failed:
            plt.scatter(*self.x[self.fail_index, :2], color="red", marker="x")

    def __len__(self) -> int:
        return len(self.x)
