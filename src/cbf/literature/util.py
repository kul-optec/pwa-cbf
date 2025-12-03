import casadi as cs
from .base import QuadraticBarrier


def compute_fushimi_hyperparams_corridor(
    barrier: QuadraticBarrier, confidence: float, stdev: float, sim_len: int
):
    opti = cs.Opti()
    a = opti.variable()
    β = opti.variable()
    σ = stdev
    c = barrier.c
    ε = 1 - confidence
    cost = cs.exp(-a * c) + β * sim_len
    constraint2 = (
        cs.exp(-a * c) + β * sim_len <= ε
    )  # opti.subject_to(a <= 1 / (2*sigma**2) - 1e-6)

    opti.subject_to(constraint2)
    constraint3 = a * c >= -cs.log(cs.exp(-a * c) + β) - 1 / 2 * cs.log(
        1 - 2 * σ**2 * a
    )

    constraint4 = a >= 1
    constraint5 = β >= 0
    opti.subject_to(constraint3)
    opti.subject_to(constraint4)
    opti.subject_to(constraint5)
    opti.set_initial(a, 1 / 2 / σ**2 - 0.1)
    opti.set_initial(β, confidence / sim_len * 0.8)
    opti.minimize(-cost)
    opts = {"print_time": False, "ipopt.print_level": 0}
    opti.solver("ipopt", opts)
    sol = opti.solve()
    return sol.value(a), sol.value(β)


def get_beta_corridor(
    barrier: QuadraticBarrier, confidence: float, stdev: float, a: float, sim_len: int
):
    c = barrier.c
    opti = cs.Opti()
    β = opti.variable()
    cost = cs.exp(-a * c) + β * sim_len
    σ = stdev
    ε = 1 - confidence
    constraint1 = cs.exp(-a * c) + β * sim_len <= ε
    # opti.subject_to(a <= 1 / (2*sigma**2) - 1e-6)
    opti.subject_to(constraint1)
    constraint2 = a * c >= -cs.log(cs.exp(-a * c) + β) - 1 / 2 * cs.log(
        1 - 2 * σ**2 * a
    )
    constraint3 = β >= 0
    opti.subject_to(constraint2)
    opti.subject_to(constraint3)
    opti.set_initial(β, confidence / sim_len * 0.8)
    opti.minimize(-cost)
    opts = {"print_time": False, "ipopt.print_level": 0}
    opti.solver("ipopt", opts)
    sol = opti.solve()
    # print(sol.value(β))
    return sol.value(β)
