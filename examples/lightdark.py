# The Light-Dark problem from https://arxiv.org/pdf/1709.06196.pdf
from quickpomdps import POMDP

from julia.Main import Float64
from julia.POMDPs import solve
from julia.QMDP import QMDPSolver
from julia.POMDPTools import stepthrough, alphavectors, Uniform, Deterministic
from julia.Distributions import Normal

r = 60
light_loc = 10

m = POMDP(
    transition=lambda s, a: Deterministic(r + 1) if a == 0 else Deterministic(min(max(s + a, -r), r)),
    observation=lambda s, a, sp: Normal(sp, abs(sp - light_loc) + 0.0001),
    reward=lambda s, a, sp: (100.0 if s == 0 else -100.0) if a == 0 else -1.0,
    states=range(-r, r + 2),
    actions=[-10, -1, 0, 1, 10],
    obstype=Float64,
    initialstate=Uniform(range(-r // 2, r // 2 + 1)),
    isterminal=lambda s: s < -r or s > r,
    discount=0.95,
)

solver = QMDPSolver()
policy = solve(solver, m.jl)

print("alpha vectors:")
for v in alphavectors(policy):
    print(v)

print()

rsum = 0.0
for step in stepthrough(m.jl, policy, max_steps=10):
    print(f"s={step.s}, a={step.a}, o={step.o:.1f}")
    rsum += step.r

print(f"Undiscounted reward was {rsum}")
