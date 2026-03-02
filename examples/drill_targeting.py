"""
Drill targeting POMDP with continuous observations and POMCPOW.

Maps DrillTargeting.jl concepts to Python via quickpomdps:
  DrillTargetingPOMDP  -> POMDP(gen=..., obs_weight=...)
  GeoModel (hidden)    -> Integer world index (static)
  DrillString (action)  -> Continuous drill position (Float64)
  DrillObservation      -> Continuous grade reading (Float64)
  EnsembleBelief        -> POMCPOW particle filter
  OptimizationPolicy    -> POMCPOWSolver (online tree search)
  CompositeReward       -> Python reward logic in gen()

Uses obs_weight for continuous observation reweighting (POMCPOW pattern).
"""
from quickpomdps import POMDP

from julia.Main import Float64, rand, randn
from julia.POMDPs import solve
from julia.POMDPTools import stepthrough, ImplicitDistribution, Uniform
from julia.POMCPOW import POMCPOWSolver
from julia.Distributions import Normal, pdf as jl_pdf
import math

# Type aliases for clarity
State = int        # world index: 0, 1, or 2
Action = float     # drill position along [0, 100]
Observation = float  # noisy grade reading

# --- Geological model ---
# Three possible "worlds" with ore body at different positions along [0, 100].
DOMAIN: tuple[float, float] = (0.0, 100.0)
NOISE_STD: float = 0.15
DRILL_COST: float = 1.0
GRADE_CUTOFF: float = 0.6


def true_grade(world: State, x: Action) -> float:
    """Grade at position x in a given geological world."""
    centers: dict[State, float] = {0: 50.0, 1: 20.0, 2: 80.0}
    return math.exp(-0.5 * ((x - centers[world]) / 15.0) ** 2)


def gen(s: State, a: Action, rng) -> dict[str, State | Observation | float]:
    """Generative model: static geology, noisy continuous grade observation.

    Args:
        s: Current world index (geology never changes).
        a: Drill position along the domain.
        rng: Julia RNG passed by the solver.

    Returns:
        {"sp": next state, "o": noisy grade reading, "r": reward}
    """
    sp: State = s
    grade: float = true_grade(int(s), float(a))
    o: Observation = grade + randn(rng) * NOISE_STD
    r: float = -DRILL_COST + (5.0 if grade >= GRADE_CUTOFF else 0.0)
    return {"sp": sp, "o": o, "r": r}


def obs_weight(s: State, a: Action, sp: State, o: Observation) -> float:
    """Observation likelihood for POMCPOW particle weighting.

    Args:
        s: Previous state (unused — geology is static).
        a: Drill position.
        sp: Current state (world index).
        o: Observed noisy grade reading.

    Returns:
        pdf(Normal(true_grade, noise), o) — how likely this observation is.
    """
    grade: float = true_grade(int(sp), float(a))
    return jl_pdf(Normal(grade, NOISE_STD), o)


m = POMDP(
    gen=gen,
    obs_weight=obs_weight,
    # Continuous action space: POMCPOW samples drill positions via progressive widening.
    actions=ImplicitDistribution(lambda rng: rand(rng) * 100.0),
    # Uniform prior over 3 geological worlds.
    initialstate=Uniform([0, 1, 2]),
    discount=0.95,
    # No states=/observations= needed — not enumerable.
    # Explicit Julia types for the continuous spaces:
    actiontype=Float64,
    obstype=Float64,
)

solver = POMCPOWSolver(
    tree_queries=1000,
    max_depth=5,
    k_action=4.0,       # progressive widening: action branching factor
    alpha_action=0.25,   # progressive widening: action growth rate
    k_observation=4.0,   # progressive widening: observation branching factor
    alpha_observation=0.1,
)
policy = solve(solver, m.jl)

print("Drill targeting simulation (continuous obs, POMCPOW)")
print("=" * 55)
for step in stepthrough(m.jl, policy, max_steps=3):
    s: State = int(step.s)
    a: Action = float(step.a)
    grade: float = true_grade(s, a)
    o: Observation = float(step.o)
    r: float = float(step.r)
    print(f"  Drill x={a:.1f}: obs={o:.2f} (true grade={grade:.2f}), r={r:.1f}")
