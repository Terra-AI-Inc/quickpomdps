import pytest
from quickpomdps import POMDP, MDP

from julia.Main import Float64, rand, randn
from julia.POMDPs import solve, value
from julia.POMDPTools import stepthrough, alphavectors, Uniform, Deterministic
from julia.Distributions import Normal
from julia.QMDP import QMDPSolver
from julia.DiscreteValueIteration import ValueIterationSolver

try:
    from julia.BasicPOMCP import POMCPSolver
except ImportError:
    POMCPSolver = None


# -- Tiger POMDP with gen + POMCP (discrete) ---------------------------------

@pytest.mark.skipif(POMCPSolver is None, reason="BasicPOMCP unavailable")
def test_tiger_gen_pomcp():
    """Static hidden state + gen + online solver."""

    def gen(s, a, rng):
        if a == "listen":
            sp = s
            o = s if rand(rng) < 0.85 else ("right" if s == "left" else "left")
            r = -1.0
        else:
            sp = "left" if rand(rng) < 0.5 else "right"
            o = "left" if rand(rng) < 0.5 else "right"
            r = -100.0 if s == a else 10.0
        return {"sp": sp, "o": o, "r": r}

    m = POMDP(
        gen=gen,
        states=["left", "right"],
        actions=["left", "right", "listen"],
        observations=["left", "right"],
        initialstate=Uniform(["left", "right"]),
        discount=0.95,
    )

    solver = POMCPSolver(tree_queries=1000)
    policy = solve(solver, m.jl)

    rsum = 0.0
    for step in stepthrough(m.jl, policy, max_steps=10):
        print(f"s={step.s}, a={step.a}, o={step.o}, r={step.r}")
        rsum += step.r
    print(f"Undiscounted reward: {rsum}")


# -- Light-Dark with explicit distributions + QMDP ---------------------------

def test_lightdark_explicit():
    """Explicit transition/observation distributions solved with QMDP."""
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

    rsum = 0.0
    for step in stepthrough(m.jl, policy, max_steps=10):
        print(f"s={step.s}, a={step.a}, o={step.o:.1f}")
        rsum += step.r
    print(f"Undiscounted reward: {rsum}")


# -- MDP with explicit transition + DiscreteValueIteration --------------------

def test_mdp_explicit():
    """MDP with explicit transition distributions."""
    S = ["a", "b", "done"]
    A = ["go", "stay"]

    def transition(s, a):
        if s == "done":
            return Deterministic("done")
        if a == "go":
            return Deterministic("b" if s == "a" else "done")
        return Deterministic(s)

    def reward(s, a):
        if s == "done":
            return 0.0
        if s == "b" and a == "go":
            return 10.0
        return -1.0

    m = MDP(
        transition=transition,
        reward=reward,
        states=S,
        actions=A,
        initialstate=Deterministic("a"),
        discount=0.95,
    )

    solver = ValueIterationSolver()
    policy = solve(solver, m.jl)
    assert value(policy, "a") > 0


# -- MDP with gen interface ---------------------------------------------------

def test_mdp_gen():
    """MDP using gen(s, a, rng) -> {sp, r} with simulation."""
    from julia.Main import eval as jl_eval

    def gen(s, a, rng):
        if s >= 10:
            return {"sp": s, "r": 0.0}
        if a == 1:
            sp = s + 1
            r = 1.0 if sp == 10 else -0.1
        else:
            sp = s
            r = -0.5
        return {"sp": sp, "r": r}

    m = MDP(
        gen=gen,
        states=list(range(11)),
        actions=[0, 1],
        initialstate=Deterministic(0),
        isterminal=lambda s: s >= 10,
        discount=0.95,
    )

    # Verify gen works via manual stepping
    rng = jl_eval("using Random; Random.MersenneTwister(42)")
    from julia.POMDPs import gen as jl_gen
    result = jl_gen(m.jl, 0, 1, rng)
    assert result.sp == 1
    assert result.r == -0.1


# -- Drill targeting: gen + discrete obs + POMCP ------------------------------

@pytest.mark.skipif(POMCPSolver is None, reason="BasicPOMCP unavailable")
def test_drill_targeting_discrete():
    """Static hidden geology, discrete observations, POMCP."""
    WORLDS = [
        [0.2, 0.3, 0.9, 0.8, 0.1],
        [0.8, 0.7, 0.2, 0.1, 0.3],
        [0.1, 0.2, 0.3, 0.7, 0.9],
    ]
    NOISE_STD = 0.15
    GRADE_CUTOFF = 0.6

    def grade_bin(grade):
        if grade < 0.3:
            return "low"
        elif grade < 0.6:
            return "medium"
        else:
            return "high"

    def gen(s, a, rng):
        sp = s
        true_grade = WORLDS[int(s)][int(a)]
        noisy = true_grade + randn(rng) * NOISE_STD
        o = grade_bin(noisy)
        r = -1.0 + (5.0 if true_grade >= GRADE_CUTOFF else 0.0)
        return {"sp": sp, "o": o, "r": r}

    m = POMDP(
        gen=gen,
        states=[0, 1, 2],
        actions=[0, 1, 2, 3, 4],
        observations=["low", "medium", "high"],
        initialstate=Uniform([0, 1, 2]),
        discount=0.95,
    )

    solver = POMCPSolver(tree_queries=500, max_depth=5)
    policy = solve(solver, m.jl)

    n_steps = 0
    for step in stepthrough(m.jl, policy, max_steps=3):
        print(f"s={step.s}, a={step.a}, o={step.o}, r={step.r}")
        n_steps += 1
    assert n_steps == 3


# -- Information-gathering composite reward -----------------------------------

@pytest.mark.skipif(POMCPSolver is None, reason="BasicPOMCP unavailable")
def test_info_gathering_reward():
    """Reward combines exploration bonus + grade value (composite reward pattern)."""

    WORLDS = [[0.1, 0.9], [0.9, 0.1]]
    EXPLORE_BONUS = 2.0

    def gen(s, a, rng):
        sp = s
        grade = WORLDS[int(s)][int(a)]
        o = "high" if grade + randn(rng) * 0.1 > 0.5 else "low"
        r = EXPLORE_BONUS + grade
        return {"sp": sp, "o": o, "r": r}

    m = POMDP(
        gen=gen,
        states=[0, 1],
        actions=[0, 1],
        observations=["low", "high"],
        initialstate=Uniform([0, 1]),
        discount=0.95,
    )

    solver = POMCPSolver(tree_queries=200, max_depth=3)
    policy = solve(solver, m.jl)

    for step in stepthrough(m.jl, policy, max_steps=2):
        print(f"s={step.s}, a={step.a}, o={step.o}, r={step.r}")
        assert step.r >= EXPLORE_BONUS  # composite reward always >= bonus
