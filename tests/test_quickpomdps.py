from quickpomdps import DiscreteExplicitPOMDP, DiscreteExplicitMDP, QuickPOMDP, QuickMDP

from julia import Pkg
Pkg.add(["POMDPs", "POMDPSimulators", "POMDPPolicies", "POMDPModelTools", "Distributions", "QMDP", "DiscreteValueIteration", "BasicPOMCP"])

from julia.QuickPOMDPs import preprocess
from julia import Main
from julia.Main import applicable, Val, Symbol, Float64

from julia.POMDPs import solve, pdf, value, action
from julia.QMDP import QMDPSolver
from julia.DiscreteValueIteration import ValueIterationSolver
from julia.BasicPOMCP import POMCPSolver
from julia.POMDPSimulators import stepthrough
from julia.POMDPPolicies import alphavectors
# for lightdark
from julia.POMDPModelTools import Uniform, Deterministic
from julia.Distributions import Normal

def test_basics():
    def T(s, a, sp):
        return s == sp

    def Z(a, sp, o):
        return 0.5

    def R(s, a):
        return -1.0

    S = ['l', 'r']
    A = ['l', 'r']
    O = ['l', 'r']
    discount = 0.95
    prob = DiscreteExplicitPOMDP(S, A, O, T, Z, R, discount)

def test_reward():
    r = True
    def reward(s, a, sp=1, *args): s**2
    jlrew = preprocess(Main.eval('Val(:reward)'), reward)
    assert applicable(jlrew, 1, 2)
    assert applicable(jlrew, 1, 2, 3)
    assert applicable(jlrew, 1, 2, 3, 4)
    assert not applicable(jlrew, 1)

# Tiger POMDP from Kaelbling et al. 98 (http://www.sciencedirect.com/science/article/pii/S000437029800023X)
def test_tiger():

    S = ['left', 'right']
    A = ['left', 'right', 'listen']
    O = ['left', 'right']
    discount = 0.95

    def T(s, a, sp):
        if a == 'listen':
            return s == sp
        else: # a door is opened
            return 0.5 #reset

    def Z(a, sp, o):
        if a == 'listen':
            if o == sp:
                return 0.85
            else:
                return 0.15
        else:
            return 0.5

    def R(s, a):
        if a == 'listen':
            return -1.0
        elif s == a: # the tiger was found
            return -100.0
        else: # the tiger was escaped
            return 10.0

    m = DiscreteExplicitPOMDP(S,A,O,T,Z,R,discount)

    solver = QMDPSolver()
    policy = solve(solver, m)

    print('alpha vectors:')
    for v in alphavectors(policy):
        print(v)

    print()

    rsum = 0.0
    for step in stepthrough(m, policy, max_steps=10):
        print('s:', step.s)
        print('b:', [pdf(step.b, x) for x in S])
        print('a:', step.a)
        print('o:', step.o, '\n')
        rsum += step.r

    print('Undiscounted reward was', rsum)

# Test using DiscreteValueIteration solver with QuickMDP to verify
# that solvers other than QMDP work with quickpomdps
def test_mdp_value_iteration():
    S = ['a', 'b', 'done']
    A = ['go', 'stay']

    def T(s, a, sp):
        if s == 'done':
            return 1.0 if sp == 'done' else 0.0
        if a == 'go':
            if s == 'a':
                return 1.0 if sp == 'b' else 0.0
            else:
                return 1.0 if sp == 'done' else 0.0
        else:  # stay
            return 1.0 if sp == s else 0.0

    def R(s, a):
        if s == 'done':
            return 0.0
        if s == 'b' and a == 'go':
            return 10.0
        return -1.0

    m = DiscreteExplicitMDP(S, A, T, R, 0.95)

    solver = ValueIterationSolver()
    policy = solve(solver, m)

    # From state 'a', going to 'b' then 'done' collects the +10 reward,
    # so value of 'a' should be positive
    assert value(policy, 'a') > 0

def test_lightdark():
    r = 60
    light_loc = 10

    def transition(s, a):
        if a == 0:
            return Deterministic(r+1)
        else:
            return Deterministic(min(max(s+a, -r), r))

    def observation(s, a, sp):
        return Normal(sp, abs(sp - light_loc) + 0.0001)

    def reward(s, a, sp):
        if a == 0:
            return 100.0 if s == 0 else -100.0
        else:
            return -1.0

    m = QuickPOMDP(
        states = range(-r, r+2),
        actions = [-10, -1, 0, 1, 10],
        discount = 0.95,
        isterminal = lambda s: s < -r or s > r,
        obstype = Float64,
        transition = transition,
        observation = observation,
        reward = reward,
        initialstate = Uniform(range(-r//2, r//2+1))
    )

    solver = QMDPSolver()
    policy = solve(solver, m)

    print('alpha vectors:')
    for v in alphavectors(policy):
        print(v)

    print()

    rsum = 0.0
    for step in stepthrough(m, policy, max_steps=10):
        print('s:', step.s)
        print('a:', step.a)
        print('o:', step.o, '\n')
        rsum += step.r

    print('Undiscounted reward was', rsum)

# Test generative model interface with an online Monte Carlo solver (BasicPOMCP).
# Uses gen(s, a, rng) instead of explicit transition/observation distributions.
def test_generative_pomcp():
    from julia.Main import rand

    S = ['left', 'right']
    A = ['left', 'right', 'listen']
    O = ['left', 'right']

    def gen(s, a, rng):
        if a == 'listen':
            sp = s
            o = s if rand(rng) < 0.85 else ('right' if s == 'left' else 'left')
            r = -1.0
        else:
            sp = 'left' if rand(rng) < 0.5 else 'right'
            o = 'left' if rand(rng) < 0.5 else 'right'
            r = -100.0 if s == a else 10.0
        return {'sp': sp, 'o': o, 'r': r}

    m = QuickPOMDP(
        states=S,
        actions=A,
        observations=O,
        discount=0.95,
        isterminal=lambda s: False,
        initialstate=Uniform(S),
        gen=gen,
    )

    solver = POMCPSolver()
    policy = solve(solver, m)

    # Run a short simulation to verify the solver+gen interface works end-to-end
    rsum = 0.0
    for step in stepthrough(m, policy, max_steps=5):
        print('s:', step.s, 'a:', step.a, 'o:', step.o)
        rsum += step.r
    print('Undiscounted reward was', rsum)
