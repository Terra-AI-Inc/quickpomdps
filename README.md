[![test](https://github.com/Terra-AI-Inc/quickpomdps/actions/workflows/ci.yml/badge.svg)](https://github.com/Terra-AI-Inc/quickpomdps/actions/workflows/ci.yml)

# quickpomdps

Define and solve POMDPs and MDPs in Python using Julia's [POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl) ecosystem. Supports offline solvers (QMDP, SARSOP, DiscreteValueIteration), online solvers (BasicPOMCP, POMCPOW), and any combination of discrete/continuous state, action, and observation spaces.

## Examples

### Tiger POMDP — generative model with POMCP

Hidden state is static. `gen(s, a, rng)` samples transitions directly for online Monte Carlo tree search.

```python
from quickpomdps import POMDP, require_julia_package

require_julia_package("BasicPOMCP", "POMDPTools")

from julia.Main import rand
from julia.POMDPs import solve
from julia.POMDPTools import stepthrough, Uniform
from julia.BasicPOMCP import POMCPSolver

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

for step in stepthrough(m.jl, policy, max_steps=10):
    print(f"s={step.s}, a={step.a}, o={step.o}, r={step.r}")
```

### Light-Dark — explicit distributions with QMDP

Transition and observation return Julia distributions. Offline solver enumerates the state space.

```python
from quickpomdps import POMDP, require_julia_package

require_julia_package("QMDP", "Distributions", "POMDPTools")

from julia.Main import Float64
from julia.POMDPs import solve
from julia.POMDPTools import stepthrough, Uniform, Deterministic
from julia.QMDP import QMDPSolver
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

for step in stepthrough(m.jl, policy, max_steps=10):
    print(f"s={step.s}, a={step.a}, o={step.o:.1f}")
```

### Drill targeting — continuous observations with POMCPOW

Continuous observation space using `gen` + `obs_weight` for particle-based belief updates.

```python
from quickpomdps import POMDP, require_julia_package

require_julia_package("POMCPOW", "Distributions", "POMDPTools")

from julia.Main import Float64, rand, randn
from julia.POMDPs import solve
from julia.POMDPTools import stepthrough, ImplicitDistribution, Uniform
from julia.POMCPOW import POMCPOWSolver
from julia.Distributions import Normal, pdf as jl_pdf
import math

NOISE_STD = 0.15

def true_grade(world, x):
    centers = {0: 50.0, 1: 20.0, 2: 80.0}
    return math.exp(-0.5 * ((x - centers[world]) / 15.0) ** 2)

def gen(s, a, rng):
    sp = s
    grade = true_grade(int(s), float(a))
    o = grade + randn(rng) * NOISE_STD
    r = -1.0 + (5.0 if grade >= 0.6 else 0.0)
    return {"sp": sp, "o": o, "r": r}

def obs_weight(s, a, sp, o):
    grade = true_grade(int(sp), float(a))
    return jl_pdf(Normal(grade, NOISE_STD), o)

m = POMDP(
    gen=gen,
    obs_weight=obs_weight,
    actions=ImplicitDistribution(lambda rng: rand(rng) * 100.0),
    initialstate=Uniform([0, 1, 2]),
    discount=0.95,
    actiontype=Float64,
    obstype=Float64,
)

solver = POMCPOWSolver(tree_queries=1000, max_depth=5)
policy = solve(solver, m.jl)

for step in stepthrough(m.jl, policy, max_steps=3):
    print(f"Drill x={float(step.a):.1f}: obs={float(step.o):.2f}, r={float(step.r):.1f}")
```

### MDP with value iteration

```python
from quickpomdps import MDP, require_julia_package

require_julia_package("DiscreteValueIteration", "POMDPTools")

from julia.POMDPs import solve, value
from julia.POMDPTools import Deterministic
from julia.DiscreteValueIteration import ValueIterationSolver

m = MDP(
    transition=lambda s, a: Deterministic("done") if s == "done" else Deterministic("b" if s == "a" and a == "go" else ("done" if a == "go" else s)),
    reward=lambda s, a: 10.0 if s == "b" and a == "go" else (0.0 if s == "done" else -1.0),
    states=["a", "b", "done"],
    actions=["go", "stay"],
    initialstate=Deterministic("a"),
    discount=0.95,
)

solver = ValueIterationSolver()
policy = solve(solver, m.jl)
print(f"V(a) = {value(policy, 'a'):.2f}")  # positive → go is optimal
```

## API

### `POMDP(**kwargs)` / `MDP(**kwargs)`

Create a problem and pass `m.jl` to any Julia solver.

| Keyword | Description |
|---|---|
| `gen` | `gen(s, a, rng) -> {"sp": ..., "o": ..., "r": ...}` — generative model |
| `transition` | `transition(s, a) -> distribution` — explicit transition |
| `observation` | `observation(s, a, sp) -> distribution` — explicit observation |
| `reward` | `reward(s, a)` or `reward(s, a, sp)` — reward function |
| `obs_weight` | `obs_weight(s, a, sp, o) -> float` — observation likelihood (for particle filters) |
| `isterminal` | `isterminal(s) -> bool` — terminal state check (default: always false) |
| `states` | Iterable of states (required for offline solvers) |
| `actions` | Iterable or distribution over actions |
| `observations` | Iterable of observations (required for offline solvers) |
| `initialstate` | Distribution over initial states |
| `discount` | Discount factor (default: 0.95) |
| `statetype` / `actiontype` / `obstype` | Explicit Julia type parameters (inferred from spaces if omitted) |

Provide **either** `gen` (generative) **or** `transition`+`observation`+`reward` (explicit). Mixing is allowed — `gen` takes priority when present, with `observation` used as fallback for belief updates.

### `require_julia_package(*names)`

Install Julia packages on demand. Call before importing solver modules.

## Installation

Requires Julia — install via [juliaup](https://github.com/JuliaLang/juliaup).

```bash
pip install quickpomdps
```

Julia dependencies (`POMDPs`, `POMDPTools`, `PyCall`) are installed automatically on first import. Solvers are installed on demand via `require_julia_package()`.

By default, Julia packages are added to the global environment. Set `JULIA_PROJECT` to use a local environment instead ([docs](https://docs.julialang.org/en/v1/manual/environment-variables/#JULIA_PROJECT)).

## Development

```bash
git clone https://github.com/Terra-AI-Inc/quickpomdps
cd quickpomdps
uv sync
JULIA_PROJECT=./tests uv run pytest
```

See [`examples/`](examples/) and [`tests/`](tests/) for more usage patterns. Solver documentation is at [POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl).
