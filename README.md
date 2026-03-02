[![test](https://github.com/Terra-AI-Inc/quickpomdps/actions/workflows/ci.yml/badge.svg)](https://github.com/Terra-AI-Inc/quickpomdps/actions/workflows/ci.yml)

# quickpomdps - python

`quickpomdps` is a package to quickly define [[PO]MDPs](https://en.wikipedia.org/wiki/Partially_observable_Markov_decision_process) in Python.
You can use any of the solvers in the [POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl) ecosystem directly from Python — both offline solvers (QMDP, SARSOP, DiscreteValueIteration) and online solvers (BasicPOMCP, POMCPOW) that work with generative models.

### Explicit distribution model (QMDP)

A hybrid continuous-discrete light-dark problem definition and QMDP solution (taken from [`examples/lightdark.py`](examples/lightdark.py)):
```python
m = QuickPOMDP(
    states = range(-r, r+2),
    actions = [-10, -1, 0, 1, 10],
    discount = 0.95,
    isterminal = lambda s: s < -r or s > r,
    obstype = Float64,
    transition = transition,    # returns a distribution
    observation = observation,  # returns a distribution
    reward = reward,
    initialstate = Uniform(range(-r//2, r//2+1))
)

solver = QMDPSolver()
policy = solve(solver, m)
```

### Generative sampler model (BasicPOMCP)

Problems can also be defined with a `gen(s, a, rng)` function that samples transitions directly, enabling online Monte Carlo solvers:
```python
from julia.Main import rand
from julia.BasicPOMCP import POMCPSolver

def gen(s, a, rng):
    if a == 'listen':
        sp = s
        o = s if rand(rng) < 0.85 else other(s)
        r = -1.0
    else:
        sp = 'left' if rand(rng) < 0.5 else 'right'
        o = 'left' if rand(rng) < 0.5 else 'right'
        r = -100.0 if s == a else 10.0
    return {'sp': sp, 'o': o, 'r': r}

m = QuickPOMDP(
    states=['left', 'right'],
    actions=['left', 'right', 'listen'],
    observations=['left', 'right'],
    discount=0.95,
    isterminal=lambda s: False,
    initialstate=Uniform(['left', 'right']),
    gen=gen,
)

solver = POMCPSolver()
policy = solve(solver, m)
```

## Installation

`quickpomdps` uses the [pyjulia package](https://github.com/JuliaPy/pyjulia) which requires Julia to be installed. We recommend using [juliaup](https://github.com/JuliaLang/juliaup) for this purpose.

```bash
pip install quickpomdps
```

Upon invocation of `import quickpomdps` in Python, all Julia dependencies will be installed if they are not already present.
By default, the Julia dependencies are added to the *global* environment.
To install to a local environment instead, set `JULIA_PROJECT` as documented [here](https://docs.julialang.org/en/v1/manual/environment-variables/#JULIA_PROJECT).

## Development

```bash
git clone https://github.com/Terra-AI-Inc/quickpomdps
cd quickpomdps
uv sync
JULIA_PROJECT=./tests uv run pytest
```

## Usage

See [`examples/`](examples/) and [`tests/`](tests/). Documentation can be found at the [QuickPOMDPs.jl](https://github.com/JuliaPOMDP/QuickPOMDPs.jl) and [POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl/blob/master/README.md) packages.
