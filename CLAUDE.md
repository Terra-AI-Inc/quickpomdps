# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**quickpomdps** is a Python package that bridges Python to Julia's POMDPs.jl ecosystem. Users define POMDPs/MDPs in Python (via `gen`, `transition`, `observation`, `reward`, `obs_weight` callables) and solve them with any POMDPs.jl-compatible Julia solver (QMDP, BasicPOMCP, POMCPOW, etc.). The bridge is in `quickpomdps/bridge.jl`, which defines `PyPOMDP{S,A,O} <: POMDP{S,A,O}` and `PyMDP{S,A} <: MDP{S,A}` structs that dispatch POMDPs.jl interface methods to stored Python callables.

## Commands

```bash
# Run tests (CI uses JULIA_PROJECT=./tests)
JULIA_PROJECT=./tests uv run pytest --cov=quickpomdps

# Run example
uv run python examples/drill_targeting.py
```

## Architecture

- `quickpomdps/__init__.py` — Package entry point. Loads `bridge.jl` via pyjulia; auto-installs Julia dependencies (`PyCall`, `POMDPs`, `POMDPTools`) on first import if missing. Exports: `POMDP`, `MDP`, `require_julia_package`.
- `quickpomdps/bridge.jl` — Julia module `PyPOMDPsBridge`. Defines `PyPOMDP{S,A,O}` and `PyMDP{S,A}` structs storing Python callables as Julia Functions. Implements POMDPs.jl interface (`gen`, `transition`, `observation`, `reward`, `obs_weight`, `isterminal`, `states`, `actions`, `observations`, `initialstate`, `discount`). Constructors `create_pypomdp`/`create_pymdp` convert PyObjects to Julia functions and infer type parameters.
- `tests/test_quickpomdps.py` — Tests: Tiger POMDP gen+POMCP, Light-Dark explicit+QMDP, MDP explicit+ValueIteration, MDP gen, drill targeting gen+POMCP, info-gathering composite reward.
- `examples/drill_targeting.py` — Continuous-observation drill targeting with POMCPOW (gen + obs_weight pattern).

## Dependencies

- **Python**: `julia >=0.5,<0.7` (pyjulia), Python ^3.7
- **Julia** (auto-installed): `PyCall`, `POMDPs`, `POMDPTools`; solvers installed on demand via `require_julia_package()`
- **Dev**: `pytest`, `pytest-cov` (managed by uv via `pyproject.toml`)
