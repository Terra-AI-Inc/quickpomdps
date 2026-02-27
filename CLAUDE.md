# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**quickpomdps** is a Python package that bridges Python to Julia's POMDPs ecosystem. Users define POMDPs/MDPs in Python (transition, observation, reward functions) and solve them with Julia solvers (e.g., QMDP). The core complexity is in `quickpomdps/setup.jl`, which uses introspection and macro-generated method overloads to convert Python's flexible function signatures into Julia's strict type system.

## Commands

```bash
# Run tests (CI uses JULIA_PROJECT=./tests)
JULIA_PROJECT=./tests uv run pytest --cov=quickpomdps

# Run example
uv run python examples/lightdark.py
```

## Architecture

- `quickpomdps/__init__.py` — Package entry point. Attempts to load Julia's `QuickPOMDPs` package via `pyjulia`; auto-installs Julia dependencies (`PyCall`, `QuickPOMDPs`) on first import if missing. Exports: `DiscreteExplicitPOMDP`, `DiscreteExplicitMDP`, `QuickMDP`, `QuickPOMDP`, `MissingQuickArgument`.
- `quickpomdps/setup.jl` — Julia module `PyQuickPOMDPs`. Preprocesses Python callables into Julia functions. The `@provide_vararg_closure` macro generates multiple Julia method overloads to handle Python functions with varying argument counts (e.g., `observation(s,a,sp)` vs `observation(s,a,sp,o)`).
- `tests/test_quickpomdps.py` — Four tests: basic discrete POMDP, reward function preprocessing, Tiger POMDP benchmark, Light-Dark continuous-discrete problem.
- `examples/lightdark.py` — End-to-end example solving the Light-Dark POMDP with QMDP.

## Dependencies

- **Python**: `julia >=0.5,<0.7` (pyjulia), Python ^3.7
- **Julia** (auto-installed): `PyCall`, `QuickPOMDPs`, plus `POMDPs`, `POMDPSimulators`, `POMDPTools`, `Distributions`, `QMDP` for tests/examples
- **Dev**: `pytest`, `pytest-cov` (managed by uv via `pyproject.toml`)
