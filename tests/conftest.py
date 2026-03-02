import os
import sys

# Diagnostic: verify Julia packages are loadable before test collection
if os.environ.get("CI"):
    from julia import Main
    for pkg in ["POMDPs", "QMDP", "DiscreteValueIteration", "BasicPOMCP",
                "POMDPSimulators", "POMDPPolicies", "POMDPModelTools", "Distributions"]:
        try:
            Main.eval(f"import {pkg}")
            print(f"  OK: {pkg}", file=sys.stderr)
        except Exception as e:
            print(f"  FAIL: {pkg}: {e}", file=sys.stderr)
