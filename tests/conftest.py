from julia.api import Julia

# Initialize Julia before test collection with compiled_modules=False.
# This is required because pyjulia's embedded Julia cannot load PyCall's
# precompiled cache (built by standalone Julia in CI). Disabling compiled
# modules forces PyCall to recompile at runtime, which works in embedded mode.
Julia(compiled_modules=False)
