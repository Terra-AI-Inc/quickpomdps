import julia

# Ensure PyCall is built for the current Python before pyjulia initializes Julia.
# This runs a Julia subprocess that validates (and if necessary rebuilds) PyCall,
# which resolves precompile-cache mismatches in embedded mode (Julia 1.12+).
julia.install()
