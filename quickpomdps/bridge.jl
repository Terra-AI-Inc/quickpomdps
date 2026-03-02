module PyPOMDPsBridge

using PyCall
using POMDPs
import POMDPTools

# ---------------------------------------------------------------------------
# Structs
# ---------------------------------------------------------------------------

struct PyPOMDP{S,A,O} <: POMDP{S,A,O}
    gen_fn::Union{Function,Nothing}
    transition_fn::Union{Function,Nothing}
    observation_fn::Union{Function,Nothing}
    reward_fn::Union{Function,Nothing}
    obs_weight_fn::Union{Function,Nothing}
    isterminal_fn::Union{Function,Nothing}
    states_val::Any          # nothing, Vector, or Distribution
    actions_val::Any
    observations_val::Any
    initialstate_val::Any
    discount_val::Float64
    state_index::Union{Dict,Nothing}
    action_index::Union{Dict,Nothing}
    obs_index::Union{Dict,Nothing}
end

struct PyMDP{S,A} <: MDP{S,A}
    gen_fn::Union{Function,Nothing}
    transition_fn::Union{Function,Nothing}
    reward_fn::Union{Function,Nothing}
    isterminal_fn::Union{Function,Nothing}
    states_val::Any
    actions_val::Any
    initialstate_val::Any
    discount_val::Float64
    state_index::Union{Dict,Nothing}
    action_index::Union{Dict,Nothing}
end

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

inspect = pyimport("inspect")

function maybe_convert_fn(v::PyObject)
    if pybuiltin("callable")(v)
        return convert(Function, v)
    end
    return v
end
maybe_convert_fn(v) = v

"""
Introspect a Python callable's positional arg count. Returns (n_required, has_varargs).
"""
function py_argcount(v::PyObject)
    params = inspect.signature(v).parameters.values()
    n_required = count(params) do p
        (p.kind == p.POSITIONAL_ONLY || p.kind == p.POSITIONAL_OR_KEYWORD) && p.default == p.empty
    end
    has_var = any(p -> p.kind == p.VAR_POSITIONAL, params)
    n_optional = count(params) do p
        (p.kind == p.POSITIONAL_ONLY || p.kind == p.POSITIONAL_OR_KEYWORD) && p.default != p.empty
    end
    return n_required, n_optional, has_var
end

"""
Wrap a Python callable so that Julia can call it with more args than it accepts.
Extra trailing args are silently dropped. This handles POMDPs.jl calling
reward(s,a,sp,o) when the user defined reward(s,a,sp) or reward(s,a).
"""
function wrap_vararg(v::PyObject)
    n_req, n_opt, has_var = py_argcount(v)
    fn = convert(Function, v)
    if has_var
        return fn  # already accepts varargs
    end
    max_args = n_req + n_opt
    return (args...) -> fn(args[1:min(length(args), max_args)]...)
end

function wrap_gen(gen_fn::Function)
    return function(args...)
        py_d = gen_fn(args...)
        jl_d = Dict{Symbol,Any}(Symbol(k) => val for (k, val) in PyDict(py_d))
        return (; jl_d...)
    end
end

function infer_type(explicit_type, space)
    explicit_type !== nothing && return explicit_type
    if space !== nothing
        try
            collected = collect(space)
            if !isempty(collected)
                return eltype(collected)
            end
        catch
        end
    end
    return Any
end

function build_index(space)
    space === nothing && return nothing
    try
        collected = collect(space)
        return Dict(v => i for (i, v) in enumerate(collected))
    catch
        return nothing
    end
end

# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------

function create_pypomdp(;
    gen=nothing,
    transition=nothing,
    observation=nothing,
    reward=nothing,
    obs_weight=nothing,
    isterminal=nothing,
    states=nothing,
    actions=nothing,
    observations=nothing,
    initialstate=nothing,
    discount=0.95,
    statetype=nothing,
    actiontype=nothing,
    obstype=nothing,
)
    gen_fn = gen !== nothing ? wrap_gen(maybe_convert_fn(gen)) : nothing
    transition_fn = transition isa PyObject ? wrap_vararg(transition) : maybe_convert_fn(transition)
    observation_fn = observation isa PyObject ? wrap_vararg(observation) : maybe_convert_fn(observation)
    reward_fn = reward isa PyObject ? wrap_vararg(reward) : maybe_convert_fn(reward)
    obs_weight_fn = obs_weight !== nothing ? maybe_convert_fn(obs_weight) : nothing
    isterminal_fn = isterminal !== nothing ? maybe_convert_fn(isterminal) : nothing

    states_val = maybe_convert_fn(states)
    actions_val = maybe_convert_fn(actions)
    observations_val = maybe_convert_fn(observations)
    initialstate_val = maybe_convert_fn(initialstate)

    S = infer_type(statetype, states_val)
    A = infer_type(actiontype, actions_val)
    O = infer_type(obstype, observations_val)

    return PyPOMDP{S,A,O}(
        gen_fn, transition_fn, observation_fn, reward_fn, obs_weight_fn,
        isterminal_fn, states_val, actions_val, observations_val,
        initialstate_val, Float64(discount),
        build_index(states_val), build_index(actions_val), build_index(observations_val),
    )
end

function create_pymdp(;
    gen=nothing,
    transition=nothing,
    reward=nothing,
    isterminal=nothing,
    states=nothing,
    actions=nothing,
    initialstate=nothing,
    discount=0.95,
    statetype=nothing,
    actiontype=nothing,
)
    gen_fn = gen !== nothing ? wrap_gen(maybe_convert_fn(gen)) : nothing
    transition_fn = transition isa PyObject ? wrap_vararg(transition) : maybe_convert_fn(transition)
    reward_fn = reward isa PyObject ? wrap_vararg(reward) : maybe_convert_fn(reward)
    isterminal_fn = isterminal !== nothing ? maybe_convert_fn(isterminal) : nothing

    states_val = maybe_convert_fn(states)
    actions_val = maybe_convert_fn(actions)
    initialstate_val = maybe_convert_fn(initialstate)

    S = infer_type(statetype, states_val)
    A = infer_type(actiontype, actions_val)

    return PyMDP{S,A}(
        gen_fn, transition_fn, reward_fn, isterminal_fn,
        states_val, actions_val, initialstate_val, Float64(discount),
        build_index(states_val), build_index(actions_val),
    )
end

# ---------------------------------------------------------------------------
# POMDPs.jl interface — POMDP
# ---------------------------------------------------------------------------

# Only define gen if user provided one; otherwise POMDPs.jl's default
# gen will sample from transition/observation/reward.
function POMDPs.gen(m::PyPOMDP, s, a, rng)
    if m.gen_fn !== nothing
        return m.gen_fn(s, a, rng)
    end
    # Fall back to default POMDPs.jl gen (sample from distributions)
    return invoke(POMDPs.gen, Tuple{POMDP, Any, Any, Any}, m, s, a, rng)
end

function POMDPs.transition(m::PyPOMDP, args...)
    m.transition_fn === nothing && error("No transition function provided")
    return m.transition_fn(args...)
end

function POMDPs.observation(m::PyPOMDP, args...)
    m.observation_fn === nothing && error("No observation function provided")
    return m.observation_fn(args...)
end

function POMDPs.reward(m::PyPOMDP, args...)
    m.reward_fn === nothing && error("No reward function provided")
    return m.reward_fn(args...)
end

function POMDPTools.obs_weight(m::PyPOMDP, s, a, sp, o)
    if m.obs_weight_fn !== nothing
        return m.obs_weight_fn(s, a, sp, o)
    end
    # Fall back to default: pdf(observation(m, s, a, sp), o)
    return POMDPs.pdf(POMDPs.observation(m, s, a, sp), o)
end

function POMDPs.isterminal(m::PyPOMDP, s)
    m.isterminal_fn === nothing && return false
    return m.isterminal_fn(s)
end

function POMDPs.states(m::PyPOMDP)
    m.states_val === nothing && error("No states provided")
    return m.states_val
end

function POMDPs.actions(m::PyPOMDP)
    m.actions_val === nothing && error("No actions provided")
    return m.actions_val
end

function POMDPs.observations(m::PyPOMDP)
    m.observations_val === nothing && error("No observations provided")
    return m.observations_val
end

function POMDPs.initialstate(m::PyPOMDP)
    m.initialstate_val === nothing && error("No initialstate provided")
    return m.initialstate_val
end

POMDPs.discount(m::PyPOMDP) = m.discount_val

function POMDPs.stateindex(m::PyPOMDP, s)
    m.state_index === nothing && error("No enumerable states provided; cannot compute stateindex")
    return m.state_index[s]
end

function POMDPs.actionindex(m::PyPOMDP, a)
    m.action_index === nothing && error("No enumerable actions provided; cannot compute actionindex")
    return m.action_index[a]
end

function POMDPs.obsindex(m::PyPOMDP, o)
    m.obs_index === nothing && error("No enumerable observations provided; cannot compute obsindex")
    return m.obs_index[o]
end

# ---------------------------------------------------------------------------
# POMDPs.jl interface — MDP
# ---------------------------------------------------------------------------

function POMDPs.gen(m::PyMDP, s, a, rng)
    if m.gen_fn !== nothing
        return m.gen_fn(s, a, rng)
    end
    return invoke(POMDPs.gen, Tuple{MDP, Any, Any, Any}, m, s, a, rng)
end

function POMDPs.transition(m::PyMDP, args...)
    m.transition_fn === nothing && error("No transition function provided")
    return m.transition_fn(args...)
end

function POMDPs.reward(m::PyMDP, args...)
    m.reward_fn === nothing && error("No reward function provided")
    return m.reward_fn(args...)
end

function POMDPs.isterminal(m::PyMDP, s)
    m.isterminal_fn === nothing && return false
    return m.isterminal_fn(s)
end

function POMDPs.states(m::PyMDP)
    m.states_val === nothing && error("No states provided")
    return m.states_val
end

function POMDPs.actions(m::PyMDP)
    m.actions_val === nothing && error("No actions provided")
    return m.actions_val
end

function POMDPs.initialstate(m::PyMDP)
    m.initialstate_val === nothing && error("No initialstate provided")
    return m.initialstate_val
end

POMDPs.discount(m::PyMDP) = m.discount_val

function POMDPs.stateindex(m::PyMDP, s)
    m.state_index === nothing && error("No enumerable states provided; cannot compute stateindex")
    return m.state_index[s]
end

function POMDPs.actionindex(m::PyMDP, a)
    m.action_index === nothing && error("No enumerable actions provided; cannot compute actionindex")
    return m.action_index[a]
end

end # module
