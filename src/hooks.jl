export

    ProgressDisplay,
    Snapshot

# --------------------------------------------------------------------

mutable struct ProgressDisplay
    dt::Float64
    bar::Progress

    function ProgressDisplay(dt::Number = .1)
        p = new()
        p.dt = Float64(dt)
        p
    end
end

function hook_init!(p::ProgressDisplay, state::State, sim::Simulator)
    p.bar = Progress(sim.maxiter, dt = p.dt)
end

function hook_update!(p::ProgressDisplay, state::State, sim::Simulator)
    next!(p.bar; showvalues = [(:time,state.t), (:iter,state.iter)])
end

# --------------------------------------------------------------------

mutable struct Snapshot
    every::Float64
    last::Float64
    index::Int
    history

    function Snapshot(every::Number)
        hook = new()
        hook.every = Float64(every)
        hook
    end
end

function hook_init!(hook::Snapshot, state::State{N,T}, sim::Simulator) where {N,T}
    count = floor(Int, sim.duration / hook.every)
    hook.last = typemin(Float64)
    hook.index = 1
    hook.history = zeros(T, (size(state.current)..., count))
    hook
end

function hook_update!(hook::Snapshot, state::State{N}, sim::Simulator) where N
    if state.t - (hook.last + hook.every) >= 0
        if hook.index <= size(hook.history, ndims(hook.history))
            copy!(view(hook.history, ntuple(_->:,Val{N})..., hook.index), state.current)
            hook.index += 1
        end
        hook.last = state.t
    end
    hook
end
