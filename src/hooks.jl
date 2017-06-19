export

    ProgressDisplay,
    StoreSnapshots,
    PointMicrophone

# --------------------------------------------------------------------

mutable struct ProgressDisplay
    dt::Float64
    bar::Progress
    showvalues::Bool

    function ProgressDisplay(dt::Number = .1, showvalues = true)
        hook = new()
        hook.dt = Float64(dt)
        hook.showvalues = showvalues
        hook
    end
end

function hook_init!(hook::ProgressDisplay, state::State, sim::Simulator)
    hook.bar = Progress(sim.maxiter, dt = hook.dt)
end

function hook_update!(hook::ProgressDisplay, state::State, sim::Simulator)
    if hook.showvalues
        next!(hook.bar; showvalues = [(:time,state.t), (:iter,state.iter)])
    else
        next!(hook.bar)
    end
end

# --------------------------------------------------------------------

mutable struct StoreSnapshots
    dt::Float64
    last::Float64
    index::Int
    snapshots

    function StoreSnapshots(dt::Number)
        hook = new()
        hook.dt = Float64(dt)
        hook
    end
end

function hook_init!(hook::StoreSnapshots, state::State{N,T}, sim::Simulator) where {N,T}
    count = floor(Int, sim.duration / hook.dt)
    hook.last = typemin(Float64)
    hook.index = 1
    hook.snapshots = zeros(T, (size(state.current)..., count))
    hook
end

function hook_update!(hook::StoreSnapshots, state::State{N}, sim::Simulator) where N
    if state.t - (hook.last + hook.dt) >= 0
        if hook.index <= size(hook.snapshots, ndims(hook.snapshots))
            copy!(view(hook.snapshots, ntuple(_->:,Val{N})..., hook.index), state.current)
            hook.index += 1
        end
        hook.last = state.t
    end
    hook
end

# --------------------------------------------------------------------

mutable struct PointMicrophone{N}
    position::NTuple{N,Float64}
    startiter::Int
    gridpos::NTuple{N,Int}
    signal

    function PointMicrophone(position::Vararg{Number,N}) where N
        hook = new{N}()
        hook.position = Float64.(position)
        hook.startiter = 1
        hook
    end
end

function hook_init!(hook::PointMicrophone, state::State{N,T}, sim::Simulator) where {N,T}
    hook.signal = zeros(T, sim.maxiter)
    hook.startiter = state.iter
    hook.gridpos = floor.(Int, hook.position ./ sim.wave.dx) .+ 1
    hook
end

function hook_update!(hook::PointMicrophone, state::State{N}, sim::Simulator) where N
    hook.signal[state.iter - hook.startiter] = state.current[hook.gridpos...]
    hook
end

# --------------------------------------------------------------------

mutable struct PointSpeaker{N,S<:AbstractVector}
    position::NTuple{N,Float64}
    gridpos::NTuple{N,Int}
    signal::S

    function PointSpeaker(position::NTuple{N,Number}, signal::S) where {N,S<:AbstractVector}
        hook = new{N,S}()
        hook.position = Float64.(position)
        hook.signal = signal
        hook
    end
end

function hook_init!(hook::PointSpeaker, state::State{N,T}, sim::Simulator) where {N,T}
    hook.gridpos = floor.(Int, hook.position ./ sim.wave.dx) .+ 1
    hook
end

function hook_update!(hook::PointSpeaker, state::State{N}, sim::Simulator) where N
    if state.iter <= length(hook.signal)
        state.current[hook.gridpos...] = hook.signal[state.iter]
    end
    hook
end
