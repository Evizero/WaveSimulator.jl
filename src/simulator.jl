struct Simulator{N,T<:AbstractFloat,W<:Wave{N,T},R<:AbstractResource,H<:Tuple}
    wave::W
    resource::R
    hooks::H
    duration::T
    maxiter::Int
end

function Simulator(wave::Wave{N,T}, hooks...; resource=CPU1(), duration=0.01) where {N,T}
    maxiter = ceil(Int, duration / wave.dt)
    Simulator{N,T,typeof(wave),typeof(resource),typeof(hooks)}(wave, resource, hooks, T(duration), maxiter)
end


function simulate!(state::AbstractState, sim::Simulator)
    backend = backend_init(sim.resource, state, sim)
    foreach(hook -> hook_init!(hook, state, sim), sim.hooks)
    #result = zeros(Float64, (size(state.current)..., Nt))
    for _ in 1:sim.maxiter
        update!(state, backend, sim)
        # update state variables
        state.previous, state.current = state.current, state.previous
        state.t    += sim.wave.dt
        state.iter += 1
        # trigger hooks
        foreach(hook -> hook_update!(hook, state, sim), sim.hooks)
        #copy!(view(result, ntuple(i->:,Val{N})..., t), state.current)
    end
    state
end

function simulate(f0, sim::Simulator{N,T}, domain::Domain{N}) where {N,T}
    state = state_init(f0, sim.wave, domain)
    simulate!(state, sim)
end

function simulate(sim::Simulator{N,T}, domain::Domain{N}; f0 = (I...)->zero(T)) where {N,T}
    simulate(f0, sim, domain)
end

function simulate_gauss(sim::Simulator{N,T}, domain::Domain{N}) where {N,T}
    simulate(gauss(domain), sim, domain)
end