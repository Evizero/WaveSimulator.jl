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

# --------------------------------------------------------------------

function simulator_init(state::State, sim::Simulator)
    foreach(hook -> hook_init!(hook, state, sim), sim.hooks)
end

function update!(state::State, backend::Backend, sim::Simulator)
    backend_update!(state, backend, sim)
    state.previous, state.current = state.current, state.previous
    state.t    += sim.wave.dt
    state.iter += 1
    foreach(hook -> hook_update!(hook, state, sim), sim.hooks)
end

function simulate!(state::State, backend::Backend, sim::Simulator)
    simulator_init(state, sim)
    for _ in 1:sim.maxiter
        update!(state, backend, sim)
    end
    state
end

# --------------------------------------------------------------------

function simulate(f0, sim::Simulator{N,T}, domain::Domain{N}) where {N,T}
    backend = backend_init(sim.resource, domain, sim)
    state   = state_init(f0, backend, domain, sim)
    simulate!(state, backend, sim)
    backend_cleanup!(backend, state, sim)
    state
end

function simulate(sim::Simulator{N,T}, domain::Domain{N}) where {N,T}
    backend = backend_init(sim.resource, domain, sim)
    state   = state_init(backend, domain, sim)
    simulate!(state, backend, sim)
    backend_cleanup!(backend, state, sim)
    state
end

function simulate_gauss(sim::Simulator{N,T}, domain::Domain{N}) where {N,T}
    simulate(gauss(domain), sim, domain)
end
