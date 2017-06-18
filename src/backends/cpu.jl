struct CPU1Backend{R<:CPU1} <: Backend{R}
    resource::R
end

function backend_init(resource::CPU1, state, sim)
    CPU1Backend(resource)
end

function update!(state::State, backend::CPU1Backend, sim)
    cpu_kernel!(state, backend.resource, sim)
end

struct CPUThreadsBackend{R<:CPUThreads,T} <: Backend{R}
    resource::R
    tiles::T
end

function backend_init(r::CPUThreads{TileSize{N}}, state, sim) where N
    inds = map(s->2:s-1, size(state.current))
    tiles = collect(TileIterator(inds, r.settings.dims))
    CPUThreadsBackend(r, tiles)
end

function backend_init(r::CPUThreads{NTuple{N,Int}}, state, sim) where N
    backend_init(CPUThreads(TileSize(r.settings)), state, sim)
end

function update!(state::State, backend::CPUThreadsBackend, sim)
    tiles = backend.tiles
    Threads.@threads for i = 1:length(tiles)
        inds = tiles[i]
        cpu_kernel!(state, CPUThreads(inds), sim)
    end
end

function cpu_kernel_impl(N, indices=:(current_domain))
    quote
        @nloops $N I $indices begin
            @nexprs $N w->(
                q₊_w = @nref($N, q, i -> i==w ? I_i+1 : I_i);
                q₋_w = @nref($N, q, i -> i==w ? I_i-1 : I_i)
            )
            # q₊_1 + q₋_1 + q₊_2 + ...
            K = @ncall($N, +, w->(q₊_w + q₋_w))
            # K̄ = $(2*N) - K

            @nexprs $N w->(
                Q_w = q₊_w * @nref($N, Ψₜ, i -> i==w ? I_i+1 : I_i) +
                      q₋_w * @nref($N, Ψₜ, i -> i==w ? I_i-1 : I_i)
            )
            # Q_1 + Q_2 + ...
            Q = @ncall($N, +, Q)

            @nexprs $N w->(γ_w = (1-q₊_w) * γ + (1-q₋_w) * γ)
            # γ_1 + γ_2 + ...
            γ̄ = @ncall($N, +, γ)

            @nref($N, Ψₜ₊₁, I) = 1/(1+λhalf*γ̄) * (
                (2-λsq*K)   * @nref($N, Ψₜ,  I)  +
                (λhalf*γ̄-1) * @nref($N, Ψₜ₋₁,I) +
                λsq*Q)
        end
    end
end

@generated function cpu_kernel!(
        state::State{N,T,D},
        resource::Union{CPU1,CPUThreads},
        sim) where {N, T, D <: HyperCube{N}}
    SETUP  = resource<:CPUThreads ? :(inds = resource.settings) : :()
    REGION = resource<:CPUThreads ? :(i->inds[i]) : :(i->2:size(Ψₜ₊₁,i)-1)
    quote
        λ = sim.wave.λ
        λsq = λ^2
        λhalf = λ/2
        γ = state.domain.γ
        Ψₜ₋₁ = state.previous
        Ψₜ   = state.current
        Ψₜ₊₁ = state.previous
        q    = state.q
        $(SETUP)
        $(cpu_kernel_impl(N, REGION))
        state
    end
end
