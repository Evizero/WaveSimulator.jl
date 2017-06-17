function cpu_kernel_impl(N, indices=:(current_domain))
    quote
        @inbounds @nloops $N I $indices begin
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

@generated function update!(
        resource::Union{CPU1,CPUThreads},
        state::State{N,T,D}) where {N, T, D <: HyperCube{N}}
    REGION = resource<:CPUThreads ? :(i->inds[i]) : :(i->2:size(Ψₜ₊₁,i)-1)
    quote
        λ = state.λ
        λsq = λ^2
        λhalf = λ/2
        γ = state.domain.γ
        Ψₜ₋₁ = state.previous
        Ψₜ   = state.current
        Ψₜ₊₁ = state.previous
        q    = state.q
        $(resource<:CPUThreads ? :(inds = resource.settings) : :())
        $(cpu_kernel_impl(N, REGION))
        state
    end
end

update!(state::State, args...) = update!(CPU1(), state, args...)

function simulate(resource::Union{CPU1,CPUThreads}, wave::Wave, domain, time)
    state = setup(wave, domain)
    simulate!(resource, state, time)
end

function simulate!(resource::CPU1, state::State, time::Number)
    Nt = ceil(Int, time / state.dt)
    p = Progress(Nt, .1)
    for _ in 1:Nt
        update!(resource, state)
        state.previous, state.current = state.current, state.previous
        state.t    += state.dt
        state.iter += 1
        next!(p)
    end
    state
end

function simulate!(resource::CPUThreads{TileSize{N}}, state::State, time::Number) where N
    Nt = ceil(Int, time / state.dt)
    p = Progress(Nt, .1)
    inner_inds = map(s->2:s-1, size(state.current))
    #result = zeros(Float64, (size(state.current)..., Nt))
    tiles = collect(TileIterator(inner_inds, resource.settings.dims))
    for t in 1:Nt
        next!(p)
        Threads.@threads for i = 1:length(tiles)
            inds = tiles[i]
            update!(CPUThreads(inds), state)
        end
        state.previous, state.current = state.current, state.previous
        state.t    += state.dt
        state.iter += 1
        #copy!(view(result, ntuple(i->:,Val{N})..., t), state.current)
        next!(p)
    end
    state
end

function simulate!(resource::CPUThreads{NTuple{N,Int}}, state::State, time::Number) where N
    simulate!(CPUThreads(TileSize(resource.settings)), state, time)
end

function simulate(wave::Wave, args...; resource=CPU1(), time=0.025)
    simulate(resource, wave, args..., time)
end

function simulate!(state::State, args...; resource=CPU1(), time=0.025)
    simulate!(resource, state, args..., time)
end
