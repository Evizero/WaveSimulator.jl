module WaveSimulator

using Base.Cartesian, Images, Colors, FixedPointNumbers, Distributions, ProgressMeter, MappedArrays, TiledIteration

export

    WaveProblem,
    BoxDomain,

    setup,
    update!,
    simulate,
    toimage

struct WaveProblem
    c::Float64 # wave speed
    # ρ::Float64 # density
    λ::Float64  # Courant Number c*dt/dx
    dx::Float64 # spatial step
    dt::Float64 # time step (dx*λ/c)
end
function WaveProblem(; c=340, dx=0.1, lambda=√(1/3), dt=-1.)
    λ  = dt <= 0 ? lambda      : c*dt/dx
    dt = dt <= 0 ? dx*lambda/c : dt
    WaveProblem(Float64(c), Float64(λ), Float64(dx), Float64(dt))
end

struct BoxDomain{N}
    size::NTuple{N,Float64} # in m
    γ::Float64 # global admittance (loss at boundary)
end
function BoxDomain(size::NTuple{N,Number}; gamma=0) where N
    BoxDomain{N}(Float64.(size), Float64(gamma))
end
function BoxDomain(size...; gamma=0)
    BoxDomain(Float64.(size), Float64(gamma))
end

mutable struct BoxDomainState{N}
    previous::Array{Float64,N}
    current::Array{Float64,N}
    indomain::Array{Int8,N}
    iter::Int
    t::Float64
end

function setup(problem::WaveProblem, domain::BoxDomain)
    setup(gauss([(domain.size./2)...],0.02), problem, domain)
end

function setup(f0, problem::WaveProblem, domain::BoxDomain{N}) where N
    dims = ceil.(Int, domain.size ./ problem.dx)
    grid1 = Float64[Float64(f0((I.I.*problem.dx)...)) for I in CartesianRange(dims)]
    grid2 = copy(grid1)
    indom = zeros(Int8, dims)
    for I in CartesianRange(CartesianIndex(ntuple(i->2, Val{N})), CartesianIndex(size(indom) .- 1))
        indom[I] = Int8(1)
    end
    BoxDomainState(grid1, grid2, indom, 0, 0.0)
end

function gauss(mean, var)
    function f0(I...)
        pdf(MvNormal(mean, sqrt(var)), [I...])
    end
    f0
end

function update_impl(N,indices=:(current_domain))
    quote
        @nloops $N I $indices begin
            @nexprs $N i->(
                qp_i = @nref($N, indomain, j -> j==i ? I_j+1 : I_j);
                qn_i = @nref($N, indomain, j -> j==i ? I_j-1 : I_j)
            )
            # qp_1 + qn_1 + qp_2 + ...
            K = @ncall($N, +, i->(qp_i + qn_i))
            Kbar = $(2*N) - K

            @nexprs $N i->(
                Q_i = qp_i * @nref($N, current, j -> j==i ? I_j+1 : I_j) +
                      qn_i * @nref($N, current, j -> j==i ? I_j-1 : I_j)
            )
            # Q_1 + Q_2 + ...
            Q = @ncall($N, +, Q)

            @nexprs $N i->(γ_i = (1-qp_i) * γ + (1-qn_i) * γ)
            # γ_1 + γ_2 + ...
            γbar = @ncall($N, +, γ)

            @nref($N,next,I) = 1/(1+λhalf*γbar) * (
                (2-λsq*K)      * @nref($N,current,I)  +
                (λhalf*γbar-1) * @nref($N,previous,I) +
                λsq*Q)
        end
    end
end

@generated function update!(state::BoxDomainState{N}, problem::WaveProblem, domain::BoxDomain{N}, inds) where N
    quote
        λ = problem.λ
        λsq = λ^2
        λhalf = λ/2
        γ = domain.γ
        previous = state.previous
        current  = state.current
        next     = state.previous
        indomain = state.indomain
        $(update_impl(N, :(i->inds[i])))
        state
    end
end

@generated function update!(state::BoxDomainState{N}, problem::WaveProblem, domain::BoxDomain{N}) where N
    quote
        λ = problem.λ
        λsq = λ^2
        λhalf = λ/2
        γ = domain.γ
        previous = state.previous
        current  = state.current
        next     = state.previous
        indomain = state.indomain
        $(update_impl(N, :(i->2:size(next,i)-1)))
        state.previous = current
        state.current  = next
        state.t += problem.dt
        state.iter += 1
        state
    end
end

function simulate(problem::WaveProblem, domain::BoxDomain{N}, tilesz; time=0.025) where N
    state = setup(problem, domain)
    Nt = ceil(Int,time/problem.dt)
    #result = zeros(Float64, (size(state.current)..., Nt))
    p = Progress(Nt, .1)
    tileinds_all = collect(TileIterator(map(s->2:s-1, size(state.current)), tilesz))
    for t in 1:Nt
        next!(p)
        Threads.@threads for i = 1:length(tileinds_all)
            tileinds = tileinds_all[i]
            update!(state, problem, domain, tileinds)
        end
        state.previous, state.current = state.current, state.previous
        state.t += problem.dt
        state.iter += 1
     #   copy!(view(result, ntuple(i->:,Val{N})..., t), state.current)
    end
    state #result
end

function simulate(problem::WaveProblem, domain::BoxDomain{N}; time=0.025) where N
    state = setup(problem, domain)
    Nt = ceil(Int,time/problem.dt)
    #result = zeros(Float64, (size(state.current)..., Nt))
    p = Progress(Nt, .1)
    for t in 1:Nt
        next!(p)
        update!(state, problem, domain)
        #copy!(view(result, ntuple(i->:,Val{N})..., t), state.current)
    end
    state #result
end

const tocolor = colorsigned(RGB{N0f8}(174/255,55/255,14/255),RGB{N0f8}(.99,.99,.99),RGB{N0f8}(5/255,105/255,97/255))
toimage(state::BoxDomainState, scale=2) = mappedarray(i->tocolor(scalesigned(scale)(i)), state.current)
toimage(img, scale=2) = mappedarray(i->tocolor(scalesigned(scale)(i)), img)

end # module
