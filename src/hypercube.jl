struct HyperCube{N,T<:AbstractFloat}
    size::NTuple{N,T} # in m
    γ::T # global admittance (loss at boundary)
end

function HyperCube(size::NTuple{N,Number}; gamma=0) where N
    HyperCube{N}(Float64.(size), Float64(gamma))
end

function HyperCube(size...; gamma=0)
    HyperCube(Float64.(size), Float64(gamma))
end

Base.size(domain::HyperCube) = domain.size

mutable struct State{N,T<:AbstractFloat,D,A<:AbstractArray{T,N},Q<:AbstractArray{Int8,N}}
    wave::Wave{T}
    domain::D
    previous::A
    current::A
    q::Q  # domain indicator grid
    dx::T # spatial step (c*dt/λ)
    dt::T # time step (dx*λ/c)
    λ::T  # courant number c*dt/dx
    t::T  # accumulated time
    iter::Int # number of discrete simulation steps
end

function setup{N,T}(f0, wave::Wave{T}, domain::HyperCube{N}; lambda=-1, dt=-1, dx=-1)
    λ, dₜ, dₓ = sampling_settings(wave, N, lambda, dt, dx)
    dims = ceil.(Int, domain.size ./ dₓ)
    Ψ₀ = T[T(f0(((I.I.-1).*dₓ)...)) for I in CartesianRange(dims)]
    Ψ₁ = copy(Ψ₀)
    q  = zeros(Int8, dims)
    for I in CartesianRange(CartesianIndex(ntuple(i->2, Val{N})), CartesianIndex(size(q) .- 1))
        q[I] = Int8(1)
    end
    State{N,T,typeof(domain),typeof(Ψ₀),typeof(q)}(wave,domain,Ψ₀,Ψ₁,q,T(dₓ),T(dₜ),T(λ),zero(T),0)
end

function setup{N,T}(wave::Wave{T}, domain::HyperCube{N}; lambda=-1, dt=-1, dx=-1)
    setup((I...)->zero(T), wave, domain, lambda=lambda, dt=dt, dx=dx)
end
