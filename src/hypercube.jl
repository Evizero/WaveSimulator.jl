struct HyperCube{N,T<:AbstractFloat} <: Domain{N}
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

mutable struct State{N,T<:AbstractFloat,D,A<:AbstractArray{T,N},Q<:AbstractArray{Int8,N}} <: AbstractState
    domain::D
    previous::A
    current::A
    q::Q  # domain indicator grid
    t::T  # accumulated time
    iter::Int # number of discrete simulation steps
end

function state_init(f0, wave::Wave{N,T}, domain::HyperCube{N}) where {N,T}
    dims = ceil.(Int, size(domain) ./ wave.dx)
    Ψ₀ = T[T(f0(((I.I.-1).*wave.dx)...)) for I in CartesianRange(dims)]
    Ψ₁ = copy(Ψ₀)
    q  = zeros(Int8, dims)
    for I in CartesianRange(CartesianIndex(ntuple(i->2, Val{N})), CartesianIndex(size(q) .- 1))
        q[I] = Int8(1)
    end
    State{N,T,typeof(domain),typeof(Ψ₀),typeof(q)}(domain,Ψ₀,Ψ₁,q,zero(T),0)
end
