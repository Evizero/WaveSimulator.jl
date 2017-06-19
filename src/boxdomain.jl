struct BoxDomain{N} <: Domain{N}
    size::NTuple{N,Float64} # in m
    γ::Float64 # global admittance (loss at boundary)
end

function BoxDomain(size::NTuple{N,Number}; gamma=0) where N
    BoxDomain{N}(Float64.(size), Float64(gamma))
end

function BoxDomain(size...; gamma=0)
    BoxDomain(Float64.(size), Float64(gamma))
end

Base.size(box::BoxDomain) = box.size

mutable struct BoxState{N,T<:AbstractFloat,D,A<:AbstractArray{T,N},Q<:AbstractArray{Int8,N}} <: State{N,T,D}
    box::D
    previous::A
    current::A
    q::Q  # domain indicator grid
    t::T  # accumulated time
    iter::Int # accumulated simulation steps
end

function state_init(f0, wave::Wave{N,T}, box::BoxDomain{N}) where {N,T}
    dims = ceil.(Int, size(box) ./ wave.dx)
    Ψ₀ = T[T(f0(((I.I.-1).*wave.dx)...)) for I in CartesianRange(dims)]
    Ψ₁ = copy(Ψ₀)
    q  = zeros(Int8, dims)
    for I in CartesianRange(CartesianIndex(ntuple(i->2, Val{N})), CartesianIndex(size(q) .- 1))
        q[I] = Int8(1)
    end
    BoxState{N,T,typeof(box),typeof(Ψ₀),typeof(q)}(box,Ψ₀,Ψ₁,q,zero(T),0)
end

function state_init(wave::Wave{N,T}, box::BoxDomain{N}) where {N,T}
    dims = ceil.(Int, size(box) ./ wave.dx)
    Ψ₀ = zeros(T, dims)
    Ψ₁ = zeros(T, dims)
    q  = zeros(Int8, dims)
    for I in CartesianRange(CartesianIndex(ntuple(i->2, Val{N})), CartesianIndex(size(q) .- 1))
        q[I] = Int8(1)
    end
    BoxState{N,T,typeof(box),typeof(Ψ₀),typeof(q)}(box,Ψ₀,Ψ₁,q,zero(T),0)
end
