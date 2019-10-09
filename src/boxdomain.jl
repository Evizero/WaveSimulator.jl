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
gridsize(domain::Domain, dx) = ceil.(Int, size(domain) ./ dx)

mutable struct BoxState{N,T<:AbstractFloat,D,A<:AbstractArray{T,N},Q<:AbstractArray} <: State{N,T,D}
    box::D
    previous::A
    current::A
    q::Q  # domain indicator grid
    t::T  # accumulated time
    iter::Int # accumulated simulation steps
end

function state_init(f0, backend::Backend, box::BoxDomain{N}, sim::Simulator{N,T}) where {N,T}
    dims = gridsize(box, sim.wave.dx)
    A = zeros(T, dims)
    for I in CartesianIndices(ntuple(i->2:(dims[i]-1), N))
        A[I] = T(f0(((I.I.-1).*sim.wave.dx)...))
    end
    Ψ₀ = move_backend(A, backend)
    Ψ₁ = move_backend(copy(A), backend)
    qhost = zeros(Int8, dims)
    for I in CartesianIndices(ntuple(i->2:(size(qhost)[i]-1), N))
        qhost[I] = Int8(1)
    end
    q = move_backend(qhost, backend)
    BoxState{N,T,typeof(box),typeof(Ψ₀),typeof(q)}(box,Ψ₀,Ψ₁,q,zero(T),0)
end

function state_init(backend::Backend, box::BoxDomain{N}, sim::Simulator{N,T}) where {N,T}
    dims = gridsize(box, sim.wave.dx)
    Ψ₀ = move_backend(zeros(T, dims), backend)
    Ψ₁ = move_backend(zeros(T, dims), backend)
    qhost = zeros(Int8, dims)
    for I in CartesianIndices(ntuple(i->3:(size(qhost)[i]-1), N))
        qhost[I] = Int8(1)
    end
    q = move_backend(qhost, backend)
    BoxState{N,T,typeof(box),typeof(Ψ₀),typeof(q)}(box,Ψ₀,Ψ₁,q,zero(T),0)
end
