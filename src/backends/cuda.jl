using CUDAdrv, CUDAnative

struct CUDABackend{R<:CUDALibs} <: Backend{R}
    resource::R
end

function backend_init(resource::CUDALibs, domain, sim)
    CUDABackend(resource)
end

function backend_cleanup!(backend::CUDABackend, state, sim)
    nothing
end

function move_backend(A::Array{T,N}, backend::CUDABackend) where {T<:Union{Float32,Float64},N}
    CuArray(A)
end

function move_backend(A::Array{Int8,N}, backend::CUDABackend) where N
    move_backend(Float32.(A), backend)
end

# --------------------------------------------------------------------

function backend_update!(state::BoxState{2,T}, backend::CUDABackend, sim) where T
    h,w = size(state.current)
    threads = (32,32)
    yblocks = ceil(Int, h / threads[2])
    xblocks = ceil(Int, w / threads[1])
    # shmem = 2 * prod(threads.+2) * sizeof(T)
    @cuda blocks=(xblocks,yblocks) threads=threads cuda_kernel!(
        state.previous, state.current, state.q,
        sim.wave.λ,
        state.box.γ)
    state
end

function cuda_kernel!(Ψₜ₋₁::CuDeviceArray{T,2}, Ψₜ::CuDeviceArray{T,2}, q::CuDeviceArray, λ::T, γ::T) where T
    y = (blockIdx().y-1) * blockDim().y + threadIdx().y
    x = (blockIdx().x-1) * blockDim().x + threadIdx().x
    h, w = size(Ψₜ₋₁)
    Ψₜ₊₁ = Ψₜ₋₁ # store next state in previous
    eins = one(T)
    zwei = eins + eins
    λsq = λ*λ
    λhalf = λ/zwei

    @inbounds if 1 < y < h && 1 < x < w
        q₁₋ = q[y-1,x]
        q₁₊ = q[y+1,x]
        q₂₋ = q[y,x-1]
        q₂₊ = q[y,x+1]
        K = q₁₋ + q₁₊ + q₂₋ + q₂₊

        Q = q₁₋ * Ψₜ[y-1,x] + q₁₊ * Ψₜ[y+1,x] +
            q₂₋ * Ψₜ[y,x-1] + q₂₊ * Ψₜ[y,x+1]

        γ̄ = γ * ((eins-q₁₋) + (eins-q₁₊) + (eins-q₂₋) + (eins-q₂₊))

        Ψₜ₊₁[y,x] = eins/(eins+λhalf*γ̄) * ((zwei-λsq*K) * Ψₜ[y,x] + (λhalf*γ̄-eins) * Ψₜ₋₁[y,x] + λsq*Q)
    end

    return nothing
end

# --------------------------------------------------------------------

function backend_update!(state::BoxState{3,T}, backend::CUDABackend, sim) where T
    h,w,v = size(state.current)
    threads = (16,8,8)
    yblocks = ceil(Int, h / threads[2])
    xblocks = ceil(Int, w / threads[1])
    zblocks = ceil(Int, v / threads[3])
    #shmem = Int(2 * prod(threads.+2) * sizeof(T))
    @cuda blocks=(xblocks,yblocks,zblocks) threads=threads cuda_kernel!(
        state.previous, state.current, state.q,
        sim.wave.λ,
        state.box.γ)
    state
end

function cuda_kernel!(Ψₜ₋₁::CuDeviceArray{T,3}, Ψₜ::CuDeviceArray{T,3}, q::CuDeviceArray, λ::T, γ::T) where T
    y = (blockIdx().y-1) * blockDim().y + threadIdx().y
    x = (blockIdx().x-1) * blockDim().x + threadIdx().x
    z = (blockIdx().z-1) * blockDim().z + threadIdx().z
    h, w, v = size(Ψₜ₋₁)
    Ψₜ₊₁ = Ψₜ₋₁ # store next state in previous
    eins = one(T)
    zwei = eins + eins
    λsq = λ*λ
    λhalf = λ/zwei

    @inbounds if 1 < y < h && 1 < x < w && 1 < z < v
        q₁₋ = q[y-1,x,z]
        q₁₊ = q[y+1,x,z]
        q₂₋ = q[y,x-1,z]
        q₂₊ = q[y,x+1,z]
        q₃₋ = q[y,x,z-1]
        q₃₊ = q[y,x,z+1]
        K = q₁₋ + q₁₊ + q₂₋ + q₂₊ + q₃₋ + q₃₊

        Q = q₁₋ * Ψₜ[y-1,x,z] + q₁₊ * Ψₜ[y+1,x,z] +
            q₂₋ * Ψₜ[y,x-1,z] + q₂₊ * Ψₜ[y,x+1,z] +
            q₃₋ * Ψₜ[y,x,z-1] + q₃₊ * Ψₜ[y,x,z+1]

        γ̄ = γ * ((eins-q₁₋) + (eins-q₁₊) + (eins-q₂₋) + (eins-q₂₊)+ (eins-q₃₋) + (eins-q₃₊))

        Ψₜ₊₁[y,x,z] = eins/(eins+λhalf*γ̄) * ((zwei-λsq*K) * Ψₜ[y,x,z] + (λhalf*γ̄-eins) * Ψₜ₋₁[y,x,z] + λsq*Q)
    end

    return nothing
end
