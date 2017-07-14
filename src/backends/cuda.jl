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

function update!(state::BoxState{2,T}, backend::CUDABackend, sim) where T
    h,w = size(state.current)
    threads = (32,32)
    yblocks = ceil(Int, h / threads[2])
    xblocks = ceil(Int, w / threads[1])
    # shmem = 2 * prod(threads.+2) * sizeof(T)
    @cuda ((xblocks,yblocks),threads) cuda_kernel!(
        pointer(state.previous), pointer(state.current),
        pointer(state.q),
        h, w,
        sim.wave.λ,
        state.box.γ)
    state
end

function cuda_kernel!(previous_ptr::Ptr{T}, current_ptr::Ptr{T}, q_ptr::Ptr, h::Int, w::Int, λ::T, γ::T) where T
    y = (blockIdx().y-1) * blockDim().y + threadIdx().y
    x = (blockIdx().x-1) * blockDim().x + threadIdx().x
    Ψₜ₋₁ = CuDeviceArray((h,w), previous_ptr)
    Ψₜ   = CuDeviceArray((h,w), current_ptr)
    Ψₜ₊₁ = CuDeviceArray((h,w), previous_ptr)
    q    = CuDeviceArray((h,w), q_ptr)
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

function update!(state::BoxState{3,T}, backend::CUDABackend, sim) where T
    h,w,v = size(state.current)
    threads = (16,8,8)
    yblocks = ceil(Int, h / threads[2])
    xblocks = ceil(Int, w / threads[1])
    zblocks = ceil(Int, v / threads[3])
    #shmem = Int(2 * prod(threads.+2) * sizeof(T))
    @cuda ((xblocks,yblocks,zblocks),threads) cuda_kernel!(
        pointer(state.previous), pointer(state.current),
        pointer(state.q),
        h, w, v,
        sim.wave.λ,
        state.box.γ)
    state
end

function cuda_kernel!(previous_ptr::Ptr{T}, current_ptr::Ptr{T}, q_ptr::Ptr, h::Int, w::Int, v::Int, λ::T, γ::T) where T
    y = (blockIdx().y-1) * blockDim().y + threadIdx().y
    x = (blockIdx().x-1) * blockDim().x + threadIdx().x
    z = (blockIdx().z-1) * blockDim().z + threadIdx().z
    Ψₜ₋₁ = CuDeviceArray((h,w,v), previous_ptr)
    Ψₜ   = CuDeviceArray((h,w,v), current_ptr)
    Ψₜ₊₁ = CuDeviceArray((h,w,v), previous_ptr)
    q    = CuDeviceArray((h,w,v), q_ptr)
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
