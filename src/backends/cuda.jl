using CUDAdrv, CUDAnative

struct CUDABackend{R<:CUDALibs} <: Backend{R}
    resource::R
    device::CuDevice
    context::CuContext
end

function backend_init(resource::CUDALibs, domain, sim)
    dev = CuDevice(0)
    ctx = CuContext(dev)
    CUDABackend(resource, dev, ctx)
end

function backend_cleanup!(backend::CUDABackend, state, sim)
    destroy!(backend.context)
end

function move_backend(A::Array{T,N}, backend::CUDABackend) where {T<:Union{Float32,Float64},N}
    CuArray(A)
end

function move_backend(A::Array{Int8,N}, backend::CUDABackend) where N
    move_backend(Float32.(A), backend)
end

# --------------------------------------------------------------------

function update!(state::BoxState{2,T}, backend::CUDABackend, sim) where T
    threads = (32,32)
    yblocks = ceil(Int, size(state.current,1) / threads[2])
    xblocks = ceil(Int, size(state.current,2) / threads[1])
    shmem = 2 * prod(threads.+2) * sizeof(T)
    @cuda ((xblocks,yblocks),threads,shmem) cuda_kernel!(
        pointer(state.previous), pointer(state.current),
        pointer(state.q),
        size(state.current,1),
        size(state.current,2),
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
    λsq = λ*λ
    λhalf = λ/2

    if 1 < y < h && 1 < x < w
        shmem = @cuDynamicSharedMem(T, (blockDim().y+2, blockDim().x+2, 2))
        if threadIdx().y == 1 || y == 2
            shmem[threadIdx().y,threadIdx().x+1,1]   = Ψₜ[y-1,x]
            shmem[threadIdx().y,threadIdx().x+1,2]   = T(q[y-1,x])
        elseif threadIdx().y == blockDim().y || y == h-1
            shmem[threadIdx().y+2,threadIdx().x+1,1] = Ψₜ[y+1,x]
            shmem[threadIdx().y+2,threadIdx().x+1,2] = T(q[y+1,x])
        end
        if threadIdx().x == 1 || x == 2
            shmem[threadIdx().y+1,threadIdx().x,1]   = Ψₜ[y,x-1]
            shmem[threadIdx().y+1,threadIdx().x,2]   = T(q[y,x-1])
        elseif threadIdx().x == blockDim().x || x == w-1
            shmem[threadIdx().y+1,threadIdx().x+2,1] = Ψₜ[y,x+1]
            shmem[threadIdx().y+1,threadIdx().x+2,2] = T(q[y,x+1])
        end
        shmem[threadIdx().y+1,threadIdx().x+1,1] = Ψₜ[y,x]
        shmem[threadIdx().y+1,threadIdx().x+1,2] = T(q[y,x])

        sync_threads()

        q₁₋ = shmem[threadIdx().y,  threadIdx().x+1,2]
        q₁₊ = shmem[threadIdx().y+2,threadIdx().x+1,2]
        q₂₋ = shmem[threadIdx().y+1,threadIdx().x,  2]
        q₂₊ = shmem[threadIdx().y+1,threadIdx().x+2,2]
        K = q₁₋ + q₁₊ + q₂₋ + q₂₊

        Ψₜₛ  = shmem[threadIdx().y+1,threadIdx().x+1,1]
        Ψₜ₁₋ = shmem[threadIdx().y,  threadIdx().x+1,1]
        Ψₜ₁₊ = shmem[threadIdx().y+2,threadIdx().x+1,1]
        Ψₜ₂₋ = shmem[threadIdx().y+1,threadIdx().x,  1]
        Ψₜ₂₊ = shmem[threadIdx().y+1,threadIdx().x+2,1]
        Q = q₁₋ * Ψₜ₁₋ + q₁₊ * Ψₜ₁₊ +
            q₂₋ * Ψₜ₂₋ + q₂₊ * Ψₜ₂₊

        γ̄ = γ * ((1-q₁₋) + (1-q₁₊) + (1-q₂₋) + (1-q₂₊))

        Ψₜ₊₁[y,x] = 1/(1+λhalf*γ̄) * ((2-λsq*K) * Ψₜₛ + (λhalf*γ̄-1) * Ψₜ₋₁[y,x] + λsq*Q)
    end

    return nothing
end

function update!(state::BoxState{3}, backend::CUDABackend, sim)
    threads = (16,8,8)
    yblocks = ceil(Int, size(state.current,1) / threads[2])
    xblocks = ceil(Int, size(state.current,2) / threads[1])
    zblocks = ceil(Int, size(state.current,3) / threads[3])
    @cuda ((xblocks,yblocks,zblocks),threads) cuda_kernel!(
        pointer(state.previous), pointer(state.current),
        pointer(state.q),
        size(state.current)...,
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
    λsq = λ*λ
    λhalf = λ/2

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

        γ̄ = γ * ((1-q₁₋) + (1-q₁₊) + (1-q₂₋) + (1-q₂₊)+ (1-q₃₋) + (1-q₃₊))

        Ψₜ₊₁[y,x,z] = 1/(1+λhalf*γ̄) * ((2-λsq*K) * Ψₜ[y,x,z] + (λhalf*γ̄-1) * Ψₜ₋₁[y,x,z] + λsq*Q)
    end

    return nothing
end
