# Not included or used because there is no speedup whatsoever

function cuda_kernel_shared!(previous_ptr::Ptr{T}, current_ptr::Ptr{T}, q_ptr::Ptr, h::Int, w::Int, λ::T, γ::T) where T
    y = (blockIdx().y-1) * blockDim().y + threadIdx().y
    x = (blockIdx().x-1) * blockDim().x + threadIdx().x
    Ψₜ₋₁ = CuDeviceArray((h,w), previous_ptr)
    Ψₜ   = CuDeviceArray((h,w), current_ptr)
    Ψₜ₊₁ = CuDeviceArray((h,w), previous_ptr)
    q    = CuDeviceArray((h,w), q_ptr)
    λsq = λ*λ
    λhalf = λ/2
    eins = one(T)
    zwei = eins + eins

    if 1 < y < h && 1 < x < w
        S = @cuDynamicSharedMem(T, (2, blockDim().y+2, blockDim().x+2))
        if threadIdx().y == 1 || y == 2
            S[1, threadIdx().y,   threadIdx().x+1] = T(q[y-1,x])
            S[2, threadIdx().y,   threadIdx().x+1] = Ψₜ[y-1,x]
        elseif threadIdx().y == blockDim().y || y == h-1
            S[1, threadIdx().y+2, threadIdx().x+1] = T(q[y+1,x])
            S[2, threadIdx().y+2, threadIdx().x+1] = Ψₜ[y+1,x]
        end
        if threadIdx().x == 1 || x == 2
            S[1, threadIdx().y+1, threadIdx().x] = T(q[y,x-1])
            S[2, threadIdx().y+1, threadIdx().x] = Ψₜ[y,x-1]
        elseif threadIdx().x == blockDim().x || x == w-1
            S[1, threadIdx().y+1, threadIdx().x+2] = T(q[y,x+1])
            S[2, threadIdx().y+1, threadIdx().x+2] = Ψₜ[y,x+1]
        end
        S[1, threadIdx().y+1, threadIdx().x+1] = T(q[y,x])
        S[2, threadIdx().y+1, threadIdx().x+1] = Ψₜ[y,x]

        sync_threads()

        q₂₋  = S[1, threadIdx().y+1, threadIdx().x  ]
        Ψₜ₂₋ = S[2, threadIdx().y+1, threadIdx().x  ]
        q₁₋  = S[1, threadIdx().y,   threadIdx().x+1]
        Ψₜ₁₋ = S[2, threadIdx().y,   threadIdx().x+1]
        Ψₜₛ  = S[2, threadIdx().y+1, threadIdx().x+1]
        q₁₊  = S[1, threadIdx().y+2, threadIdx().x+1]
        Ψₜ₁₊ = S[2, threadIdx().y+2, threadIdx().x+1]
        q₂₊  = S[1, threadIdx().y+1, threadIdx().x+2]
        Ψₜ₂₊ = S[2, threadIdx().y+1, threadIdx().x+2]

        K = q₁₋ + q₁₊ + q₂₋ + q₂₊

        Q = q₁₋ * Ψₜ₁₋ + q₁₊ * Ψₜ₁₊ +
            q₂₋ * Ψₜ₂₋ + q₂₊ * Ψₜ₂₊

        γ̄ = γ * ((eins-q₁₋) + (eins-q₁₊) + (eins-q₂₋) + (eins-q₂₊))

        Ψₜ₊₁[y,x] = eins/(eins+λhalf*γ̄) * ((zwei-λsq*K) * Ψₜₛ + (λhalf*γ̄-eins) * Ψₜ₋₁[y,x] + λsq*Q)
    end

    return nothing
end


function cuda_kernel_shared!(previous_ptr::Ptr{T}, current_ptr::Ptr{T}, q_ptr::Ptr, h::Int, w::Int, v::Int, λ::T, γ::T) where T
    y = (blockIdx().y-1) * blockDim().y + threadIdx().y
    x = (blockIdx().x-1) * blockDim().x + threadIdx().x
    z = (blockIdx().z-1) * blockDim().z + threadIdx().z
    Ψₜ₋₁ = CuDeviceArray((h,w,v), previous_ptr)
    Ψₜ   = CuDeviceArray((h,w,v), current_ptr)
    Ψₜ₊₁ = CuDeviceArray((h,w,v), previous_ptr)
    q    = CuDeviceArray((h,w,v), q_ptr)
    λsq = λ*λ
    λhalf = λ/2
    eins = one(T)
    zwei = eins + eins

    if 1 < y < h && 1 < x < w && 1 < z < v
        S = @cuDynamicSharedMem(T, (2, blockDim().y+2, blockDim().x+2, blockDim().z+2))
        sy = threadIdx().y+1
        sx = threadIdx().x+1
        sz = threadIdx().z+1
        if threadIdx().y == 1 || y == 2
            S[1, sy-1, sx, sz] = T(q[y-1, x, z])
            S[2, sy-1, sx, sz] =  Ψₜ[y-1, x, z]
        elseif threadIdx().y == blockDim().y || y == h-1
            S[1, sy+1, sx, sz] = T(q[y+1, x, z])
            S[2, sy+1, sx, sz] =  Ψₜ[y+1, x, z]
        end
        if threadIdx().x == 1 || x == 2
            S[1, sy, sx-1, sz] = T(q[y, x-1, z])
            S[2, sy, sx-1, sz] =  Ψₜ[y, x-1, z]
        elseif threadIdx().x == blockDim().x || x == w-1
            S[1, sy, sx+1, sz] = T(q[y, x+1, z])
            S[2, sy, sx+1, sz] =  Ψₜ[y, x+1, z]
        end
        if threadIdx().z == 1 || z == 2
            S[1, sy, sx, sz-1] = T(q[y, x, z-1])
            S[2, sy, sx, sz-1] =  Ψₜ[y, x, z-1]
        elseif threadIdx().z == blockDim().z || z == v-1
            S[1, sy, sx, sz+1] = T(q[y, x, z+1])
            S[2, sy, sx, sz+1] =  Ψₜ[y, x, z+1]
        end
        S[1, sy, sx, sz] = T(q[y, x, z])
        S[2, sy, sx, sz] =  Ψₜ[y, x, z]

        sync_threads()

        q₃₋  = S[1, sy,   sx,   sz-1]
        Ψₜ₃₋ = S[2, sy,   sx,   sz-1]
        q₂₋  = S[1, sy,   sx-1, sz]
        Ψₜ₂₋ = S[2, sy,   sx-1, sz]
        q₁₋  = S[1, sy-1, sx,   sz]
        Ψₜ₁₋ = S[2, sy-1, sx,   sz]
        Ψₜₛ  = S[2, sy,   sx,   sz]
        q₁₊  = S[1, sy+1, sx,   sz]
        Ψₜ₁₊ = S[2, sy+1, sx,   sz]
        q₂₊  = S[1, sy,   sx+1, sz]
        Ψₜ₂₊ = S[2, sy,   sx+1, sz]
        q₃₊  = S[1, sy,   sx,   sz+1]
        Ψₜ₃₊ = S[2, sy,   sx,   sz+1]

        K = q₁₋ + q₁₊ + q₂₋ + q₂₊ + q₃₋ + q₃₊

        Q = q₁₋ * Ψₜ₁₋ + q₁₊ * Ψₜ₁₊ +
            q₂₋ * Ψₜ₂₋ + q₂₊ * Ψₜ₂₊ +
            q₃₋ * Ψₜ₃₋ + q₃₊ * Ψₜ₃₊

        γ̄ = γ * ((eins-q₁₋) + (eins-q₁₊) + (eins-q₂₋) + (eins-q₂₊) + (eins-q₃₋) + (eins-q₃₊))

        Ψₜ₊₁[y,x,z] = eins/(eins+λhalf*γ̄) * ((zwei-λsq*K) * Ψₜₛ + (λhalf*γ̄-eins) * Ψₜ₋₁[y,x,z] + λsq*Q)
    end

    return nothing
end
