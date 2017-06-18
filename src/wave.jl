struct UniformWave{N,T<:AbstractFloat} <: Wave{N,T}
    c::T    # constant wave speed
    fmax::T # max supported frequency
    dx::T   # spatial step (c*dt/λ)
    dt::T   # time step (dx*λ/c)
    λ::T    # courant number c*dt/dx
end

function (::Type{UniformWave{N}})(::Type{T}=Float64; c=340, fmax=-1, lambda=-1, dt=-1, dx=-1) where {N, T <: AbstractFloat}
    λ = T(lambda == -1 ? √(1/N) : lambda)
    if fmax == -1 && dt == -1 && dx == -1
        dₜ = T(λ / (2*2e3))
        dₓ = T(c * dₜ / λ)
    elseif dt == -1 && dx == -1
        dₜ = T(λ / (2*fmax))
        dₓ = T(c * dₜ / λ)
    elseif dt != -1 && dx == -1
        dₜ = T(dt)
        dₓ = T(c * dₜ / λ)
    elseif dt == -1 && dx != -1
        dₓ = T(dx)
        dₜ = T(dₓ * λ / c)
    elseif dt != -1 && dx != -1
        dₜ = T(dt)
        dₓ = T(dx)
        λ = T(c * dₜ / dₓ)
    end
    fₘₐₓ = T(λ / (2*dₜ))
    fmax == -1 || fₘₐₓ ≈ fmax || warn("Overwriting specified fₘₐₓ = $(round(fmax/1000,2)) kHz with implied fₘₐₓ = $(round(fₘₐₓ/1000,2)) kHz")
    UniformWave{N,T}(c, fₘₐₓ, dₓ, dₜ, λ)
end
