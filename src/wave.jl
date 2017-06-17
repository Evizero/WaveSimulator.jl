struct Wave{T<:AbstractFloat}
    c::T    # constant wave speed
    fₘₐₓ::T # max supported frequency
end

function Wave(::Type{T}=Float64; c=340, fmax=2e3) where T<:AbstractFloat
    Wave(T(c), T(fmax))
end

function sampling_settings(wave::Wave{T}, N, lambda=-1, dt=-1, dx=-1) where T <: AbstractFloat
    λ = T(lambda == -1 ? √(1/N) : lambda)
    if dt == -1 && dx == -1
        dₜ = T(λ / (2*wave.fₘₐₓ))
        dₓ = T(wave.c * dₜ / λ)
    elseif dt != -1 && dx == -1
        dₜ = T(dt)
        dₓ = T(wave.c * dₜ / λ)
        warn("Overwriting specified fₘₐₓ = $(round(wave.fₘₐₓ/1000,2)) kHz with implied fₘₐₓ = $(round(λ/(2*dₜ)/1000,2)) kHz by setting dₜ = $(dₜ) s")
    elseif dt == -1 && dx == -1
        dₓ = T(dx)
        dₜ = T(dₓ * λ / wave.c)
    elseif dt != -1 && dx != -1
        dₜ = T(dt)
        dₓ = T(dx)
        λ = T(wave.c * dₜ / dₓ)
    end
    λ, dₜ, dₓ
end
