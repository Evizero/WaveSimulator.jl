module WaveSimulator

using Base.Cartesian, ComputationalResources, TiledIteration, ProgressMeter, ValueHistories

export

    CPU1,
    CPUThreads,
    CUDALibs,

    UniformWave,
    BoxDomain,
    Simulator,
    simulate,
    simulate_gauss,
    simulate!,
    update!,
    toimage

abstract type Domain{N} end
abstract type State{N,T,D<:Domain} end
abstract type Wave{N,T<:AbstractFloat} end
abstract type Backend{R<:AbstractResource} end
abstract type CPUBackend{R<:AbstractResource} <: Backend{R} end

include("wave.jl")
include("simulator.jl")
include("boxdomain.jl")
include("utils.jl")
include("hooks.jl")
include("backends/cpu.jl")
include("backends/cuda.jl")

end
