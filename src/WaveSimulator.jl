module WaveSimulator

using Base.Cartesian, ComputationalResources, TiledIteration, ProgressMeter

export

    UniformWave,
    BoxDomain,
    Simulator,
    simulate,
    simulate_gauss,
    simulate!,
    toimage

abstract type Domain{N} end
abstract type State{N,T,D<:Domain} end
abstract type Wave{N,T<:AbstractFloat} end
abstract type Backend{R<:AbstractResource} end

include("wave.jl")
include("boxdomain.jl")
include("utils.jl")
include("simulator.jl")
include("hooks.jl")
include("backends/cpu.jl")

end
