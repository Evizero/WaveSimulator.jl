module WaveSimulator

using Base.Cartesian, ComputationalResources, TiledIteration, ProgressMeter

export

    Wave,
    HyperCube,
    setup,
    setup_gauss,
    simulate,
    simulate!,
    toimage

include("wave.jl")
include("hypercube.jl")
include("utils.jl")
include("backends/cpu.jl")

end
