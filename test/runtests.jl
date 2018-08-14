using WaveSimulator
using ComputationalResources
using Base.Test

if Base.Threads.nthreads() > 1
    addresource(CPUThreads)
end
if CUDAdrv.vendor() != "none"
    addresource(CUDALibs)
end
info("Testing resources: $(ComputationalResources.resources)")

# Generate resource instances
test_resources = AbstractResource[]
for resource_type in ComputationalResources.resources
    if resource_type == CPU1
        push!(test_resources, CPU1())
    elseif resource_type == CPUThreads
        push!(test_resources, CPUThreads((64,64,1)))
    elseif resource_type == CUDALibs
        push!(test_resources, CUDALibs())
    end
end

@testset "Backend Simulation State" begin
    test_results = Dict()
    for resource in test_resources
        # Setup
        wave = UniformWave{3}(fmax=2e3)
        sim = Simulator(wave, resource=resource, duration=0.01)
        domain = BoxDomain(6,8,4, gamma=0.05)
        f0 = WaveSimulator.gauss(domain)
        backend = WaveSimulator.backend_init(sim.resource, domain, sim)
        state = WaveSimulator.state_init(f0, backend, domain, sim)
        # Simulation
        for i in 1:5
            simulate!(state, backend, sim)
        end
        test_results[typeof(resource)] = state.current
    end

    # Test that all considered backends return similar results
    cpu1_result = first(values(filter((res,_)->res<:CPU1, test_results)))
    filter!((res,result)->!(res isa CPU1), test_results)
    for resource in keys(test_results)
        info("Comparing CPU1 to $(resource)")
        @test isapprox(cpu1_result, Array(test_results[resource]))
    end
end
