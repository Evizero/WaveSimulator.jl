using WaveSimulator
using PkgBenchmark

@benchgroup "backend" ["init", "allocation", "kernel"] begin
    for (resource, tags) in (
            (CPU1(), ["CPU", "CPU1"]),
            (CPUThreads((100,100,1)), ["CPU", "CPUThreads"]),
            (CUDALibs(), ["GPU"]))
        @benchgroup "$(typeof(resource).name.name)" tags begin
            wave = UniformWave{3}(fmax=2e3)
            sim = Simulator(wave, resource=resource, duration=0.01)
            domain = BoxDomain(6,8,4, gamma=0.05)
            backend = WaveSimulator.backend_init(sim.resource, domain, sim)
            @bench(("state_init", "zeros"),
                WaveSimulator.state_init($backend, $domain, $sim),
                teardown = gc(),
                seconds = 30,
                samples = 10
            )
            @bench(("state_init", "gauss"),
                WaveSimulator.state_init(f0, $backend, $domain, $sim),
                setup = begin
                    f0 = WaveSimulator.gauss($domain)
                end,
                teardown = gc(),
                seconds = 30,
                samples = 10
            )
            @bench("update!",
                WaveSimulator.update_backend!(state, backend, sim),
                setup = begin
                    sim = Simulator($wave, resource=$(resource), duration=0.01)
                    domain = BoxDomain(6,8,4, gamma=0.05)
                    f0 = WaveSimulator.gauss(domain)
                    backend = WaveSimulator.backend_init(sim.resource, domain, sim)
                    state   = WaveSimulator.state_init(f0, backend, domain, sim)
                end,
                teardown = begin
                    backend = nothing
                    state = nothing
                    gc()
                end,
                seconds = 30,
                samples = 50
            )
        end
    end
end

@benchgroup "integration" ["simulate!"] begin
    for (resource, tags) in (
            (CPU1(), ["CPU", "CPU1"]),
            (CPUThreads((100,100,1)), ["CPU", "CPUThreads"]),
            (CUDALibs(), ["GPU"]))
        @benchgroup "$(typeof(resource).name.name)" tags begin
            @bench("simulate!",
                simulate!(state, backend, sim),
                setup = begin
                    wave = UniformWave{3}(fmax=2e3)
                    sim = Simulator(wave, resource=$(resource), duration=0.01)
                    domain = BoxDomain(6,8,4, gamma=0.05)
                    f0 = WaveSimulator.gauss(domain)
                    backend = WaveSimulator.backend_init(sim.resource, domain, sim)
                    state   = WaveSimulator.state_init(f0, backend, domain, sim)
                end,
                teardown = begin
                    backend = nothing
                    state = nothing
                    gc()
                end,
                seconds = 30,
                samples = 50
            )
        end
    end
end
