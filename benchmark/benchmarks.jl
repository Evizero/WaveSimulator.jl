using WaveSimulator
using BenchmarkTools

const SUITE = BenchmarkGroup()

SUITE["backend"] = BenchmarkGroup(["init", "allocation", "kernel"])
for (resource, tags) in (
        (CPU1(), ["CPU", "CPU1"]),
        (CPUThreads((100,100,1)), ["CPU", "CPUThreads"]),
        (CUDALibs(), ["GPU"]))
    GRP = SUITE["backend"][string(typeof(resource).name.name)] = BenchmarkGroup(tags)
    wave = UniformWave{3}(fmax=2e3)
    sim = Simulator(wave, resource=resource, duration=0.01)
    domain = BoxDomain(6,8,4, gamma=0.05)
    backend = WaveSimulator.backend_init(sim.resource, domain, sim)
    GRP["state_init_zeros"] = @benchmarkable(
        WaveSimulator.state_init($backend, $domain, $sim),
        teardown = gc(),
        seconds = 30,
        samples = 10
    )
    GRP["state_init_gaus"] = @benchmarkable(
        WaveSimulator.state_init(f0, $backend, $domain, $sim),
        setup = begin
            f0 = WaveSimulator.gauss($domain)
        end,
        teardown = gc(),
        seconds = 30,
        samples = 10
    )
    GRP["update!"] = @benchmarkable(
        WaveSimulator.backend_update!(state, backend, sim),
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

SUITE["integration"] = BenchmarkGroup(["simulate!"])
for (resource, tags) in (
        (CPU1(), ["CPU", "CPU1"]),
        (CPUThreads((100,100,1)), ["CPU", "CPUThreads"]),
        (CUDALibs(), ["GPU"]))
    GRP = SUITE["integration"][string(typeof(resource).name.name)] = BenchmarkGroup(tags)
    GRP["simulate!"] = @benchmarkable(
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
