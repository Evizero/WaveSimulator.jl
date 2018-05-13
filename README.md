# WaveSimulator

[![Build Status](https://travis-ci.org/Evizero/WaveSimulator.jl.svg?branch=master)](https://travis-ci.org/Evizero/WaveSimulator.jl) [![Coverage Status](https://coveralls.io/repos/Evizero/WaveSimulator.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/Evizero/WaveSimulator.jl?branch=master) [![codecov.io](http://codecov.io/github/Evizero/WaveSimulator.jl/coverage.svg?branch=master)](http://codecov.io/github/Evizero/WaveSimulator.jl?branch=master)

Simple Gauss impulse response to reproduce Fig 3.8 of
https://www.era.lib.ed.ac.uk/handle/1842/22940 on page 70 (83 of
PDF)


```julia
using WaveSimulator, Images

wave = UniformWave{3}(fmax=2e3)
sim = Simulator(wave, resource=CUDALibs(), duration=0.0245)
domain = BoxDomain(6,8,4, gamma=0.0)

res = simulate_gauss(sim, domain)

A = res.current # 71×95×48 CUDAdrv.CuArray{Float64,3}

# visualization of the acoustic field (seen from top-down)
# at around half the room's height
toimage(Array(A), .6)[:,:,24]
```

![acoustic field](https://user-images.githubusercontent.com/10854026/39971172-376db93c-56f7-11e8-96e2-2da7fd3e7b74.png)

Simple signal response. Point speaker plays the sound in the
right part of the room while a point microphone listens in the
left part of the room. See implementation for details.

```julia
using WaveSimulator, Images, ComputationalResources, SampledSignals, LibSndFile
test = load("test.wav");
signal = vec(Float64.(test.data))
SampleBuf(signal, test.samplerate) # can play in notebook

input, output, sim = WaveSimulator.signal_response(
    signal, fps=120, gamma=0.02, duration=1.,
    dt=1/test.samplerate, showvalues=false
)

out = SampleBuf(output, 1/sim.wave.dt)
save("test_out.wav", out)
```
