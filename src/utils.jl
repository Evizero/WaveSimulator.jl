using Distributions, ImageCore, MappedArrays, Colors, FixedPointNumbers

function gauss(mean, var)
    function f0(I...)
        pdf(MvNormal(mean, sqrt(var)), [I...])
    end
    f0
end

function gauss(domain::Domain, var = 0.02)
    gauss([(size(domain)./2)...], var)
end

const tocolor = colorsigned(RGBA{N0f8}(174/255,55/255,14/255,1.),RGBA{N0f8}(.99,.99,.99,0.),RGBA{N0f8}(5/255,105/255,97/255,1.))

function toimage(state::State, scale = 2)
    mappedarray(i->tocolor(scalesigned(scale)(i)), state.current)
end

function toimage(img, scale = 2)
    mappedarray(i->tocolor(scalesigned(scale)(i)), img)
end

function dc_block(signal; R=0.995)
    cor_out = zeros(signal)
    cor_out[1] = signal[1]
    for i in 2:length(signal)
        cor_out[i] = signal[i] - signal[i-1] + R * cor_out[i-1]
    end
    cor_out
end

function make_sine(dt; A=1, f=220, duration=2., phase=0)
    Float64[A*sin(2π*f*t+phase) for t in 0:dt:duration]
end

function sin_response(; A=1, f=220, phase=0, length = 0.5, showvalues=true, duration=2., resource=CPUThreads((100,100,1)), gamma=0.05, fps=5, kw...)
    wave = UniformWave{3}(; kw...)
    signal = make_sine(wave.dt, A=A, f=f, phase=phase, duration = length)
    signal_response(signal, showvalues=showvalues, duration=duration, resource=resource, gamma=gamma, kw...)
end

function signal_response(signal; showvalues=true, fps=5, duration=2., resource=CPUThreads((100,100,1)), gamma=0.05, kw...)
    wave = UniformWave{3}(;kw...)
    speaker = PointSpeaker((3,1,2), signal)
    micro = PointMicrophone(3,7,2)
    sim = Simulator(wave, StoreSnapshots(1/fps), speaker, micro, ProgressDisplay(0.1, showvalues), resource=resource, duration=duration)
    domain = BoxDomain(6,8,4, gamma=gamma)
    backend = backend_init(sim.resource, domain, sim)
    state = state_init(backend, domain, sim);
    simulate!(state, backend, sim)
    backend_cleanup!(backend, state, sim)
    speaker.signal, micro.signal, sim
end

function impulse_response(; gamma=0., duration=2., resource=CPUThreads((100,100,1)), kw...)
    micro_in  = PointMicrophone(4,5,3)
    micro_out = PointMicrophone(1.7,2.8,0.9)
    sim = Simulator(UniformWave{3}(;kw...), micro_in, micro_out, ProgressDisplay(), resource=resource, duration=duration)
    domain = BoxDomain(6,8,4, gamma=gamma)
    backend = backend_init(sim.resource, domain, sim)
    state = state_init(backend, domain, sim);
    gridpos = floor.(Int, (4,5,3) ./ sim.wave.dx) .+ 1
    state.current[gridpos...] = sim.wave.c^2 * sim.wave.dt / sim.wave.dx^3
    simulate!(state, backend, sim)
    backend_cleanup!(backend, state, sim)
    micro_in.signal, micro_out.signal, sim
end
