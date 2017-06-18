using Distributions, ImageCore, MappedArrays, Colors, FixedPointNumbers

function gauss(mean, var)
    function f0(I...)
        pdf(MvNormal(mean, sqrt(var)), [I...])
    end
    f0
end

function gauss(domain::HyperCube, var = 0.02)
    gauss([(size(domain)./2)...], var)
end

const tocolor = colorsigned(RGB{N0f8}(174/255,55/255,14/255),RGB{N0f8}(.99,.99,.99),RGB{N0f8}(5/255,105/255,97/255))

function toimage(state::State, scale = 2)
    mappedarray(i->tocolor(scalesigned(scale)(i)), state.current)
end

function toimage(img, scale = 2)
    mappedarray(i->tocolor(scalesigned(scale)(i)), img)
end
