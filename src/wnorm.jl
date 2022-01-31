@doc raw"""
    gu2w(g, u) -> w, un

Returns `w, un` (as a `NamedTuple`), from `g, u`, where:

```math
\mathbf{w} = g \frac{\mathbf{u}}{\|\mathbf{u}\|}
```

and `un` are the norms ``\|\mathbf{u}\|``.
"""
function gu2w(g::AbstractArray, u::AbstractArray)
    un = sqrt.(sum!(similar(g), abs2.(u)))
    return (w = g .* u ./ un, un = un)
end

@doc raw"""
    w2gu(w, un) -> g, u

Returns `g, u` (as a `NamedTuple`), such that

```math
\mathbf{w} = g \frac{\mathbf{u}}{\|\mathbf{u}\|}
```

where the norms ``\|\mathbf{u}\|`` are given by `un`.
"""
function w2gu(w::AbstractArray, un::AbstractArray)
    g = sqrt.(sum!(similar(un), abs2.(w)))
    u = w ./ g .* un
    return (g = g, u = u)
end

@doc raw"""
    ∂wnorm(∂w, g, u)

Given the gradients `∂w` of a function `f(w)` with respect to `w`,
returns the gradients `∂g, ∂u` of `f` with respect to the re-parameterization:

```math
\mathbf{w} = g \frac{\mathbf{u}}{\|\mathbf{u}\|}
```
"""
function ∂wnorm(∂w::AbstractArray, g::AbstractArray, u::AbstractArray)
    @assert size(∂w) == size(u)
    # see Eqs. (3) and (4) of Salimans & Kingma 2016
    un = sqrt.(sum!(similar(g), abs2.(u)))
    ∂g = sum!(similar(g), ∂w .* u) ./ un
    ∂u = (∂w - ∂g .* u ./ un) .* g ./ un
    return (g = ∂g, u = ∂u)
end
