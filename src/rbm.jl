struct WeightNormRBM{V<:AbstractLayer, H<:AbstractLayer, G<:AbstractArray, U<:AbstractArray}
    visible::V
    hidden::H
    g::G # weight norms for each hidden unit
    u::U # weight (unormalized) directions
    function WeightNormRBM(visible::AbstractLayer, hidden::AbstractLayer, g::AbstractArray, u::AbstractArray)
        @assert all(size(g)[1:ndims(visible)] .== 1)
        @assert size(g)[(ndims(visible) + 1):end] == size(hidden)
        @assert size(u) == (size(visible)..., size(hidden)...)
        return new{typeof(visible), typeof(hidden), typeof(g), typeof(u)}(visible, hidden, g, u)
    end
end

function RBMs.RBM(wrbm::WeightNormRBM)
    w, _ = gu2w(wrbm.g, wrbm.u)
    return RBMs.RBM(wrbm.visible, wrbm.hidden, w)
end

function WeightNormRBM(rbm::RBM, un::AbstractArray)
    g, u = w2gu(rbm.w, un)
    return WeightNormRBM(rbm.visible, rbm.hidden, g, u)
end

function WeightNormRBM(rbm::RBM)
    g = weight_norms(rbm)
    u = copy(rbm.w)
    return WeightNormRBM(rbm.visible, rbm.hidden, g, u)
end

"""
    weight_norms(rbm)

Norms of weight patterns attached to each hidden unit.
"""
weight_norms(rbm::RBM) = sqrt.(sum(abs2, rbm.w; dims=1:ndims(rbm.visible)))

"""
    ∂free_energy(wrbm, v; wts, moments)

Gradient of `free_energy(RBM(wrbm), v)` with respect to the weight-normalized
parameterization. Returns a `NamedTuple` with fields `visible`, `hidden`
(gradients with respect to the layer parameter arrays `layer.par`), and
`g`, `u` (gradients with respect to the weight norms and directions).
"""
function RBMs.∂free_energy(
    wrbm::WeightNormRBM, v::AbstractArray;
    wts::AbstractArray{<:Real} = RBMs.uniform_wts(wrbm.visible, v),
    moments = RBMs.moments_from_samples(wrbm.visible, v; wts)
)
    ∂ = RBMs.∂free_energy(RBM(wrbm), v; wts, moments)
    ∂g, ∂u = ∂wnorm(∂.w, wrbm.g, wrbm.u)
    return (visible = ∂.visible, hidden = ∂.hidden, g = ∂g, u = ∂u)
end
