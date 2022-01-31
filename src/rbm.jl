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

function RBMs.∂free_energy(
    wrbm::WeightNormRBM, v::AbstractArray; wts = nothing,
    stats = RBMs.sufficient_statistics(wrbm.visible, v; wts)
)
    ∂ = RBMs.∂free_energy(RBM(wrbm), v; wts, stats)
    ∂g, ∂u = ∂wnorm(∂.w, wrbm.g, wrbm.u)
    return (visible = ∂.visible, hidden = ∂.hidden, g = ∂g, u = ∂u)
end
