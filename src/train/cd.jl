function RBMs.∂contrastive_divergence(
    rbm::WeightNormRBM, vd::AbstractArray, vm::AbstractArray; wd = nothing, wm = nothing,
    stats = RBMs.sufficient_statistics(rbm.visible, vd; wts = wd)
)
    ∂d = RBMs.∂free_energy(rbm, vd; wts = wd, stats)
    ∂m = RBMs.∂free_energy(rbm, vm; wts = wm)
    return RBMs.subtract_gradients(∂d, ∂m)
end

function RBMs.update!(optimizer, rbm::WeightNormRBM, ∂::NamedTuple)
    RBMs.update!(optimizer, rbm.g, ∂.g)
    RBMs.update!(optimizer, rbm.u, ∂.u)
    RBMs.update!(optimizer, rbm.visible, ∂.visible)
    RBMs.update!(optimizer, rbm.hidden, ∂.hidden)
end
