"""
    pcd!(wrbm::WeightNormRBM, data; kwargs...)

Train a weight-normalized RBM with Persistent Contrastive Divergence (PCD),
following the same conventions as `RestrictedBoltzmannMachines.pcd!`, but
optimizing the weight norms `g` and directions `u` instead of the weights `w`.

Returns `(state, ps)`, the optimizer state and the optimized parameters.
"""
function RBMs.pcd!(
    rbm::WeightNormRBM, data::AbstractArray;
    batchsize::Int = 1,
    iters::Int = 1, # number of gradient updates
    wts::AbstractVector{<:Real} = RBMs.uniform_wts(rbm.visible, data), # data weights
    steps::Int = 1, # MC steps to update fantasy chains
    optim::AbstractRule = Adam(), # optimizer rule
    moments = RBMs.moments_from_samples(rbm.visible, data; wts), # sufficient statistics for visible layer
    callback = Returns(nothing), # called for every batch
    # init fantasy chains
    vm::AbstractArray = RBMs.sample_from_inputs(
        rbm.visible, falses(size(rbm.visible)..., min(batchsize, size(data)[end]))
    ),
    shuffle::Bool = true,
    # parameters to optimize
    ps = (; visible = rbm.visible.par, hidden = rbm.hidden.par, g = rbm.g, u = rbm.u),
    state = setup(optim, ps),
)
    @assert size(data) == (size(rbm.visible)..., size(data)[end])
    batchsize > 0 || throw(ArgumentError("batchsize must be positive"))
    size(data, ndims(data)) > 0 ||
        throw(ArgumentError("data must contain at least one sample"))
    length(wts) == size(data, ndims(data)) ||
        throw(DimensionMismatch("length(wts) must equal the number of data samples"))
    RBMs.validate_wts(wts)
    wts_mean = sum(wts) / length(wts)
    batchsize = min(batchsize, length(wts))

    for (iter, (vd, wd)) in zip(1:iters, RBMs.infinite_minibatches(data, wts; batchsize, shuffle))
        # positive phase
        ∂d = RBMs.∂free_energy(rbm, vd; wts = wd, moments)

        # negative phase: update persistent fantasy chains
        vm .= RBMs.sample_v_from_v(RBM(rbm), vm; steps)
        ∂m = RBMs.∂free_energy(rbm, vm)

        # weighted minibatch bias correction, in the gradient eltype
        batch_weight = convert(float(real(eltype(∂d.u))), (sum(wd) / length(wd)) / wts_mean)
        gs = map((d, m) -> (d - m) * batch_weight, ∂d, ∂m)

        # feed gradient to Optimiser rule
        state, ps = update!(state, ps, gs)

        callback(; rbm, optim, state, ps, iter, vd, wd, ∂ = gs, vm)
    end
    return state, ps
end
