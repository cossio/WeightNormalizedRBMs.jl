function RBMs.cd!(rbm::WeightNormRBM, data::AbstractArray;
    batchsize = 1,
    epochs = 1,
    optimizer = Flux.ADAM(), # optimizer algorithm
    history::ValueHistories.MVHistory = ValueHistories.MVHistory(), # stores training log
    wts = nothing, # data point weights
    steps::Int = 1, # Monte Carlo steps to update fantasy particles
)
    @assert size(data) == (size(rbm.visible)..., size(data)[end])
    @assert isnothing(wts) || _nobs(data) == _nobs(wts)

    stats = RBMs.sufficient_statistics(rbm.visible, data; wts)

    for epoch in 1:epochs
        batches = RBMs.minibatches(data, wts; batchsize = batchsize)
        Δt = @elapsed for (vd, wd) in batches
            vm = RBMs.sample_v_from_v(RBM(rbm), vd; steps = steps)
            ∂ = RBMs.∂contrastive_divergence(rbm, vd, vm; wd = wd, wm = wd, stats)
            RBMs.update!(optimizer, rbm, ∂)
            push!(history, :∂, RBMs.gradnorms(∂))
        end

        lpl = RBMs.wmean(RBMs.log_pseudolikelihood(RBM(rbm), data); wts)
        push!(history, :lpl, lpl)
        push!(history, :epoch, epoch)
        push!(history, :Δt, Δt)

        Δt_ = round(Δt, digits=2)
        lpl_ = round(lpl, digits=2)
        @debug "epoch $epoch/$epochs ($(Δt_)s), log(PL)=$lpl_"
    end
    return history
end

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
