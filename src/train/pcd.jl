function RBMs.pcd!(rbm::WeightNormRBM, data::AbstractArray;
    batchsize::Int = 1,
    epochs::Int = 1,
    optimizer = Flux.ADAM(),
    history::ValueHistories.MVHistory = ValueHistories.MVHistory(),
    wts = nothing,
    steps::Int = 1,
    vm::AbstractArray = RBMs.transfer_sample(rbm.visible, falses(size(rbm.visible)..., batchsize))
)
    @assert size(data) == (size(rbm.visible)..., size(data)[end])
    @assert isnothing(wts) || _nobs(data) == _nobs(wts)

    stats = RBMs.sufficient_statistics(rbm.visible, data; wts)

    for epoch in 1:epochs
        batches = RBMs.minibatches(data, wts; batchsize = batchsize)
        Δt = @elapsed for (vd, wd) in batches
            vm .= RBMs.sample_v_from_v(RBM(rbm), vm; steps = steps)
            ∂ = RBMs.∂contrastive_divergence(rbm, vd, vm; wd, stats)
            RBMs.update!(optimizer, rbm, ∂)
            push!(history, :∂, RBMs.gradnorms(∂))
        end

        lpl = RBMs.wmean(RBMs.log_pseudolikelihood(RBM(rbm), data); wts)
        push!(history, :lpl, lpl)
        push!(history, :epoch, epoch)
        push!(history, :Δt, Δt)
        push!(history, :vm, copy(vm))

        Δt_ = round(Δt, digits=2)
        lpl_ = round(lpl, digits=2)
        @debug "epoch $epoch/$epochs ($(Δt_)s), log(PL)=$lpl_"
    end

    return history
end
