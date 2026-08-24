module WeightNormalizedRBMs
    import RestrictedBoltzmannMachines as RBMs
    using Optimisers: AbstractRule, Adam, setup, update!
    using RestrictedBoltzmannMachines: RBM, AbstractLayer

    include("rbm.jl")
    include("wnorm.jl")
    include("train/pcd.jl")
end
