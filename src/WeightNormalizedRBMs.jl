module WeightNormalizedRBMs
    import Flux
    import ValueHistories
    import RestrictedBoltzmannMachines as RBMs
    using RestrictedBoltzmannMachines: RBM, AbstractLayer

    include("rbm.jl")
    include("wnorm.jl")
    include("train/cd.jl")
    include("train/pcd.jl")
end
