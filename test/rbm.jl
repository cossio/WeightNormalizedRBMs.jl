using Test: @test, @testset
import Random
import Statistics
import Zygote
import RestrictedBoltzmannMachines as RBMs
import WeightNormalizedRBMs as wRBMs

@testset "weight_norms" begin
    rbm = RBMs.BinaryRBM(randn(3,2,4), randn(2,5), randn(3,2,4,2,5))
    wn = wRBMs.weight_norms(rbm)
    @test size(wn) == (1,1,1,2,5)
    @test wn ≈ sqrt.(sum(abs2.(rbm.w); dims=1:3))
end

@testset "WeightNormRBM" begin
    rbm = RBMs.BinaryRBM(randn(3,2,4), randn(2,5), randn(3,2,4,2,5))
    un = rand(1,1,1,2,5)
    wrbm = wRBMs.WeightNormRBM(rbm, un)
    rbm1 = RBMs.RBM(wrbm)
    @test rbm1.w ≈ rbm.w
    @test rbm1.visible.θ ≈ rbm.visible.θ
    @test rbm1.hidden.θ ≈ rbm.hidden.θ
end

@testset "∂free_energy" begin
    rbm = RBMs.BinaryRBM(randn(3,2,4), randn(2,5), randn(3,2,4,2,5))
    wrbm = wRBMs.WeightNormRBM(rbm)
    v = Random.bitrand(3,2,4,5,2)
    ∂ = RBMs.∂free_energy(wrbm, v)
    gs, = Zygote.gradient(wrbm) do wrbm
        w = wrbm.g .* wrbm.u ./ sqrt.(sum(abs2, wrbm.u; dims=1:ndims(rbm.visible)))
        return Statistics.mean(RBMs.free_energy(RBMs.RBM(wrbm.visible, wrbm.hidden, w), v))
    end
    @test ∂.g ≈ gs.g
    @test ∂.u ≈ gs.u
    @test ∂.visible ≈ gs.visible.par
    @test ∂.hidden ≈ gs.hidden.par
end

@testset "pcd!" begin
    Random.seed!(3)
    rbm = RBMs.BinaryRBM(randn(7), randn(2), randn(7,2) / √7)
    wrbm = wRBMs.WeightNormRBM(rbm)
    data = Random.bitrand(7, 128)
    iters_seen = Int[]
    state, ps = RBMs.pcd!(
        wrbm, data;
        iters = 20, batchsize = 16, steps = 2,
        callback = (; iter, _...) -> push!(iters_seen, iter)
    )
    @test iters_seen == 1:20
    @test all(isfinite, wrbm.g)
    @test all(isfinite, wrbm.u)
    @test all(isfinite, wrbm.visible.par)
    @test all(isfinite, wrbm.hidden.par)
    # parameters actually moved
    @test !(wrbm.u ≈ rbm.w)
    # the trained model produces finite free energies
    @test all(isfinite, RBMs.free_energy(RBMs.RBM(wrbm), data))
end
