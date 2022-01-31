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
    @test ∂.visible.θ ≈ gs.visible.θ
    @test ∂.hidden.θ ≈ gs.hidden.θ
end
