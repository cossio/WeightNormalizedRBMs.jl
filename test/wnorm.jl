using Test: @test, @testset
import Zygote
import WeightNormalizedRBMs as wRBMs

@testset "gu2w, w2gu" begin
    g = abs.(randn(1,1,4,2))
    u = randn(5,3,4,2)
    nt = wRBMs.gu2w(g, u)
    @test nt.un ≈ sqrt.(sum!(similar(g), abs2.(u)))
    @test nt.w ≈ g .* u ./ nt.un
    @test sqrt.(sum!(similar(g), abs2.(nt.w))) ≈ g

    nt1 = wRBMs.w2gu(nt.w, nt.un)
    @test nt1.g ≈ g
    @test nt1.u ≈ u
end

@testset "∂wnorm" begin
    foo(w) = sum(sin.(w))
    function foo(g, u)
        un = sqrt.(sum(abs2.(u); dims=findall(isone, size(g))))
        w = g .* u ./ un
        return foo(w)
    end
    g = abs.(randn(1,1,4,2))
    u = randn(5,3,4,2)
    w = wRBMs.gu2w(g, u).w
    ∂w, = Zygote.gradient(foo, w)
    ∂g, ∂u = Zygote.gradient(foo, g, u)
    ∂ = wRBMs.∂wnorm(∂w, g, u)
    @test ∂.g ≈ ∂g
    @test ∂.u ≈ ∂u
end
