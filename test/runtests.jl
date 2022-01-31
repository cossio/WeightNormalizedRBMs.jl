#= As far as I know, Github Actions uses Intel CPUs.
So it is faster to use MKL than OpenBLAS.
It is recommended to load MKL before ANY other package.=#
import MKL
using SafeTestsets: @safetestset

@time @safetestset "wnorm" begin include("wnorm.jl") end
@time @safetestset "rbm" begin include("rbm.jl") end
