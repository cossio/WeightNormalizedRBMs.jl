var documenterSearchIndex = {"docs":
[{"location":"literate/MNIST/","page":"MNIST","title":"MNIST","text":"EditURL = \"https://github.com/cossio/WeightNormalizedRBMs.jl/blob/master/docs/src/literate/MNIST.jl\"","category":"page"},{"location":"literate/MNIST/#Weight-normalization","page":"MNIST","title":"Weight normalization","text":"","category":"section"},{"location":"literate/MNIST/","page":"MNIST","title":"MNIST","text":"The authors of https://arxiv.org/abs/1602.07868 introduce weight normalization to boost learning. Let's try it here.","category":"page"},{"location":"literate/MNIST/","page":"MNIST","title":"MNIST","text":"Preliminaries:","category":"page"},{"location":"literate/MNIST/","page":"MNIST","title":"MNIST","text":"import CairoMakie\nimport MLDatasets\nimport RestrictedBoltzmannMachines as RBMs\nimport WeightNormalizedRBMs as wRBMs\nnothing #hide","category":"page"},{"location":"literate/MNIST/","page":"MNIST","title":"MNIST","text":"The following is a convenience function to plot grids of digits. Given a four dimensional tensor A of size (width, height, ncols, nrows) containing width x height images in a grid of nrows x ncols, this returns a matrix of size (width * ncols, height * nrows), that can be plotted in a heatmap to display all images.","category":"page"},{"location":"literate/MNIST/","page":"MNIST","title":"MNIST","text":"function imggrid(A::AbstractArray{<:Any,4})\n    reshape(permutedims(A, (1,3,2,4)), size(A,1)*size(A,3), size(A,2)*size(A,4))\nend","category":"page"},{"location":"literate/MNIST/","page":"MNIST","title":"MNIST","text":"Load data. Keep only 0, 1 digits for speed.","category":"page"},{"location":"literate/MNIST/","page":"MNIST","title":"MNIST","text":"Float = Float32\ntrain_x, train_y = MLDatasets.MNIST.traindata()\ntests_x, tests_y = MLDatasets.MNIST.testdata()\ntrain_x = Array{Float}(train_x[:, :, train_y .∈ Ref((0,1))] .> 0.5)\ntests_x = Array{Float}(tests_x[:, :, tests_y .∈ Ref((0,1))] .> 0.5)\nselected_digits = (0, 1)\ntrain_y = train_y[train_y .∈ Ref(selected_digits)]\ntests_y = tests_y[tests_y .∈ Ref(selected_digits)]\ntrain_nsamples = length(train_y)\ntests_nsamples = length(tests_y)\n(train_nsamples, tests_nsamples)","category":"page"},{"location":"literate/MNIST/","page":"MNIST","title":"MNIST","text":"Init RBM and train.","category":"page"},{"location":"literate/MNIST/","page":"MNIST","title":"MNIST","text":"rbm = RBMs.BinaryRBM(zeros(Float,28,28), zeros(Float,400), zeros(Float,28,28,400))\nRBMs.initialize!(rbm, train_x)\nwrbm = wRBMs.WeightNormRBM(rbm) # weight normalization reparameterization\n@time history = RBMs.pcd!(wrbm, train_x; epochs=500, batchsize=256, steps=5)\nrbm = RBMs.RBM(wrbm)\nnothing #hide","category":"page"},{"location":"literate/MNIST/","page":"MNIST","title":"MNIST","text":"Let's see what the learning curves look like.","category":"page"},{"location":"literate/MNIST/","page":"MNIST","title":"MNIST","text":"fig = CairoMakie.Figure(resolution=(800, 300))\nax = CairoMakie.Axis(fig[1,1], xlabel=\"train time\", ylabel=\"log(pseudolikelihood)\")\nCairoMakie.lines!(ax, get(history, :lpl)...)\nfig","category":"page"},{"location":"literate/MNIST/","page":"MNIST","title":"MNIST","text":"Let's look at some samples generated by this RBM.","category":"page"},{"location":"literate/MNIST/","page":"MNIST","title":"MNIST","text":"nrows, ncols = 10, 15\nfantasy_x = train_x[:, :, rand(1:train_nsamples, nrows * ncols)]\n@time fantasy_x .= RBMs.sample_v_from_v(rbm, fantasy_x; steps=10000)\nnothing #hide","category":"page"},{"location":"literate/MNIST/","page":"MNIST","title":"MNIST","text":"fig = CairoMakie.Figure(resolution=(40ncols, 40nrows))\nax = CairoMakie.Axis(fig[1,1], yreversed=true)\nCairoMakie.image!(ax, imggrid(reshape(fantasy_x, 28, 28, ncols, nrows)), colorrange=(1,0))\nCairoMakie.hidedecorations!(ax)\nCairoMakie.hidespines!(ax)\nfig","category":"page"},{"location":"literate/MNIST/","page":"MNIST","title":"MNIST","text":"","category":"page"},{"location":"literate/MNIST/","page":"MNIST","title":"MNIST","text":"This page was generated using Literate.jl.","category":"page"},{"location":"reference/#Reference","page":"Reference","title":"Reference","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"Modules = [WeightNormalizedRBMs]","category":"page"},{"location":"reference/#WeightNormalizedRBMs.gu2w-Tuple{AbstractArray, AbstractArray}","page":"Reference","title":"WeightNormalizedRBMs.gu2w","text":"gu2w(g, u) -> w, un\n\nReturns w, un (as a NamedTuple), from g, u, where:\n\nmathbfw = g fracmathbfumathbfu\n\nand un are the norms mathbfu.\n\n\n\n\n\n","category":"method"},{"location":"reference/#WeightNormalizedRBMs.w2gu-Tuple{AbstractArray, AbstractArray}","page":"Reference","title":"WeightNormalizedRBMs.w2gu","text":"w2gu(w, un) -> g, u\n\nReturns g, u (as a NamedTuple), such that\n\nmathbfw = g fracmathbfumathbfu\n\nwhere the norms mathbfu are given by un.\n\n\n\n\n\n","category":"method"},{"location":"reference/#WeightNormalizedRBMs.weight_norms-Tuple{RestrictedBoltzmannMachines.RBM}","page":"Reference","title":"WeightNormalizedRBMs.weight_norms","text":"weight_norms(rbm)\n\nNorms of weight patterns attached to each hidden unit.\n\n\n\n\n\n","category":"method"},{"location":"reference/#WeightNormalizedRBMs.∂wnorm-Tuple{AbstractArray, AbstractArray, AbstractArray}","page":"Reference","title":"WeightNormalizedRBMs.∂wnorm","text":"∂wnorm(∂w, g, u)\n\nGiven the gradients ∂w of a function f(w) with respect to w, returns the gradients ∂g, ∂u of f with respect to the re-parameterization:\n\nmathbfw = g fracmathbfumathbfu\n\n\n\n\n\n","category":"method"},{"location":"#WeightNormalizedRBMs.jl-Documentation","page":"Home","title":"WeightNormalizedRBMs.jl Documentation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"A Julia package to train and simulate Restricted Boltzmann Machines, with the weight normalization trick (see: http://papers.nips.cc/paper/6114-weight-normalization-a-simple-reparameterization-to-accelerate-training-of-deep-neural-networks.pdf).","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The package is not registered. Install with:","category":"page"},{"location":"","page":"Home","title":"Home","text":"using Pkg\nPkg.add(url=\"https://github.com/cossio/WeightNormalizedRBMs.jl\")","category":"page"},{"location":"","page":"Home","title":"Home","text":"This package doesn't export any symbols.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Most of the functions have a helpful docstring. See Reference section.","category":"page"},{"location":"#Related","page":"Home","title":"Related","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"See also https://github.com/cossio/RestrictedBoltzmannMachines.jl.","category":"page"}]
}