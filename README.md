# WeightNormalizedRBMs Julia package

[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/cossio/WeightNormalizedRBMs.jl/blob/master/LICENSE.md)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://cossio.github.io/WeightNormalizedRBMs.jl/dev)
![](https://github.com/cossio/WeightNormalizedRBMs.jl/workflows/CI/badge.svg)
[![codecov](https://codecov.io/gh/cossio/WeightNormalizedRBMs.jl/branch/master/graph/badge.svg?token=JylwfiCsUJ)](https://codecov.io/gh/cossio/WeightNormalizedRBMs.jl)
![GitHub repo size](https://img.shields.io/github/repo-size/cossio/WeightNormalizedRBMs.jl)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/cossio/WeightNormalizedRBMs.jl)

Train and sample [Restricted Boltzmann machines](https://en.wikipedia.org/wiki/Restricted_Boltzmann_machine) in Julia, with the weight normalization trick.
See Salimans & Kingma 2016,
<https://proceedings.neurips.cc/paper/2016/hash/ed265bc903a5a097f61d3ec064d96d2e-Abstract.html>.

## Installation

This package is not registered.
Install with:

```julia
using Pkg
Pkg.add(url="https://github.com/cossio/WeightNormalizedRBMs.jl")
```

This package does not export any symbols.

## Related

See <https://github.com/cossio/RestrictedBoltzmannMachines.jl>.