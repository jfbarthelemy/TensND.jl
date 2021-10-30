# TensND

*Package allowing tensor calculations in arbitrary coordinate systems.*

<!-- [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://jfbarthelemy.github.io/TensND.jl/stable) -->
<!-- [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jfbarthelemy.github.io/TensND.jl/dev) -->
<!-- [![Build Status](https://github.com/jfbarthelemy/TensND.jl/workflows/CI/badge.svg)](https://github.com/jfbarthelemy/TensND.jl/actions) -->

| **Documentation**                       | **Build Status**                  |
|:---------------------------------------:|:---------------------------------:|
| [![Dev][docs-dev-img]][docs-dev-url]    | [![Build Status][ci-img]][ci-url] |

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://jfbarthelemy.github.io/TensND.jl/dev

[ci-img]: https://github.com/jfbarthelemy/TensND.jl/workflows/CI/badge.svg?branch=main
[ci-url]: https://github.com/jfbarthelemy/TensND.jl/actions/workflows/CI.yml?query=branch%3Amain

[issues-url]: https://github.com/Ferrite-FEM/TensND.jl/issues

## Introduction

This Julia package provides tools to perform tensor calculations of any order and any dimension in arbitrary coordinate systems (cartesian, polar, cylindrical, spherical, spheroidal or any user defined coordinate systems...). In particular differential operators are available: gradient, symmetrized gradient, divergence, Laplace, Hessian. The implementation of this library is much inspired by the Maple library [Tens3d](http://jean.garrigues.perso.centrale-marseille.fr/tens3d.html) developped by Jean Guarrigues.

## Installation

The package can be installed with the Julia package manager. From the Julia REPL, type `]` to enter the Pkg REPL mode and run:

```julia
pkg> add https://github.com/jfbarthelemy/TensND.jl.git
```

Or, equivalently, via the `Pkg` API:

```julia
julia> import Pkg; Pkg.add("https://github.com/jfbarthelemy/TensND.jl.git")
```

## Documentation

<!-- - [**STABLE**][docs-stable-url] &mdash; **most recently tagged version of the documentation.** -->
- [**DEV**][docs-dev-url] &mdash; **development version of the documentation.**

## Citing TensND.jl

```latex
@misc{TensND.jl,
  author  = {Jean-François Barthélémy},
  title   = {TensND.jl},
  url     = {https://github.com/jfbarthelemy/TensND.jl},
  version = {v0.1.0},
  year    = {2021},
  month   = {8}
}
```

## Related packages

- [SymPy.jl](https://github.com/JuliaPy/SymPy.jl)
- [Tensors.jl](https://github.com/Ferrite-FEM/Tensors.jl)
- [OMEinsum.jl](https://github.com/under-Peter/OMEinsum.jl)
- [Rotations.jl](https://github.com/JuliaGeometry/Rotations.jl)

## References

1. [Tens3d](http://jean.garrigues.perso.centrale-marseille.fr/tens3d.html)
1. [Sébastien Brisard's blog](https://sbrisard.github.io/posts/20140226-decomposition_of_transverse_isotropic_fourth-rank_tensors.html)