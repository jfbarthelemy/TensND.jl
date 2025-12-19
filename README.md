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

[issues-url]: https://github.com/jfbarthelemy/TensND.jl/issues

[![DOI](https://zenodo.org/badge/394914455.svg)](https://doi.org/10.5281/zenodo.17985768)

## Introduction

This Julia package provides tools to perform tensor calculations of any order and any dimension in arbitrary coordinate systems (cartesian, polar, cylindrical, spherical, spheroidal or any user defined coordinate systems...). In particular differential operators are available: gradient, symmetrized gradient, divergence, Laplace, Hessian. The implementation of this library is much inspired by the Maple library [Tens3d](http://jean.garrigues.perso.centrale-marseille.fr/tens3d.html) developed by Jean Garrigues.

This package and its manual are still under construction. The API may vary before official release.

The following example is provided to illustrate the purpose of the library

```julia
julia> using SymPy, TensND

julia> Spherical = coorsys_spherical() ; θ, ϕ, r = getcoords(Spherical) ; 𝐞ᶿ, 𝐞ᵠ, 𝐞ʳ = unitvec(Spherical) ;

julia> @set_coorsys Spherical

julia> GRAD(𝐞ʳ) |> intrinsic
(1/r)𝐞ᶿ⊗𝐞ᶿ + (1/r)𝐞ᵠ⊗𝐞ᵠ

julia> DIV(𝐞ʳ ⊗ 𝐞ʳ) |> intrinsic
(2/r)𝐞ʳ

julia> LAPLACE(1/r) |> intrinsic
0

julia> f = SymFunction("f", real = true)
f

julia> DIV(f(r) * 𝐞ʳ ⊗ 𝐞ʳ) |> intrinsic
(Derivative(f(r), r) + 2*f(r)/r)𝐞ʳ

julia> LAPLACE(f(r)) |> intrinsic
              d       
  2         2⋅──(f(r))
 d            dr
───(f(r)) + ──────────
  2             r
dr

julia> for σⁱʲ ∈ ("σʳʳ", "σᶿᶿ", "σᵠᵠ") @eval $(Symbol(σⁱʲ)) = SymFunction($σⁱʲ, real = true)($r) end

julia> 𝛔 = σʳʳ * 𝐞ʳ ⊗ 𝐞ʳ + σᶿᶿ * 𝐞ᶿ ⊗ 𝐞ᶿ + σᵠᵠ * 𝐞ᵠ ⊗ 𝐞ᵠ ; intrinsic(𝛔)
(σᶿᶿ(r))𝐞ᶿ⊗𝐞ᶿ + (σᵠᵠ(r))𝐞ᵠ⊗𝐞ᵠ + (σʳʳ(r))𝐞ʳ⊗𝐞ʳ

julia> div𝛔 = simplify(DIV(𝛔)) ; intrinsic(div𝛔)
((-σᵠᵠ(r) + σᶿᶿ(r))/(r*tan(θ)))𝐞ᶿ + ((r*Derivative(σʳʳ(r), r) + 2*σʳʳ(r) - σᵠᵠ(r) - σᶿᶿ(r))/r)𝐞ʳ
```

## Installation

The package can be installed with the Julia package manager. From the Julia REPL, type `]` to enter the Pkg REPL mode and run:

```julia
pkg> add TensND
```

Or, equivalently, via the `Pkg` API:

```julia
julia> import Pkg; Pkg.add("TensND")
```

## Documentation

<!-- - [**STABLE**][docs-stable-url] &mdash; **most recently tagged version of the documentation.** -->
- [**DEV**][docs-dev-url] &mdash; **development version of the documentation.**

## Citation

[![DOI](https://zenodo.org/badge/394914455.svg)](https://doi.org/10.5281/zenodo.17985768)

See [CITATION.cff](CITATION.cff) for citation details.

## Related packages

- [SymPy.jl](https://github.com/JuliaPy/SymPy.jl)
- [Tensors.jl](https://github.com/Ferrite-FEM/Tensors.jl)
- [OMEinsum.jl](https://github.com/under-Peter/OMEinsum.jl)
- [Rotations.jl](https://github.com/JuliaGeometry/Rotations.jl)

## References

1. [Tens3d](http://jean.garrigues.perso.centrale-marseille.fr/tens3d.html)
1. [Sébastien Brisard's blog](https://sbrisard.github.io/posts/20140226-decomposition_of_transverse_isotropic_fourth-rank_tensors.html)
