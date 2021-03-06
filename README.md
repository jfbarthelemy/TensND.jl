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

This Julia package provides tools to perform tensor calculations of any order and any dimension in arbitrary coordinate systems (cartesian, polar, cylindrical, spherical, spheroidal or any user defined coordinate systems...). In particular differential operators are available: gradient, symmetrized gradient, divergence, Laplace, Hessian. The implementation of this library is much inspired by the Maple library [Tens3d](http://jean.garrigues.perso.centrale-marseille.fr/tens3d.html) developed by Jean Garrigues.

The following example is provided to illustrate the purpose of the library

```julia
julia> using SymPy, TensND

julia> Spherical = coorsys_spherical() ; ฮธ, ฯ, r = getcoords(Spherical) ; ๐แถฟ, ๐แต , ๐สณ = unitvec(Spherical) ;

julia> @set_coorsys Spherical

julia> GRAD(๐สณ)
(1/r)๐แถฟโ๐แถฟ + (1/r)๐แต โ๐แต 

julia> DIV(๐สณ โ ๐สณ)
(2/r)๐สณ

julia> LAPLACE(1/r)
0

julia> f = SymFunction("f", real = true)
f

julia> DIV(f(r) * ๐สณ โ ๐สณ)
(Derivative(f(r), r) + 2*f(r)/r)๐สณ

julia> LAPLACE(f(r))
              d       
  2         2โโโ(f(r))
 d            dr
โโโ(f(r)) + โโโโโโโโโโ
  2             r
dr

julia> for ฯโฑสฒ โ ("ฯสณสณ", "ฯแถฟแถฟ", "ฯแต แต ") @eval $(Symbol(ฯโฑสฒ)) = SymFunction($ฯโฑสฒ, real = true)($r) end

julia> ๐ = ฯสณสณ * ๐สณ โ ๐สณ + ฯแถฟแถฟ * ๐แถฟ โ ๐แถฟ + ฯแต แต  * ๐แต  โ ๐แต 
(ฯแถฟแถฟ(r))๐แถฟโ๐แถฟ + (ฯแต แต (r))๐แต โ๐แต  + (ฯสณสณ(r))๐สณโ๐สณ

julia> div๐ = simplify(DIV(๐))
((-ฯแต แต (r) + ฯแถฟแถฟ(r))/(r*tan(ฮธ)))๐แถฟ + ((r*Derivative(ฯสณสณ(r), r) + 2*ฯสณสณ(r) - ฯแต แต (r) - ฯแถฟแถฟ(r))/r)๐สณ
```

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
  author  = {Jean-Franรงois Barthรฉlรฉmy},
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
1. [Sรฉbastien Brisard's blog](https://sbrisard.github.io/posts/20140226-decomposition_of_transverse_isotropic_fourth-rank_tensors.html)