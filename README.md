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

This Julia package provides tools to perform tensor calculations of any order and any dimension in arbitrary coordinate systems (cartesian,
polar, cymindrical, spherical, spheroidal or any user defined coordinate systems...). In particular differential operators are available:
gradient, symmetrized gradient, divergence, Laplace, Hessian.

## Installation

The package can be installed with the Julia package manager. From the Julia REPL, type `]` to enter the Pkg REPL mode and run:

```
pkg> add https://github.com/jfbarthelemy/TensND.jl.git
```

Or, equivalently, via the `Pkg` API:

```julia
julia> import Pkg; Pkg.add("https://github.com/jfbarthelemy/TensND.jl.git")
```

## Brief description of the package

The package relies on the definition of

- **bases** which can be of the following types (`T` denotes the scalar type, subtype of `Number`)
  - `CanonicalBasis{dim,T}`: fundamental canonical basis in `ℝᵈⁱᵐ` in which the metric tensor is the second-order identity
  - `RotatedBasis{dim,T}`: orthonormal basis in `ℝᵈⁱᵐ` obtained by rotation of the canonical basis by means of one angle if `dim=2` or three Euler angles if `dim=3`, the metric tensor is again the second-order identity
  - `OrthogonalBasis{dim,T}`: orthogonal basis in `ℝᵈⁱᵐ` obtained from a given orthonormal rotated basis by applying a scaling factor along each unit vector, the metric tensor is then diagonal
  - `Basis{dim,T}`: arbitrary basis not entering the previous cases

- **tensors**
  - a tensor is determined by a set of data (array or synthetic parameters) corresponding to its `order`, a basis and a tuple of variances
  - depending on the type of basis, the type of tensor can be `TensCanonical{order,dim,T,A}`, `TensRotated{order,dim,T,A}`, `TensOrthogonal{order,dim,T,A}` or `Tens{order,dim,T,A}` if the data are stored under the form of an array or a `Tensor` object (see [Tensors.jl](https://github.com/Ferrite-FEM/Tensors.jl)), or the type can be `TensISO{order,dim,T,N}` if the tensor is isotropic and data are stored under the form of a set of parameters (1 for order 2 and 2 for order 4).
  
  !!! note
      More material symmetry types such as transverse isotropy or orthotropy will be added in the future.

- **coordinate systems**
  - a coordinate system contains all information required to perform differential operations on tensor fields: position vector `OM` expressed in the canonical basis, coordinate names `xⁱ`, natural basis `aᵢ=∂ᵢOM`, normalized basis `eᵢ=aᵢ/||aᵢ||`, Christoffel coefficients `Γᵢⱼᵏ=∂ᵢaⱼ⋅aᵏ` where `aⁱ` form the dual basis associated to the natural one
  - predefined coordinate systems are available: cartesian, polar, cylindrical, spherical and spheroidal but the user can define new systems

  !!! note
      Note that for the moment the coordinate systems and differential operators have been implemented only for symbolic calculations (using [SymPy.jl](https://github.com/JuliaPy/SymPy.jl)). Numerical coordinate systems and differential operators based on automatic differentiation will be implemented in the future.

## Examples

### Differential operators in polar coordinates
```julia
julia> using TensND, SymPy

julia> Polar = CS_polar() ; r, θ = getcoords(Polar) ; 𝐞ʳ, 𝐞ᶿ = unitvec(Polar) ;

julia> LAPLACE(SymFunction("f", real = true)(r, θ), Polar)
                               2
                              ∂
               ∂             ───(f(r, θ))
  2            ──(f(r, θ))     2
 ∂             ∂r            ∂θ
───(f(r, θ)) + ─────────── + ────────────
  2                 r              2
∂r                                r

julia> n = symbols("n", integer = true)
n

julia> simplify(HESS(r^n,Polar))
TensND.TensRotated{2, 2, Sym, Tensors.Tensor{2, 2, Sym, 4}}
→ data: 2×2 Tensors.Tensor{2, 2, Sym, 4}:
 n*r^(n - 2)*(n - 1)            0
                   0  n*r^(n - 2)
→ basis: 2×2 Matrix{Sym}:
 cos(θ)  -sin(θ)
 sin(θ)   cos(θ)
→ var: (:cont, :cont)
```