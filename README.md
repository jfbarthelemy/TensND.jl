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

## Brief description of the package

The package relies on the definition of

- **bases** which can be of the following types (`T` denotes the scalar type, subtype of `Number`)

  - `CanonicalBasis{dim,T}`: fundamental canonical basis in `ℝᵈⁱᵐ` in which the metric tensor is the second-order identity
  - `RotatedBasis{dim,T}`: orthonormal basis in `ℝᵈⁱᵐ` obtained by rotation of the canonical basis by means of one angle if `dim=2` or three Euler angles if `dim=3`, the metric tensor is again the second-order identity
  - `OrthogonalBasis{dim,T}`: orthogonal basis in `ℝᵈⁱᵐ` obtained from a given orthonormal rotated basis by applying a scaling factor along each unit vector, the metric tensor is then diagonal
  - `Basis{dim,T}`: arbitrary basis not entering the previous cases

- **tensors**

  - a tensor is determined by a set of data (array or synthetic parameters) corresponding to its `order`, a basis and a tuple of variances
  - depending on the type of basis, the type of tensor can be `TensCanonical{order,dim,T,A}`, `TensRotated{order,dim,T,A}`, `TensOrthogonal{order,dim,T,A}` or `Tens{order,dim,T,A}` if the data are stored under the form of an array or a `Tensor` object (see [Tensors.jl](https://github.com/Ferrite-FEM/Tensors.jl)), or the type can be `TensISO{order,dim,T,N}` if the tensor is isotropic and data are stored under the form of a set of parameters (one for order 2 and two for order 4).
  
  **_NOTE:_**
      More material symmetry types such as transverse isotropy or orthotropy will be added in the future.

- **coordinate systems**

  - a coordinate system contains all information required to perform differential operations on tensor fields: position vector `OM` expressed in the canonical basis, coordinate names `xⁱ`, natural basis `aᵢ=∂ᵢOM`, normalized basis `eᵢ=aᵢ/||aᵢ||`, Christoffel coefficients `Γᵢⱼᵏ=∂ᵢaⱼ⋅aᵏ` where `(aⁱ)(1≤i≤dim)` form the dual basis associated to the natural one
  - predefined coordinate systems are available: cartesian, polar, cylindrical, spherical and spheroidal but the user can define new systems

  **_NOTE:_**
      Note that for the moment the coordinate systems and differential operators have been implemented only for symbolic calculations (using [SymPy.jl](https://github.com/JuliaPy/SymPy.jl)). Numerical coordinate systems and differential operators based on automatic differentiation will be implemented in the future.

## Examples

Before detailing examples, it is worth recalling that the use of the libraries TensND and SymPy requires to start scripts by

```julia
julia> using TensND, SymPy
```

### Definition of bases, tensors and tensor products

An arbitrary basis contains four matrices

- one in which columns correspond to the covariant vectors of new basis with respect to the canonical one `𝐞ᵢ`,
- one defining the contravariant (or dual) basis `𝐞ⁱ`,
- one defining the metric tensor `gᵢⱼ=𝐞ᵢ⋅𝐞ⱼ`,
- one defining the inverse of the  metric tensor `gⁱʲ=𝐞ⁱ⋅𝐞ʲ`.

and is built by one the following constructors

- `Basis(eᵢ::AbstractMatrix{T},eⁱ::AbstractMatrix{T},gᵢⱼ::AbstractMatrix{T},gⁱʲ::AbstractMatrix{T}) where {T}`
- `Basis(ℬ::AbstractBasis{dim,T}, χᵢ::V) where {dim,T,V}` where `χᵢ` is a list of scaling factors applied on the vectors of the basis `ℬ`
- `Basis(v::AbstractMatrix{T}, var::Symbol)`
- `Basis(θ::T1, ϕ::T2, ψ::T3 = 0) where {T1,T2,T3}`
- `Basis(θ::T) where {T}`
- `Basis{dim,T}() where {dim,T}`

Depending on the property of the basis (canonical, orthonormal, orthogonal...), the most relevant type (`CanonicalBasis`, `RotatedBasis`, `OrthogonalBasis` or `Basis`) is implicitly created by calling `Basis`.

```julia
julia> ℬ = Basis(Sym[1 0 0; 0 1 0; 0 1 1])
Basis{3, Sym}
→ basis: 3×3 Matrix{Sym}:
 1  0  0
 0  1  0
 0  1  1      
→ dual basis: 3×3 Matrix{Sym}:
 1  0   0
 0  1  -1
 0  0   1
→ covariant metric tensor: 3×3 Symmetric{Sym, Matrix{Sym}}:
 1  0  0
 0  2  1
 0  1  1
→ contravariant metric tensor: 3×3 Symmetric{Sym, Matrix{Sym}}:
 1   0   0
 0   1  -1
 0  -1   2

julia> ℬ₂ = Basis(symbols("θ, ϕ, ψ", real = true)...)
RotatedBasis{3, Sym}
→ basis: 3×3 Matrix{Sym}:
 -sin(ψ)⋅sin(ϕ) + cos(θ)⋅cos(ψ)⋅cos(ϕ)  -sin(ψ)⋅cos(θ)⋅cos(ϕ) - sin(ϕ)⋅cos(ψ)  sin(θ)⋅cos(ϕ)
  sin(ψ)⋅cos(ϕ) + sin(ϕ)⋅cos(θ)⋅cos(ψ)  -sin(ψ)⋅sin(ϕ)⋅cos(θ) + cos(ψ)⋅cos(ϕ)  sin(θ)⋅sin(ϕ)
                        -sin(θ)⋅cos(ψ)                          sin(θ)⋅sin(ψ)         cos(θ)
→ dual basis: 3×3 Matrix{Sym}:
 -sin(ψ)⋅sin(ϕ) + cos(θ)⋅cos(ψ)⋅cos(ϕ)  -sin(ψ)⋅cos(θ)⋅cos(ϕ) - sin(ϕ)⋅cos(ψ)  sin(θ)⋅cos(ϕ)
  sin(ψ)⋅cos(ϕ) + sin(ϕ)⋅cos(θ)⋅cos(ψ)  -sin(ψ)⋅sin(ϕ)⋅cos(θ) + cos(ψ)⋅cos(ϕ)  sin(θ)⋅sin(ϕ)
                        -sin(θ)⋅cos(ψ)                          sin(θ)⋅sin(ψ)         cos(θ)
→ covariant metric tensor: 3×3 TensND.Id2{3, Sym}:
 1  ⋅  ⋅
 ⋅  1  ⋅
 ⋅  ⋅  1
→ contravariant metric tensor: 3×3 TensND.Id2{3, Sym}:
 1  ⋅  ⋅
 ⋅  1  ⋅
 ⋅  ⋅  1
```

Predefined symbolic or numerical coordinates and basis vectors can be obtained from

- `init_cartesian(dim::Integer)`
- `init_polar(coords = (symbols("r", positive = true), symbols("θ", real = true)); canonical = false)`
- `init_cylindrical(coords = (symbols("r", positive = true), symbols("θ", real = true), symbols("z", real = true)); canonical = false)`
- `init_spherical(coords = (symbols("θ", real = true), symbols("ϕ", real = true), symbols("r", positive = true)); canonical = false)`
- `init_rotated(coords = symbols("θ ϕ ψ", real = true); canonical = false)`

The option `canonical` specifies whether the vector is expressed as a tensor with components in the canonical basis or directly in the rotated basis. The second option (ie `canonical = false` by default) is often preferable for further calculations in the rotated basis.

```julia
julia> (x, y, z), (𝐞₁, 𝐞₂, 𝐞₃), ℬ = init_cartesian() ;

julia> (r, θ), (𝐞ʳ, 𝐞ᶿ), ℬᵖ = init_polar() ;

julia> (r, θ, z), (𝐞ʳ, 𝐞ᶿ, 𝐞ᶻ), ℬᶜ = init_cylindrical() ;

julia> display(ℬᶜ)
RotatedBasis{3, Sym}
→ basis: 3×3 Matrix{Sym}:
 cos(θ)  -sin(θ)  0
 sin(θ)   cos(θ)  0
      0        0  1
→ dual basis: 3×3 Matrix{Sym}:
 cos(θ)  -sin(θ)  0
 sin(θ)   cos(θ)  0
      0        0  1
→ covariant metric tensor: 3×3 TensND.Id2{3, Sym}:
 1  ⋅  ⋅
 ⋅  1  ⋅
 ⋅  ⋅  1
→ contravariant metric tensor: 3×3 TensND.Id2{3, Sym}:
 1  ⋅  ⋅
 ⋅  1  ⋅
 ⋅  ⋅  1

julia> (θ, ϕ, r), (𝐞ᶿ, 𝐞ᵠ, 𝐞ʳ), ℬˢ = init_spherical() ;

julia> display(𝐞ʳ)
TensND.TensRotated{1, 3, Sym, Vec{3, Sym}}
→ data: 3-element Vec{3, Sym}:
 0
 0
 1
→ basis: 3×3 Matrix{Sym}:
 cos(θ)⋅cos(ϕ)  -sin(ϕ)  sin(θ)⋅cos(ϕ)
 sin(ϕ)⋅cos(θ)   cos(ϕ)  sin(θ)⋅sin(ϕ)
       -sin(θ)        0         cos(θ)
→ var: (:cont,)

julia> (θ, ϕ, ψ), (𝐞ᶿ, 𝐞ᵠ, 𝐞ʳ), ℬʳ = init_rotated() ;
```

**_NOTE:_**
it is worth noting the unusual order of coordinates and vectors of the spherical basis which have been chosen here so that `θ = ϕ = 0` corresponds to the cartesian basis in the correct order.

A tensor, parametrized by an order and a dimension, is in general defined by

- an array or a set of condensed parameters (e.g. isotropic tensors),
- a basis,
- a set of variances (covariant `:cov` or contravariant `:cont`) useful if the basis is not orthonormal.

In practice, the type of basis conditions the type of tensor (`TensCanonical`, `TensRotated`, `TensOrthogonal`, `Tens` or even `TensISO` in case of isotropic tensor).

```julia
julia> ℬ = Basis(Sym[0 1 1; 1 0 1; 1 1 0])
Basis{3, Sym}
→ basis: 3×3 Matrix{Sym}:
 0  1  1
 1  0  1
 1  1  0
→ dual basis: 3×3 Matrix{Sym}:
 -1/2   1/2   1/2
  1/2  -1/2   1/2
  1/2   1/2  -1/2
→ covariant metric tensor: 3×3 Symmetric{Sym, Matrix{Sym}}:
 2  1  1
 1  2  1
 1  1  2
→ contravariant metric tensor: 3×3 Symmetric{Sym, Matrix{Sym}}:
  3/4  -1/4  -1/4
 -1/4   3/4  -1/4
 -1/4  -1/4   3/4

julia> V = Tens(Tensor{1,3}(i -> symbols("v$i", real = true)))
TensND.TensCanonical{1, 3, Sym, Vec{3, Sym}}
→ data: 3-element Vec{3, Sym}:
 v₁
 v₂
 v₃
→ basis: 3×3 TensND.Id2{3, Sym}:
 1  ⋅  ⋅
 ⋅  1  ⋅
 ⋅  ⋅  1
→ var: (:cont,)

julia> components(V, ℬ, (:cont,))
3-element Vector{Sym}:
 -v1/2 + v2/2 + v3/2
  v1/2 - v2/2 + v3/2
  v1/2 + v2/2 - v3/2

julia> components(V, ℬ, (:cov,))
3-element Vector{Sym}:
 v₂ + v₃
 v₁ + v₃
 v₁ + v₂

julia> ℬ̄ = normalize(ℬ)
Basis{3, Sym}
→ basis: 3×3 Matrix{Sym}:
         0  sqrt(2)/2  sqrt(2)/2
 sqrt(2)/2          0  sqrt(2)/2
 sqrt(2)/2  sqrt(2)/2          0
→ dual basis: 3×3 Matrix{Sym}:
 -sqrt(2)/2   sqrt(2)/2   sqrt(2)/2
  sqrt(2)/2  -sqrt(2)/2   sqrt(2)/2
  sqrt(2)/2   sqrt(2)/2  -sqrt(2)/2
→ covariant metric tensor: 3×3 Symmetric{Sym, Matrix{Sym}}:
   1  1/2  1/2
 1/2    1  1/2
 1/2  1/2    1
→ contravariant metric tensor: 3×3 Symmetric{Sym, Matrix{Sym}}:
  3/2  -1/2  -1/2
 -1/2   3/2  -1/2
 -1/2  -1/2   3/2

julia> components(V, ℬ̄, (:cov,))
3-element Vector{Sym}:
 sqrt(2)*v2/2 + sqrt(2)*v3/2
 sqrt(2)*v1/2 + sqrt(2)*v3/2
 sqrt(2)*v1/2 + sqrt(2)*v2/2

julia> T = Tens(Tensor{2,3}((i, j) -> symbols("t$i$j", real = true)))
TensND.TensCanonical{2, 3, Sym, Tensor{2, 3, Sym, 9}}
→ data: 3×3 Tensor{2, 3, Sym, 9}:
 t₁₁  t₁₂  t₁₃
 t₂₁  t₂₂  t₂₃
 t₃₁  t₃₂  t₃₃
→ basis: 3×3 TensND.Id2{3, Sym}:
 1  ⋅  ⋅
 ⋅  1  ⋅
 ⋅  ⋅  1
→ var: (:cont, :cont)

julia> components(T, ℬ, (:cov, :cov))
3×3 Matrix{Sym}:
 t₂₂ + t₂₃ + t₃₂ + t₃₃  t₂₁ + t₂₃ + t₃₁ + t₃₃  t₂₁ + t₂₂ + t₃₁ + t₃₂
 t₁₂ + t₁₃ + t₃₂ + t₃₃  t₁₁ + t₁₃ + t₃₁ + t₃₃  t₁₁ + t₁₂ + t₃₁ + t₃₂
 t₁₂ + t₁₃ + t₂₂ + t₂₃  t₁₁ + t₁₃ + t₂₁ + t₂₃  t₁₁ + t₁₂ + t₂₁ + t₂₂

julia> factor(simplify(components(T, ℬ, (:cont, :cov))))
3×3 Matrix{Sym}:
 -(t12 + t13 - t22 - t23 - t32 - t33)/2  -(t11 + t13 - t21 - t23 - t31 - t33)/2  -(t11 + t12 - t21 - t22 - t31 - t32)/2
  (t12 + t13 - t22 - t23 + t32 + t33)/2   (t11 + t13 - t21 - t23 + t31 + t33)/2   (t11 + t12 - t21 - t22 + t31 + t32)/2
  (t12 + t13 + t22 + t23 - t32 - t33)/2   (t11 + t13 + t21 + t23 - t31 - t33)/2   (t11 + t12 + t21 + t22 - t31 - t32)/2
```

Special tensors are available

- `tensId2(::Val{dim} = Val(3), ::Val{T} = Val(Sym)) where {dim,T<:Number}`: second-order identity (`𝟏ᵢⱼ = δᵢⱼ = 1 if i=j otherwise 0`)
- `tensId4(::Val{dim} = Val(3), ::Val{T} = Val(Sym)) where {dim,T<:Number}`: fourth-order identity with minor symmetries (`𝕀 = 𝟏 ⊠ˢ 𝟏` i.e. `(𝕀)ᵢⱼₖₗ = (δᵢₖδⱼₗ+δᵢₗδⱼₖ)/2`)
- `tensJ4(::Val{dim} = Val(3), ::Val{T} = Val(Sym)) where {dim,T<:Number}`: fourth-order spherical projector (`𝕁 = (𝟏 ⊗ 𝟏) / dim` i.e. `(𝕁)ᵢⱼₖₗ = δᵢⱼδₖₗ/dim`)
- `tensK4(::Val{dim} = Val(3), ::Val{T} = Val(Sym)) where {dim,T<:Number}`: fourth-order deviatoric projector (`𝕂 = 𝕀 - 𝕁` i.e. `(𝕂)ᵢⱼₖₗ = (δᵢₖδⱼₗ+δᵢₗδⱼₖ)/2 - δᵢⱼδₖₗ/dim`)
- `ISO(::Val{dim} = Val(3), ::Val{T} = Val(Sym)) where {dim,T<:Number}`: returns `𝕀, 𝕁, 𝕂`

The useful tensor products are the following:

- `⊗` tensor product
- `⊗ˢ` symmetrized tensor product
- `⊠` modified tensor product
- `⊠ˢ` symmetrized modified tensor product
- `⋅` contracted product
- `⊡` double contracted product
- `⊙` quadruple contracted product

**_NOTE:_**
more information about modified tensor products can be found in [Sébastien Brisard's blog](https://sbrisard.github.io/posts/20140226-decomposition_of_transverse_isotropic_fourth-rank_tensors.html).

```julia
julia> 𝟏 = tensId2(Val(3), Val(Sym))
TensISO{2, 3, Sym, 1}
→ data: 3×3 Matrix{Sym}:
 1  0  0
 0  1  0
 0  0  1
→ basis: 3×3 TensND.Id2{3, Sym}:
 1  ⋅  ⋅
 ⋅  1  ⋅
 ⋅  ⋅  1
→ var: (:cont, :cont)

julia> 𝕀, 𝕁, 𝕂 = ISO(Val(3),Val(Sym)) ;

julia> 𝕀 == 𝟏 ⊠ˢ 𝟏
true

julia> 𝕁 == (𝟏 ⊗ 𝟏)/3
true

julia> a = Tens(Vec{3}((i,) -> symbols("a$i", real = true))) ;

julia> b = Tens(Vec{3}((i,) -> symbols("b$i", real = true))) ;

julia> a ⊗ b
TensND.TensCanonical{2, 3, Sym, Tensor{2, 3, Sym, 9}}
→ data: 3×3 Tensor{2, 3, Sym, 9}:
 a₁⋅b₁  a₁⋅b₂  a₁⋅b₃
 a₂⋅b₁  a₂⋅b₂  a₂⋅b₃
 a₃⋅b₁  a₃⋅b₂  a₃⋅b₃
→ basis: 3×3 TensND.Id2{3, Sym}:
 1  ⋅  ⋅
 ⋅  1  ⋅
 ⋅  ⋅  1
→ var: (:cont, :cont)

julia> a ⊗ˢ b
TensND.TensCanonical{2, 3, Sym, Tensor{2, 3, Sym, 9}}
→ data: 3×3 Tensor{2, 3, Sym, 9}:
             a₁⋅b₁  a1*b2/2 + a2*b1/2  a1*b3/2 + a3*b1/2
 a1*b2/2 + a2*b1/2              a₂⋅b₂  a2*b3/2 + a3*b2/2
 a1*b3/2 + a3*b1/2  a2*b3/2 + a3*b2/2              a₃⋅b₃
→ basis: 3×3 TensND.Id2{3, Sym}:
 1  ⋅  ⋅
 ⋅  1  ⋅
 ⋅  ⋅  1
→ var: (:cont, :cont)

julia> (θ, ϕ, r), (𝐞ᶿ, 𝐞ᵠ, 𝐞ʳ), ℬˢ = init_spherical()
((θ, ϕ, r), (Sym[1, 0, 0], Sym[0, 1, 0], Sym[0, 0, 1]), Sym[cos(θ)*cos(ϕ) -sin(ϕ) sin(θ)*cos(ϕ); sin(ϕ)*cos(θ) cos(ϕ) sin(θ)*sin(ϕ); -sin(θ) 0 cos(θ)])

julia> R = rot3(θ, ϕ)
3×3 RotZYZ{Sym} with indices SOneTo(3)×SOneTo(3)(ϕ, θ, 0):
 cos(θ)⋅cos(ϕ)  -sin(ϕ)  sin(θ)⋅cos(ϕ)
 sin(ϕ)⋅cos(θ)   cos(ϕ)  sin(θ)⋅sin(ϕ)
       -sin(θ)        0         cos(θ)

julia> A = Tens(R * a)
TensND.TensCanonical{1, 3, Sym, Vec{3, Sym}}
→ data: 3-element Vec{3, Sym}:
 a₁⋅cos(θ)⋅cos(ϕ) - a₂⋅sin(ϕ) + a₃⋅sin(θ)⋅cos(ϕ)
 a₁⋅sin(ϕ)⋅cos(θ) + a₂⋅cos(ϕ) + a₃⋅sin(θ)⋅sin(ϕ)
                          -a₁⋅sin(θ) + a₃⋅cos(θ)
→ basis: 3×3 TensND.Id2{3, Sym}:
 1  ⋅  ⋅
 ⋅  1  ⋅
 ⋅  ⋅  1
→ var: (:cont,)

julia> simplify(change_tens(A, ℬˢ))
TensND.TensRotated{1, 3, Sym, Vec{3, Sym}}
→ data: 3-element Vec{3, Sym}:
 a₁
 a₂
 a₃
→ basis: 3×3 Matrix{Sym}:
 cos(θ)⋅cos(ϕ)  -sin(ϕ)  sin(θ)⋅cos(ϕ)
 sin(ϕ)⋅cos(θ)   cos(ϕ)  sin(θ)⋅sin(ϕ)
       -sin(θ)        0         cos(θ)
→ var: (:cont,)
```

### Coordinate systems and differential operators

```julia
julia> Polar = coorsys_polar() ; r, θ = getcoords(Polar) ; 𝐞ʳ, 𝐞ᶿ = unitvec(Polar) ;

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

```julia
julia> Spherical = coorsys_spherical() ;

julia> θ, ϕ, r = getcoords(Spherical) ;

julia> 𝐞ᶿ, 𝐞ᵠ, 𝐞ʳ = unitvec(Spherical) ;

julia> getChristoffel(Spherical)
3×3×3 Array{Sym, 3}:
[:, :, 1] =
   0               0  1/r
   0  -sin(θ)⋅cos(θ)    0
 1/r               0    0

[:, :, 2] =
             0  cos(θ)/sin(θ)    0
 cos(θ)/sin(θ)              0  1/r
             0            1/r    0

[:, :, 3] =
 -r            0  0
  0  -r*sin(θ)^2  0
  0            0  0

julia> ℬˢ = get_normalized_basis(Spherical)
RotatedBasis{3, Sym}
→ basis: 3×3 Matrix{Sym}:
 cos(θ)⋅cos(ϕ)  -sin(ϕ)  sin(θ)⋅cos(ϕ)
 sin(ϕ)⋅cos(θ)   cos(ϕ)  sin(θ)⋅sin(ϕ)
       -sin(θ)        0         cos(θ)
→ dual basis: 3×3 Matrix{Sym}:
 cos(θ)⋅cos(ϕ)  -sin(ϕ)  sin(θ)⋅cos(ϕ)
 sin(ϕ)⋅cos(θ)   cos(ϕ)  sin(θ)⋅sin(ϕ)
       -sin(θ)        0         cos(θ)
→ covariant metric tensor: 3×3 TensND.Id2{3, Sym}:
 1  ⋅  ⋅
 ⋅  1  ⋅
 ⋅  ⋅  1
→ contravariant metric tensor: 3×3 TensND.Id2{3, Sym}:
 1  ⋅  ⋅
 ⋅  1  ⋅
 ⋅  ⋅  1

julia> for σⁱʲ ∈ ("σʳʳ", "σᶿᶿ", "σᵠᵠ") @eval $(Symbol(σⁱʲ)) = SymFunction($σⁱʲ, real = true)($r) end

julia> 𝛔 = σʳʳ * 𝐞ʳ ⊗ 𝐞ʳ + σᶿᶿ * 𝐞ᶿ ⊗ 𝐞ᶿ + σᵠᵠ * 𝐞ᵠ ⊗ 𝐞ᵠ
TensND.TensRotated{2, 3, Sym, Tensor{2, 3, Sym, 9}}
→ data: 3×3 Tensor{2, 3, Sym, 9}:
 σᶿᶿ(r)       0       0
      0  σᵠᵠ(r)       0
      0       0  σʳʳ(r)
→ basis: 3×3 Matrix{Sym}:
 cos(θ)⋅cos(ϕ)  -sin(ϕ)  sin(θ)⋅cos(ϕ)
 sin(ϕ)⋅cos(θ)   cos(ϕ)  sin(θ)⋅sin(ϕ)
       -sin(θ)        0         cos(θ)
→ var: (:cont, :cont)

julia>         div𝛔 = simplify(DIV(𝛔, Spherical))
TensND.TensRotated{1, 3, Sym, Vec{3, Sym}}
→ data: 3-element Vec{3, Sym}:
                            (-σᵠᵠ(r) + σᶿᶿ(r))/(r*tan(θ))
                                                        0
 (r*Derivative(σʳʳ(r), r) + 2*σʳʳ(r) - σᵠᵠ(r) - σᶿᶿ(r))/r
→ basis: 3×3 Matrix{Sym}:
 cos(θ)⋅cos(ϕ)  -sin(ϕ)  sin(θ)⋅cos(ϕ)
 sin(ϕ)⋅cos(θ)   cos(ϕ)  sin(θ)⋅sin(ϕ)
       -sin(θ)        0         cos(θ)
→ var: (:cont,)
```

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