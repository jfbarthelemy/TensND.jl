# Coordinate systems and differential operators

Both symbolic (`CoorSystemSym`) and numerical (`CoorSystemNum`) coordinate systems are available. Symbolic systems support exact derivation; numerical systems evaluate differential operators pointwise via automatic differentiation — see the [tutorial](../tuto/coorsystems_num.md) for examples.

```@repl coorsys
using TensND, SymPy
Polar = coorsys_polar() ; r, θ = getcoords(Polar) ; 𝐞ʳ, 𝐞ᶿ = unitvec(Polar) ;
@set_coorsys Polar
LAPLACE(SymFunction("f", real = true)(r, θ))
n = symbols("n", integer = true)
simplify(HESS(r^n))
```

```@repl coorsys2
using TensND, SymPy
Spherical = coorsys_spherical() ; θ, ϕ, r = getcoords(Spherical) ; 𝐞ᶿ, 𝐞ᵠ, 𝐞ʳ = unitvec(Spherical) ;
@set_coorsys Spherical
Christoffel(Spherical)
ℬˢ = normalized_basis(Spherical)
σʳʳ = SymFunction("σʳʳ", real = true)(r) ;
σᶿᶿ = SymFunction("σᶿᶿ", real = true)(r) ;
σᵠᵠ = SymFunction("σᵠᵠ", real = true)(r) ;
𝛔 = σʳʳ * 𝐞ʳ ⊗ 𝐞ʳ + σᶿᶿ * 𝐞ᶿ ⊗ 𝐞ᶿ + σᵠᵠ * 𝐞ᵠ ⊗ 𝐞ᵠ
div𝛔 = simplify(DIV(𝛔))
```
