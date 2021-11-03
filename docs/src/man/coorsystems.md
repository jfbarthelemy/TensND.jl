# Coordinate systems and differential operators

For the moment only symbolic coordinate systems are available. Their numerical counterpart will be later developed. 

```julia
julia> Polar = coorsys_polar() ; r, θ = getcoords(Polar) ; 𝐞ʳ, 𝐞ᶿ = unitvec(Polar) ;

julia> @set_coorsys Polar

julia> LAPLACE(SymFunction("f", real = true)(r, θ))
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

julia> simplify(HESS(r^n))
(n*r^(n - 2)*(n - 1))𝐞ʳ⊗𝐞ʳ + (n*r^(n - 2))𝐞ᶿ⊗𝐞ᶿ
```

```julia
julia> Spherical = coorsys_spherical() ; θ, ϕ, r = getcoords(Spherical) ; 𝐞ᶿ, 𝐞ᵠ, 𝐞ʳ = unitvec(Spherical) ;

julia> @set_coorsys Spherical

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
(σᶿᶿ(r))𝐞ᶿ⊗𝐞ᶿ + (σᵠᵠ(r))𝐞ᵠ⊗𝐞ᵠ + (σʳʳ(r))𝐞ʳ⊗𝐞ʳ

julia> div𝛔 = simplify(DIV(𝛔))
((-σᵠᵠ(r) + σᶿᶿ(r))/(r*tan(θ)))𝐞ᶿ + ((r*Derivative(σʳʳ(r), r) + 2*σʳʳ(r) - σᵠᵠ(r) - σᶿᶿ(r))/r)𝐞ʳ
```
