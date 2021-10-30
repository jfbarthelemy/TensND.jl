# Coordinate systems and differential operators

For the moment only symbolic coordinate systems are available. Their numerical counterpart will be later developed. 

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
