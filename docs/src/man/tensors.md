# Tensors

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
