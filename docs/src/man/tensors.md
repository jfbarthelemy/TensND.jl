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
(v1)𝐞¹ + (v2)𝐞² + (v3)𝐞³

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
(t11)𝐞¹⊗𝐞¹ + (t21)𝐞²⊗𝐞¹ + (t31)𝐞³⊗𝐞¹ + (t12)𝐞¹⊗𝐞² + (t22)𝐞²⊗𝐞² + (t32)𝐞³⊗𝐞² + (t13)𝐞¹⊗𝐞³ + (t23)𝐞²⊗𝐞³ + (t33)𝐞³⊗𝐞³

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

```julia
julia> 𝟏 = tensId2(3, Sym)
(1) 𝟏

julia> 𝕀, 𝕁, 𝕂 = ISO(3, Sym) ;

julia> 𝕀 == 𝟏 ⊠ˢ 𝟏
true

julia> 𝕁 == (𝟏 ⊗ 𝟏)/3
true

julia> a = Tens(Vec{3}((i,) -> symbols("a$i", real = true))) ;

julia> b = Tens(Vec{3}((i,) -> symbols("b$i", real = true))) ;

julia> a ⊗ b
(a1*b1)𝐞¹⊗𝐞¹ + (a2*b1)𝐞²⊗𝐞¹ + (a3*b1)𝐞³⊗𝐞¹ + (a1*b2)𝐞¹⊗𝐞² + (a2*b2)𝐞²⊗𝐞² + (a3*b2)𝐞³⊗𝐞² + (a1*b3)𝐞¹⊗𝐞³ + (a2*b3)𝐞²⊗𝐞³ + (a3*b3)𝐞³⊗𝐞³

julia> a ⊗ˢ b
(a1*b1)𝐞¹⊗𝐞¹ + (a1*b2/2 + a2*b1/2)𝐞²⊗𝐞¹ + (a1*b3/2 + a3*b1/2)𝐞³⊗𝐞¹ + (a1*b2/2 + a2*b1/2)𝐞¹⊗𝐞² + (a2*b2)𝐞²⊗𝐞² + (a2*b3/2 + a3*b2/2)𝐞³⊗𝐞² + (a1*b3/2 + a3*b1/2)𝐞¹⊗𝐞³
 + (a2*b3/2 + a3*b2/2)𝐞²⊗𝐞³ + (a3*b3)𝐞³⊗𝐞³
```

The predefined spherical coordinate system `init_spherical()` provides the local orthonormal basis
``(\mathbf{e}_\theta, \mathbf{e}_\varphi, \mathbf{e}_r)`` in terms of polar angle ``\theta`` (from the ``z``-axis) and azimuthal angle ``\varphi``:

```math
\mathbf{e}_\theta = \cos\theta\cos\varphi\,\mathbf{e}_1 + \cos\theta\sin\varphi\,\mathbf{e}_2 - \sin\theta\,\mathbf{e}_3
```

```math
\mathbf{e}_\varphi = -\sin\varphi\,\mathbf{e}_1 + \cos\varphi\,\mathbf{e}_2
```

```math
\mathbf{e}_r = \sin\theta\cos\varphi\,\mathbf{e}_1 + \sin\theta\sin\varphi\,\mathbf{e}_2 + \cos\theta\,\mathbf{e}_3
```

The rotation matrix ``R = [\mathbf{e}_\theta \mid \mathbf{e}_\varphi \mid \mathbf{e}_r]`` encodes this change of basis.
For any vector ``\mathbf{A}`` in the canonical frame, `change_tens(A, ℬˢ)` returns its components in the spherical basis.
The example below verifies that if ``\mathbf{A} = R\,\mathbf{a}``, then expressing ``\mathbf{A}`` in ``\mathcal{B}^s`` recovers the original components ``(a_1, a_2, a_3)``:

```julia
julia> (θ, ϕ, r), (𝐞ᶿ, 𝐞ᵠ, 𝐞ʳ), ℬˢ = init_spherical()
((θ, ϕ, r), (Sym[1, 0, 0], Sym[0, 1, 0], Sym[0, 0, 1]), Sym[cos(θ)*cos(ϕ) -sin(ϕ) sin(θ)*cos(ϕ); sin(ϕ)*cos(θ) cos(ϕ) sin(θ)*sin(ϕ); -sin(θ) 0 cos(θ)])

julia> R = rot3(θ, ϕ)
3×3 RotZYZ{Sym} with indices SOneTo(3)×SOneTo(3)(ϕ, θ, 0):
 cos(θ)⋅cos(ϕ)  -sin(ϕ)  sin(θ)⋅cos(ϕ)
 sin(ϕ)⋅cos(θ)   cos(ϕ)  sin(θ)⋅sin(ϕ)
       -sin(θ)        0         cos(θ)

julia> A = Tens(R * a)
(a1*cos(θ)*cos(ϕ) - a2*sin(ϕ) + a3*sin(θ)*cos(ϕ))𝐞¹ + (a1*sin(ϕ)*cos(θ) + a2*cos(ϕ) + a3*sin(θ)*sin(ϕ))𝐞² + (-a1*sin(θ) + a3*cos(θ))𝐞³

julia> simplify(change_tens(A, ℬˢ))
(a1)𝐞¹ + (a2)𝐞² + (a3)𝐞³
```

## Transverse isotropy and orthotropy

### TensWalpole

A transversely isotropic 4th-order tensor with symmetry axis ``\mathbf{n}`` is decomposed in the Walpole basis:

```math
L = \ell_1 W_1 + \ell_2 W_2 + \ell_3 W_3 + \ell_4 W_4 + \ell_5 W_5 + \ell_6 W_6
```

where ``\mathbf{n}_n = \mathbf{n}\otimes\mathbf{n}``, ``\mathbf{n}_T = \mathbf{1} - \mathbf{n}_n`` and

| Tensor | Expression |
| ------ | ----------- |
| ``W_1`` | ``\mathbf{n}_n\otimes\mathbf{n}_n`` |
| ``W_2`` | ``(\mathbf{n}_T\otimes\mathbf{n}_T)/2`` |
| ``W_3`` | ``(\mathbf{n}_n\otimes\mathbf{n}_T)/\sqrt{2}`` |
| ``W_4`` | ``(\mathbf{n}_T\otimes\mathbf{n}_n)/\sqrt{2}`` |
| ``W_5`` | ``\mathbf{n}_T\,\overline{\boxtimes}^s\,\mathbf{n}_T - (\mathbf{n}_T\otimes\mathbf{n}_T)/2`` |
| ``W_6`` | ``\mathbf{n}_T\,\overline{\boxtimes}^s\,\mathbf{n}_n + \mathbf{n}_n\,\overline{\boxtimes}^s\,\mathbf{n}_T`` |

The double contraction follows the **synthetic Walpole rule**:

```math
L\colon M \equiv \left(\begin{bmatrix}\ell_1 & \ell_3\\\ell_4 & \ell_2\end{bmatrix}\begin{bmatrix}m_1 & m_3\\m_4 & m_2\end{bmatrix},\; \ell_5 m_5,\; \ell_6 m_6\right)
```

For major-symmetric tensors (``\ell_3=\ell_4``), use `N=5`; for general tensors, `N=6`.

```julia
julia> n = 𝐞(3) ;

julia> W1, W2, W3, W4, W5, W6 = Walpole(n) ;

julia> L = TensWalpole(2., 1., 0.5, 0.3, 0.8, n)
(2.0) W₁ˢ + (1.0) W₂ˢ + (0.5) W₃ˢ + (0.3) W₄ˢ + (0.8) W₅ˢ
  axis n = (0.0, 0.0, 1.0)

julia> maximum(abs.(getarray(L ⊡ inv(L)) - getarray(tensId4(Val(3), Val(Float64)))))
0.0

julia> 𝕀, 𝕁, 𝕂 = ISO() ; L2 = fromISO(3𝕁 + 2𝕂, n)
(1.6666666666666667) W₁ˢ + (2.3333333333333335) W₂ˢ + (0.4714045207910317) W₃ˢ + (2.0) W₄ˢ + (2.0) W₅ˢ
  axis n = (0.0, 0.0, 1.0)
```

### TensOrtho

An orthotropic 4th-order tensor in material frame ``(\mathbf{e}_1,\mathbf{e}_2,\mathbf{e}_3)``
with ``P_m = \mathbf{e}_m\otimes\mathbf{e}_m`` has 9 independent elastic constants:

```math
\mathbb{C} = C_{11}P_1{\otimes}P_1 + C_{22}P_2{\otimes}P_2 + C_{33}P_3{\otimes}P_3
+ C_{12}(P_1{\otimes}P_2+P_2{\otimes}P_1) + C_{13}(P_1{\otimes}P_3+P_3{\otimes}P_1) + C_{23}(P_2{\otimes}P_3+P_3{\otimes}P_2)
+ 2C_{44}(P_2\,\overline{\boxtimes}^s P_3) + 2C_{55}(P_1\,\overline{\boxtimes}^s P_3) + 2C_{66}(P_1\,\overline{\boxtimes}^s P_2)
```

The Kelvin-Mandel matrix in the material frame (ordering ``11,22,33,23,13,12``) is block-diagonal.
Use `KM_material(t)` to retrieve it; `KM(t)` gives the matrix in the canonical frame.

```julia
julia> ℬ = CanonicalBasis{3,Float64}() ;

julia> t = TensOrtho(10., 8., 9., 3., 2., 4., 2.5, 3., 1.5, ℬ) ;

julia> KM_material(t)
6×6 Matrix{Float64}:
 10.0   3.0   2.0  0.0  0.0  0.0
  3.0   8.0   4.0  0.0  0.0  0.0
  2.0   4.0   9.0  0.0  0.0  0.0
  0.0   0.0   0.0  5.0  0.0  0.0
  0.0   0.0   0.0  0.0  6.0  0.0
  0.0   0.0   0.0  0.0  0.0  3.0

julia> maximum(abs.(getarray(t) ⊡ getarray(inv(t)) - getarray(tensId4(Val(3), Val(Float64)))))
0.0
```
