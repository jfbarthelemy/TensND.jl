# Bases

An arbitrary basis contains four matrices

- one in which columns correspond to the covariant vectors of new basis `(𝐞ᵢ)` with respect to the canonical one,
- one defining the contravariant (or dual) basis `(𝐞ⁱ)`,
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

```@repl bases
using TensND, SymPy
ℬ = Basis(Sym[1 0 0; 0 1 0; 0 1 1])
ℬ₂ = Basis(symbols("θ, ϕ, ψ", real = true)...)
```

Predefined symbolic or numerical coordinates and basis vectors can be obtained from

- `init_cartesian(dim::Integer)`
- `init_polar(coords = (symbols("r", positive = true), symbols("θ", real = true)); canonical = false)`
- `init_cylindrical(coords = (symbols("r", positive = true), symbols("θ", real = true), symbols("z", real = true)); canonical = false)`
- `init_spherical(coords = (symbols("θ", real = true), symbols("ϕ", real = true), symbols("r", positive = true)); canonical = false)`
- `init_rotated(coords = symbols("θ ϕ ψ", real = true); canonical = false)`

The option `canonical` specifies whether the vector is expressed as a tensor with components in the canonical basis or directly in the rotated basis. The second option (ie `canonical = false` by default) is often preferable for further calculations in the rotated basis.

```@repl bases
(x, y, z), (𝐞₁, 𝐞₂, 𝐞₃), ℬ = init_cartesian() ;
(r, θ), (𝐞ʳ, 𝐞ᶿ), ℬᵖ = init_polar() ;
(r, θ, z), (𝐞ʳ, 𝐞ᶿ, 𝐞ᶻ), ℬᶜ = init_cylindrical() ;
ℬᶜ
(θ, ϕ, r), (𝐞ᶿ, 𝐞ᵠ, 𝐞ʳ), ℬˢ = init_spherical() ;
components_canon(𝐞ʳ)
(θ, ϕ, ψ), (𝐞ᶿ, 𝐞ᵠ, 𝐞ʳ), ℬʳ = init_rotated() ;
```

**_NOTE:_**
it is worth noting the unusual order of coordinates and vectors of the spherical basis which have been chosen here so that `θ = ϕ = 0` corresponds to the cartesian basis in the correct order.
