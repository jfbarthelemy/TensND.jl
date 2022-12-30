using TensND, LinearAlgebra, SymPy, Tensors, OMEinsum, Rotations
sympy.init_printing(use_unicode=true)

Cartesian = coorsys_cartesian(symbols("x y z", real = true))
𝐞₁, 𝐞₂, 𝐞₃ = unitvec(Cartesian)
x₁, x₂, x₃ = getcoords(Cartesian)

Spherical = coorsys_spherical((symbols("θ ϕ", real = true)..., symbols("ξ", positive = true)))
θ, ϕ, ξ = getcoords(Spherical) ; 𝐞ᶿ, 𝐞ᵠ, 𝐞ʳ = unitvec(Spherical) ;
ℬˢ = normalized_basis(Spherical)
# @set_coorsys Spherical
𝕀, 𝕁, 𝕂 = ISO(Val(3),Val(Sym))
𝟏 = tensId2(Val(3),Val(Sym))

𝛏 = getOM(Spherical)

E, k, μ = symbols("E k μ", positive = true)
ν = symbols("ν", real = true)
# k = E / (3(1-2ν)) ; μ = E / (2(1+ν))
# λ = k -2μ/3
λ = symbols("λ", real = true)

ℂ = 3λ*𝕁 + 2μ*𝕀 ;
𝐊 = 𝛏⋅ℂ⋅𝛏 ;
ℾ = 𝛏 ⊗ˢ 𝐊^(-1) ⊗ˢ 𝛏 ;
𝚲 = tsimplify(ℂ ⊡ ℾ ⊡ ℂ) ;  
𝚲₂ = tsimplify(λ^2/(λ+2μ)*𝟏⊗𝟏 + 2λ*μ/(λ+2μ)*(𝟏 ⊗ 𝐞ʳ ⊗ 𝐞ʳ + 𝐞ʳ ⊗ 𝐞ʳ ⊗ 𝟏) + 4μ*(𝐞ʳ ⊗ˢ 𝟏 ⊗ˢ 𝐞ʳ - (λ+μ)/(λ+2μ) * 𝐞ʳ ⊗ 𝐞ʳ ⊗ 𝐞ʳ ⊗ 𝐞ʳ)) ;
intrinsic(𝚲-𝚲₂,Spherical)

f(𝐧) = 𝐧⋅𝚲⋅𝐧
h(𝐧) = 𝐧⋅𝚲₂⋅𝐧
g(𝐧) = λ^2/(λ+2μ)*𝐧⊗𝐧 + μ*(3λ+2μ)/(λ+2μ)*(𝐧⋅𝐞ʳ)*(𝐧⊗𝐞ʳ+𝐞ʳ⊗𝐧) -4μ*(λ+μ)/(λ+2μ)*(𝐧⋅𝐞ʳ)^2*𝐞ʳ⊗𝐞ʳ + μ*(𝐧⋅𝐞ʳ)^2*𝟏 + μ*𝐞ʳ⊗𝐞ʳ

𝐧 = 𝐞₃

f(𝐧)
intrinsic(tsimplify(g(𝐧)-f(𝐧)),Spherical)

F(𝐧) = 𝐧⋅(𝐞ʳ ⊗ˢ 𝟏 ⊗ˢ 𝐞ʳ)⋅𝐧
G(𝐧) = (𝐞ʳ⊗𝐞ʳ+(𝐧⋅𝐞ʳ)*(𝐧⊗𝐞ʳ+𝐞ʳ⊗𝐧)+(𝐧⋅𝐞ʳ)^2*𝟏)/4

G(𝐧) = tsimplify(𝐧⋅𝚲⋅𝐧-𝐧⋅ℂ⋅𝐧)
