using TensND, LinearAlgebra, SymPy, Tensors, OMEinsum, Rotations
# sympy.init_printing(use_unicode=true)

Polar = coorsys_polar() ; r, θ = getcoords(Polar) ; 𝐞ʳ, 𝐞ᶿ = unitvec(Polar)
@set_coorsys Polar
ℬˢ = normalized_basis(Polar)
𝐱 = getOM(Polar)
Cartesian = coorsys_cartesian(symbols("x y", real = true))
𝐞₁, 𝐞₂ = unitvec(Cartesian)
x₁, x₂ = getcoords(Cartesian)
𝕀, 𝕁, 𝕂 = ISO(Val(2),Val(Sym))
𝟏 = tensId2(Val(2),Val(Sym))

E, k, μ = symbols("E k μ", positive = true)
ν, κ = symbols("ν κ", real = true)
k = E / (3(1-2ν)) ; μ = E / (2(1+ν))
λ = k -2μ/3

𝐆 =tsimplify(1/(8 * PI * μ * (1-ν)) * (𝐞ʳ ⊗ 𝐞ʳ -(3-4ν) * log(r) * 𝟏))
HG = -tsimplify(HESS(𝐆))
aHG = getarray(HG)
𝕄 = SymmetricTensor{4,2}((i,j,k,l)->(aHG[i,k,j,l]+aHG[j,k,i,l]+aHG[i,l,j,k]+aHG[j,l,i,k])/4)
ℾ = tsimplify(Tens(𝕄,ℬˢ))
ℾ₂ = tsimplify(1/(8PI * μ * (1-ν) * r^2) * (-2𝕁 +2(1-2ν)*𝕀 + 2(𝟏⊗𝐞ʳ⊗𝐞ʳ + 𝐞ʳ⊗𝐞ʳ⊗𝟏) + 8ν*𝐞ʳ⊗ˢ𝟏⊗ˢ𝐞ʳ -8𝐞ʳ⊗𝐞ʳ⊗𝐞ʳ⊗𝐞ʳ))
tsimplify(ℾ-ℾ₂)

ℂ = 2λ * 𝕁 + 2μ * 𝕀

𝕜 = tsimplify(ℾ ⊡ ℂ)
d = Dict(r => sqrt(x₁^2+x₂^2), sin(θ) => x₂/sqrt(x₁^2+x₂^2), cos(θ) => x₁/sqrt(x₁^2+x₂^2), ν => (3-κ)/4)


Spherical = coorsys_spherical() ; θ, ϕ, r = getcoords(Spherical) ; 𝐞ᶿ, 𝐞ᵠ, 𝐞ʳ = unitvec(Spherical) ;
ℬˢ = normalized_basis(Spherical)
@set_coorsys Spherical
𝕀, 𝕁, 𝕂 = ISO(Val(3),Val(Sym))
𝟏 = tensId2(Val(3),Val(Sym))

E, k, μ = symbols("E k μ", positive = true)
ν = symbols("ν", real = true)
k = E / (3(1-2ν)) ; μ = E / (2(1+ν))
λ = k -2μ/3

𝐆 = 1/ (8PI * μ * (3k+4μ) * r) * ( (3k+7μ) * 𝟏 + (3k+μ) * 𝐞ʳ⊗𝐞ʳ)
𝐆₂ = 1 / (16PI * μ * (1-ν) * r) * ( (3-4ν) * 𝟏 + 𝐞ʳ⊗𝐞ʳ)
tsimplify(𝐆-𝐆₂)

HG = -tsimplify(HESS(𝐆))
aHG = getarray(HG)
𝕄 = SymmetricTensor{4,3}((i,j,k,l)->(aHG[i,k,j,l]+aHG[j,k,i,l]+aHG[i,l,j,k]+aHG[j,l,i,k])/4)
ℾ = tsimplify(Tens(𝕄,ℬˢ))
ℾ₂ = tsimplify(1/(16PI * μ * (1-ν) * r^3) * (-3𝕁 +2(1-2ν)*𝕀 + 3(𝟏⊗𝐞ʳ⊗𝐞ʳ + 𝐞ʳ⊗𝐞ʳ⊗𝟏) + 12ν*𝐞ʳ⊗ˢ𝟏⊗ˢ𝐞ʳ -15𝐞ʳ⊗𝐞ʳ⊗𝐞ʳ⊗𝐞ʳ))
tsimplify(ℾ-ℾ₂)

Cartesian = coorsys_cartesian(symbols("x y z", real = true))
𝐞₁, 𝐞₂, 𝐞3 = unitvec(Cartesian)
F = symbols("F", real = true)
J = tsimplify(det(𝟏 + F*GRAD(𝐆⋅𝐞₁)))
factor(tsimplify(subs(J, θ => PI/2, ϕ => 0)))
