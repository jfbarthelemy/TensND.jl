using TensND, LinearAlgebra, SymPy, Tensors, OMEinsum, Rotations
sympy.init_printing(use_unicode=true)

Polar = coorsys_polar()
@set_coorsys Polar
r, θ = getcoords(Polar)
𝐞ʳ, 𝐞ᶿ = unitvec(Polar)
ℬˢ = get_normalized_basis(Polar)
𝐱 = getOM(Polar)
Cartesian = coorsys_cartesian(symbols("x y", real = true))
𝐞₁, 𝐞₂ = unitvec(Cartesian)
x₁, x₂ = getcoords(Cartesian)
𝕀, 𝕁, 𝕂 = ISO(Val(2),Val(Sym))
𝟏 = tensId2(Val(2),Val(Sym))

E, k, μ, κ = symbols("E k μ κ", positive = true)
ν = symbols("ν", real = true)
k = E / (3(1-2ν)) ; μ = E / (2(1+ν))
λ = k -2μ/3

𝐆 =simplify(1/(8 * PI * μ * (1-ν)) * (𝐞ʳ ⊗ 𝐞ʳ -(3-4ν) * ln(r) * 𝟏))
HG = -simplify(HESS(𝐆))
ℾ = simplify(Tens(minorsymmetric(getarray(HG)),ℬˢ))
ℾ₂ = simplify(1/(8 * PI * μ * (1-ν) * r^2) * (-2𝕁 +2(1-2ν)*𝕀 + 2𝟏⊗𝐞ʳ⊗𝐞ʳ +2𝐞ʳ⊗𝐞ʳ⊗𝟏 + 8ν*𝐞ʳ⊗ˢ𝟏⊗ˢ𝐞ʳ -8𝐞ʳ⊗𝐞ʳ⊗𝐞ʳ⊗𝐞ʳ))
simplify(ℾ-ℾ₂)

ℂ = 2λ * 𝕁 + 2μ * 𝕀

𝕜 = simplify(ℾ ⊡ ℂ)

d = Dict(r => sqrt(x₁^2+x₂^2), sin(θ) => x₂/sqrt(x₁^2+x₂^2), cos(θ) => x₁/sqrt(x₁^2+x₂^2), ν => (3-κ)/4)

mk = factor(simplify(subs(components_canon(𝕜),d...))) ;
mk[1,2,1,2]

𝛆 = 𝕜 ⊡ 𝟏
m𝛆 = factor(simplify(subs(components_canon(𝛆),d...)))

simplify((𝐞ʳ⊗𝐞ʳ + 𝐞ᶿ⊗𝐞ᶿ)⊗𝐞ʳ⊗𝐞ʳ -𝟏⊗𝐞ʳ⊗𝐞ʳ)

𝐞ᶿ⊗𝐞ᶿ⊗𝐞ʳ⊗𝐞ʳ

𝟏 = tensId2(Val(2),Val(Sym))

U = 𝟏 ⊗ 𝐞ʳ

V = 𝟏 ⊗ (𝐞ʳ⊗𝐞ʳ)

T = 𝟏 ⊗ 𝐞ʳ⊗𝐞ʳ
