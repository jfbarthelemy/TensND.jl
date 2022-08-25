using TensND, LinearAlgebra, SymPy, Tensors, OMEinsum, Rotations
sympy.init_printing(use_unicode=true)

Spherical = coorsys_spherical()
θ, ϕ, r = getcoords(Spherical)
𝐞ᶿ, 𝐞ᵠ, 𝐞ʳ = unitvec(Spherical)
ℬˢ = normalized_basis(Spherical)
𝐱 = getOM(Spherical)
@set_coorsys Spherical
𝐞₁, 𝐞₂, 𝐞₃ = unitvec(coorsys_cartesian())
# 𝟏, 𝟙, 𝕀, 𝕁, 𝕂 = init_isotropic(basis = ℬˢ)
𝕀, 𝕁, 𝕂 = ISO(Val(3),Val(Sym))
𝟏 = tensId2(Val(3),Val(Sym))
k, μ = symbols("k μ", positive = true)
λ = k -2μ/3

u = SymFunction("u", real = true)
𝐮ˢᵖʰ = u(r) * 𝐞ʳ
𝛆ˢᵖʰ = SYMGRAD(𝐮ˢᵖʰ, Spherical)
𝛔ˢᵖʰ = λ * tr(𝛆ˢᵖʰ) * 𝟏 + 2μ * 𝛆ˢᵖʰ
𝐓ˢᵖʰ = 𝛔ˢᵖʰ ⋅ 𝐞ʳ
div𝛔ˢᵖʰ = DIV(𝛔ˢᵖʰ, Spherical) ;
eqˢᵖʰ = factor(simplify(div𝛔ˢᵖʰ ⋅ 𝐞ʳ))
solˢᵖʰ = dsolve(eqˢᵖʰ, u(r))
ûˢᵖʰ = solˢᵖʰ.rhs()
T̂ˢᵖʰ = factor(simplify(subs(𝐓ˢᵖʰ ⋅ 𝐞ʳ, u(r) => ûˢᵖʰ)))

𝐄 = 𝟏 - 3𝐞₃⊗𝐞₃
fᶿ = simplify(𝐞ᶿ ⋅ 𝐄 ⋅ 𝐞ʳ)
fʳ = simplify(𝐞ʳ ⋅ 𝐄 ⋅ 𝐞ʳ)
# fᶿ = SymFunction("fᶿ", real = true)(θ)
# fʳ = SymFunction("fʳ", real = true)(θ)
uᶿ = SymFunction("uᶿ", real = true)
uʳ = SymFunction("uʳ", real = true)
𝐮ᵈᵉᵛ = uᶿ(r) * fᶿ * 𝐞ᶿ + uʳ(r) * fʳ * 𝐞ʳ
𝛆ᵈᵉᵛ = SYMGRAD(𝐮ᵈᵉᵛ, Spherical)
𝛔ᵈᵉᵛ = λ * tr(𝛆ᵈᵉᵛ) * 𝟏 + 2μ * 𝛆ᵈᵉᵛ
𝐓ᵈᵉᵛ = 𝛔ᵈᵉᵛ ⋅ 𝐞ʳ
div𝛔ᵈᵉᵛ = simplify(DIV(𝛔ᵈᵉᵛ, Spherical))
eqᶿᵈᵉᵛ = factor(simplify(div𝛔ᵈᵉᵛ ⋅ 𝐞ᶿ / fᶿ))
eqʳᵈᵉᵛ = factor(simplify(div𝛔ᵈᵉᵛ ⋅ 𝐞ʳ / fʳ))
α, Λ = symbols("α Λ", real = true)
eqᵈᵉᵛ = factor(simplify(subs([eqᶿᵈᵉᵛ,eqʳᵈᵉᵛ], uᶿ(r) => r^α, uʳ(r) => Λ*r^α)))
αΛ = solve([eq.doit() for eq ∈ eqᵈᵉᵛ], [α, Λ])
ûᶿᵈᵉᵛ = sum([Sym("C$(i+2)") * r^αΛ[i][1] for i ∈ 1:length(αΛ)])
ûʳᵈᵉᵛ = sum([Sym("C$(i+2)") * αΛ[i][2] * r^αΛ[i][1] for i ∈ 1:length(αΛ)])
T̂ᶿᵈᵉᵛ = factor(simplify(subs(simplify(𝐓ᵈᵉᵛ ⋅ 𝐞ᶿ / fᶿ), uᶿ(r) => ûᶿᵈᵉᵛ, uʳ(r) => ûʳᵈᵉᵛ)))
T̂ʳᵈᵉᵛ = factor(simplify(subs(simplify(𝐓ᵈᵉᵛ ⋅ 𝐞ʳ / fʳ), uᶿ(r) => ûᶿᵈᵉᵛ, uʳ(r) => ûʳᵈᵉᵛ)))
