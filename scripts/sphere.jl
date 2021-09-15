using TensND, LinearAlgebra, SymPy, Tensors, OMEinsum, Rotations
sympy.init_printing(use_unicode=true)

Spherical = CS_spherical()
θ, ϕ, r = getcoords(Spherical)
𝐞ᶿ, 𝐞ᵠ, 𝐞ʳ = unitvec(Spherical)
𝐱 = getOM(Spherical)
𝐞₁, 𝐞₂, 𝐞₃ = unitvec(CS_cartesian())

𝟏, 𝟙, 𝕀, 𝕁, 𝕂 = init_isotropic()
k, μ = symbols("k μ", positive = true)
λ = k -2μ/3

u = SymFunction("u", real = true)
𝐮ˢᵖʰ = u(r) * 𝐞ʳ
𝛆ˢᵖʰ = tenssimp(SYMGRAD(𝐮ˢᵖʰ, Spherical))
𝛔ˢᵖʰ = tenssimp(λ * tr(𝛆ˢᵖʰ) * 𝟏 + 2μ * 𝛆ˢᵖʰ)
𝐓ˢᵖʰ = tenssimp(𝛔ˢᵖʰ ⋅ 𝐞ʳ)
div𝛔ˢᵖʰ = DIV(𝛔ˢᵖʰ, Spherical) ;
eqˢᵖʰ = factor(simplify(div𝛔ˢᵖʰ ⋅ 𝐞ʳ))
solˢᵖʰ = dsolve(eqˢᵖʰ, u(r))
ûˢᵖʰ = solˢᵖʰ.rhs()
T̂ˢᵖʰ = factor(simplify(subs(𝐓ˢᵖʰ ⋅ 𝐞ʳ, u(r) => ûˢᵖʰ)))

𝐄 = 𝟏 - 3𝐞₃⊗𝐞₃
fᶿ = 𝐞ᶿ ⋅ 𝐄 ⋅ 𝐞ʳ
fʳ = 𝐞ʳ ⋅ 𝐄 ⋅ 𝐞ʳ
uᶿ = SymFunction("uᶿ", real = true)
uʳ = SymFunction("uʳ", real = true)
𝐮ᵈᵉᵛ = uᶿ(r)* fᶿ * 𝐞ᶿ + uʳ(r)* fʳ * 𝐞ʳ
𝛆ᵈᵉᵛ = tenssimp(SYMGRAD(𝐮ᵈᵉᵛ, Spherical))
𝛔ᵈᵉᵛ = tenssimp(λ * tr(𝛆ᵈᵉᵛ) * 𝟏 + 2μ * 𝛆ᵈᵉᵛ)
𝐓ᵈᵉᵛ = tenssimp(𝛔ᵈᵉᵛ ⋅ 𝐞ʳ)
div𝛔ᵈᵉᵛ = DIV(𝛔ᵈᵉᵛ, Spherical) ;
eqᶿᵈᵉᵛ = factor(simplify(div𝛔ᵈᵉᵛ ⋅ 𝐞ᶿ / fᶿ))
eqʳᵈᵉᵛ = factor(simplify(div𝛔ᵈᵉᵛ ⋅ 𝐞ʳ / fʳ))
α, Λ = symbols("α Λ", real = true)
eqᵈᵉᵛ = simplify.(subs.([eqᶿᵈᵉᵛ,eqʳᵈᵉᵛ], uᶿ(r) => r^α, uʳ(r) => Λ*r^α))
αΛ = solve(eqᵈᵉᵛ, [α, Λ])
ûᶿᵈᵉᵛ = sum([Sym("C$(i+2)") * r^αΛ[i][1] for i ∈ 1:length(αΛ)])
ûʳᵈᵉᵛ = sum([Sym("C$(i+2)") * αΛ[i][2] * r^αΛ[i][1] for i ∈ 1:length(αΛ)])
T̂ᶿᵈᵉᵛ = factor(simplify(subs(𝐓ᵈᵉᵛ ⋅ 𝐞ᶿ / fᶿ, uᶿ(r) => ûᶿᵈᵉᵛ, uʳ(r) => ûʳᵈᵉᵛ)))
T̂ʳᵈᵉᵛ = factor(simplify(subs(𝐓ᵈᵉᵛ ⋅ 𝐞ʳ / fʳ, uᶿ(r) => ûᶿᵈᵉᵛ, uʳ(r) => ûʳᵈᵉᵛ)))




