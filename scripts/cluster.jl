using TensND, LinearAlgebra, SymPy, Tensors, OMEinsum, Rotations
sympy.init_printing(use_unicode=true)

θ, ϕ, r = (symbols("θ", real=true), symbols("ϕ", real=true), symbols("r", positive=true))
S = coorsys_spherical((θ, ϕ, r))
𝐞ᶿ, 𝐞ᵠ, 𝐞ʳ = unitvec(S)
𝐱 = getOM(S)

θ′, ϕ′, r′ = (symbols("θ′", real=true), symbols("ϕ′", real=true), symbols("r′", positive=true))
S′ = coorsys_spherical((θ′, ϕ′, r′))
𝐞ᶿ′, 𝐞ᵠ′, 𝐞ʳ′ = unitvec(S′)

Cartesian = coorsys_cartesian(symbols("x y z", real = true))
𝐞₁, 𝐞₂, 𝐞₃ = unitvec(Cartesian)

R = symbols("R", positive = true)

𝐱′ = getOM(S′) + R * 𝐞₃

𝕀, 𝕁, 𝕂 = ISO(Val(3),Val(Sym))
𝟏 = tensId2(Val(3),Val(Sym))

E, k, μ = symbols("E k μ", positive = true)
ν = symbols("ν", real = true)
k = E / (3(1-2ν)) ; μ = E / (2(1+ν))
λ = k -2μ/3

Δ𝐱 = 𝐱-𝐱′
ρ = tsimplify(norm(Δ𝐱))
𝐍 = Δ𝐱/ρ
𝐧 = tsimplify(change_tens_canon(𝐍))

ℾ = 1/(16PI * μ * (1-ν) * ρ^3) * (-3𝕁 +2(1-2ν)*𝕀 + 3(𝟏⊗𝐧⊗𝐧 + 𝐧⊗𝐧⊗𝟏) + 12ν*𝐧⊗ˢ𝟏⊗ˢ𝐧 -15𝐧⊗𝐧⊗𝐧⊗𝐧)

