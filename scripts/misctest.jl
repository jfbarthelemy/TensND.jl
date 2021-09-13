using Revise, TensND, LinearAlgebra, SymPy, Tensors, OMEinsum, Rotations, Test

Cartesian, 𝐗, 𝐄, ℬ = CS_cartesian()
𝛔 = Tensnd(SymmetricTensor{2,3}((i, j) -> SymFunction("σ$i$j", real = true)(𝐗...)))
@show DIV(𝛔, Cartesian)
@show sum([sum([diff(getdata(𝛔)[i, j], 𝐗[j]) for j ∈ 1:3]) * 𝐄[i] for i ∈ 1:3])

# Polar
Polar, (r, θ), (𝐞ʳ, 𝐞ᶿ), ℬᵖ = CS_polar()
f = SymFunction("f", real = true)(r, θ)
@show simplify(LAPLACE(f, Polar))
@show simplify(diff(r * diff(f, r), r) / r + diff(f, θ, θ) / r^2)

# Cylindrical
Cylindrical, rθz, (𝐞ʳ, 𝐞ᶿ, 𝐞ᶻ), ℬᶜ = CS_cylindrical()
𝐯 = Tensnd(Vec{3}(i -> SymFunction("v$(rθz[i])", real = true)(rθz...)), ℬᶜ)
vʳ, vᶿ, vᶻ = getdata(𝐯)
@show simplify(DIV(𝐯, Cylindrical))
@show simplify(diff(vʳ, r) + vʳ / r + diff(vᶿ, θ) / r + diff(vᶻ, z))

# Spherical
Spherical, (θ, ϕ, r), (𝐞ᶿ, 𝐞ᵠ, 𝐞ʳ), ℬˢ = CS_spherical()
for σⁱʲ ∈ ("σʳʳ", "σᶿᶿ", "σᵠᵠ")
    @eval $(Symbol(σⁱʲ)) = SymFunction($σⁱʲ, real = true)($r)
end
𝛔 = σʳʳ * 𝐞ʳ ⊗ 𝐞ʳ + σᶿᶿ * 𝐞ᶿ ⊗ 𝐞ᶿ + σᵠᵠ * 𝐞ᵠ ⊗ 𝐞ᵠ
div𝛔 = DIV(𝛔, Spherical)
@show simplify(div𝛔 ⋅ 𝐞ʳ)
@show simplify(diff(σʳʳ, r) + (2 * σʳʳ - σᶿᶿ - σᵠᵠ) / r)

Spherical, (θ, ϕ, r), (𝐞ᶿ, 𝐞ᵠ, 𝐞ʳ), ℬˢ = CS_spherical() ;
𝟏, 𝟙, 𝕀, 𝕁, 𝕂 = init_isotropic() ;
k, μ = symbols("k μ", positive = true) ; ℂ = 3k * 𝕁 + 2μ * 𝕂 ;
u = SymFunction("u", real = true)(r) ;
𝐮 = u * 𝐞ʳ ;
𝛆 = SYMGRAD(𝐮, Spherical) ;
𝛔 = ℂ ⊡ 𝛆 ;
dsolve(simplify(DIV(𝛔, Spherical) ⋅ 𝐞ʳ), u)

Spherical, (θ, ϕ, r), (𝐞ᶿ, 𝐞ᵠ, 𝐞ʳ), ℬˢ = CS_spherical() ;
𝟏, 𝟙, 𝕀, 𝕁, 𝕂 = init_isotropic() ;
k, μ = symbols("k μ", positive = true) ; ℂ = 3k * 𝕁 + 2μ * 𝕂 ;
u = SymFunction("u", real = true)(r) ;
dsolve(simplify(DIV(ℂ ⊡ GRAD(u * 𝐞ʳ, Spherical), Spherical) ⋅ 𝐞ʳ), u)

# Spheroidal
ϕ = symbols("ϕ", real = true)
p = symbols("p", real = true)
p̄ = √(1 - p^2)
q = symbols("q", positive = true)
q̄ = √(q^2 - 1)
c = symbols("c", positive = true)
coords = (ϕ, p, q)
OM = Tensnd(c * [p̄ * q̄ * cos(ϕ), p̄ * q̄ * sin(ϕ), p * q])
rules = Dict(sqrt(1-p^2) * sqrt(q^2-1) => sqrt(-(p^2 - 1)*(q^2 - 1)), sqrt((p^2 - q^2)/(p^2 - 1))*sqrt(1 - p^2) => sqrt(q^2 - p^2))
Spheroidal = CoorSystemSym(OM, coords; rules = rules)

