using Revise, TensND, LinearAlgebra, SymPy, Tensors, OMEinsum, Rotations, Test


# Spheroidal
ϕ = symbols("ϕ", real = true)
p = symbols("p", real = true)
p̄ = √(1 - p^2)
q = symbols("q", positive = true)
q̄ = √(q^2 - 1)
c = symbols("c", positive = true)
coords = (ϕ, p, q)
OM = Tens(c * [p̄ * q̄ * cos(ϕ), p̄ * q̄ * sin(ϕ), p * q])
rules = Dict(
    sqrt(1 - p^2) * sqrt(q^2 - 1) => sqrt(-(p^2 - 1) * (q^2 - 1)),
    sqrt((p^2 - q^2) / (p^2 - 1)) * sqrt(1 - p^2) => sqrt(q^2 - p^2),
)
rules = Dict(
    sqrt(-(p^2 - 1) * (q^2 - 1)) => sqrt(1 - p^2) * sqrt(q^2 - 1),
    sqrt((p^2 - q^2) / (p^2 - 1)) * sqrt(1 - p^2) => sqrt(q^2 - p^2),
)
Spheroidal = CoorSystemSym(OM, coords; rules = rules)



ϕ, p = symbols("ϕ p", real = true);
p̄, q, q̄, c = symbols("p̄ q q̄ c", positive = true);
coords = (ϕ, p, q);
tmp_coords = (p̄, q̄);
params = (c,);
OM = Tens(c * [p̄ * q̄ * cos(ϕ), p̄ * q̄ * sin(ϕ), p * q]);
Spheroidal = CoorSystemSym(
    OM,
    coords,
    tmp_coords,
    params;
    tmp_var = Dict(1 - p^2 => p̄^2, q^2 - 1 => q̄^2),
    to_coords = Dict(p̄ => √(1 - p^2), q̄ => √(q^2 - 1)),
);
simplify(LAPLACE(OM[1]^2, Spheroidal))
m = 2;
n = 5;
P = sympy.assoc_legendre;
T = P(n, m, p) * P(n, m, q) * cos(m * ϕ);
simplify(LAPLACE(T, Spheroidal))



θ, ϕ, R = symbols("θ ϕ", real = true)..., symbols("R", positive = true)
OM =  Tens(R*[sin(θ)*cos(ϕ), sin(θ)*sin(ϕ), cos(θ)])
SM = TensND.SubManifoldSym(OM, (θ,ϕ); rules = Dict(abs(sin(θ)) => sin(θ)))
𝐞ᶿ, 𝐞ᵠ, 𝐞ʳ = unitvec(SM)
@set_coorsys SM
GRAD(𝐞ʳ) |> intrinsic
GRAD(𝐞ʳ) == -curvature(SM)
GRAD(𝐞ʳ) + curvature(SM) |> intrinsic

x, y = symbols("x y", real = true)
OM =  Tens([x,y,x^2+y^2-x*y])
SM = TensND.SubManifoldSym(OM, (x,y))
𝐄ˣ, 𝐄ʸ, 𝐍 = unitvec(SM)
@set_coorsys SM

x, y = symbols("x y", real = true)
α, β, γ = symbols("α β γ", positive = true)
OM =  Tens([x,y,γ*√(1-(x/α)^2-(y/β)^2)])
SM = TensND.SubManifoldSym(OM, (x,y))
𝐄ˣ, 𝐄ʸ, 𝐍 = unitvec(SM)
@set_coorsys SM
