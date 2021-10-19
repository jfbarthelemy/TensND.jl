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
