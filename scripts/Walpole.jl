using TensND, LinearAlgebra, SymPy, Tensors, OMEinsum, Rotations, Latexify
sympy.init_printing(use_unicode=true)

𝕀, 𝕁, 𝕂 = ISO(Val(3),Val(Sym))
𝟏 = tensId2(Val(3),Val(Sym))

E, k, μ = symbols("E k μ", positive = true)
ν = symbols("ν", real = true)
k = E / (3(1-2ν)) ; μ = E / (2(1+ν))
λ = k -2μ/3

θ, ϕ, ψ = symbols("θ ϕ ψ", real = true) ;
cθ, cϕ, cψ, sθ, sϕ, sψ = symbols("cθ cϕ cψ sθ sϕ sψ", real = true) ;
d = Dict(cos(θ) => cθ, cos(ϕ) => cϕ, cos(ψ) => cψ, sin(θ) => sθ, sin(ϕ) => sϕ, sin(ψ) => sψ) ;
R = Tens(tsubs(rot3(θ, ϕ, ψ),d...)) ;
R6 = invKM(tsubs(KM(rot6(θ, ϕ, ψ)),d...)) ;

for i ∈ 1:3, j ∈ 1:3
    @eval $(Symbol("R$i$j")) = symbols($"R$(TensND.subscriptnumber(i))$(TensND.subscriptnumber(j))", real = true)
end
R = Tens(Tensor{2,3}((i, j) -> eval(Symbol("R$i$j"))))
