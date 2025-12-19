using TensND, LinearAlgebra, SymPy, Tensors, OMEinsum, Rotations
sympy.init_printing(use_unicode=true)

𝕀, 𝕁, 𝕂 = ISO(Val(3),Val(Sym))

k, μ = symbols("k μ", positive = true)
ℂ = 3k*𝕁+2μ*𝕂
ℂ = Tens(SymmetricTensor{4,3}((i, j, k, l) -> symbols("C$i$j$k$l", real = true)))

μ = simplify((ℂ ⊙ 𝕂)/10)
k = (ℂ ⊙ 𝕁)/3
λ = k-2μ/3
