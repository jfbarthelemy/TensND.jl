using TensND, LinearAlgebra, SymPy, Tensors, OMEinsum, Rotations
sympy.init_printing(use_unicode=true)

CS = coorsys_cylindrical() ; r, θ, z = getcoords(CS) ; 𝐞ʳ, 𝐞ᶿ, 𝐞ᶻ = unitvec(CS) ;
ℬ = get_normalized_basis(CS)
@set_coorsys CS

ξʳ, ξᶻ = SymFunction("ξʳ, ξᶻ", real = true)
𝛏 = ξʳ(r,z)*𝐞ʳ+ξᶻ(r,z)*𝐞ᶻ
𝛜 = SYMGRAD(𝛏)
