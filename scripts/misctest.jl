using Revise, TensND, LinearAlgebra, SymPy, Tensors, OMEinsum, Rotations, Test

(x, y, z), (𝐞₁, 𝐞₂, 𝐞₃), ℬ = init_cartesian()
for i ∈ 1:3, j ∈ 1:3
    @eval $(Symbol("σ$i$j")) = SymFunction($"σ$i$j", real = true)(coords...)
end
𝛔 = Tensnd(SymmetricTensor{2,3}((i, j) -> eval(Symbol("σ$i$j"))))
OM = x * 𝐞₁ + y * 𝐞₂ + z * 𝐞₃
CS = CoorSystemSym(OM, coords)
div𝛔 = DIV(𝛔, CS)


coords, vectors, ℬᶜ = init_cylindrical()
r, θ, z = coords
𝐞ʳ, 𝐞ᶿ, 𝐞ᶻ = vectors
OM = r * 𝐞ʳ + z * 𝐞ᶻ
for i ∈ 1:3
    @eval $(Symbol("v$(coords[i])")) = SymFunction($"v$(coords[i])", real = true)(coords...)
end
𝐯 = Tensnd(Vec{3}((i) -> eval(Symbol("v$(coords[i])"))), ℬᶜ)
CS = CoorSystemSym(OM, coords)
div𝐯 = DIV(𝐯, CS)