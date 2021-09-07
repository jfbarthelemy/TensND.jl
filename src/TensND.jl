module TensND

import Base.@pure
import Base: eltype
import LinearAlgebra: normalize, dot

using LinearAlgebra, SymPy, Tensors, OMEinsum, Rotations

export AbstractBasis, Basis, CanonicalBasis, RotatedBasis, CylindricalBasis
export angles
export eltype
export vecbasis, metric
export normal_basis, normalize, isorthogonal, isorthonormal
export Tensnd, components, components_canon
export KM, invKM
export qcontract, otimesul, ⊙, ⊠, ⊠ˢ, ⊗ˢ#, ⊗̅, ⊗̲, ⊗̲̅

export fϵ, ϵ
export tensId2, tensId4, tensId4s, tensJ4, tensK4
export t𝟏, t𝟙, t𝕀, t𝕁, t𝕂
export 𝐞, 𝐞ᵖ, 𝐞ᶜ, 𝐞ˢ
export 𝐞𝐞, 𝐞𝐞ˢ, 𝐞ᵖ𝐞ᵖ, 𝐞ᵖ𝐞ᵖs, 𝐞ᶜ𝐞ᶜ, 𝐞ᶜ𝐞ᶜs, 𝐞ˢ𝐞ˢ, 𝐞ˢ𝐞ˢs

include("bases.jl")
include("tensnd_struct.jl")
include("special_tens.jl")
include("coorsystems.jl")

end # module
