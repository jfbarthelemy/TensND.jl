module TensND

import Base.@pure
import Base: eltype
import LinearAlgebra: normalize, dot

using LinearAlgebra, SymPy, Tensors, OMEinsum, Rotations

export AbstractBasis, Basis, CanonicalBasis, RotatedBasis
export angles
export eltype
export vecbasis, metric
export normal_basis, normalize, isorthogonal, isorthonormal
export Tensnd, components
export KM, invKM
export qcontract, otimesul, ⊙, ⊠, ⊠ˢ, ⊗ˢ, ⊗̅, ⊗̲, ⊗̲̅

export fϵ, ϵ
export tensId2, tensId4, tensId4s, tensJ4, tensK4
export t𝟏, t𝟙, t𝕀, t𝕁, t𝕂

include("bases.jl")
include("tensnd_struct.jl")
include("special_tens.jl")

end # module
