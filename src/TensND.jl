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
export qcontract, otimesul, ⊙, ⊠, ⊠ˢ, ⊗ˢ#, ⊗̅, ⊗̲, ⊗̲̅

export fϵ, ϵ
export tensId2, tensId4, tensId4s, tensJ4, tensK4
export t𝟏, t𝟙, t𝕀, t𝕁, t𝕂
export 𝐞, 𝐞p, 𝐞c, 𝐞s
export 𝐞𝐞, 𝐞𝐞s, 𝐞p𝐞p, 𝐞p𝐞ps, 𝐞c𝐞c, 𝐞c𝐞cs, 𝐞s𝐞s, 𝐞s𝐞ss

include("bases.jl")
include("tensnd_struct.jl")
include("special_tens.jl")
include("coorsystems.jl")

end # module
