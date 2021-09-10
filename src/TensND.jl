module TensND

import Base.@pure
import Base: eltype
import LinearAlgebra: normalize, dot

using LinearAlgebra, SymPy, Tensors, OMEinsum, Rotations

export AbstractBasis, Basis, CanonicalBasis, RotatedBasis, CylindricalBasis
export angles
export eltype
export vecbasis, metric
export isorthogonal, isorthonormal
export Tensnd, components, components_canon, change_tens, change_tens_canon
export KM, invKM
export qcontract, otimesul, ⊙, ⊠, ⊠ˢ, ⊗ˢ#, ⊗̅, ⊗̲, ⊗̲̅

export LeviCivita
export tensId2, tensId4, tensId4s, tensJ4, tensK4
export t𝟏, t𝟙, t𝕀, t𝕁, t𝕂
export 𝐞, 𝐞ᵖ, 𝐞ᶜ, 𝐞ˢ
export init_canonical, init_isotropic, init_polar, init_cylindrical, init_spherical, init_rotated
export rot2, rot3, rot6

export ∂
export CoorSystemSym


include("bases.jl")
include("tensnd_struct.jl")
include("special_tens.jl")
include("coorsystems.jl")


# function __init__()
# end

end # module
