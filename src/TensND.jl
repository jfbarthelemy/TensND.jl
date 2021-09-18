module TensND

import Base.@pure
import Base: eltype
import LinearAlgebra: normalize, dot

using LinearAlgebra, SymPy, Tensors, OMEinsum, Rotations

export Id2, Isotropic2, Id4, I4, J4, K4, Isotropic4

export Basis, CanonicalBasis, RotatedBasis, CylindricalBasis, SphericalBasis, OrthogonalBasis, AllOrthogonalBasis
export dim, vecbasis, metric, angles, isorthogonal, isorthonormal, isidentity

# export Tens
# export ndims, arraytype, array, basis
# export components, components_canon, change_tens, change_tens_canon
# export trigsimp, expand_trig
# export KM, invKM
# export contract, qcontract, otimesul, ⊙, ⊠, ⊠ˢ, ⊗ˢ
# export getdata, getbasis, getvar, natvec, unitvec, getcoords, getOM

# export LeviCivita
# export tensId2, tensId4, tensId4s, tensJ4, tensK4
# export t𝟏, t𝟙, t𝕀, t𝕁, t𝕂
# # export 𝐞, 𝐞ᵖ, 𝐞ᶜ, 𝐞ˢ
# export init_isotropic
# export rot2, rot3, rot6

# export ∂, CoorSystemSym, GRAD, SYMGRAD, DIV, LAPLACE, HESS
# export init_cartesian, init_polar, init_cylindrical, init_spherical, init_rotated
# export CS_cartesian, CS_polar, CS_cylindrical, CS_spherical, CS_spheroidal


include("isotropic_arrays.jl")
include("bases.jl")
# include("tens.jl")
# include("special_tens.jl")
# include("coorsystems.jl")


# function __init__()
# end

end # module
