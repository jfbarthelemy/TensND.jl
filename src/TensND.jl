module TensND

import Base.@pure
import Base: eltype
import LinearAlgebra: normalize, dot

using LinearAlgebra, SymPy, Tensors, OMEinsum, Rotations

export contract, qcontract, otimesu, otimesul, sboxtimes, sotimes, ⊙, ⊠, ⊠ˢ, ⊗ˢ

export Id2, Isotropic2, Id4, I4, J4, K4, Isotropic4
export KM

export Basis, CanonicalBasis, RotatedBasis, CylindricalBasis, SphericalBasis, OrthogonalBasis, AllOrthogonalBasis
export getdim, vecbasis, metric, angles, isorthogonal, isorthonormal, isidentity

export AbstractTens, Tens
export getorder, arraytype, getdata, getarray, getbasis, getvar
export components, components_canon, change_tens, change_tens_canon
export trigsimp, expand_trig
export KM, invKM
export getbasis, getvar

# natvec, unitvec, getcoords, getOM

# export LeviCivita
# export tensId2, tensId4, tensId4s, tensJ4, tensK4
# export t𝟏, t𝟙, t𝕀, t𝕁, t𝕂
# # export 𝐞, 𝐞ᵖ, 𝐞ᶜ, 𝐞ˢ
# export init_isotropic
# export rot2, rot3, rot6

# export ∂, CoorSystemSym, GRAD, SYMGRAD, DIV, LAPLACE, HESS
# export init_cartesian, init_polar, init_cylindrical, init_spherical, init_rotated
# export CS_cartesian, CS_polar, CS_cylindrical, CS_spherical, CS_spheroidal


include("array_utils.jl")
# include("isotropic_arrays.jl")
include("bases.jl")
include("tens.jl")
# include("special_tens.jl")
# include("coorsystems.jl")


# function __init__()
# end

end # module
