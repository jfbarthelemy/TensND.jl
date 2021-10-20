module TensND

import Base.@pure
import Base: eltype
import LinearAlgebra: normalize, dot, tr

using LinearAlgebra, SymPy, Tensors, OMEinsum, Rotations

export contract, qcontract, otimesu, otimesul, sboxtimes, sotimes, ⊙, ⊠, ⊠ˢ, ⊗ˢ

export Basis, CanonicalBasis, RotatedBasis, CylindricalBasis, SphericalBasis, OrthogonalBasis, AllOrthogonalBasis
export getdim, vecbasis, metric, angles, isorthogonal, isorthonormal, isidentity

export AbstractTens, Tens
export getorder, arraytype, getdata, getarray, getbasis, getvar
export components, components_canon, change_tens, change_tens_canon
export trigsimp, expand_trig
export KM, invKM
export getbasis, getvar

export TensISO, tensId2, tensId4, tensJ4, tensK4, ISO, isotropify, isISO
export t𝟏, t𝕀, t𝕁, t𝕂

export proj_tens

export LeviCivita, 𝐞, 𝐞ᵖ, 𝐞ᶜ, 𝐞ˢ, rot2, rot3, rot6

export ∂, CoorSystemSym, getChristoffel, GRAD, SYMGRAD, DIV, LAPLACE, HESS
export get_normalized_basis, get_natural_basis, natvec, unitvec, getcoords, getOM
export init_cartesian, init_polar, init_cylindrical, init_spherical, init_rotated
export CS_cartesian, CS_polar, CS_cylindrical, CS_spherical, CS_spheroidal


include("array_utils.jl")
include("bases.jl")
include("tens.jl")
include("tens_isotropic.jl")
include("special_tens.jl")
include("coorsystems.jl")


# function __init__()
# end

end # module
