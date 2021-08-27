module TensND

import Base.@pure
import Base: eltype
import LinearAlgebra: normalize, dot

using LinearAlgebra, SymPy, Tensors, OMEinsum, Rotations

export AbstractBasis, Basis, CanonicalBasis, RotatedBasis
export eltype
export vecbasis, metric
export normal_basis, normalize, isorthogonal
export Tensnd, components
export KM, invKM

export fϵ, ϵ
export tensId2, tensId4, tensJ4, tensK4

include("bases.jl")
include("tensnd_struct.jl")
include("special_tens.jl")

end # module
