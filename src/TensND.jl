module TensND

import Base.@pure
import Base: eltype
import LinearAlgebra: normalize

using LinearAlgebra, SymPy, Tensors, OMEinsum, Rotations

export AbstractBasis, Basis, CanonicalBasis
export eltype
export basis, metric
export normal_basis, normalize, isorthogonal
export fϵ, ϵ

include("bases.jl")
include("special_tens.jl")

end # module