module TensND

using LinearAlgebra, SymPy, Tensors, OMEinsum
import LinearAlgebra: normalize
import Base: eltype

export AbstractBasis, Basis, CanonicalBasis
export eltype
export basis, metric
export normal_basis, normalize, isorthogonal
export fϵ, ϵ

include("bases.jl")
include("special_tens.jl")

end # module
