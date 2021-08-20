module TensND

import Base.@pure
import Base: eltype
import LinearAlgebra: normalize

using LinearAlgebra, SymPy, Tensors, OMEinsum, Rotations

export AbstractBasis, Basis, CanonicalBasis
export eltype
export vecbasis, metric
export normal_basis, normalize, isorthogonal
export fϵ, ϵ
export Tensnd, components


include("bases.jl")
include("special_tens.jl")
include("tensnd_struct.jl")

end # module
