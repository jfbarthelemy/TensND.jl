module TensND

import Base: @pure, eltype
import LinearAlgebra: normalize, dot, tr

using LinearAlgebra, SymPy, Tensors, OMEinsum, Rotations

include("array_utils.jl")
include("bases.jl")
include("tens.jl")
include("tens_isotropic.jl")
include("special_tens.jl")
include("coorsystems.jl")


# function __init__()
# end

end # module
