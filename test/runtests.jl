if lowercase(get(ENV, "CI", "false")) == "true"
    include("install_dependencies.jl")
end

using TensND
using Test
using TimerOutputs
using LinearAlgebra, SymPy, Tensors, OMEinsum, Rotations

macro testsection(str, block)
    return quote
        @timeit "$($(esc(str)))" begin
            @testset "$($(esc(str)))" begin
                $(esc(block))
            end
        end
    end
end

reset_timer!()

opequal(x,y) = x ≈ y
opequal(x::Sym,y) = x == y
opequal(x::AbstractArray{Sym},y) = x == y


include("test_isotropic_arrays.jl")
include("test_bases.jl")
# include("test_tens.jl")
# include("test_special_tens.jl")
# include("test_coorsystems.jl")

print_timer()
println()

