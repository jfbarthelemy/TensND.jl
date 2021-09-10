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

include("test_bases.jl")
include("test_tensnd.jl")
include("test_special_tens.jl")

print_timer()
println()

