if lowercase(get(ENV, "CI", "false")) == "true"
    include("install_dependencies.jl")
end

using TensND
using Test
using LinearAlgebra, SymPy


@testset "TensND.jl" begin
    v = Sym[1 0 0; 0 1 0; 0 1 1]
    b = Basis(v)
    @test eltype(b) == Sym
    @test basis(b) == b.e == Sym[1 0 0; 0 1 0; 0 1 1]
    @test basis(b, :cont) == b.E == Sym[1 0 0; 0 1 -1; 0 0 1]
    @test metric(b) == b.g == Sym[1 0 0; 0 2 1; 0 1 1]
    @test metric(b, :cont) == b.G == Sym[1 0 0; 0 1 -1; 0 -1 2]
    @test !isorthogonal(b)
end
