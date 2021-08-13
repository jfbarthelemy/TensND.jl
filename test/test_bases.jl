@testsection "Bases" begin
    for T ∈ (Sym, Float64)
        @testsection "type $T" begin
            sq2 = √T(2)
            v = T[1 0 0; 0 1 0; 0 1 1]
            bv = Basis(v)
            @test eltype(bv) == T
            @test basis(bv) == bv.e == T[1 0 0; 0 1 0; 0 1 1]
            @test basis(bv, :cont) == bv.E == T[1 0 0; 0 1 -1; 0 0 1]
            @test metric(bv) == bv.g == T[1 0 0; 0 2 1; 0 1 1]
            @test metric(bv, :cont) == bv.G == T[1 0 0; 0 1 -1; 0 -1 2]
            @test !isorthogonal(bv)
            nb = normal_basis(v)
            if T == Sym
                @test basis(nb) == nb.e == T[1 0 0; 0 sq2/2 0; 0 sq2/2 1]
                θ, ϕ, ψ = symbols("θ, ϕ, ψ", real = true)
                br = Basis(θ, ϕ, ψ)
                @test metric(br) == I
            else
                @test basis(nb) ≈ nb.e ≈ T[1 0 0; 0 sq2/2 0; 0 sq2/2 1]
            end

            bw = Basis(bv.E, :cont)
            @test eltype(bw) == T
            @test basis(bw) == bw.e == bv.e == T[1 0 0; 0 1 0; 0 1 1]
            @test basis(bw, :cont) == bw.E == bv.E == T[1 0 0; 0 1 -1; 0 0 1]
            @test metric(bw) == bw.g == bv.g == T[1 0 0; 0 2 1; 0 1 1]
            @test metric(bw, :cont) == bw.G == bv.G == T[1 0 0; 0 1 -1; 0 -1 2]
            @test !isorthogonal(bw)
        end
    end
end
