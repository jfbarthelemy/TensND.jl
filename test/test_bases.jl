@testsection "Bases" begin
    for T ∈ (Sym, Float64)
        @testsection "type $T" begin
            sq2 = √T(2)
            v = T[1 0 0; 0 1 0; 0 1 1]
            bv = Basis(v)
            @test eltype(bv) == T
            @test vecbasis(bv) == bv.e == T[1 0 0; 0 1 0; 0 1 1]
            @test vecbasis(bv, :cont) == bv.E == T[1 0 0; 0 1 -1; 0 0 1]
            @test metric(bv) == bv.g == T[1 0 0; 0 2 1; 0 1 1]
            @test metric(bv, :cont) == bv.G == T[1 0 0; 0 1 -1; 0 -1 2]
            @test simplify.(bv.E' ⋅ bv.e) == I
            @test simplify.(bv.G' ⋅ bv.g) == I
            @test !isorthogonal(bv)

            bw = Basis(bv.E, :cont)
            @test eltype(bw) == T
            @test vecbasis(bw) == bw.e == bv.e == T[1 0 0; 0 1 0; 0 1 1]
            @test vecbasis(bw, :cont) == bw.E == bv.E == T[1 0 0; 0 1 -1; 0 0 1]
            @test metric(bw) == bw.g == bv.g == T[1 0 0; 0 2 1; 0 1 1]
            @test metric(bw, :cont) == bw.G == bv.G == T[1 0 0; 0 1 -1; 0 -1 2]
            @test simplify.(bv.E' ⋅ bv.e) == I
            @test simplify.(bv.G' ⋅ bv.g) == I
            @test !isorthogonal(bw)

            nb = normal_basis(v)
            if T == Sym
                @test vecbasis(nb) == nb.e == T[1 0 0; 0 sq2/2 0; 0 sq2/2 1]
                θ, ϕ, ψ = symbols("θ, ϕ, ψ", real = true)
                br = Basis(θ, ϕ, ψ)
                @test metric(br) == I
                @test br.e == br.E
            else
                @test vecbasis(nb) ≈ nb.e ≈ T[1 0 0; 0 sq2/2 0; 0 sq2/2 1]
            end

            if T ≠ Sym
                v = rand(T, 3, 3)
                while det(v) ≈ 0
                    v = rand(T, 3, 3)
                end
                @test eltype(bv) == T
                @test vecbasis(bv) == bv.e
                @test vecbasis(bv, :cont) == bv.E
                @test metric(bv) == bv.g
                @test metric(bv, :cont) == bv.G
                @test bv.E' ⋅ bv.e == I
                @test bv.G' ⋅ bv.g == I
            end

        end
    end
end
