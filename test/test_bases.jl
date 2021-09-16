@testsection "Bases" begin
    for T ∈ (Sym, Float64)
        @testsection "type $T" begin
            sq2 = √T(2)
            v = T[1 0 0; 0 1 0; 0 1 1]
            bv = Basis(v)
            @test eltype(bv) == T
            @test vecbasis(bv) == T[1 0 0; 0 1 0; 0 1 1]
            @test vecbasis(bv, :cont) == T[1 0 0; 0 1 -1; 0 0 1]
            @test metric(bv) == T[1 0 0; 0 2 1; 0 1 1]
            @test metric(bv, :cont) == bv.G == T[1 0 0; 0 1 -1; 0 -1 2]
            @test simplify.(vecbasis(bv, :cont)' ⋅ vecbasis(bv, :cov)) == I
            @test simplify.(metric(bv, :cont)' ⋅ metric(bv, :cov)) == I
            @test !isorthogonal(bv)

            bw = Basis(bv.E, :cont)
            @test eltype(bw) == T
            @test vecbasis(bw) == T[1 0 0; 0 1 0; 0 1 1]
            @test vecbasis(bw, :cont) == T[1 0 0; 0 1 -1; 0 0 1]
            @test metric(bw) == T[1 0 0; 0 2 1; 0 1 1]
            @test metric(bw, :cont) == T[1 0 0; 0 1 -1; 0 -1 2]
            @test simplify.(vecbasis(bw, :cont)' ⋅ vecbasis(bw, :cov)) == I
            @test simplify.(metric(bw, :cont)' ⋅ metric(bw, :cov)) == I
            @test !isorthogonal(bw)

            nb = normalize(bv)
            if T == Sym
                @test vecbasis(nb) == vecbasis(nb, :cov) == T[1 0 0; 0 sq2/2 0; 0 sq2/2 1]
                θ, ϕ, ψ = symbols("θ, ϕ, ψ", real = true)
                br = Basis(θ, ϕ, ψ)
                @test metric(br) == I
                @test vecbasis(br, :cont) == vecbasis(br, :cov)
                @test angles(br) == (θ = θ, ϕ = ϕ, ψ = ψ)
                br = Basis(θ)
                @test metric(br) == I
                @test vecbasis(br, :cont) == vecbasis(br, :cov)
                @test angles(br) == (θ = θ,)
            else
                @test vecbasis(nb) ≈ nb.e ≈ T[1 0 0; 0 sq2/2 0; 0 sq2/2 1]
                θ, ϕ, ψ = rand(T, 3)
                br = Basis(θ, ϕ, ψ)
                @test metric(br) ≈ I
                @test vecbasis(br, :cont) == vecbasis(br, :cov)
                angbr = angles(br)
                @test angbr.θ ≈ θ && angbr.ϕ ≈ ϕ && angbr.ψ ≈ ψ
                angbr = angles(Array(br))
                @test angbr.θ ≈ θ && angbr.ϕ ≈ ϕ && angbr.ψ ≈ ψ
                br = Basis(θ)
                @test metric(br) ≈ I
                @test vecbasis(br, :cont) == vecbasis(br, :cov)
                @test angles(br).θ ≈ θ
                @test angles(Array(br)).θ ≈ θ
            end

            if T ≠ Sym
                v = rand(T, 3, 3)
                while det(v) ≈ 0
                    v = rand(T, 3, 3)
                end
                @test eltype(bv) == T
                @test simplify.(vecbasis(bv, :cont)' ⋅ vecbasis(bv, :cov)) == I
                @test simplify.(metric(bv, :cont)' ⋅ metric(bv, :cov)) == I
                end

        end
    end
end
