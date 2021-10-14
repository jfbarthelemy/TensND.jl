@testsection "Isotropic tensors" begin
    for T ∈ (Sym, Float64), dim ∈ (2, 3)
        @testsection "type $T, dim $dim" begin
            𝟏 = tensId2(Val(dim), Val(T))
            @test opequal(𝟏, I)
            @test opequal(tr(𝟏), dim)

            if T == Sym
                α = symbols("α", real = true)
            else
                α = rand()
                while opequal(α, one(T)) || opequal(α, zero(T))
                    α = rand()
                end
            end
            @test α * 𝟏 isa TensISO{2}
            t = α * 𝟏 + (1 - α) * 𝟏
            @test opequal(t, 𝟏)
            @test t isa TensISO{2}
            @test opequal(inv(α * 𝟏), inv(α) * 𝟏)
            @test isISO(t)

            𝕀, 𝕁, 𝕂 = ISO(Val(dim), Val(T))
            @test opequal(𝕁 + 𝕂, 𝕀)
            @test opequal((𝟏 ⊗ 𝟏) / dim, 𝕁)
            if T == Sym
                α, β = symbols("α β", real = true)
            else
                α = rand()
                β = rand()
                while opequal(α + β, one(T)) || opequal(α, zero(T)) || opequal(β, zero(T))
                    α = rand()
                    β = rand()
                end
            end
            𝕋 = α * 𝕁 + β * 𝕂
            @test 𝕋 isa TensISO{4}
            @test opequal(inv(𝕋), inv(α) * 𝕁 + inv(β) * 𝕂)
            @test isISO(𝕋)

            if T == Sym && dim == 3
                E, ν = symbols("E ν", real = true)
                k = E / 3(1 - 2ν)
                μ = E / 2(1 + ν)
                λ = E * ν / ((1 + ν) * (1 - 2ν))
                ℂ = simplify(3k * 𝕁 + 2μ * 𝕂)
                @test ℂ == simplify(TensISO{dim}(3k, 2μ))
                𝕊 = simplify(inv(ℂ))
                @test simplify.(KM(𝕊)) == [
                    1/E -ν/E -ν/E 0 0 0
                    -ν/E 1/E -ν/E 0 0 0
                    -ν/E -ν/E 1/E 0 0 0
                    0 0 0 (1+ν)/E 0 0
                    0 0 0 0 (1+ν)/E 0
                    0 0 0 0 0 (1+ν)/E
                ]
                @test simplify(ℂ ⊡ 𝕊) == 𝕀       
                
                n = 𝐞(3)
                Eᵒᵉᵈᵒ = E * (1 - ν) / ((1 + ν) * (1 - 2ν))
                Kref = simplify.([μ 0 0; 0 μ 0; 0 0 Eᵒᵉᵈᵒ])
                @test factor(n ⋅ ℂ ⋅ n) == factor(dotdot(n, ℂ, n)) == Kref
                # Hooke law
                for i ∈ 1:3, j ∈ 1:3
                    @eval $(Symbol("ε$i$j")) = symbols($"ε$i$j", real = true)
                end
                𝛆 = Tens(SymmetricTensor{2,3}((i, j) -> eval(Symbol("ε$i$j"))))
                𝛔 = ℂ ⊡ 𝛆
                @test factor(𝛔) == factor(λ * tr(𝛆) * 𝟏 + 2μ * 𝛆)
                @test factor(simplify(𝛔 ⊡ 𝛆)) == factor(simplify(λ * tr(𝛆)^2 + 2μ * 𝛆 ⊡ 𝛆))
                            
                @test 𝕀 == 𝟏 ⊠ˢ 𝟏
                @test 3𝕁 == 𝟏 ⊗ 𝟏
                @test 𝕀 ⊙ 𝕀 == 6
                @test 𝕁 ⊙ 𝕀 == 𝕁 ⊙ 𝕁 == 1
                @test 𝕂 ⊙ 𝕀 == 𝕂 ⊙ 𝕂 == 5
                @test 𝕂 ⊙ 𝕁 == 𝕁 ⊙ 𝕂 == 0
                @test simplify(ℂ ⊙ 𝕁) == simplify(3k)
                @test simplify(ℂ ⊙ 𝕂) == simplify(10μ)
            

            end

        end
    end
end
