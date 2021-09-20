@testsection "Isotropic arrays" begin
    for T ∈ (Sym, Float64), dim ∈ (2, 3)
        @testsection "type $T, dim $dim" begin
            i2 = Id2{dim,T}()
            @test opequal(i2, I)
            @test opequal(tr(i2), dim)

            if T == Sym
                α = symbols("α", real = true)
            else
                α = rand()
                while opequal(α, one(T)) || opequal(α, zero(T))
                    α = rand()
                end
            end
            @test α * i2 isa Isotropic2
            @test !(α * i2 isa Id2)
            n2 = α * i2 + (1 - α) * i2
            @test opequal(n2, i2)
            @test n2 isa Id2
            @test opequal(inv(α * i2), inv(α) * i2)

            𝕀 = I4{dim,T}()
            𝕁 = J4{dim,T}()
            𝕂 = K4{dim,T}()
            @test opequal(𝕁 + 𝕂, 𝕀)
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
            @test 𝕋 isa Isotropic4
            @test !(𝕋 isa Id4)
            @test opequal(inv(𝕋), inv(α) * 𝕁 + inv(β) * 𝕂)

        end
    end
end
