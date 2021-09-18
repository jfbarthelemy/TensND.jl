@testsection "Isotropic arrays" begin
    for T ∈ (Sym, Float64), dim ∈ (2, 3)
        @testsection "type $T, dim $dim" begin
            i2 = Id2{dim,T}()
            @test opequal(i2, I)
            @test opequal(tr(i2), dim)

            if T == Sym
                α = one(T) / 3
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
            @test opequal(inv(α * i2),  inv(α) * i2)



        end
    end
end
