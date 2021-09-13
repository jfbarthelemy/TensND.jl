@testsection "Coordinate systems" begin

    (x, y, z), (𝐞₁, 𝐞₂, 𝐞₃), ℬ = init_cartesian()
    (θ, ϕ, r), (𝐞ᶿ, 𝐞ᵠ, 𝐞ʳ), ℬˢ = init_spherical()

    @testsection "Usual coordinate systems" begin
        @test components(𝐞ʳ ⊗ 𝐞ᵠ, ℬˢ) == components(𝐞₃ ⊗ 𝐞₂, ℬ) == components_canon(𝐞₃ ⊗ 𝐞₂)
    end

    @testsection "Partial derivatives" begin
        @test ∂(𝐞ʳ, θ) == 𝐞ᶿ
        @test ∂(𝐞ʳ, ϕ) == sin(θ) * 𝐞ᵠ
        @test ∂(𝐞ᵠ ⊗ 𝐞ᶿ, ϕ) == ∂(𝐞ᵠ, ϕ) ⊗ 𝐞ᶿ + 𝐞ᵠ ⊗ ∂(𝐞ᶿ, ϕ)
        @test ∂(𝐞ʳ ⊗ˢ 𝐞ᵠ, ϕ) == ∂(𝐞ʳ, ϕ) ⊗ˢ 𝐞ᵠ + 𝐞ʳ ⊗ˢ ∂(𝐞ᵠ, ϕ)
    end

    @testsection "Coordinate systems" begin
        # Cartesian
        Cartesian, 𝐗, 𝐄, ℬ = CS_cartesian()
        𝛔 = Tensnd(SymmetricTensor{2,3}((i, j) -> SymFunction("σ$i$j", real = true)(𝐗...)))
        @test DIV(𝛔, Cartesian) ==
              sum([sum([diff(getdata(𝛔)[i, j], 𝐗[j]) for j ∈ 1:3]) * 𝐄[i] for i ∈ 1:3])

        # Polar
        Polar, (r, θ), (𝐞ʳ, 𝐞ᶿ), ℬᵖ = CS_polar()
        f = SymFunction("f", real = true)(r, θ)
        @test simplify(LAPLACE(f, Polar)) ==
              simplify(diff(r * diff(f, r), r) / r + diff(f, θ, θ) / r^2)

        # Cylindrical
        Cylindrical, rθz, (𝐞ʳ, 𝐞ᶿ, 𝐞ᶻ), ℬᶜ = CS_cylindrical()
        𝐯 = Tensnd(Vec{3}(i -> SymFunction("v$(rθz[i])", real = true)(rθz...)), ℬᶜ)
        vʳ, vᶿ, vᶻ = getdata(𝐯)
        @test simplify(DIV(𝐯, Cylindrical)) == simplify(diff(vʳ, r) + vʳ / r + diff(vᶿ, θ) / r + diff(vᶻ, z))

        # Spherical
        Spherical, (θ, ϕ, r), (𝐞ᶿ, 𝐞ᵠ, 𝐞ʳ), ℬˢ = CS_spherical()
        for σⁱʲ ∈ ("σʳʳ", "σᶿᶿ", "σᵠᵠ")
            @eval $(Symbol(σⁱʲ)) = SymFunction($σⁱʲ, real = true)($r)
        end
        𝛔 = σʳʳ * 𝐞ʳ ⊗ 𝐞ʳ + σᶿᶿ * 𝐞ᶿ ⊗ 𝐞ᶿ + σᵠᵠ * 𝐞ᵠ ⊗ 𝐞ᵠ
        div𝛔 = DIV(𝛔, Spherical)
        @test simplify(div𝛔 ⋅ 𝐞ʳ) == simplify(diff(σʳʳ, r) + (2 * σʳʳ - σᶿᶿ - σᵠᵠ) / r)



    end


end
