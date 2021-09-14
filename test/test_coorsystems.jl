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
              sum([sum([∂(𝛔[i, j], 𝐗[j]) for j ∈ 1:3]) * 𝐄[i] for i ∈ 1:3])

        # Polar
        Polar, (r, θ), (𝐞ʳ, 𝐞ᶿ), ℬᵖ = CS_polar()
        f = SymFunction("f", real = true)(r, θ)
        @test simplify(LAPLACE(f, Polar)) ==
              simplify(∂(r * ∂(f, r), r) / r + ∂(f, θ, θ) / r^2)

        # Cylindrical
        Cylindrical, rθz, (𝐞ʳ, 𝐞ᶿ, 𝐞ᶻ), ℬᶜ = CS_cylindrical()
        r, θ, z = rθz
        𝐯 = Tensnd(Vec{3}(i -> SymFunction("v$(rθz[i])", real = true)(rθz...)), ℬᶜ)
        vʳ, vᶿ, vᶻ = getdata(𝐯)
        @test simplify(DIV(𝐯, Cylindrical)) ==
              simplify(∂(vʳ, r) + vʳ / r + ∂(vᶿ, θ) / r + ∂(vᶻ, z))

        # Spherical
        Spherical, (θ, ϕ, r), (𝐞ᶿ, 𝐞ᵠ, 𝐞ʳ), ℬˢ = CS_spherical()
        for σⁱʲ ∈ ("σʳʳ", "σᶿᶿ", "σᵠᵠ")
            @eval $(Symbol(σⁱʲ)) = SymFunction($σⁱʲ, real = true)($r)
        end
        𝛔 = σʳʳ * 𝐞ʳ ⊗ 𝐞ʳ + σᶿᶿ * 𝐞ᶿ ⊗ 𝐞ᶿ + σᵠᵠ * 𝐞ᵠ ⊗ 𝐞ᵠ
        div𝛔 = DIV(𝛔, Spherical)
        @test simplify(div𝛔 ⋅ 𝐞ʳ) == simplify(∂(σʳʳ, r) + (2 * σʳʳ - σᶿᶿ - σᵠᵠ) / r)

        # Concentric sphere - hydrostatic part
        Spherical, (θ, ϕ, r), (𝐞ᶿ, 𝐞ᵠ, 𝐞ʳ), ℬˢ = CS_spherical()
        𝟏, 𝟙, 𝕀, 𝕁, 𝕂 = init_isotropic()
        k, μ = symbols("k μ", positive = true)
        ℂ = 3k * 𝕁 + 2μ * 𝕂
        u = SymFunction("u", real = true)(r)
        𝐮 = u * 𝐞ʳ
        𝛆 = SYMGRAD(𝐮, Spherical)
        𝛔 = ℂ ⊡ 𝛆
        @test dsolve(simplify(DIV(𝛔, Spherical) ⋅ 𝐞ʳ), u) ==
              Eq(u, symbols("C1") / r^2 + symbols("C2") * r)



    end


end
