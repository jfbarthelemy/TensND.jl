@testsection "Coordinate systems" begin
    s∂ = SymPy.simplify ∘ ∂
    (x, y, z), (𝐞₁, 𝐞₂, 𝐞₃), ℬ = init_cartesian()
    (θ, ϕ, r), (𝐞ᶿ, 𝐞ᵠ, 𝐞ʳ), ℬˢ = init_spherical()

    @testsection "Usual coordinate systems" begin
        @test components(𝐞ʳ ⊗ 𝐞ᵠ, ℬˢ) == components(𝐞₃ ⊗ 𝐞₂, ℬ) == components_canon(𝐞₃ ⊗ 𝐞₂)
    end

    @testsection "Partial derivatives" begin
        @test s∂(𝐞ʳ, θ) == 𝐞ᶿ
        @test s∂(𝐞ʳ, ϕ) == sin(θ) * 𝐞ᵠ
        @test s∂(𝐞ᵠ ⊗ 𝐞ᶿ, ϕ) == s∂(𝐞ᵠ, ϕ) ⊗ 𝐞ᶿ + 𝐞ᵠ ⊗ s∂(𝐞ᶿ, ϕ)
        @test s∂(𝐞ʳ ⊗ˢ 𝐞ᵠ, ϕ) == s∂(𝐞ʳ, ϕ) ⊗ˢ 𝐞ᵠ + 𝐞ʳ ⊗ˢ s∂(𝐞ᵠ, ϕ)
    end

    @testsection "Coordinate systems" begin
        # Cartesian
        Cartesian = coorsys_cartesian()
        𝐗 = getcoords(Cartesian)
        𝐄 = unitvec(Cartesian)
        ℬ = normalized_basis(Cartesian)
        𝛔 = Tens(SymmetricTensor{2,3}((i, j) -> SymFunction("σ$i$j", real = true)(𝐗...)))
        @test DIV(𝛔, Cartesian) ==
              sum([sum([∂(𝛔[i, j], 𝐗[j]) for j ∈ 1:3]) * 𝐄[i] for i ∈ 1:3])

        # Polar
        Polar = coorsys_polar()
        r, θ = getcoords(Polar)
        𝐞ʳ, 𝐞ᶿ = unitvec(Polar)
        ℬᵖ = normalized_basis(Polar)
        f = SymFunction("f", real = true)(r, θ)
        @test SymPy.simplify(LAPLACE(f, Polar)) ==
              SymPy.simplify(∂(r * ∂(f, r), r) / r + ∂(f, θ, θ) / r^2)

        # Cylindrical
        Cylindrical = coorsys_cylindrical()
        rθz = getcoords(Cylindrical)
        𝐞ʳ, 𝐞ᶿ, 𝐞ᶻ = unitvec(Cylindrical)
        ℬᶜ = normalized_basis(Cylindrical)
        r, θ, z = rθz
        𝐯 = Tens(Vec{3}(i -> SymFunction("v$(rθz[i])", real = true)(rθz...)), ℬᶜ)
        vʳ, vᶿ, vᶻ = getarray(𝐯)
        @test SymPy.simplify(DIV(𝐯, Cylindrical)) ==
              SymPy.simplify(∂(vʳ, r) + vʳ / r + ∂(vᶿ, θ) / r + ∂(vᶻ, z))

        # Spherical
        Spherical = coorsys_spherical()
        θ, ϕ, r = getcoords(Spherical)
        𝐞ᶿ, 𝐞ᵠ, 𝐞ʳ = unitvec(Spherical)
        ℬˢ = normalized_basis(Spherical)
        for σⁱʲ ∈ ("σʳʳ", "σᶿᶿ", "σᵠᵠ")
            @eval $(Symbol(σⁱʲ)) = SymFunction($σⁱʲ, real = true)($r)
        end
        𝛔 = σʳʳ * 𝐞ʳ ⊗ 𝐞ʳ + σᶿᶿ * 𝐞ᶿ ⊗ 𝐞ᶿ + σᵠᵠ * 𝐞ᵠ ⊗ 𝐞ᵠ
        div𝛔 = SymPy.simplify(DIV(𝛔, Spherical))
        @test SymPy.simplify(div𝛔 ⋅ 𝐞ʳ) == SymPy.simplify(∂(σʳʳ, r) + (2σʳʳ - σᶿᶿ - σᵠᵠ) / r)

        # Concentric sphere - hydrostatic part
        θ, ϕ, r = getcoords(Spherical)
        𝐞ᶿ, 𝐞ᵠ, 𝐞ʳ = unitvec(Spherical)
        ℬˢ = normalized_basis(Spherical)
        𝕀, 𝕁, 𝕂 = ISO(Val(3), Val(Sym))
        𝟏 = tensId2(Val(3), Val(Sym))
        k, μ = symbols("k μ", positive = true)
        λ = k - 2μ / 3
        ℂ = 3k * 𝕁 + 2μ * 𝕂
        u = SymFunction("u", real = true)(r)
        𝐮 = u * 𝐞ʳ
        𝛆 = SymPy.simplify(SYMGRAD(𝐮, Spherical))
        𝛔 = SymPy.simplify(ℂ ⊡ 𝛆)
        # 𝛔 = SymPy.simplify(λ * tr(𝛆) * 𝟏 + 2μ * 𝛆)
        @test dsolve(factor(SymPy.simplify(DIV(𝛔, Spherical) ⋅ 𝐞ʳ)), u) ==
              Eq(u, symbols("C1") / r^2 + symbols("C2") * r)

        # Spheroidal
        Spheroidal = coorsys_spheroidal()
        OM = getOM(Spheroidal)
        @test SymPy.simplify(LAPLACE(OM[1]^2, Spheroidal)) == 2


    end


end
