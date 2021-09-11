@testsection "Coordinate systems" begin

    @testsection "Usual coordinate systems" begin
        (x, y, z), (𝐞₁, 𝐞₂, 𝐞₃), ℬ = init_canonical()
        (θ, ϕ, r), (𝐞ᶿ, 𝐞ᵠ, 𝐞ʳ), ℬˢ = init_spherical()
        @test components(𝐞ʳ ⊗ 𝐞ᵠ, ℬˢ) == components(𝐞₃ ⊗ 𝐞₂, ℬ) == components_canon(𝐞₃ ⊗ 𝐞₂)
    end

    @testsection "Partial derivatives" begin
        (x, y, z), (𝐞₁, 𝐞₂, 𝐞₃), ℬ = init_canonical()
        (θ, ϕ, r), (𝐞ᶿ, 𝐞ᵠ, 𝐞ʳ), ℬˢ = init_spherical()
        @test ∂(𝐞ʳ, θ) == 𝐞ᶿ
        @test ∂(𝐞ʳ, ϕ) == sin(θ) * 𝐞ᵠ
        @test ∂(𝐞ᵠ ⊗ 𝐞ᶿ, ϕ) == ∂(𝐞ᵠ, ϕ) ⊗ 𝐞ᶿ + 𝐞ᵠ ⊗ ∂(𝐞ᶿ, ϕ)
        @test ∂(𝐞ʳ ⊗ˢ 𝐞ᵠ, ϕ) == ∂(𝐞ʳ, ϕ) ⊗ˢ 𝐞ᵠ + 𝐞ʳ ⊗ˢ ∂(𝐞ᵠ, ϕ)
    end

    @testsection "Coordinate systems" begin
        coords, vectors, ℬˢ = init_spherical()
        θ, ϕ, r = coords
        𝐞ᶿ, 𝐞ᵠ, 𝐞ʳ = vectors
        OM = r * 𝐞ʳ
        CS = CoorSystemSym(OM, coords; simp = Dict(abs(sin(θ)) => sin(θ)))

    end


end
