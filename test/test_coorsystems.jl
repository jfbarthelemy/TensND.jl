@testsection "Coordinate systems" begin
    @testsection "Partial derivatives" begin
        ℬ, 𝐞₁, 𝐞₂, 𝐞₃ = init_canonical()
        θ, ϕ, ℬˢ, 𝐞ᶿ, 𝐞ᵠ, 𝐞ʳ = init_spherical(symbols("θ ϕ", real = true)...)
        @test ∂(𝐞ʳ, θ) == 𝐞ᶿ
        @test ∂(𝐞ʳ, ϕ) == sin(θ) * 𝐞ᵠ
        @test ∂(𝐞ᵠ ⊗ 𝐞ᶿ, ϕ) == ∂(𝐞ᵠ, ϕ) ⊗ 𝐞ᶿ + 𝐞ᵠ ⊗ ∂(𝐞ᶿ, ϕ)
        @test ∂(𝐞ʳ ⊗ˢ 𝐞ᵠ, ϕ) == ∂(𝐞ʳ, ϕ) ⊗ˢ 𝐞ᵠ + 𝐞ʳ ⊗ˢ ∂(𝐞ᵠ, ϕ)
    end

    @testsection "Coordinate systems" begin
        θ, ϕ, ℬˢ, 𝐞ᶿ, 𝐞ᵠ, 𝐞ʳ = init_spherical(symbols("θ ϕ", real = true)...)
        r = symbols("r", positive = true)
        x = [r, θ, ϕ]
        OM = r * 𝐞ʳ
        CS = CoorSystemSym(OM, x; simp = Dict(abs(sin(θ)) => sin(θ)))

    end


end
