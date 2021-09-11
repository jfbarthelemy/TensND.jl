@testsection "Special tensors" begin
    # Isotropic stiffness and compliance tensors
    𝟏, 𝟙, 𝕀, 𝕁, 𝕂 = init_isotropic()
    E, ν = symbols("E ν", real = true)
    λ = E * ν / ((1 + ν) * (1 - 2ν))
    μ = E / (2(1 + ν))
    ℂ = 3λ * 𝕁 + 2μ * 𝕀
    𝕊 = inv(ℂ)
    @test simplify.(KM(𝕊)) == [
        1/E -ν/E -ν/E 0 0 0
        -ν/E 1/E -ν/E 0 0 0
        -ν/E -ν/E 1/E 0 0 0
        0 0 0 (1+ν)/E 0 0
        0 0 0 0 (1+ν)/E 0
        0 0 0 0 0 (1+ν)/E
    ]
    @test invKM(KM(𝕊)) == 𝕊
    
    # Acoustic tensor
    n = Tensnd(Sym[0, 0, 1])
    Eᵒᵉᵈᵒ = E * (1 - ν) / ((1 + ν) * (1 - 2ν))
    Kref = simplify.([μ 0 0; 0 μ 0; 0 0 Eᵒᵉᵈᵒ])
    @test factor.(n ⋅ ℂ ⋅ n) == factor.(dotdot(n, ℂ, n)) == Kref
    # Hooke law
    for i ∈ 1:3, j ∈ 1:3
        @eval $(Symbol("ε$i$j")) = symbols($"ε$i$j", real = true)
    end
    𝛆 = Tensnd(SymmetricTensor{2,3}((i, j) -> eval(Symbol("ε$i$j"))))
    𝛔 = ℂ ⊡ 𝛆
    @test factor.(𝛔) == factor.(λ * tr(𝛆) * 𝟏 + 2μ * 𝛆)
    @test factor(simplify(𝛔 ⊡ 𝛆)) == factor(simplify(λ * tr(𝛆)^2 + 2μ * 𝛆 ⊡ 𝛆))

    @test 𝟙 == 𝟏 ⊠ 𝟏
    @test 𝕀 == 𝟏 ⊠ˢ 𝟏
    @test 3𝕁 == 𝟏 ⊗ 𝟏
    @test 𝕀 ⊙ 𝕀 == 6
    @test 𝕁 ⊙ 𝕀 == 𝕁 ⊙ 𝕁 == 1
    @test 𝕂 ⊙ 𝕀 == 𝕂 ⊙ 𝕂 == 5
    @test 𝕂 ⊙ 𝕁 == 𝕁 ⊙ 𝕂 == 0
    k = E / (3(1 - 2ν))
    @test simplify(ℂ ⊙ 𝕁) == simplify(3k)
    @test simplify(ℂ ⊙ 𝕂) == simplify(10μ)

    # Rotations
    θ, ϕ, ψ = symbols("θ ϕ ψ", real = true) ;
    cθ, cϕ, cψ, sθ, sϕ, sψ = symbols("cθ cϕ cψ sθ sϕ sψ", real = true) ;
    d = Dict(cos(θ) => cθ, cos(ϕ) => cϕ, cos(ψ) => cψ, sin(θ) => sθ, sin(ϕ) => sϕ, sin(ψ) => sψ) ;
    R = Tensnd(subs.(rot3(θ, ϕ, ψ),d...)) ;
    R6 = invKM(subs.(KM(rot6(θ, ϕ, ψ)),d...)) ;
    @test R6 == R ⊠ˢ R

end
