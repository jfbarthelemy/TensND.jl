@testsection "Tensnd" begin
    sq2 = √Sym(2)
    v = Sym[0 1 1; 1 0 1; 1 1 0]
    b = Basis(v)
    bn = normalize(b)
    for i ∈ 1:3
        @eval $(Symbol("v$i")) = symbols($"v$i", real = true)
    end
    V = Tensnd(Tensor{1,3}(i -> eval(Symbol("v$i"))))
    @test simplify.(components(V, (:cont,), b)) ==
          [(-v1 + v2 + v3) / 2, (v1 - v2 + v3) / 2, (v1 + v2 - v3) / 2]
    @test simplify.(components(V, (:cov,), b)) == [v2 + v3, v1 + v3, v1 + v2]
    @test simplify.(components(V, (:cov,), bn)) ==
          [sq2 * (v2 + v3) / 2, sq2 * (v1 + v3) / 2, sq2 * (v1 + v2) / 2]

    for i ∈ 1:3, j ∈ 1:3
        @eval $(Symbol("t$i$j")) = symbols($"t$i$j", real = true)
    end
    T = Tensnd(Tensor{2,3}((i, j) -> eval(Symbol("t$i$j"))))
    @test simplify.(components(T, (:cov, :cov), b)) == [
        t22+t23+t32+t33 t21+t23+t31+t33 t21+t22+t31+t32
        t12+t13+t32+t33 t11+t13+t31+t33 t11+t12+t31+t32
        t12+t13+t22+t23 t11+t13+t21+t23 t11+t12+t21+t22
    ]
    @test simplify.(components(T, (:cont, :cov), b)) ==
          [
        -t12-t13+t22+t23+t32+t33 -t11-t13+t21+t23+t31+t33 -t11-t12+t21+t22+t31+t32
        t12+t13-t22-t23+t32+t33 t11+t13-t21-t23+t31+t33 t11+t12-t21-t22+t31+t32
        t12+t13+t22+t23-t32-t33 t11+t13+t21+t23-t31-t33 t11+t12+t21+t22-t31-t32
    ] / 2

    # Isotropic stiffness and compliance tensors
    𝟏 = t𝟏(Sym)
    𝟙 = t𝟙(Sym)
    𝕀 = t𝕀(Sym)
    𝕁 = t𝕁(Sym)
    𝕂 = t𝕂(Sym)
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
    @test 𝕀 ⊙ 𝕀 == 6
    @test 𝕁 ⊙ 𝕀 == 𝕁 ⊙ 𝕁 == 1
    @test 𝕂 ⊙ 𝕀 == 𝕂 ⊙ 𝕂 == 5
    @test 𝕂 ⊙ 𝕁 == 𝕁 ⊙ 𝕂 == 0
    k = E / (3(1 - 2ν))
    @test simplify(ℂ ⊙ 𝕁) == simplify(3k)
    @test simplify(ℂ ⊙ 𝕂) == simplify(10μ)

    for i ∈ 1:3
        @eval $(Symbol("a$i")) = symbols($"a$i", real = true)
        @eval $(Symbol("b$i")) = symbols($"b$i", real = true)
    end
    a = Tensnd(Vec{3}((i,) -> eval(Symbol("a$i"))))
    b = Tensnd(Vec{3}((i,) -> eval(Symbol("b$i"))))
    @test a ⊗ b == Sym[a1*b1 a1*b2 a1*b3; a2*b1 a2*b2 a2*b3; a3*b1 a3*b2 a3*b3]
    @test a ⊗ˢ b == Sym[
        a1*b1 a1*b2/2+a2*b1/2 a1*b3/2+a3*b1/2
        a1*b2/2+a2*b1/2 a2*b2 a2*b3/2+a3*b2/2
        a1*b3/2+a3*b1/2 a2*b3/2+a3*b2/2 a3*b3
    ]

    θ, ϕ, ψ = symbols("θ, ϕ, ψ", real = true) ; br = Basis(θ, ϕ, ψ) ;
    ts = 𝐞ˢ𝐞ˢ(3,2,θ,ϕ,ψ)
    @test components(ts, br) == 𝐞𝐞(3,2)

end
