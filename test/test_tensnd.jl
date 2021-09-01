@testsection "Tensnd" begin
    sq2 = √Sym(2)
    v = Sym[0 1 1; 1 0 1; 1 1 0]
    b = Basis(v)
    bn = normal_basis(b)
    for i ∈ 1:3
        @eval $(Symbol("v$i")) = symbols($"v$i", real = true)
    end
    V = Tensor{1,3}(i -> eval(Symbol("v$i")))
    TV = Tensnd(V) # TV = Tensnd(V, (:cont,), CanonicalBasis())
    @test simplify.(components(TV, (:cont,), b)) ==
          [(-v1 + v2 + v3) / 2, (v1 - v2 + v3) / 2, (v1 + v2 - v3) / 2]
    @test simplify.(components(TV, (:cov,), b)) == [v2 + v3, v1 + v3, v1 + v2]
    @test simplify.(components(TV, (:cov,), bn)) ==
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
    E, ν = symbols("E ν", real = true)
    λ = E * ν / ((1 + ν) * (1 - 2ν))
    μ = E / (2(1 + ν))
    ℂ = 3λ * 𝕁() + 2μ * 𝕀()
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
    K = factor.(n ⋅ ℂ ⋅ n)
    Eᵒᵉᵈᵒ = E * (1 - ν) / ((1 + ν) * (1 - 2ν))
    @test K == simplify.([μ 0 0; 0 μ 0; 0 0 Eᵒᵉᵈᵒ])
    # Hooke law
    for i ∈ 1:3, j ∈ 1:3
        @eval $(Symbol("ε$i$j")) = symbols($"ε$i$j", real = true)
    end
    𝛆 = Tensnd(SymmetricTensor{2,3}((i, j) -> eval(Symbol("ε$i$j"))))
    𝛔 = ℂ ⊡ 𝛆
    𝛔2 = λ * tr(𝛆) * 𝟏() + 2μ * 𝛆
    @test factor.(𝛔) == factor.(𝛔2)

end
