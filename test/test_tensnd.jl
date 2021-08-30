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
    T = Tensor{2,3}((i, j) -> eval(Symbol("t$i$j")))
    TT = Tensnd(T)
    @test simplify.(components(TT, (:cov, :cov), b)) == [
        t22+t23+t32+t33 t21+t23+t31+t33 t21+t22+t31+t32
        t12+t13+t32+t33 t11+t13+t31+t33 t11+t12+t31+t32
        t12+t13+t22+t23 t11+t13+t21+t23 t11+t12+t21+t22
    ]
    @test simplify.(components(TT, (:cont, :cov), b)) ==
          [
        -t12-t13+t22+t23+t32+t33 -t11-t13+t21+t23+t31+t33 -t11-t12+t21+t22+t31+t32
        t12+t13-t22-t23+t32+t33 t11+t13-t21-t23+t31+t33 t11+t12-t21-t22+t31+t32
        t12+t13+t22+t23-t32-t33 t11+t13+t21+t23-t31-t33 t11+t12+t21+t22-t31-t32
    ] / 2

    # Isotropic stiffness and compliance tensors
    E, ν = symbols("E ν", real = true)
    λ = E * ν / ((1 + ν) * (1 - 2ν))
    μ = E / (2 * (1 + ν))
    C = 3λ * tensJ4() + 2μ * tensId4()
    S = inv(C)
    @test simplify.(KM(S)) == [
        1/E -ν/E -ν/E 0 0 0
        -ν/E 1/E -ν/E 0 0 0
        -ν/E -ν/E 1/E 0 0 0
        0 0 0 (1+ν)/E 0 0
        0 0 0 0 (1+ν)/E 0
        0 0 0 0 0 (1+ν)/E
    ]
    @test invKM(KM(S)) == S
    # Acoustic tensor
    n = Tensnd(Sym[0, 0, 1])
    K = factor.(n ⋅ C ⋅ n)
    Eᵒᵉᵈᵒ = E * (1 - ν) / ((1 + ν) * (1 - 2ν))
    @test K == simplify.([μ 0 0; 0 μ 0; 0 0 Eᵒᵉᵈᵒ])

end
