@testsection "Tens" begin
    sq2 = √Sym(2)
    v = Sym[0 1 1; 1 0 1; 1 1 0]
    b = Basis(v)
    bn = normalize(b)
    for i ∈ 1:3
        @eval $(Symbol("v$i")) = symbols($"v$i", real = true)
    end
    V = Tens(Tensor{1,3}(i -> eval(Symbol("v$i"))))
    @test tsimplify(components(V, b, (:cont,))) ==
          [(-v1 + v2 + v3) / 2, (v1 - v2 + v3) / 2, (v1 + v2 - v3) / 2]
    @test tsimplify(components(V, b, (:cov,))) == [v2 + v3, v1 + v3, v1 + v2]
    @test tsimplify(components(V, bn, (:cov,))) ==
          [sq2 * (v2 + v3) / 2, sq2 * (v1 + v3) / 2, sq2 * (v1 + v2) / 2]

    for i ∈ 1:3, j ∈ 1:3
        @eval $(Symbol("t$i$j")) = symbols($"t$i$j", real = true)
    end
    T = Tens(Tensor{2,3}((i, j) -> eval(Symbol("t$i$j"))))
    @test tsimplify(components(T, b, (:cov, :cov))) == [
        t22+t23+t32+t33 t21+t23+t31+t33 t21+t22+t31+t32
        t12+t13+t32+t33 t11+t13+t31+t33 t11+t12+t31+t32
        t12+t13+t22+t23 t11+t13+t21+t23 t11+t12+t21+t22
    ]
    @test tsimplify(components(T, b, (:cont, :cov))) ==
          [
        -t12-t13+t22+t23+t32+t33 -t11-t13+t21+t23+t31+t33 -t11-t12+t21+t22+t31+t32
        t12+t13-t22-t23+t32+t33 t11+t13-t21-t23+t31+t33 t11+t12-t21-t22+t31+t32
        t12+t13+t22+t23-t32-t33 t11+t13+t21+t23-t31-t33 t11+t12+t21+t22-t31-t32
    ] / 2

    for i ∈ 1:3
        @eval $(Symbol("a$i")) = symbols($"a$i", real = true)
        @eval $(Symbol("b$i")) = symbols($"b$i", real = true)
    end
    a = Tens(Vec{3}((i,) -> eval(Symbol("a$i"))))
    b = Tens(Vec{3}((i,) -> eval(Symbol("b$i"))))
    @test a ⊗ b == Sym[a1*b1 a1*b2 a1*b3; a2*b1 a2*b2 a2*b3; a3*b1 a3*b2 a3*b3]
    @test a ⊗ˢ b == Sym[
        a1*b1 a1*b2/2+a2*b1/2 a1*b3/2+a3*b1/2
        a1*b2/2+a2*b1/2 a2*b2 a2*b3/2+a3*b2/2
        a1*b3/2+a3*b1/2 a2*b3/2+a3*b2/2 a3*b3
    ]

    θ, ϕ, ψ = symbols("θ, ϕ, ψ", real = true)
    R = rot3(θ, ϕ, ψ)
    Λ = [symbols("λ$i", positive = true) for i ∈ 1:3]
    bg = Basis(R .* Λ')
    bo = Basis(R)
    𝕀, 𝕁, 𝕂 = ISO()
    α, β = symbols("α β", real = true)
    𝕋 = α * 𝕁 + β * 𝕂
    @test change_tens(𝕋, bg) == 𝕋
    @test change_tens(𝕋, bo) == 𝕋
    A = components(𝕋, bg)
    𝕋₂ = Tens(A, bg)
    @test 𝕋₂ == 𝕋


end
