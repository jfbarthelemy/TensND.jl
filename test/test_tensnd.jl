@testsection "Tensnd" begin
    sq2 = √Sym(2)
    v = Sym[0 1 1; 1 0 1; 1 1 0]
    b = Basis(v)
    bn = normalize(b)
    for i ∈ 1:3
        @eval $(Symbol("v$i")) = symbols($"v$i", real = true)
    end
    V = Tensnd(Tensor{1,3}(i -> eval(Symbol("v$i"))))
    @test simplify.(components(V, b, (:cont,))) ==
          [(-v1 + v2 + v3) / 2, (v1 - v2 + v3) / 2, (v1 + v2 - v3) / 2]
    @test simplify.(components(V, b, (:cov,))) == [v2 + v3, v1 + v3, v1 + v2]
    @test simplify.(components(V, bn, (:cov,))) ==
          [sq2 * (v2 + v3) / 2, sq2 * (v1 + v3) / 2, sq2 * (v1 + v2) / 2]

    for i ∈ 1:3, j ∈ 1:3
        @eval $(Symbol("t$i$j")) = symbols($"t$i$j", real = true)
    end
    T = Tensnd(Tensor{2,3}((i, j) -> eval(Symbol("t$i$j"))))
    @test simplify.(components(T, b, (:cov, :cov))) == [
        t22+t23+t32+t33 t21+t23+t31+t33 t21+t22+t31+t32
        t12+t13+t32+t33 t11+t13+t31+t33 t11+t12+t31+t32
        t12+t13+t22+t23 t11+t13+t21+t23 t11+t12+t21+t22
    ]
    @test simplify.(components(T, b, (:cont, :cov))) ==
          [
        -t12-t13+t22+t23+t32+t33 -t11-t13+t21+t23+t31+t33 -t11-t12+t21+t22+t31+t32
        t12+t13-t22-t23+t32+t33 t11+t13-t21-t23+t31+t33 t11+t12-t21-t22+t31+t32
        t12+t13+t22+t23-t32-t33 t11+t13+t21+t23-t31-t33 t11+t12+t21+t22-t31-t32
    ] / 2

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


end
