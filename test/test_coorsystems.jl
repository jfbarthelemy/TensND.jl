@testsection "Coordinate systems" begin
    sโ = simplify โ โ
    (x, y, z), (๐โ, ๐โ, ๐โ), โฌ = init_cartesian()
    (ฮธ, ฯ, r), (๐แถฟ, ๐แต , ๐สณ), โฌหข = init_spherical()

    @testsection "Usual coordinate systems" begin
        @test components(๐สณ โ ๐แต , โฌหข) == components(๐โ โ ๐โ, โฌ) == components_canon(๐โ โ ๐โ)
    end

    @testsection "Partial derivatives" begin
        @test sโ(๐สณ, ฮธ) == ๐แถฟ
        @test sโ(๐สณ, ฯ) == sin(ฮธ) * ๐แต 
        @test sโ(๐แต  โ ๐แถฟ, ฯ) == sโ(๐แต , ฯ) โ ๐แถฟ + ๐แต  โ sโ(๐แถฟ, ฯ)
        @test sโ(๐สณ โหข ๐แต , ฯ) == sโ(๐สณ, ฯ) โหข ๐แต  + ๐สณ โหข sโ(๐แต , ฯ)
    end

    @testsection "Coordinate systems" begin
        # Cartesian
        Cartesian = coorsys_cartesian()
        ๐ = getcoords(Cartesian)
        ๐ = unitvec(Cartesian)
        โฌ = get_normalized_basis(Cartesian)
        ๐ = Tens(SymmetricTensor{2,3}((i, j) -> SymFunction("ฯ$i$j", real = true)(๐...)))
        @test DIV(๐, Cartesian) ==
              sum([sum([โ(๐[i, j], ๐[j]) for j โ 1:3]) * ๐[i] for i โ 1:3])

        # Polar
        Polar = coorsys_polar()
        r, ฮธ = getcoords(Polar)
        ๐สณ, ๐แถฟ = unitvec(Polar)
        โฌแต = get_normalized_basis(Polar)
        f = SymFunction("f", real = true)(r, ฮธ)
        @test simplify(LAPLACE(f, Polar)) ==
              simplify(โ(r * โ(f, r), r) / r + โ(f, ฮธ, ฮธ) / r^2)

        # Cylindrical
        Cylindrical = coorsys_cylindrical()
        rฮธz = getcoords(Cylindrical)
        ๐สณ, ๐แถฟ, ๐แถป = unitvec(Cylindrical)
        โฌแถ = get_normalized_basis(Cylindrical)
        r, ฮธ, z = rฮธz
        ๐ฏ = Tens(Vec{3}(i -> SymFunction("v$(rฮธz[i])", real = true)(rฮธz...)), โฌแถ)
        vสณ, vแถฟ, vแถป = getarray(๐ฏ)
        @test simplify(DIV(๐ฏ, Cylindrical)) ==
              simplify(โ(vสณ, r) + vสณ / r + โ(vแถฟ, ฮธ) / r + โ(vแถป, z))

        # Spherical
        Spherical = coorsys_spherical()
        ฮธ, ฯ, r = getcoords(Spherical)
        ๐แถฟ, ๐แต , ๐สณ = unitvec(Spherical)
        โฌหข = get_normalized_basis(Spherical)
        for ฯโฑสฒ โ ("ฯสณสณ", "ฯแถฟแถฟ", "ฯแต แต ")
            @eval $(Symbol(ฯโฑสฒ)) = SymFunction($ฯโฑสฒ, real = true)($r)
        end
        ๐ = ฯสณสณ * ๐สณ โ ๐สณ + ฯแถฟแถฟ * ๐แถฟ โ ๐แถฟ + ฯแต แต  * ๐แต  โ ๐แต 
        div๐ = simplify(DIV(๐, Spherical))
        @test simplify(div๐ โ ๐สณ) == simplify(โ(ฯสณสณ, r) + (2ฯสณสณ - ฯแถฟแถฟ - ฯแต แต ) / r)

        # Concentric sphere - hydrostatic part
        ฮธ, ฯ, r = getcoords(Spherical)
        ๐แถฟ, ๐แต , ๐สณ = unitvec(Spherical)
        โฌหข = get_normalized_basis(Spherical)
        ๐, ๐, ๐ = ISO(Val(3), Val(Sym))
        ๐ = tensId2(Val(3), Val(Sym))
        k, ฮผ = symbols("k ฮผ", positive = true)
        ฮป = k - 2ฮผ / 3
        โ = 3k * ๐ + 2ฮผ * ๐
        u = SymFunction("u", real = true)(r)
        ๐ฎ = u * ๐สณ
        ๐ = simplify(SYMGRAD(๐ฎ, Spherical))
        ๐ = simplify(โ โก ๐)
        # ๐ = simplify(ฮป * tr(๐) * ๐ + 2ฮผ * ๐)
        @test dsolve(factor(simplify(DIV(๐, Spherical) โ ๐สณ)), u) ==
              Eq(u, symbols("C1") / r^2 + symbols("C2") * r)

        # Spheroidal
        Spheroidal = coorsys_spheroidal()
        OM = getOM(Spheroidal)
        @test simplify(LAPLACE(OM[1]^2, Spheroidal)) == 2


    end


end
