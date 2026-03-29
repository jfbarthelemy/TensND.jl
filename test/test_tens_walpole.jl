@testsection "Walpole & Ortho tensors" begin

    # ─── helpers shared across sub-sections ───────────────────────────────────
    n3 = 𝐞(Val(3), Val(3), Val(Float64))   # e₃ as Float64 Vec
    n3s = 𝐞(Val(3), Val(3), Val(Sym))       # e₃ as Sym Vec
    atol_num = 1e-12

    # ═══════════════════════════════════════════════════════════════════════════
    @testsection "TensWalpole — construction & traits" begin
        W1, W2, W3, W4, W5, W6 = Walpole(n3)
        @test W1 isa TensWalpole{Float64,6}
        @test W2 isa TensWalpole{Float64,6}
        @test size(W1) == (3, 3, 3, 3)
        @test getbasis(W1) isa CanonicalBasis{3,Float64}
        @test getvar(W1) == (:cont, :cont, :cont, :cont)

        # N=5 (symmetric) basis
        W1s, W2s, W3s, W4s, W5s = Walpole(n3; sym = true)
        @test W1s isa TensWalpole{Float64,5}
        @test size(W1s) == (3, 3, 3, 3)

        # individual constructors
        @test tensW1(n3) isa TensWalpole{Float64,6}
        @test tensW3(n3) isa TensWalpole{Float64,6}

        # axis accessor
        @test getaxis(W1) == (0.0, 0.0, 1.0)
    end

    # ═══════════════════════════════════════════════════════════════════════════
    @testsection "TensWalpole — getarray & KM (n=e₃, Float64)" begin
        W1, W2, W3, W4, W5, W6 = Walpole(n3)
        sq2 = sqrt(2.0)

        # W₁ = nₙ⊗nₙ  →  only entry [3,3,3,3] = 1
        A1 = getarray(W1)
        @test A1[3,3,3,3] ≈ 1.0   atol=atol_num
        @test all(abs.(A1[i,j,k,l]) < atol_num
                  for i in 1:3, j in 1:3, k in 1:3, l in 1:3
                  if !(i==3 && j==3 && k==3 && l==3))

        # W₂ = (nT⊗nT)/2 with n=e₃ → nT = diag(1,1,0)
        # so W₂[i,j,k,l] = δᵢⱼ(1-δᵢ₃)δₖₗ(1-δₖ₃)/2
        A2 = getarray(W2)
        @test A2[1,1,1,1] ≈ 0.5   atol=atol_num
        @test A2[1,1,2,2] ≈ 0.5   atol=atol_num
        @test A2[2,2,2,2] ≈ 0.5   atol=atol_num
        @test A2[3,3,3,3] ≈ 0.0   atol=atol_num

        # W₃[3,3,1,1] = 1/√2
        A3 = getarray(W3)
        @test A3[3,3,1,1] ≈ 1.0/sq2   atol=atol_num
        @test A3[3,3,2,2] ≈ 1.0/sq2   atol=atol_num
        @test A3[3,3,3,3] ≈ 0.0       atol=atol_num

        # W₄[1,1,3,3] = 1/√2
        A4 = getarray(W4)
        @test A4[1,1,3,3] ≈ 1.0/sq2   atol=atol_num

        # W₅: shear in transverse plane (e₁,e₂)
        # W₅[1,2,1,2] = W₅[2,1,1,2] = W₅[1,2,2,1] = W₅[2,1,2,1] = 1/2
        A5 = getarray(W5)
        @test A5[1,2,1,2] ≈ 0.5   atol=atol_num
        @test A5[1,1,1,1] ≈ 0.5   atol=atol_num
        @test A5[3,3,3,3] ≈ 0.0   atol=atol_num

        # W₆: shear between transverse and axial
        A6 = getarray(W6)
        @test A6[1,3,1,3] ≈ 0.5   atol=atol_num
        @test A6[2,3,2,3] ≈ 0.5   atol=atol_num
        @test A6[1,1,1,1] ≈ 0.0   atol=atol_num
        @test A6[3,3,3,3] ≈ 0.0   atol=atol_num

        # KM structure for n=e₃: blocks should separate
        L = TensWalpole(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, n3)
        Km = KM(L)
        @test size(Km) == (6, 6)
        # Off-diagonal shear coupling should be zero for axis n=e₃
        @test abs(Km[1,4]) < atol_num
        @test abs(Km[1,5]) < atol_num
        @test abs(Km[4,6]) < atol_num
    end

    # ═══════════════════════════════════════════════════════════════════════════
    @testsection "TensWalpole — Walpole product rule (Float64)" begin
        W1, W2, W3, W4, W5, W6 = Walpole(n3)

        # Idempotents: W₁⊡W₁=W₁, W₂⊡W₂=W₂, W₅⊡W₅=W₅, W₆⊡W₆=W₆
        @test opequal(getarray(W1 ⊡ W1), getarray(W1))
        @test opequal(getarray(W2 ⊡ W2), getarray(W2))
        @test opequal(getarray(W5 ⊡ W5), getarray(W5))
        @test opequal(getarray(W6 ⊡ W6), getarray(W6))

        # Cross products
        @test opequal(getarray(W3 ⊡ W4), getarray(W1))
        @test opequal(getarray(W4 ⊡ W3), getarray(W2))

        # Zero cross products between incompatible blocks
        zero4 = zeros(3, 3, 3, 3)
        @test opequal(getarray(W1 ⊡ W2), zero4)
        @test opequal(getarray(W1 ⊡ W5), zero4)
        @test opequal(getarray(W5 ⊡ W6), zero4)
        @test opequal(getarray(W6 ⊡ W5), zero4)

        # General product: Walpole vs direct array contraction
        L = TensWalpole(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, n3)
        M = TensWalpole(0.5, 1.5, 2.0, 0.3, 0.8, 1.2, n3)
        LM_walpole = getarray(L ⊡ M)
        LM_direct  = Tensor{4,3}(getarray(L)) ⊡ Tensor{4,3}(getarray(M))
        @test opequal(LM_walpole, Array(LM_direct))
    end

    # ═══════════════════════════════════════════════════════════════════════════
    @testsection "TensWalpole — inverse (Float64)" begin
        𝕀 = tensId4(Val(3), Val(Float64))
        n = [1.0/sqrt(3.0), 1.0/sqrt(3.0), 1.0/sqrt(3.0)]

        # N=5 (symmetric)
        L5 = TensWalpole(2.0, 3.0, 1.0, 4.0, 5.0, n)
        Li5 = inv(L5)
        @test Li5 isa TensWalpole{Float64,5}
        LLi5 = getarray(L5 ⊡ Li5)
        𝕀arr = getarray(𝕀)
        @test opequal(LLi5, 𝕀arr)

        # N=6 (general)
        L6 = TensWalpole(2.0, 3.0, 1.0, 0.5, 4.0, 5.0, n)
        Li6 = inv(L6)
        @test Li6 isa TensWalpole{Float64,6}
        LLi6 = getarray(L6 ⊡ Li6)
        @test opequal(LLi6, 𝕀arr)
    end

    # ═══════════════════════════════════════════════════════════════════════════
    @testsection "TensWalpole — fromISO" begin
        𝕀, 𝕁, 𝕂 = ISO(Val(3), Val(Float64))
        α, β = 3.0, 2.0
        ℂiso = α * 𝕁 + β * 𝕂

        # For any axis, fromISO should give the same array as the isotropic tensor
        for n ∈ ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0],
                 [1.0/sqrt(3), 1.0/sqrt(3), 1.0/sqrt(3)])
            Wiso = fromISO(ℂiso, n)
            @test Wiso isa TensWalpole{Float64,5}
            @test isTI(Wiso)
            @test opequal(getarray(Wiso), getarray(ℂiso))
        end

        # Symbolic — use Sym ISO tensors to avoid Float64/Sym residuals
        αs, βs = symbols("α β", real = true)
        𝕁s, 𝕂s = tensJ4(Val(3), Val(Sym)), tensK4(Val(3), Val(Sym))
        ℂisos = αs * 𝕁s + βs * 𝕂s
        Wisos = fromISO(ℂisos, n3s)
        @test Wisos isa TensWalpole{<:Any,5}
        for i in 1:3, j in 1:3, k in 1:3, l in 1:3
            @test simplify(Wisos[i,j,k,l] - ℂisos[i,j,k,l]) == 0
        end
    end

    # ═══════════════════════════════════════════════════════════════════════════
    @testsection "TensWalpole — isISO / isTI / isOrtho" begin
        W = tensW1(n3)
        @test !isISO(W)
        @test  isTI(W)
        @test !isOrtho(W)
        𝕀, 𝕁, 𝕂 = ISO(Val(3), Val(Float64))
        @test !isTI(𝕁)
        @test !isOrtho(𝕁)
    end

    # ═══════════════════════════════════════════════════════════════════════════
    @testsection "TensOrtho — isISO / isTI / isOrtho" begin
        frame3 = CanonicalBasis{3,Float64}()
        t = TensOrtho(10., 8., 9., 3., 2., 4., 2.5, 3., 1.5, frame3)
        @test !isISO(t)
        @test !isTI(t)
        @test  isOrtho(t)
        # universal fallback
        @test !isOrtho(42)
        @test !isOrtho("string")
    end

    # ═══════════════════════════════════════════════════════════════════════════
    @testsection "TensWalpole — show" begin
        L5 = TensWalpole(1.0, 2.0, 0.5, 3.0, 4.0, n3)   # N=5
        L6 = TensWalpole(1.0, 2.0, 0.5, 0.3, 3.0, 4.0, n3)   # N=6
        buf5 = IOBuffer()
        show(buf5, L5)
        s5 = String(take!(buf5))
        @test contains(s5, "W") && contains(s5, "axis")

        buf6 = IOBuffer()
        show(buf6, L6)
        s6 = String(take!(buf6))
        @test contains(s6, "W") && contains(s6, "axis")
    end

    # ═══════════════════════════════════════════════════════════════════════════
    @testsection "TensOrtho — show" begin
        frame3 = CanonicalBasis{3,Float64}()
        t = TensOrtho(10., 8., 9., 3., 2., 4., 2.5, 3., 1.5, frame3)
        buf = IOBuffer()
        show(buf, t)
        s = String(take!(buf))
        @test contains(s, "P₁⊗P₁") && contains(s, "frame")
    end

    # ═══════════════════════════════════════════════════════════════════════════
    @testsection "TensWalpole — tsimplify (symbolic, _rebuild)" begin
        ℓ₁, ℓ₂, ℓ₃ = symbols("ℓ₁ ℓ₂ ℓ₃", real = true)
        L = TensWalpole(ℓ₁, ℓ₂, ℓ₃, ℓ₁+ℓ₂, ℓ₂+ℓ₃, n3s)   # N=6
        Ls = tsimplify(L)
        @test Ls isa TensWalpole   # _rebuild preserves type
        @test get_ℓ(Ls)[1] == ℓ₁   # simplification is a no-op here
    end

    # ═══════════════════════════════════════════════════════════════════════════
    @testsection "TensOrtho — tsimplify (_rebuild)" begin
        # TensOrtho frames are always numeric; test _rebuild with Float64 data.
        # tsimplify on a non-symbolic NTuple is a no-op, but _rebuild must still
        # reconstruct the TensOrtho, verifying the dispatch path.
        frame3 = CanonicalBasis{3,Float64}()
        t = TensOrtho(10., 8., 9., 3., 2., 4., 2.5, 3., 1.5, frame3)
        ts = tsimplify(t)
        @test ts isa TensOrtho   # _rebuild preserves type
        @test getdata(ts)[1] ≈ 10.0
    end

    # ═══════════════════════════════════════════════════════════════════════════
    @testsection "TensWalpole — arithmetic" begin
        W1, W2, W3, W4, W5, W6 = Walpole(n3)
        L = TensWalpole(1.0, 2.0, 0.5, 0.5, 3.0, 4.0, n3)   # N=6
        M = TensWalpole(0.5, 1.0, 0.25, 0.25, 1.5, 2.0, n3)

        @test opequal(getarray(L + M), getarray(L) .+ getarray(M))
        @test opequal(getarray(L - M), getarray(L) .- getarray(M))
        @test opequal(getarray(2.0 * L), 2.0 .* getarray(L))
        @test opequal(getarray(-L), .-getarray(L))

        # Symmetric N=5
        Ls = TensWalpole(1.0, 2.0, 0.5, 3.0, 4.0, n3)
        Ms = TensWalpole(0.5, 1.0, 0.25, 1.5, 2.0, n3)
        @test (Ls + Ms) isa TensWalpole{Float64,5}
        @test opequal(getarray(Ls + Ms), getarray(Ls) .+ getarray(Ms))
    end

    # ═══════════════════════════════════════════════════════════════════════════
    @testsection "TensWalpole — symbolic inverse" begin
        ℓ₁, ℓ₂, ℓ₃, ℓ₅, ℓ₆ = symbols("ℓ₁ ℓ₂ ℓ₃ ℓ₅ ℓ₆", real = true)
        L = TensWalpole(ℓ₁, ℓ₂, ℓ₃, ℓ₅, ℓ₆, n3s)
        Li = inv(L)
        @test Li isa TensWalpole{<:Any,5}
        # L⊡inv(L) should be identity
        𝕀 = tensId4(Val(3), Val(Sym))
        prod = L ⊡ Li
        for i in 1:3, j in 1:3, k in 1:3, l in 1:3
            @test simplify(prod[i,j,k,l] - 𝕀[i,j,k,l]) == 0
        end
    end

    # ═══════════════════════════════════════════════════════════════════════════
    @testsection "TensOrtho — construction & traits" begin
        frame3 = CanonicalBasis{3,Float64}()
        C11, C22, C33 = 10.0, 8.0, 12.0
        C12, C13, C23 = 3.0, 4.0, 2.5
        C44, C55, C66 = 2.0, 3.0, 1.5
        t = TensOrtho(C11, C22, C33, C12, C13, C23, C44, C55, C66, frame3)
        @test t isa TensOrtho{Float64}
        @test size(t) == (3, 3, 3, 3)
        @test getbasis(t) isa CanonicalBasis{3,Float64}
        @test getvar(t) == (:cont, :cont, :cont, :cont)
        @test getframe(t) === frame3
    end

    # ═══════════════════════════════════════════════════════════════════════════
    @testsection "TensOrtho — KM_material (canonical frame)" begin
        frame3 = CanonicalBasis{3,Float64}()
        C11, C22, C33 = 10.0, 8.0, 12.0
        C12, C13, C23 = 3.0, 4.0, 2.5
        C44, C55, C66 = 2.0, 3.0, 1.5
        t = TensOrtho(C11, C22, C33, C12, C13, C23, C44, C55, C66, frame3)
        Km = KM_material(t)
        @test size(Km) == (6, 6)
        # Diagonal blocks
        @test Km[1,1] ≈ C11  atol=atol_num
        @test Km[2,2] ≈ C22  atol=atol_num
        @test Km[3,3] ≈ C33  atol=atol_num
        @test Km[4,4] ≈ 2*C44 atol=atol_num
        @test Km[5,5] ≈ 2*C55 atol=atol_num
        @test Km[6,6] ≈ 2*C66 atol=atol_num
        # Off-diagonal within normal block
        @test Km[1,2] ≈ C12  atol=atol_num
        @test Km[1,3] ≈ C13  atol=atol_num
        @test Km[2,3] ≈ C23  atol=atol_num
        # Zeros between normal and shear blocks
        @test Km[1,4] ≈ 0.0  atol=atol_num
        @test Km[2,5] ≈ 0.0  atol=atol_num
        @test Km[3,6] ≈ 0.0  atol=atol_num
        @test Km[4,5] ≈ 0.0  atol=atol_num
        @test Km[4,6] ≈ 0.0  atol=atol_num
        @test Km[5,6] ≈ 0.0  atol=atol_num
        # Symmetry
        @test Km ≈ Km'  atol=atol_num
        # KM in canonical frame should match
        Km2 = KM(t)
        @test Km2 ≈ Km  atol=atol_num
    end

    # ═══════════════════════════════════════════════════════════════════════════
    @testsection "TensOrtho — from KM matrix" begin
        frame3 = CanonicalBasis{3,Float64}()
        C11, C22, C33 = 10.0, 8.0, 12.0
        C12, C13, C23 = 3.0, 4.0, 2.5
        C44, C55, C66 = 2.0, 3.0, 1.5
        t1 = TensOrtho(C11, C22, C33, C12, C13, C23, C44, C55, C66, frame3)
        Km = KM_material(t1)
        t2 = TensOrtho(Km, frame3)
        @test opequal(getarray(t1), getarray(t2))
    end

    # ═══════════════════════════════════════════════════════════════════════════
    @testsection "TensOrtho — inverse (canonical frame)" begin
        frame3 = CanonicalBasis{3,Float64}()
        C11, C22, C33 = 10.0, 8.0, 12.0
        C12, C13, C23 = 3.0, 4.0, 2.5
        C44, C55, C66 = 2.0, 3.0, 1.5
        t = TensOrtho(C11, C22, C33, C12, C13, C23, C44, C55, C66, frame3)
        ti = inv(t)
        @test ti isa TensOrtho{Float64}
        𝕀 = tensId4(Val(3), Val(Float64))
        A  = Tensor{4,3}(getarray(t))
        Ai = Tensor{4,3}(getarray(ti))
        prod = Array(A ⊡ Ai)
        𝕀arr = getarray(𝕀)
        @test opequal(prod, 𝕀arr)
    end

    # ═══════════════════════════════════════════════════════════════════════════
    @testsection "TensOrtho — rotated frame" begin
        # Rotate material frame: e₁→e₂, e₂→e₃, e₃→e₁ (cyclic permutation)
        R = Float64[0 0 1; 1 0 0; 0 1 0]
        frame_rot = RotatedBasis(R)
        C11, C22, C33 = 10.0, 10.0, 12.0
        C12, C13, C23 = 3.0, 3.0, 3.0
        C44, C55, C66 = 2.0, 2.0, 2.0
        t = TensOrtho(C11, C22, C33, C12, C13, C23, C44, C55, C66, frame_rot)
        # KM_material should still have block-diagonal structure
        Km = KM_material(t)
        @test Km[1,4] ≈ 0.0  atol=atol_num
        @test Km[4,5] ≈ 0.0  atol=atol_num
        @test Km[5,6] ≈ 0.0  atol=atol_num
        # change_tens to canonical basis should give a valid Tens
        tc = change_tens(t, CanonicalBasis{3,Float64}())
        @test tc isa AbstractTens
    end

    # ═══════════════════════════════════════════════════════════════════════════
    @testsection "TensOrtho — TI consistency" begin
        # A TI tensor (C11=C22, C13=C23, C44=C55, C66=(C11-C12)/2)
        # should give same array as corresponding TensWalpole (with n=e₃)
        frame3 = CanonicalBasis{3,Float64}()
        C11 = 10.0; C33 = 12.0; C12 = 3.0; C13 = 2.5; C44 = 2.0
        C22 = C11; C23 = C13; C55 = C44; C66 = (C11 - C12) / 2
        to = TensOrtho(C11, C22, C33, C12, C13, C23, C44, C55, C66, frame3)

        # Build equivalent TensWalpole (n=e₃) via the engineering constants.
        # For a TI material with n=e₃:
        #   C₁₁=C₂₂=(ℓ₂+ℓ₅)/2, C₃₃=ℓ₁, C₁₂=(ℓ₂-ℓ₅)/2, C₁₃=C₂₃=ℓ₃/√2, C₄₄=ℓ₆/2, C₆₆=ℓ₅/2
        # Inverting: ℓ₂=C₁₁+C₁₂, ℓ₅=C₁₁−C₁₂=2C₆₆, ℓ₃=C₁₃√2, ℓ₆=2C₄₄
        sq2 = sqrt(2.0)
        ℓ₁ = C33
        ℓ₅ = C66 * 2       # = C11 - C12
        ℓ₂ = C11 + C12     # NOT (C11+C12)/2
        ℓ₃ = C13 * sq2
        ℓ₆ = C44 * 2
        tw = TensWalpole(ℓ₁, ℓ₂, ℓ₃, ℓ₅, ℓ₆, n3)
        @test opequal(getarray(to), getarray(tw))
    end

    # ═══════════════════════════════════════════════════════════════════════════
    @testsection "TensWalpole — dcontract with TensISO" begin
        𝕀, 𝕁, 𝕂 = ISO(Val(3), Val(Float64))
        α, β = 3.0, 2.0
        ℂiso = α * 𝕁 + β * 𝕂
        L = TensWalpole(2.0, 3.0, 1.0, 4.0, 5.0, n3)   # N=5

        # L⊡ℂiso via Walpole product rule (convert ISO first)
        res_w = L ⊡ ℂiso
        @test res_w isa TensWalpole
        # Compare with direct array contraction
        res_direct = Tensor{4,3}(getarray(L)) ⊡ Tensor{4,3}(getarray(ℂiso))
        @test opequal(getarray(res_w), Array(res_direct))
    end

end  # "Walpole & Ortho tensors"
