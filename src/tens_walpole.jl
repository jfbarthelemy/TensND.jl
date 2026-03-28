
##############################################################################
# TensWalpole — transversely isotropic 4th-order tensors (Walpole basis)    #
# TensOrtho  — orthotropic 4th-order tensors                                #
##############################################################################

# ─────────────────────────────────────────────────────────────────────────────
# TensWalpole
# ─────────────────────────────────────────────────────────────────────────────
#
# A transversely isotropic (TI) 4th-order tensor with symmetry axis n can be
# written in the Walpole basis {W₁,…,W₆} as
#
#   L = ℓ₁W₁ + ℓ₂W₂ + ℓ₃W₃ + ℓ₄W₄ + ℓ₅W₅ + ℓ₆W₆
#
# where (nₙ = n⊗n, nT = 1 − nₙ):
#   W₁ = nₙ⊗nₙ
#   W₂ = (nT⊗nT)/2
#   W₃ = (nₙ⊗nT)/√2
#   W₄ = (nT⊗nₙ)/√2
#   W₅ = nT⊠ˢnT − (nT⊗nT)/2
#   W₆ = nT⊠ˢnₙ + nₙ⊠ˢnT
#
# For major-symmetric tensors ℓ₃ = ℓ₄ → stored with N=5 data scalars.
# General (non-major-sym) tensors use N=6.
#
# Synthetic notation: L ≡ ([[ℓ₁,ℓ₃],[ℓ₄,ℓ₂]], ℓ₅, ℓ₆)
#   Product:  (L⊡M)_mat = L_mat × M_mat  ,  (L⊡M)₅ = ℓ₅m₅ , (L⊡M)₆ = ℓ₆m₆
#   Inverse:  (L⁻¹)_mat = (L_mat)⁻¹      ,  1/ℓ₅           , 1/ℓ₆
# ─────────────────────────────────────────────────────────────────────────────

"""
    TensWalpole{T,N} <: AbstractTens{4,3,T}

Transversely isotropic 4th-order tensor stored in the Walpole basis {W₁,…,W₆}
with symmetry axis `n` (assumed unit vector):

    L = ℓ₁W₁ + ℓ₂W₂ + ℓ₃W₃ + ℓ₄W₄ + ℓ₅W₅ + ℓ₆W₆

where (`nₙ = n⊗n`, `nT = 𝟏 − nₙ`):

| Tensor | Expression |
|--------|-----------|
| W₁ | `nₙ⊗nₙ` |
| W₂ | `(nT⊗nT)/2` |
| W₃ | `(nₙ⊗nT)/√2` |
| W₄ | `(nT⊗nₙ)/√2` |
| W₅ | `nT⊠ˢnT − (nT⊗nT)/2` |
| W₆ | `nT⊠ˢnₙ + nₙ⊠ˢnT` |

`N=5` (major-symmetric, `ℓ₃=ℓ₄`): `data=(ℓ₁,ℓ₂,ℓ₃,ℓ₅,ℓ₆)`.
`N=6` (general): `data=(ℓ₁,ℓ₂,ℓ₃,ℓ₄,ℓ₅,ℓ₆)`.

Synthetic notation: `L ≡ ([[ℓ₁,ℓ₃],[ℓ₄,ℓ₂]], ℓ₅, ℓ₆)`.
- Double contraction: `(L⊡M)_mat = L_mat × M_mat`, `(L⊡M)₅ = ℓ₅m₅`, `(L⊡M)₆ = ℓ₆m₆`
- Inverse: `(L⁻¹)_mat = (L_mat)⁻¹`, `1/ℓ₅`, `1/ℓ₆`
"""
struct TensWalpole{T,N} <: AbstractTens{4,3,T}
    data::NTuple{N,T}   # N=5: (ℓ₁,ℓ₂,ℓ₃,ℓ₅,ℓ₆)  N=6: (ℓ₁,ℓ₂,ℓ₃,ℓ₄,ℓ₅,ℓ₆)
    n::NTuple{3,T}      # symmetry axis (assumed to be a unit vector)
end

# ── Traits ────────────────────────────────────────────────────────────────────

@pure Base.eltype(::Type{TensWalpole{T,N}}) where {T,N} = T
@pure Base.length(::TensWalpole) = 81   # 3^4
@pure Base.size(::TensWalpole)   = (3, 3, 3, 3)

getbasis(::TensWalpole{T}) where {T} = CanonicalBasis{3,T}()
getvar(::TensWalpole)                = (:cont, :cont, :cont, :cont)
getvar(::TensWalpole, ::Integer)     = :cont
getdata(t::TensWalpole) = t.data

# ── Rebuild helper (used by symbolic ops) ─────────────────────────────────────
_rebuild(t::TensWalpole, new_data) =
    TensWalpole{eltype(new_data), length(new_data)}(new_data, getaxis(t))

# ── Accessors ─────────────────────────────────────────────────────────────────

"""
    get_ℓ(t::TensWalpole) → NTuple{6}

Always returns a 6-tuple `(ℓ₁,ℓ₂,ℓ₃,ℓ₄,ℓ₅,ℓ₆)`.
For N=5 (symmetric), ℓ₃ = ℓ₄ is stored once so data[3] is duplicated.
"""
get_ℓ(t::TensWalpole{T,5}) where {T} =
    (t.data[1], t.data[2], t.data[3], t.data[3], t.data[4], t.data[5])
get_ℓ(t::TensWalpole{T,6}) where {T} = t.data

"""
    getaxis(t::TensWalpole) → NTuple{3}

Returns the symmetry axis as a 3-tuple.
"""
getaxis(t::TensWalpole) = t.n

# Helper: 2×2 Walpole matrix [[ℓ₁,ℓ₃],[ℓ₄,ℓ₂]]
function _walpole_mat(t::TensWalpole)
    ℓ₁, ℓ₂, ℓ₃, ℓ₄ = get_ℓ(t)[1:4]
    return SMatrix{2,2}(ℓ₁, ℓ₄, ℓ₃, ℓ₂)   # column-major: [col1, col2] = [[ℓ₁,ℓ₄],[ℓ₃,ℓ₂]]
end

# ── Constructors ──────────────────────────────────────────────────────────────

"""
    TensWalpole(ℓ₁,ℓ₂,ℓ₃,ℓ₄,ℓ₅,ℓ₆, n) → TensWalpole{T,6}

General (not necessarily major-symmetric) Walpole tensor with axis `n`.
"""
function TensWalpole(ℓ₁, ℓ₂, ℓ₃, ℓ₄, ℓ₅, ℓ₆, n)
    T = promote_type(typeof(ℓ₁), typeof(ℓ₂), typeof(ℓ₃), typeof(ℓ₄),
                     typeof(ℓ₅), typeof(ℓ₆), eltype(n))
    nv = _extract_vec(n)
    TensWalpole{T,6}((T(ℓ₁), T(ℓ₂), T(ℓ₃), T(ℓ₄), T(ℓ₅), T(ℓ₆)),
                     (T(nv[1]), T(nv[2]), T(nv[3])))
end

"""
    TensWalpole(ℓ₁,ℓ₂,ℓ₃,ℓ₅,ℓ₆, n) → TensWalpole{T,5}

Major-symmetric Walpole tensor (ℓ₃ = ℓ₄), 5 independent scalars, with axis `n`.
"""
function TensWalpole(ℓ₁, ℓ₂, ℓ₃, ℓ₅, ℓ₆, n)
    T = promote_type(typeof(ℓ₁), typeof(ℓ₂), typeof(ℓ₃),
                     typeof(ℓ₅), typeof(ℓ₆), eltype(n))
    nv = _extract_vec(n)
    TensWalpole{T,5}((T(ℓ₁), T(ℓ₂), T(ℓ₃), T(ℓ₅), T(ℓ₆)),
                     (T(nv[1]), T(nv[2]), T(nv[3])))
end

# Extract a plain 3-vector from various input types
_extract_vec(n::NTuple{3}) = n
_extract_vec(n::AbstractVector) = (n[1], n[2], n[3])
_extract_vec(n::AbstractTens) = _extract_vec(getarray(n))
_extract_vec(n::Vec{3}) = (n[1], n[2], n[3])
_extract_vec(n::AbstractArray) = (n[1], n[2], n[3])

# ── Basis tensors Wᵢ ─────────────────────────────────────────────────────────

"""
    tensW1(n) → TensWalpole{T,6}   (W₁ = nₙ⊗nₙ, coeffs (1,0,0,0,0,0))
"""
tensW1(n) = TensWalpole(one(eltype_of(n)), zero(eltype_of(n)), zero(eltype_of(n)),
                        zero(eltype_of(n)), zero(eltype_of(n)), zero(eltype_of(n)), n)

"""
    tensW2(n) → TensWalpole{T,6}   (W₂ = (nT⊗nT)/2, coeffs (0,1,0,0,0,0))
"""
tensW2(n) = TensWalpole(zero(eltype_of(n)), one(eltype_of(n)), zero(eltype_of(n)),
                        zero(eltype_of(n)), zero(eltype_of(n)), zero(eltype_of(n)), n)

"""
    tensW3(n) → TensWalpole{T,6}   (W₃ = (nₙ⊗nT)/√2, coeffs (0,0,1,0,0,0))
"""
tensW3(n) = TensWalpole(zero(eltype_of(n)), zero(eltype_of(n)), one(eltype_of(n)),
                        zero(eltype_of(n)), zero(eltype_of(n)), zero(eltype_of(n)), n)

"""
    tensW4(n) → TensWalpole{T,6}   (W₄ = (nT⊗nₙ)/√2, coeffs (0,0,0,1,0,0))
"""
tensW4(n) = TensWalpole(zero(eltype_of(n)), zero(eltype_of(n)), zero(eltype_of(n)),
                        one(eltype_of(n)), zero(eltype_of(n)), zero(eltype_of(n)), n)

"""
    tensW5(n) → TensWalpole{T,6}   (W₅ = nT⊠ˢnT − (nT⊗nT)/2, coeffs (0,0,0,0,1,0))
"""
tensW5(n) = TensWalpole(zero(eltype_of(n)), zero(eltype_of(n)), zero(eltype_of(n)),
                        zero(eltype_of(n)), one(eltype_of(n)), zero(eltype_of(n)), n)

"""
    tensW6(n) → TensWalpole{T,6}   (W₆ = nT⊠ˢnₙ + nₙ⊠ˢnT, coeffs (0,0,0,0,0,1))
"""
tensW6(n) = TensWalpole(zero(eltype_of(n)), zero(eltype_of(n)), zero(eltype_of(n)),
                        zero(eltype_of(n)), zero(eltype_of(n)), one(eltype_of(n)), n)

# Helper: get element type from various axis representations
eltype_of(::AbstractArray{T}) where {T} = T
eltype_of(::NTuple{N,T}) where {N,T}    = T
eltype_of(::AbstractTens{1,3,T}) where {T} = T

"""
    Walpole(n)           → (W₁,W₂,W₃,W₄,W₅,W₆)
    Walpole(n; sym=true) → (W₁ˢ,W₂ˢ,W₃ˢ,W₄ˢ,W₅ˢ) where W₃ˢ = W₃+W₄
"""
function Walpole(n; sym::Bool = false)
    if sym
        T = eltype_of(n)
        o, z = one(T), zero(T)
        W1s = TensWalpole(o, z, z, z, z, n)         # N=5: ℓ₁=1
        W2s = TensWalpole(z, o, z, z, z, n)         # N=5: ℓ₂=1
        W3s = TensWalpole(z, z, o, z, z, n)         # N=5: ℓ₃=1  (W₃+W₄)
        W4s = TensWalpole(z, z, z, o, z, n)         # N=5: ℓ₅=1
        W5s = TensWalpole(z, z, z, z, o, n)         # N=5: ℓ₆=1
        return W1s, W2s, W3s, W4s, W5s
    else
        return tensW1(n), tensW2(n), tensW3(n), tensW4(n), tensW5(n), tensW6(n)
    end
end

# ── getarray ─────────────────────────────────────────────────────────────────

"""
    getarray(t::TensWalpole{T}) → Array{T,4}

Compute the 3×3×3×3 component array from the Walpole coefficients and axis.
"""
function getarray(t::TensWalpole{T}) where {T}
    ℓ₁, ℓ₂, ℓ₃, ℓ₄, ℓ₅, ℓ₆ = get_ℓ(t)
    n = t.n
    sq2 = sqrt(T(2))
    δ(i, j) = i == j ? one(T) : zero(T)
    nn(i, j) = n[i] * n[j]
    nT(i, j) = δ(i, j) - nn(i, j)
    result = Array{T,4}(undef, 3, 3, 3, 3)
    for i in 1:3, j in 1:3, k in 1:3, l in 1:3
        W1 = nn(i,j) * nn(k,l)
        W2 = nT(i,j) * nT(k,l) / 2
        W3 = nn(i,j) * nT(k,l) / sq2
        W4 = nT(i,j) * nn(k,l) / sq2
        W5 = (nT(i,k)*nT(j,l) + nT(i,l)*nT(j,k)) / 2 - nT(i,j)*nT(k,l) / 2
        W6 = (nT(i,k)*nn(j,l) + nT(i,l)*nn(j,k) + nn(i,k)*nT(j,l) + nn(i,l)*nT(j,k)) / 2
        result[i,j,k,l] = ℓ₁*W1 + ℓ₂*W2 + ℓ₃*W3 + ℓ₄*W4 + ℓ₅*W5 + ℓ₆*W6
    end
    return result
end

Base.getindex(t::TensWalpole, i::Integer, j::Integer, k::Integer, l::Integer) =
    getarray(t)[i, j, k, l]

# ── Kelvin-Mandel matrix ──────────────────────────────────────────────────────

"""
    KM(t::TensWalpole)

Kelvin-Mandel (6×6) matrix of the Walpole tensor.
"""
KM(t::TensWalpole) = tomandel(tensor_or_array(getarray(t)))

# ── Arithmetic ────────────────────────────────────────────────────────────────

@inline Base.:-(A::TensWalpole{T,N}) where {T,N} =
    TensWalpole{T,N}(.-(getdata(A)), getaxis(A))

@inline Base.:*(α::Number, A::TensWalpole{T,N}) where {T,N} =
    TensWalpole{T,N}(α .* getdata(A), getaxis(A))
@inline Base.:*(A::TensWalpole{T,N}, α::Number) where {T,N} =
    TensWalpole{T,N}(getdata(A) .* α, getaxis(A))
@inline Base.:/(A::TensWalpole{T,N}, α::Number) where {T,N} =
    TensWalpole{T,N}(getdata(A) ./ α, getaxis(A))

@inline function Base.:+(A::TensWalpole{T1,N}, B::TensWalpole{T2,N}) where {T1,T2,N}
    @assert A.n == B.n "TensWalpole addition requires the same axis"
    TensWalpole{promote_type(T1,T2),N}(getdata(A) .+ getdata(B), A.n)
end
@inline function Base.:-(A::TensWalpole{T1,N}, B::TensWalpole{T2,N}) where {T1,T2,N}
    @assert A.n == B.n "TensWalpole subtraction requires the same axis"
    TensWalpole{promote_type(T1,T2),N}(getdata(A) .- getdata(B), A.n)
end

# ── Double contraction (Walpole product rule) ─────────────────────────────────

"""
    dcontract(A::TensWalpole, B::TensWalpole) → TensWalpole{T,6}

Product rule via 2×2 matrix product + scalar products for ℓ₅, ℓ₆.
Always returns N=6 since the product of two symmetric tensors need not be symmetric.
"""
function Tensors.dcontract(A::TensWalpole, B::TensWalpole)
    @assert A.n == B.n "dcontract(TensWalpole,TensWalpole) requires the same axis"
    ℓA₁, ℓA₂, ℓA₃, ℓA₄, ℓA₅, ℓA₆ = get_ℓ(A)
    ℓB₁, ℓB₂, ℓB₃, ℓB₄, ℓB₅, ℓB₆ = get_ℓ(B)
    # 2×2 matrix rule: M_A × M_B where M = [[ℓ₁,ℓ₃],[ℓ₄,ℓ₂]]
    n₁ = ℓA₁*ℓB₁ + ℓA₃*ℓB₄
    n₃ = ℓA₁*ℓB₃ + ℓA₃*ℓB₂
    n₄ = ℓA₄*ℓB₁ + ℓA₂*ℓB₄
    n₂ = ℓA₄*ℓB₃ + ℓA₂*ℓB₂
    n₅ = ℓA₅ * ℓB₅
    n₆ = ℓA₆ * ℓB₆
    T = promote_type(eltype(A), eltype(B))
    return TensWalpole{T,6}((T(n₁), T(n₂), T(n₃), T(n₄), T(n₅), T(n₆)), A.n)
end

# ── Inverse ───────────────────────────────────────────────────────────────────

"""
    inv(t::TensWalpole{T,5}) → TensWalpole{T,5}
    inv(t::TensWalpole{T,6}) → TensWalpole{T,6}

Inverse via the 2×2 Walpole matrix and scalar inverses for ℓ₅, ℓ₆.
"""
function Base.inv(t::TensWalpole{T,5}) where {T}
    ℓ₁, ℓ₂, ℓ₃, _, ℓ₅, ℓ₆ = get_ℓ(t)   # ℓ₄=ℓ₃ for N=5
    det = ℓ₁*ℓ₂ - ℓ₃*ℓ₃
    TensWalpole{T,5}((ℓ₂/det, ℓ₁/det, -ℓ₃/det, one(T)/ℓ₅, one(T)/ℓ₆), t.n)
end

function Base.inv(t::TensWalpole{T,6}) where {T}
    ℓ₁, ℓ₂, ℓ₃, ℓ₄, ℓ₅, ℓ₆ = get_ℓ(t)
    det = ℓ₁*ℓ₂ - ℓ₃*ℓ₄
    TensWalpole{T,6}((ℓ₂/det, ℓ₁/det, -ℓ₃/det, -ℓ₄/det, one(T)/ℓ₅, one(T)/ℓ₆), t.n)
end

@inline Base.literal_pow(::typeof(^), A::TensWalpole, ::Val{-1}) = inv(A)

# ── Symmetry tests ────────────────────────────────────────────────────────────

LinearAlgebra.issymmetric(::TensWalpole{T,5}) where {T} = true
LinearAlgebra.issymmetric(t::TensWalpole{T,6}) where {T} = isequal(t.data[3], t.data[4])
Tensors.isminorsymmetric(::TensWalpole) = true
Tensors.ismajorsymmetric(::TensWalpole{T,5}) where {T} = true
Tensors.ismajorsymmetric(t::TensWalpole{T,6}) where {T} = isequal(t.data[3], t.data[4])

# ── fromISO ───────────────────────────────────────────────────────────────────

"""
    fromISO(A::TensISO{4,3}, n) → TensWalpole{T,5}

Convert an isotropic 4th-order tensor `αJ + βK` into its Walpole representation.

Formulas: ℓ₁=(α+2β)/3, ℓ₂=(2α+β)/3 (note: dim=3 → these are (3k,2μ) related),
          ℓ₃=ℓ₄=√2(α−β)/3, ℓ₅=ℓ₆=β.
Here `α` = data[1] and `β` = data[2] in TensISO (coefficients of J and K).
"""
function fromISO(A::TensISO{4,3,T}, n) where {T}
    α, β = getdata(A)    # A = α*J + β*K
    sq2 = sqrt(T(2))
    ℓ₁ = (α + 2β) / 3
    ℓ₂ = (2α + β) / 3   # Note: for 3D, 1-1/dim = 2/3 and 1/dim = 1/3
    ℓ₃ = sq2 * (α - β) / 3
    ℓ₅ = β
    ℓ₆ = β
    TensWalpole(ℓ₁, ℓ₂, ℓ₃, ℓ₅, ℓ₆, n)
end

"""
    dcontract(A::TensWalpole, B::TensISO{4,3}) → TensWalpole{T,6}
    dcontract(A::TensISO{4,3}, B::TensWalpole) → TensWalpole{T,6}
"""
function Tensors.dcontract(A::TensWalpole, B::TensISO{4,3})
    Tensors.dcontract(A, fromISO(B, A.n))
end
function Tensors.dcontract(A::TensISO{4,3}, B::TensWalpole)
    Tensors.dcontract(fromISO(A, B.n), B)
end

# ── isISO / isTI ─────────────────────────────────────────────────────────────

"""
    isTI(A)

Return `true` if `A` is a `TensWalpole`, indicating transverse isotropy.
"""
isTI(::TensWalpole)     = true
isTI(::Any)             = false
isISO(::TensWalpole)    = false
isOrtho(::TensWalpole)  = false

# ── Symbolic helpers ──────────────────────────────────────────────────────────

for OP in (:tsimplify, :tfactor, :tsubs, :tdiff, :ttrigsimp, :texpand_trig)
    @eval $OP(A::TensWalpole, args...; kwargs...) =
        _rebuild(A, $OP(getdata(A), args...; kwargs...))
end
# Explicit Num dispatch to avoid ambiguity with Symbolics.jl
for OP in (:tsimplify, :tsubs, :tdiff)
    @eval $OP(A::TensWalpole{Num,N}, args...; kwargs...) where {N} =
        _rebuild(A, $OP(getdata(A), args...; kwargs...))
end

# ── Display ───────────────────────────────────────────────────────────────────

function Base.show(io::IO, A::TensWalpole{<:Any,5})
    ℓ₁, ℓ₂, ℓ₃, _, ℓ₅, ℓ₆ = get_ℓ(A)
    print(io, "(", ℓ₁, ") W₁ˢ + (", ℓ₂, ") W₂ˢ + (", ℓ₃,
             ") W₃ˢ + (", ℓ₅, ") W₄ˢ + (", ℓ₆, ") W₅ˢ")
    print(io, "\n  axis n = ", A.n)
end
function Base.show(io::IO, A::TensWalpole{<:Any,6})
    ℓ₁, ℓ₂, ℓ₃, ℓ₄, ℓ₅, ℓ₆ = get_ℓ(A)
    print(io, "(", ℓ₁, ") W₁ + (", ℓ₂, ") W₂ + (", ℓ₃,
             ") W₃ + (", ℓ₄, ") W₄ + (", ℓ₅, ") W₅ + (", ℓ₆, ") W₆")
    print(io, "\n  axis n = ", A.n)
end

##############################################################################
# TensOrtho — orthotropic 4th-order tensor
##############################################################################
#
# In the material frame (e₁,e₂,e₃) with Pₘ = eₘ⊗eₘ:
#
#   ℂ = C₁₁P₁⊗P₁ + C₂₂P₂⊗P₂ + C₃₃P₃⊗P₃
#     + C₁₂(P₁⊗P₂+P₂⊗P₁) + C₁₃(P₁⊗P₃+P₃⊗P₁) + C₂₃(P₂⊗P₃+P₃⊗P₂)
#     + 2C₄₄(P₂⊠ˢP₃) + 2C₅₅(P₁⊠ˢP₃) + 2C₆₆(P₁⊠ˢP₂)
#
# where C₄₄=C₂₃₂₃, C₅₅=C₁₃₁₃, C₆₆=C₁₂₁₂.
#
# KM in the material frame (Kelvin-Mandel, ordering 11,22,33,23,13,12):
#
#   [[C₁₁,C₁₂,C₁₃, 0,  0,  0 ],
#    [C₁₂,C₂₂,C₂₃, 0,  0,  0 ],
#    [C₁₃,C₂₃,C₃₃, 0,  0,  0 ],
#    [ 0,  0,  0, 2C₄₄, 0,  0 ],
#    [ 0,  0,  0,  0, 2C₅₅, 0 ],
#    [ 0,  0,  0,  0,  0, 2C₆₆]]
# ─────────────────────────────────────────────────────────────────────────────

"""
    TensOrtho{T} <: AbstractTens{4,3,T}

Orthotropic 4th-order tensor with material frame `(e₁,e₂,e₃)` and 9 independent
elastic constants `(C₁₁,C₂₂,C₃₃,C₁₂,C₁₃,C₂₃,C₄₄,C₅₅,C₆₆)` where
`C₄₄=C₂₃₂₃`, `C₅₅=C₁₃₁₃`, `C₆₆=C₁₂₁₂`:

    ℂ = C₁₁P₁⊗P₁ + C₂₂P₂⊗P₂ + C₃₃P₃⊗P₃
      + C₁₂(P₁⊗P₂+P₂⊗P₁) + C₁₃(P₁⊗P₃+P₃⊗P₁) + C₂₃(P₂⊗P₃+P₃⊗P₂)
      + 2C₄₄(P₂⊠ˢP₃) + 2C₅₅(P₁⊠ˢP₃) + 2C₆₆(P₁⊠ˢP₂)

with `Pₘ = eₘ⊗eₘ`. The Kelvin-Mandel matrix in the material frame is block-diagonal:

    [[C₁₁,C₁₂,C₁₃, 0,   0,   0  ],
     [C₁₂,C₂₂,C₂₃, 0,   0,   0  ],
     [C₁₃,C₂₃,C₃₃, 0,   0,   0  ],
     [ 0,  0,  0,  2C₄₄, 0,   0  ],
     [ 0,  0,  0,   0,  2C₅₅, 0  ],
     [ 0,  0,  0,   0,   0,  2C₆₆]]
"""
struct TensOrtho{T} <: AbstractTens{4,3,T}
    data::NTuple{9,T}            # (C₁₁,C₂₂,C₃₃,C₁₂,C₁₃,C₂₃,C₄₄,C₅₅,C₆₆)
    frame::OrthonormalBasis{3,T} # material frame (e₁,e₂,e₃)
end

# ── Traits ────────────────────────────────────────────────────────────────────

@pure Base.eltype(::Type{TensOrtho{T}}) where {T} = T
@pure Base.length(::TensOrtho) = 81
@pure Base.size(::TensOrtho)   = (3, 3, 3, 3)

getbasis(::TensOrtho{T}) where {T}  = CanonicalBasis{3,T}()
getvar(::TensOrtho)                  = (:cont, :cont, :cont, :cont)
getvar(::TensOrtho, ::Integer)       = :cont
getdata(t::TensOrtho)               = t.data
getframe(t::TensOrtho)              = t.frame

# ── Rebuild helper (used by symbolic ops) ─────────────────────────────────────
_rebuild(t::TensOrtho, new_data) = TensOrtho{eltype(new_data)}(new_data, getframe(t))

# ── Constructors ──────────────────────────────────────────────────────────────

"""
    TensOrtho(C11,C22,C33,C12,C13,C23,C44,C55,C66, frame)

Orthotropic tensor from the 9 elastic constants in the material frame `frame`.
"""
function TensOrtho(C11, C22, C33, C12, C13, C23, C44, C55, C66,
                   frame::OrthonormalBasis{3})
    T = promote_type(typeof(C11), typeof(C22), typeof(C33),
                     typeof(C12), typeof(C13), typeof(C23),
                     typeof(C44), typeof(C55), typeof(C66), eltype(frame))
    TensOrtho{T}((T(C11), T(C22), T(C33), T(C12), T(C13), T(C23),
                  T(C44), T(C55), T(C66)), frame)
end

"""
    TensOrtho(KMmat::AbstractMatrix, frame)

Build a `TensOrtho` from a 6×6 Kelvin-Mandel matrix expressed in the material frame.
The matrix must have the block-diagonal orthotropic structure:
upper-left 3×3 for normal stresses and lower-right 3×3 diagonal for shear.
"""
function TensOrtho(KMmat::AbstractMatrix, frame::OrthonormalBasis{3})
    T = eltype(KMmat)
    C11 = KMmat[1,1]; C22 = KMmat[2,2]; C33 = KMmat[3,3]
    C12 = KMmat[1,2]; C13 = KMmat[1,3]; C23 = KMmat[2,3]
    C44 = KMmat[4,4] / 2
    C55 = KMmat[5,5] / 2
    C66 = KMmat[6,6] / 2
    TensOrtho{T}((T(C11), T(C22), T(C33), T(C12), T(C13), T(C23),
                  T(C44), T(C55), T(C66)), frame)
end

# ── getarray ─────────────────────────────────────────────────────────────────

"""
    getarray(t::TensOrtho{T}) → Array{T,4}

Compute the 3×3×3×3 component array in the canonical frame.
"""
function getarray(t::TensOrtho{T}) where {T}
    C11, C22, C33, C12, C13, C23, C44, C55, C66 = getdata(t)
    # Frame vectors as columns of vecbasis(frame, :cov) → e[m] = frame vector m
    E = vecbasis(t.frame, :cov)   # 3×3 matrix, column m = eₘ
    result = Array{T,4}(undef, 3, 3, 3, 3)
    # Pₘ[i,j] = E[i,m]*E[j,m]
    P(m, i, j) = E[i,m] * E[j,m]
    # (A ⊠ˢ B)[i,j,k,l] = (A[i,k]*B[j,l] + A[i,l]*B[j,k] + A[j,k]*B[i,l] + A[j,l]*B[i,k])/4
    # Note: the factor 2C in the formula accounts for the 2 in "2Cₘₘ(Pₘ⊠ˢPₙ + Pₙ⊠ˢPₘ)"
    # which is the standard Voigt-to-tensor conversion for shear moduli.
    for i in 1:3, j in 1:3, k in 1:3, l in 1:3
        val = (C11 * P(1,i,j)*P(1,k,l)
             + C22 * P(2,i,j)*P(2,k,l)
             + C33 * P(3,i,j)*P(3,k,l)
             + C12 * (P(1,i,j)*P(2,k,l) + P(2,i,j)*P(1,k,l))
             + C13 * (P(1,i,j)*P(3,k,l) + P(3,i,j)*P(1,k,l))
             + C23 * (P(2,i,j)*P(3,k,l) + P(3,i,j)*P(2,k,l))
             + C44 * (E[i,2]*E[k,3]*E[j,3]*E[l,2] + E[i,2]*E[l,3]*E[j,3]*E[k,2] +
                      E[j,2]*E[k,3]*E[i,3]*E[l,2] + E[j,2]*E[l,3]*E[i,3]*E[k,2] +
                      E[i,3]*E[k,2]*E[j,2]*E[l,3] + E[i,3]*E[l,2]*E[j,2]*E[k,3] +
                      E[j,3]*E[k,2]*E[i,2]*E[l,3] + E[j,3]*E[l,2]*E[i,2]*E[k,3]) / 2
             + C55 * (E[i,1]*E[k,3]*E[j,3]*E[l,1] + E[i,1]*E[l,3]*E[j,3]*E[k,1] +
                      E[j,1]*E[k,3]*E[i,3]*E[l,1] + E[j,1]*E[l,3]*E[i,3]*E[k,1] +
                      E[i,3]*E[k,1]*E[j,1]*E[l,3] + E[i,3]*E[l,1]*E[j,1]*E[k,3] +
                      E[j,3]*E[k,1]*E[i,1]*E[l,3] + E[j,3]*E[l,1]*E[i,1]*E[k,3]) / 2
             + C66 * (E[i,1]*E[k,2]*E[j,2]*E[l,1] + E[i,1]*E[l,2]*E[j,2]*E[k,1] +
                      E[j,1]*E[k,2]*E[i,2]*E[l,1] + E[j,1]*E[l,2]*E[i,2]*E[k,1] +
                      E[i,2]*E[k,1]*E[j,1]*E[l,2] + E[i,2]*E[l,1]*E[j,1]*E[k,2] +
                      E[j,2]*E[k,1]*E[i,1]*E[l,2] + E[j,2]*E[l,1]*E[i,1]*E[k,2]) / 2)
        result[i,j,k,l] = val
    end
    return result
end

Base.getindex(t::TensOrtho, i::Integer, j::Integer, k::Integer, l::Integer) =
    getarray(t)[i, j, k, l]

# ── KM in the material frame ──────────────────────────────────────────────────

"""
    KM(t::TensOrtho)

Returns the 6×6 Kelvin-Mandel matrix in the **canonical** frame.
Use `KM_material(t)` for the block-diagonal form in the material frame.
"""
KM(t::TensOrtho) = tomandel(tensor_or_array(getarray(t)))

"""
    KM_material(t::TensOrtho)

Returns the 6×6 Kelvin-Mandel matrix in the material frame (block-diagonal).
"""
function KM_material(t::TensOrtho{T}) where {T}
    C11, C22, C33, C12, C13, C23, C44, C55, C66 = getdata(t)
    z = zero(T)
    return [C11  C12  C13   z    z    z  ;
            C12  C22  C23   z    z    z  ;
            C13  C23  C33   z    z    z  ;
             z    z    z  2C44   z    z  ;
             z    z    z    z  2C55   z  ;
             z    z    z    z    z  2C66 ]
end

# ── Arithmetic ────────────────────────────────────────────────────────────────

@inline Base.:-(A::TensOrtho{T}) where {T} =
    TensOrtho{T}(.-(getdata(A)), getframe(A))
@inline Base.:*(α::Number, A::TensOrtho{T}) where {T} =
    TensOrtho{T}(α .* getdata(A), getframe(A))
@inline Base.:*(A::TensOrtho{T}, α::Number) where {T} =
    TensOrtho{T}(getdata(A) .* α, getframe(A))
@inline Base.:/(A::TensOrtho{T}, α::Number) where {T} =
    TensOrtho{T}(getdata(A) ./ α, getframe(A))

@inline function Base.:+(A::TensOrtho{T1}, B::TensOrtho{T2}) where {T1,T2}
    @assert A.frame == B.frame "TensOrtho addition requires the same material frame"
    T = promote_type(T1,T2)
    TensOrtho{T}(getdata(A) .+ getdata(B), A.frame)
end
@inline function Base.:-(A::TensOrtho{T1}, B::TensOrtho{T2}) where {T1,T2}
    @assert A.frame == B.frame "TensOrtho subtraction requires the same material frame"
    T = promote_type(T1,T2)
    TensOrtho{T}(getdata(A) .- getdata(B), A.frame)
end

# ── Inverse ───────────────────────────────────────────────────────────────────

"""
    inv(t::TensOrtho) → TensOrtho

Inverse via the KM matrix in the material frame (block-diagonal, efficiently invertible).
"""
function Base.inv(t::TensOrtho{T}) where {T}
    Km = KM_material(t)
    Km_inv = inv(Km)
    TensOrtho(Km_inv, t.frame)
end

@inline Base.literal_pow(::typeof(^), A::TensOrtho, ::Val{-1}) = inv(A)

# ── Symmetry ──────────────────────────────────────────────────────────────────

LinearAlgebra.issymmetric(::TensOrtho)    = true
Tensors.isminorsymmetric(::TensOrtho)     = true
Tensors.ismajorsymmetric(::TensOrtho)     = true

# ── isISO / isTI / isOrtho ───────────────────────────────────────────────────

isISO(::TensOrtho)   = false
isTI(::TensOrtho)    = false
isOrtho(::TensOrtho) = true
isOrtho(::Any)       = false   # universal fallback

# ── Symbolic helpers ──────────────────────────────────────────────────────────

for OP in (:tsimplify, :tfactor, :tsubs, :tdiff, :ttrigsimp, :texpand_trig)
    @eval $OP(A::TensOrtho, args...; kwargs...) =
        _rebuild(A, $OP(getdata(A), args...; kwargs...))
end
# Explicit Num dispatch to avoid ambiguity with Symbolics.jl
for OP in (:tsimplify, :tsubs, :tdiff)
    @eval $OP(A::TensOrtho{Num}, args...; kwargs...) =
        _rebuild(A, $OP(getdata(A), args...; kwargs...))
end

# ── Display ───────────────────────────────────────────────────────────────────

function Base.show(io::IO, A::TensOrtho)
    C11, C22, C33, C12, C13, C23, C44, C55, C66 = getdata(A)
    print(io, "TensOrtho: C₁₁=", C11, " C₂₂=", C22, " C₃₃=", C33,
             " C₁₂=", C12, " C₁₃=", C13, " C₂₃=", C23,
             " C₄₄=", C44, " C₅₅=", C55, " C₆₆=", C66)
    print(io, "\n  frame: ", vecbasis(A.frame, :cov))
end

##############################################################################
# Shared change_tens / components for TensWalpole and TensOrtho
# (both are 3D order-4 tensors stored in the canonical frame)
##############################################################################

for TT in (:TensWalpole, :TensOrtho)
    # T used to link tensor eltype with basis eltype:
    @eval change_tens(t::$TT{T}, ℬ::OrthonormalBasis{3,T}) where {T} =
        Tens(tensor_or_array(getarray(t)), ℬ)
    @eval components(t::$TT{T}, ::OrthonormalBasis{3,T}, ::NTuple{4,Symbol}) where {T} =
        getarray(t)
    # T not needed for these:
    @eval components(t::$TT) = getarray(t)
    @eval components(t::$TT, ::NTuple{4,Symbol}) = getarray(t)
end

##############################################################################
# Exports
##############################################################################

export TensWalpole, TensOrtho
export tensW1, tensW2, tensW3, tensW4, tensW5, tensW6, Walpole
export get_ℓ, getaxis, getframe
export fromISO, isTI, isOrtho
export KM_material
