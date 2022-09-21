struct TensISO{order,dim,T,N} <: AbstractTens{order,dim,T}
    data::NTuple{N,T}
    TensISO{dim}(λ::T) where {dim,T} = new{2,dim,T,1}((λ,))
    TensISO{dim}(α::T1, β::T2) where {dim,T1,T2} = new{4,dim,promote_type(T1, T2),2}((α, β))
    TensISO{dim}(data::NTuple{N,T}) where {dim,N,T} = TensISO{dim}(data...)
    TensISO{order,dim,T}() where {order,dim,T} =
        new{order,dim,T,order ÷ 2}(ntuple(_ -> one(T), Val(order ÷ 2)))
end

@pure getorder(::TensISO{order}) where {order} = order
@pure getdim(::TensISO{order,dim}) where {order,dim} = dim
@pure Base.eltype(::Type{TensISO{order,dim,T}}) where {order,dim,T} = T
@pure Base.length(::TensISO{order,dim,T,N}) where {order,dim,T,N} = dim^order
@pure datanumber(::TensISO{order,dim,T,N}) where {order,dim,T,N} = N
@pure Base.size(::TensISO{order,dim}) where {order,dim} = ntuple(_ -> dim, Val(order))
Base.getindex(t::TensISO{2}, i::Integer, j::Integer) = t.data[1] * I[i, j]
Base.getindex(
    t::TensISO{4,dim},
    i::Integer,
    j::Integer,
    k::Integer,
    l::Integer,
) where {dim} =
    (t.data[1] - t.data[2]) * I[i, j] * I[k, l] / dim +
    t.data[2] * (I[i, k] * I[j, l] + I[i, l] * I[j, k]) / 2
function Base.replace_in_print_matrix(
    ::TensISO{2},
    i::Integer,
    j::Integer,
    s::AbstractString,
)
    i == j ? s : Base.replace_with_centered_mark(s)
end

"""
    tensId2(::Val{dim}, ::Val{T}) where {dim,T<:Number}

Identity tensor of second order `𝟏ᵢⱼ = δᵢⱼ = 1 if i=j otherwise 0`

# Examples
```julia
julia> 𝟏 = t𝟏() ; KM(𝟏)
6-element Vector{Sym}:
 1
 1
 1
 0
 0
 0

julia> 𝟏.data
3×3 SymmetricTensor{2, 3, Sym, 6}:
 1  0  0
 0  1  0
 0  0  1
```  
"""
tensId2(::Val{dim} = Val(3), ::Val{T} = Val(Sym)) where {dim,T<:Number} = TensISO{2,dim,T}()

"""
    tensId4(::Val{dim} = Val(3), ::Val{T} = Val(Sym))

Symmetric identity tensor of fourth order  `𝕀 = 𝟏 ⊠ˢ 𝟏` i.e. `(𝕀)ᵢⱼₖₗ = (δᵢₖδⱼₗ+δᵢₗδⱼₖ)/2`

# Examples
```julia
julia> 𝕀 = t𝕀() ; KM(𝕀)
6×6 Matrix{Sym}:
 1  0  0  0  0  0
 0  1  0  0  0  0
 0  0  1  0  0  0
 0  0  0  1  0  0
 0  0  0  0  1  0
 0  0  0  0  0  1
``` 
"""
tensId4(::Val{dim} = Val(3), ::Val{T} = Val(Sym)) where {dim,T<:Number} = TensISO{4,dim,T}()

"""
    tensJ4(::Val{dim} = Val(3), ::Val{T} = Val(Sym))

Spherical projector of fourth order  `𝕁 = (𝟏 ⊗ 𝟏) / dim` i.e. `(𝕁)ᵢⱼₖₗ = δᵢⱼδₖₗ/dim`

# Examples
```julia
julia> 𝕁 = t𝕁() ; KM(𝕁)
6×6 Matrix{Sym}:
 1/3  1/3  1/3  0  0  0
 1/3  1/3  1/3  0  0  0
 1/3  1/3  1/3  0  0  0
   0    0    0  0  0  0
   0    0    0  0  0  0
   0    0    0  0  0  0
``` 
"""
tensJ4(::Val{dim} = Val(3), ::Val{T} = Val(Sym)) where {dim,T<:Number} =
    TensISO{dim}(one(T), zero(T))

"""
    tensK4(::Val{dim} = Val(3), ::Val{T} = Val(Sym))

Deviatoric projector of fourth order  `𝕂 = 𝕀 - 𝕁` i.e. `(𝕂)ᵢⱼₖₗ = (δᵢₖδⱼₗ+δᵢₗδⱼₖ)/2 - δᵢⱼδₖₗ/dim`

# Examples
```julia
julia> 𝕂 = t𝕂() ; KM(𝕂)
6×6 Matrix{Sym}:
  2/3  -1/3  -1/3  0  0  0
 -1/3   2/3  -1/3  0  0  0
 -1/3  -1/3   2/3  0  0  0
    0     0     0  1  0  0
    0     0     0  0  1  0
    0     0     0  0  0  1
``` 
"""
tensK4(::Val{dim} = Val(3), ::Val{T} = Val(Sym)) where {dim,T<:Number} =
    TensISO{dim}(zero(T), one(T))


"""
    ISO(::Val{dim} = Val(3), ::Val{T} = Val(Sym))

Return the three fourth-order isotropic tensors `𝕀, 𝕁, 𝕂`

# Examples
```julia
julia> 𝕀, 𝕁, 𝕂 = ISO() ;
``` 
"""
ISO(::Val{dim} = Val(3), ::Val{T} = Val(Sym)) where {dim,T<:Number} =
    tensId4(Val(dim), Val(T)), tensJ4(Val(dim), Val(T)), tensK4(Val(dim), Val(T))

for FUNC in (:tensId2, :tensId4, :tensJ4, :tensK4, :ISO)
    @eval $FUNC(args...) = $FUNC(Val.(args)...)
end

getdata(t::TensISO) = t.data
getarray(t::TensISO) = Array(t)
getbasis(::TensISO{order,dim,T}) where {order,dim,T} = CanonicalBasis{dim,T}()
getvar(::TensISO{order}) where {order} = ntuple(_ -> :cont, Val(order))
getvar(::TensISO, i::Integer) = :cont
components(t::TensISO{order,dim,T}) where {order,dim,T} = getarray(t)
components(
    t::TensISO{order,dim,T},
    ::OrthonormalBasis{dim,T},
    ::NTuple{order,Symbol},
) where {order,dim,T} = getarray(t)
components(t::TensISO{order,dim,T}, ::NTuple{order,Symbol}) where {order,dim,T} =
    getarray(t)

change_tens(t::TensISO{order,dim,T}, ℬ::OrthonormalBasis{dim,T}) where {order,dim,T} =
    Tens(getarray(t), ℬ)
change_tens(
    t::TensISO{order,dim,T},
    ℬ::OrthonormalBasis{dim,T},
    ::NTuple{order,Symbol},
) where {order,dim,T} = Tens(getarray(t), ℬ)

@inline Base.:-(A::TensISO{order,dim}) where {order,dim} = TensISO{dim}(.-(getdata(A)))
@inline Base.:*(α::Number, A::TensISO{order,dim}) where {order,dim} =
    TensISO{dim}(α .* getdata(A))
@inline Base.:*(A::TensISO{order,dim}, α::Number) where {order,dim} =
    TensISO{dim}(getdata(A) .* α)
@inline Base.:/(A::TensISO{order,dim}, α::Number) where {order,dim} =
    TensISO{dim}(getdata(A) ./ α)
for OP in (:+, :-, :*)
    @eval @inline Base.$OP(
        A1::TensISO{order,dim},
        A2::TensISO{order,dim},
    ) where {order,dim} = TensISO{dim}($OP.(getdata(A1), getdata(A2)))
    @eval @inline Base.$OP(
        A1::TensISO{order,dim,T,N},
        A2::UniformScaling,
    ) where {order,dim,T,N} = TensISO{dim}($OP.(getdata(A1), ntuple(_ -> A2.λ, N)))
    @eval @inline Base.$OP(
        A1::UniformScaling,
        A2::TensISO{order,dim},
    ) where {order,dim,T,N} = TensISO{dim}($OP.(ntuple(_ -> A1.λ, N), getdata(A2)))
    @eval @inline function Base.$OP(
        A1::TensISO{order,dim,T,N},
        A2::AbstractTens{order,dim},
    ) where {order,dim,T,N}
        m1 = components(A1, getbasis(A2), getvar(A2))
        return Tens($OP(m1, getarray(A2)), getbasis(A2), getvar(A2))
    end
    @eval @inline function Base.$OP(
        A1::AbstractTens{order,dim},
        A2::TensISO{order,dim,T,N},
    ) where {order,dim,T,N}
        m2 = components(A2, getbasis(A1), getvar(A1))
        return Tens($OP(getarray(A1), m2), getbasis(A1), getvar(A1))
    end
end
for OP ∈ (:(==), :(<=), :(>=), :(<), :(>))
    @eval @inline Base.$OP(
        A1::TensISO{order,dim},
        A2::TensISO{order,dim},
    ) where {order,dim} = all($OP.(getdata(A1), getdata(A2)))
end
@inline Base.inv(A::TensISO{order,dim,T}) where {order,dim,T} =
    TensISO{dim}(one(T) ./ getdata(A))
@inline Base.one(A::TensISO{order,dim,T}) where {order,dim,T} =
    TensISO{dim}(one.(getdata(A)))
@inline Base.zero(A::TensISO{order,dim,T}) where {order,dim,T} =
    TensISO{dim}(zero.(getdata(A)))

for FUNC ∈ (:one, :zero)
    @eval begin
        @inline Base.$FUNC(A::AbstractTens{4,dim,T}) where {dim,T} =
            TensISO{dim}($FUNC(T), $FUNC(T))
        @inline Base.$FUNC(A::AbstractTens{2,dim,T}) where {dim,T} = TensISO{dim}($FUNC(T))
    end
end

@inline Base.literal_pow(::typeof(^), A::TensISO, ::Val{-1}) = inv(A)
@inline Base.literal_pow(::typeof(^), A::TensISO, ::Val{0}) = one(A)
@inline Base.literal_pow(::typeof(^), A::TensISO, ::Val{1}) = A
@inline Base.literal_pow(
    ::typeof(^),
    A::TensISO{order,dim,T},
    ::Val{p},
) where {order,dim,T,p} = TensISO{dim}(getdata(A) .^ (p))

@inline Base.transpose(A::TensISO) = A
@inline Base.adjoint(A::TensISO) = A

for OP in (:show, :print, :display)
    @eval begin
        function Base.$OP(A::TensISO{4})
            println("(", getdata(A)[1], ") 𝕁 + (", getdata(A)[2], ") 𝕂")
        end
        function Base.$OP(A::TensISO{2})
            println("(", getdata(A)[1], ") 𝟏")
        end
    end
end
intrinsic(A::TensISO{4}) = println("(", getdata(A)[1], ") 𝕁 + (", getdata(A)[2], ") 𝕂")
intrinsic(A::TensISO{2}) = println("(", getdata(A)[1], ") 𝟏")

for OP in (:(tsimplify), :(tfactor), :(tsubs), :(tdiff), :(ttrigsimp), :(texpand_trig))
    @eval $OP(A::TensISO{order,dim,Sym}, args...; kwargs...) where {order,dim} =
        TensISO{dim}($OP(getdata(A), args...; kwargs...))
end

for OP in (:(tsimplify), :(tsubs), :(tdiff))
    @eval $OP(A::TensISO{order,dim,Num}, args...; kwargs...) where {order,dim} =
        TensISO{dim}($OP(getdata(A), args...; kwargs...))
end




"""
    KM(v::AllIsotropic{dim}; kwargs...)

Kelvin-Mandel vector or matrix representation
"""
KM(A::TensISO{order,dim}) where {order,dim} = tomandel(SymmetricTensor{order,dim}(A))

Tensors.otimes(A::TensISO{2,dim}, B::TensISO{2,dim}) where {dim} =
    TensISO{dim}(dim * getdata(A)[1] * getdata(B)[1], zero(eltype(A)))

scontract(A::TensISO{2,dim}, B::TensISO{2,dim}) where {dim} =
    TensISO{dim}(getdata(A)[1] * getdata(B)[1])

Tensors.otimes(A::TensISO{2,dim}) where {dim} =
    TensISO{dim}(dim * getdata(A)[1]^2, zero(eltype(A)))

scontract(A::TensISO{2,dim}) where {dim} =
    TensISO{dim}(getdata(A)[1]^2)

scontract(A::TensISO{2,dim}, B::AbstractArray) where {dim} = getdata(A)[1] * B
scontract(A::AbstractArray, B::TensISO{2,dim}) where {dim} = A * getdata(B)[1]

LinearAlgebra.dot(A::TensISO{2,dim}, B::TensISO{2,dim}) where {dim} = scontract(A, B)
for T ∈ (AbstractArray, AbstractTens)
    @eval LinearAlgebra.dot(A::TensISO{2,dim}, B::$T) where {dim} = scontract(A, B)
    @eval LinearAlgebra.dot(A::$T, B::TensISO{2,dim}) where {dim} = scontract(A, B)
end

Tensors.dcontract(A::TensISO{2,dim}, B::TensISO{2,dim}) where {dim} =
    dim * getdata(A)[1] * getdata(B)[1]
Tensors.dcontract(A::TensISO{4,dim}, B::TensISO{2,dim}) where {dim} =
    TensISO{dim}(getdata(A)[1] * getdata(B)[1])
Tensors.dcontract(A::TensISO{2,dim}, B::TensISO{4,dim}) where {dim} =
    TensISO{dim}(getdata(A)[1] * getdata(B)[1])
Tensors.dcontract(A::TensISO{4,dim}, B::TensISO{4,dim}) where {dim} =
    TensISO{dim}(getdata(A)[1] * getdata(B)[1], getdata(A)[2] * getdata(B)[2])

Tensors.dcontract(A::TensISO{2}, B::AbstractTens) = getdata(A)[1] * contract(B, 1, 2)
Tensors.dcontract(A::AbstractTens{order}, B::TensISO{2}) where {order} =
    contract(A, order - 1, order) * getdata(B)[1]


Tensors.dcontract(A::TensISO{4,dim}, B::TensOrthonormal{2}) where {dim} =
    getdata(A)[2] * B + (getdata(A)[1] - getdata(A)[2]) * tr(B) * I / dim
Tensors.dcontract(A::TensOrthonormal{2}, B::TensISO{4,dim}) where {dim} =
    A * getdata(B)[2] + tr(A) * (getdata(B)[1] - getdata(B)[2]) * I / dim

function Tensors.dcontract(
    A::TensISO{4,dim,T},
    B::AllTensOrthogonal{order,dim},
) where {order,dim,T}
    nB = TensOrthonormal(B)
    m = getarray(nB)
    ec1 = ntuple(i -> i, order)
    ec2 = (2, 1, ntuple(i -> i + 2, order - 2)...)
    m2 = einsum(EinCode((ec1,), ec2), (m,))
    newm =
        getdata(A)[2] * (m + m2) / 2 +
        (getdata(A)[1] - getdata(A)[2]) * Id2{dim,T}() ⊗ contract(m, 1, 2) / dim
    return Tens(newm, getbasis(nB))
end

function Tensors.dcontract(
    A::AllTensOrthogonal{order,dim},
    B::TensISO{4,dim,T},
) where {order,dim,T}
    nA = TensOrthonormal(A)
    m = getarray(nA)
    ec1 = ntuple(i -> i, order)
    ec2 = (ntuple(i -> i, order - 2)..., order, order - 1)
    m2 = einsum(EinCode((ec1,), ec2), (m,))
    newm =
        (m + m2) * getdata(B)[2] / 2 +
        contract(m, order - 1, order) ⊗ Id2{dim,T}() * (getdata(B)[1] - getdata(B)[2]) / dim
    return Tens(newm, getbasis(nA))
end


for order ∈ (2, 4)
    for OP ∈ (:+, :-, :*)
        @eval @inline Base.$OP(
            A1::AbstractTensor{$order,dim,T},
            A2::UniformScaling{T},
        ) where {dim,T<:Number} = $OP(A1, A2.λ * one(A1))
        @eval @inline Base.$OP(
            A1::UniformScaling{T},
            A2::AbstractTensor{$order,dim,T},
        ) where {dim,T<:Number} = $OP(A1.λ * one(A2), A2)
        @eval @inline Base.$OP(
            A1::AbstractTensor{$order,dim,Sym},
            A2::UniformScaling{Sym},
        ) where {dim} = $OP(A1, A2.λ * one(A1))
        @eval @inline Base.$OP(
            A1::UniformScaling{Sym},
            A2::AbstractTensor{$order,dim,Sym},
        ) where {dim} = $OP(A1.λ * one(A2), A2)
    end
end

Tensors.dotdot(v1::AbstractTens{1}, S::TensISO{2,dim}, v2::AbstractTens{1}) where {dim} =
    getdata(S)[1] * v1 ⋅ v2
Tensors.dotdot(v1::AbstractTens{1}, S::TensISO{4,dim}, v2::AbstractTens{1}) where {dim} =
    (getdata(S)[1] - getdata(S)[2]) * (v1 ⊗ v2) / dim +
    getdata(S)[2] * (v2 ⊗ v1 + v1 ⋅ v2 * I) / 2

Tensors.dotdot(a1::AbstractTens{2}, S::TensISO{4,dim}, a2::AbstractTens{2}) where {dim} =
    (getdata(S)[1] - getdata(S)[2]) * tr(a1) * tr(a2) / dim + getdata(S)[2] * a1 ⊡ a2

qcontract(A::TensISO{4,3}, B::TensISO{4,3}) =
    getdata(A)[1] * getdata(B)[1] + 5 * getdata(A)[2] * getdata(B)[2]

qcontract(A::TensISO{4,2}, B::TensISO{4,2}) =
    getdata(A)[1] * getdata(B)[1] + 2 * getdata(A)[2] * getdata(B)[2]

function qcontract(A::TensISO{4,dim,T}, B::AllTensOrthogonal{order,dim}) where {order,dim,T}
    nB = TensOrthonormal(B)
    m = getarray(nB)
    newm =
        getdata(A)[2] *
        (contract(contract(m, 1, 3), 1, 2) + contract(contract(m, 1, 4), 1, 2)) / 2 +
        (getdata(A)[1] - getdata(A)[2]) * contract(contract(m, 1, 2), 1, 2) / dim
    return Tens(newm, getbasis(nB))
end

function qcontract(A::AllTensOrthogonal{order,dim}, B::TensISO{4,dim,T}) where {order,dim,T}
    nA = TensOrthonormal(A)
    m = getarray(nA)
    newm =
        (
            contract(contract(m, order - 2, order), order - 1, order) +
            contract(contract(m, order - 3, order), order - 1, order)
        ) * getdata(B)[2] / 2 +
        contract(contract(m, order - 1, order), order - 1, order) *
        (getdata(B)[1] - getdata(B)[2]) / dim
    return Tens(newm, getbasis(nA))
end

isotropify(A::AbstractArray{T,2}) where {T} = TensISO{size(A)[1]}(tr(A) / size(A)[1])

function isotropify(A::AbstractArray{T,4}) where {T}
    dim = size(A)[1]
    α = tensJ4(dim, T) ⊙ A
    β = (tensK4(dim, T) ⊙ A) / 5
    return TensISO{dim}(α, β)
end

TensISO(A::AbstractArray) = isotropify(Tens(A))

function proj_tens(::Val{:ISO}, A::AbstractArray)
    norm = x -> simplify(√(sum(x .^ 2)))
    nA = norm(A)
    if nA == zero(eltype(A))
        return zero(A), nA, nA
    else
        B = isotropify(A)
        d = norm(B - A)
        return B, d, d / nA
    end
end

isISO(::TensISO) = true
isISO(A::AbstractArray) = isotropify(A) == A

LinearAlgebra.issymmetric(::TensISO) = true
Tensors.isminorsymmetric(::TensISO{4}) = true
Tensors.ismajorsymmetric(::TensISO{4}) = true

export TensISO, tensId2, tensId4, tensJ4, tensK4, ISO, isotropify, isISO
