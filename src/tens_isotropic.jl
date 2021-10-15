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
    t𝟏(::Val{dim}, ::Val{T}) where {dim,T<:Number}

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
    t𝕀(::Val{dim} = Val(3), ::Val{T} = Val(Sym))

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
    t𝕁(::Val{dim} = Val(3), ::Val{T} = Val(Sym))

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
    t𝕂(::Val{dim} = Val(3), ::Val{T} = Val(Sym))

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

Returns the three fourth-order isotropic tensors `𝕀, 𝕁, 𝕂`

# Examples
```julia
julia> 𝕀, 𝕁, 𝕂 = ISO() ;
``` 
"""
ISO(::Val{dim} = Val(3), ::Val{T} = Val(Sym)) where {dim,T<:Number} =
    tensId4(Val(dim), Val(T)), tensJ4(Val(dim), Val(T)), tensK4(Val(dim), Val(T))

const t𝟏 = tensId2
const t𝕀 = tensId4
const t𝕁 = tensJ4
const t𝕂 = tensK4

getdata(t::TensISO) = t.data
getarray(t::TensISO) = Array(t)
getbasis(::TensISO{order,dim,T}) where {order,dim,T} = CanonicalBasis{dim,T}()
getvar(::TensISO{order}) where {order} = ntuple(_ -> :cont, Val(order))
getvar(::TensISO, i::Int) = :cont
components(t::TensISO{order, dim, T}) where {order, dim, T} = getarray(t)
components(t::TensISO{order, dim, T}, ::OrthonormalBasis{dim, T}, ::NTuple{order, Symbol}) where {order, dim, T} = getarray(t)
components(t::TensISO{order, dim, T}, ::NTuple{order, Symbol}) where {order, dim, T} = getarray(t)

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
end
for OP in (:(==), :(<=), :(>=), :(<), :(>))
    @eval @inline Base.$OP(
        A1::TensISO{order,dim},
        A2::TensISO{order,dim},
    ) where {order,dim} = all($OP.(getdata(A1), getdata(A2)))
end
@inline Base.inv(A::TensISO{order,dim,T}) where {order,dim,T} =
    TensISO{dim}(one(T) ./ getdata(A))
@inline Base.one(A::TensISO{order,dim,T}) where {order,dim,T} =
    TensISO{dim}(one.(getdata(A)))

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

function Base.display(A::TensISO{4,dim,T}) where {dim,T}
    print("(", getdata(A)[1], ") 𝕁 + (", getdata(A)[2], ") 𝕂")
end

for OP in (:(simplify), :(factor), :(subs), :(diff))
    @eval SymPy.$OP(A::TensISO{order,dim,Sym}, args...; kwargs...) where {order,dim} =
        TensISO{dim}($OP.(getdata(A), args...; kwargs...))
end

for OP in (:(trigsimp), :(expand_trig))
    @eval $OP(A::TensISO{order,dim,Sym}, args...; kwargs...) where {order,dim} =
        TensISO{dim}(sympy.$OP.(getdata(A), args...; kwargs...))
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
Tensors.dcontract(A::AbstractTens{order}, B::TensISO{2}) where {order} = contract(A, order - 1, order) * getdata(B)[1] 


# TODO
# A:ISO for upper orders for order 2 and 4
# Case of Tensors with abitrary bases
# Idem for qcontract


Tensors.dcontract(A::TensISO{4,dim}, B::AbstractTens) where {dim} =
        getdata(A)[2] * B + (getdata(A)[1] - getdata(A)[2]) * tr(B) * I / dim
Tensors.dcontract(A::AbstractTens, B::TensISO{4,dim}) where {dim} =
        A * getdata(B)[2] + tr(A) * (getdata(B)[1] - getdata(B)[2]) * I / dim


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


qcontract(A::TensISO{4,dim}, B::TensISO{4,dim}) where {dim} =
    getdata(A)[1] * getdata(B)[1] + 5 * getdata(A)[2] * getdata(B)[2]

for T ∈ (AbstractArray4, AbstractTens{4})
    @eval qcontract(A::$T, B::TensISO{4,dim}) where {dim} =
        (ein"ijij->"(A) + ein"ijji->"(A)) * getdata(B)[2] / 2 +
        ein"iijj->"(A) * (getdata(B)[1] - getdata(B)[2]) / dim
    @eval qcontract(A::TensISO{4,dim}, B::$T) where {dim} =
        getdata(A)[2] * (ein"ijij->"(B) + ein"ijji->"(B)) / 2 +
        (getdata(A)[1] - getdata(A)[2]) * ein"iijj->"(B) / dim
end

isotropify(A::AbstractMatrix) = TensISO{size(A)[1]}(tr(A))

function isotropify(A::AbstractArray{T,4}) where {T}
    dim = size(A)[1]
    α = ein"iijj->"(A)[1] / dim
    β = ((ein"ijij->"(A)[1] + ein"ijji->"(A)[1]) / 2 - α) / 5
    return TensISO{dim}(α, β)
end

TensISO(A::AbstractArray) = isotropify(A)

function proj_array(::Val{:ISO}, A::AbstractArray)
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

isISO(A::TensISO) = true
isISO(A::AbstractArray) = isotropify(A) == A