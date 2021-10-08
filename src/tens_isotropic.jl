struct TensISO{order,dim,T,N} <: AbstractTens{order,dim,T}
    data::NTuple{N,T}
    TensISO{dim}(λ::T) where {dim,T} = new{2,dim,T,1}((λ,))
    TensISO{dim}(α::T1, β::T2) where {dim,T1,T2} =
        new{4,dim,promote_type(T1, T2),2}((α, β))
    TensISO{dim}(data::NTuple{N,T}) where {dim,N,T} = TensISO{dim}(data...)
    TensISO{order,dim,T}() where {order,dim,T} =
        new{order,dim,T,order ÷ 2}(ntuple(_ -> one(T), Val(order ÷ 2)))
end

@pure getorder(::TensISO{order}) where {order} = order
@pure getdim(::TensISO{order,dim}) where {order,dim} = dim
@pure Base.eltype(::TensISO{order,dim,T}) where {order,dim,T} = T
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

TensId2(::Val{dim}, ::Val{T}) where {dim,T<:Number} = TensISO{2, dim, T}()
TensId4(::Val{dim}, ::Val{T}) where {dim,T<:Number} = TensISO{4, dim, T}()
TensJ4(::Val{dim}, ::Val{T}) where {dim,T<:Number} = TensISO{dim}(one(T),zero(T))
TensK4(::Val{dim}, ::Val{T}) where {dim,T<:Number} = TensISO{dim}(zero(T),one(T))
ISO(::Val{dim}, ::Val{T}) where {dim,T<:Number} = TensId4(Val(dim),Val(T)), TensJ4(Val(dim),Val(T)), TensK4(Val(dim),Val(T))

for f ∈ (:TensId2, :TensId4, :TensJ4, :TensJ4, :BaseISO)
    @eval $f() = $f(Val(3), Val(Sym))
end

getdata(t::TensISO) = t.data
getarray(t::TensISO) = Array(t)
getbasis(::TensISO{order,dim,T}) where {order,dim,T} = CanonicalBasis{dim,T}()
getvar(::TensISO{order}) where {order} = ntuple(_ -> :cont, Val(order))
getvar(::TensISO, i::Int) = :cont

@inline Base.:*(α::Number, A::TensISO{order,dim}) where {order,dim} = TensISO{dim}(α .* getdata(A))
@inline Base.:*(A::TensISO{order,dim}, α::Number) where {order,dim} = TensISO{dim}(getdata(A) .* α)
@inline Base.:/(A::TensISO{order,dim}, α::Number) where {order,dim} = TensISO{dim}(getdata(A) ./ α)
for OP in (:+, :-, :*)
    @eval @inline Base.$OP(A1::TensISO{order,dim}, A2::TensISO{order,dim}) where {order, dim} =
        TensISO{dim}($OP.(getdata(A1), getdata(A2)))
    @eval @inline Base.$OP(A1::TensISO{order,dim, T, N}, A2::UniformScaling) where {order, dim, T, N} =
        TensISO{dim}($OP.(getdata(A1), ntuple(_ -> A2.λ, N)))
    @eval @inline Base.$OP(A1::UniformScaling, A2::TensISO{order,dim}) where {order, dim, T, N} =
        TensISO{dim}($OP.(ntuple(_ -> A1.λ, N), getdata(A1)))
end
for OP in (:(==), :(<=), :(>=), :(<), :(>))
    @eval @inline Base.$OP(A1::TensISO{order,dim}, A2::TensISO{order,dim}) where {order, dim} =
        all($OP.(getdata(A1), getdata(A2)))
end
@inline Base.inv(A::TensISO{order,dim, T}) where {order,dim,T} = TensISO{dim}(one(T) ./ getdata(A))
@inline Base.one(A::TensISO{order,dim, T}) where {order,dim,T} = TensISO{dim}(one.(getdata(A)))

@inline Base.literal_pow(::typeof(^), A::TensISO, ::Val{-1}) = inv(A)
@inline Base.literal_pow(::typeof(^), A::TensISO, ::Val{0}) = one(A)
@inline Base.literal_pow(::typeof(^), A::TensISO, ::Val{1}) = A
@inline Base.literal_pow(::typeof(^), A::TensISO{order,dim,T}, ::Val{p}) where {order,dim,T,p} =
    TensISO{dim}(getdata(A).^(p))

@inline Base.transpose(A::TensISO) = A
@inline Base.adjoint(A::TensISO) = A

function Base.display(A::TensISO{4,dim,T}) where {dim,T}
    print("(",A.data[1],") 𝕁 + (",A.data[2],") 𝕂")
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

TO BE CONTINUED

Tensors.otimes(A::TensISO{2,dim}, B::TensISO{2,dim}) where {dim} =
    TensISO{dim}(dim * λ(A) * λ(B), zero(eltype(A)))

scontract(A::AllIsotropic2{dim}, B::AllIsotropic2{dim}) where {dim} =
    Isotropic2{dim}(λ(A) * λ(B))
scontract(A::AllIsotropic2{dim}, B::AbstractArray) where {dim} = λ(A) * B
scontract(A::AbstractArray, B::AllIsotropic2{dim}) where {dim} = A * λ(B)

LinearAlgebra.dot(A::AllIsotropic2{dim}, B::AllIsotropic2{dim}) where {dim} =
    scontract(A, B)
LinearAlgebra.dot(A::AllIsotropic2{dim}, B::AbstractArray) where {dim} = scontract(A, B)
LinearAlgebra.dot(A::AbstractArray, B::AllIsotropic2{dim}) where {dim} = scontract(A, B)

Tensors.dcontract(A::AllIsotropic2{dim}, B::AllIsotropic2{dim}) where {dim} =
    dim * λ(A) * λ(B)
Tensors.dcontract(A::AllIsotropic4{dim}, B::AllIsotropic2{dim}) where {dim} =
    Isotropic2{dim}(aJ(A) * λ(B))
Tensors.dcontract(A::AllIsotropic2{dim}, B::AllIsotropic4{dim}) where {dim} =
    Isotropic2{dim}(λ(A) * aJ(B))
Tensors.dcontract(A::AllIsotropic4{dim}, B::AllIsotropic4{dim}) where {dim} =
    Isotropic4{dim}(aJ(A) * aJ(B), aK(A) * aK(B))
Tensors.dcontract(A::AllIsotropic4{dim}, B::AbstractArray) where {dim} =
    aK(A) * B + (aJ(A) - aK(A)) * tr(B) * I / dim
Tensors.dcontract(A::AbstractArray, B::AllIsotropic4{dim}) where {dim} =
    A * aK(B) + tr(A) * (aJ(B) - aK(B)) * I / dim


for order ∈ (2, 4)
    for OP ∈ (:+, :-, :*)
        @eval @inline Base.$OP(
            A1::AbstractTensor{$order,dim,T},
            A2::UniformScaling,
        ) where {dim,T<:Number} = $OP(A1, A2.λ * one(A1))
        @eval @inline Base.$OP(
            A1::UniformScaling,
            A2::AbstractTensor{$order,dim,T},
        ) where {dim,T<:Number} = $OP(A1.λ * one(A2), A2)
        @eval @inline Base.$OP(
            A1::AbstractTensor{$order,dim,Sym},
            A2::UniformScaling,
        ) where {dim} = $OP(A1, A2.λ * one(A1))
        @eval @inline Base.$OP(
            A1::UniformScaling,
            A2::AbstractTensor{$order,dim,Sym},
        ) where {dim} = $OP(A1.λ * one(A2), A2)
    end
end

Tensors.dotdot(v1::AbstractVector, S::AllIsotropic2{dim}, v2::AbstractVector) where {dim} =
    λ(S) * v1 ⋅ v2


Tensors.dotdot(v1::AbstractVector, S::AllIsotropic4{dim}, v2::AbstractVector) where {dim} =
    (aJ(S) - aK(S)) * (v1 ⊗ v2) / dim + aK(S) * (v2 ⊗ v1 + v1 ⋅ v2 * I) / 2

Tensors.dotdot(a1::AbstractMatrix, S::AllIsotropic4{dim}, a2::AbstractMatrix) where {dim} =
    (aJ(S) - aK(S)) * tr(a1) * tr(a2) / dim + aK(S) * a1 ⊡ a2

qcontract(A::AllIsotropic4{dim}, B::AllIsotropic4{dim}) where {dim} =
    aJ(A) * aJ(B) + 5 * aK(A) * aK(B)
