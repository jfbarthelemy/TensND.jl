δkron(T::Type{<:Number}, i::Integer, j::Integer) = i == j ? one(T) : zero(T)

struct Id2{dim,T<:Number} <: AbstractMatrix{T} end
@pure Base.size(::Id2{dim}) where {dim} = (dim, dim)
Base.getindex(::Id2{dim,T}, i::Integer, j::Integer) where {dim,T} = δkron(T, i, j)
function Base.replace_in_print_matrix(::Id2, i::Integer, j::Integer, s::AbstractString)
    i == j ? s : Base.replace_with_centered_mark(s)
end

isidentity(a::AbstractMatrix{T}) where {T} = a ≈ Id2{size(a, 1),T}()
isidentity(a::AbstractMatrix{Sym}) = a == Id2{size(a, 1),Sym}()

isdiagonal(a::AbstractMatrix{T}) where {T} = norm(a - Diagonal(a)) <= eps(T)
isdiagonal(a::AbstractMatrix{Sym}) = isdiag(a)

simplifyif(x) = x
simplifyif(x::Sym) = simplify(x)
simplifyif(m::Matrix{Sym}) = simplify.(m)
simplifyif(m::Symmetric{Sym}) = Symmetric(simplify.(m))

function Tensors.otimes(
    t1::AbstractArray{T1,order1},
    t2::AbstractArray{T2,order2},
) where {T1,T2,order1,order2}
    ec1 = ntuple(i -> i, order1)
    ec2 = ntuple(i -> order1 + i, order2)
    ec3 = ntuple(i -> i, order1 + order2)
    return einsum(EinCode((ec1, ec2), ec3), (AbstractArray{T1}(t1), AbstractArray{T2}(t2)))
end

function contract(t::AbstractArray{T,order}, i::Int, j::Int) where {T,order}
    m = min(i, j)
    M = max(i, j)
    ec1 = ntuple(k -> k == j ? i : k, order)
    ec2 = (Tuple(1:m-1)..., Tuple(m+1:M-1)..., Tuple(M+1:order)...)
    return einsum(EinCode((ec1,), ec2), (AbstractArray{T}(t),))
end

contract(t::AbstractArray{T,2}, ::Int, ::Int) where {T} = tr(t)

function Tensors.dcontract(
    t1::AbstractArray{T1,order1},
    t2::AbstractArray{T2,order2},
) where {T1,T2,order1,order2}
    newc = order1 + order2
    ec1 = (ntuple(i -> i, order1 - 2)..., newc, newc + 1)
    ec2 = (newc, newc + 1, ntuple(i -> order1 - 2 + i, order2 - 2)...)
    ec3 = ntuple(i -> i, order1 + order2 - 4)
    return einsum(EinCode((ec1, ec2), ec3), (AbstractArray{T1}(t1), AbstractArray{T2}(t2)))
end

Tensors.dcontract(t1::AbstractArray{T1,2}, t2::AbstractArray{T2,2}) where {T1,T2} =
    dot(AbstractArray{T1}(t1), AbstractArray{T2}(t2))


function Tensors.dotdot(
    v1::AbstractArray{T1,order1},
    S::AbstractArray{TS,orderS},
    v2::AbstractArray{T2,order2},
) where {T1,TS,T2,order1,orderS,order2}
    newc = order1 + orderS
    ec1 = (ntuple(i -> i, order1 - 1)..., newc)
    ecS = (newc, ntuple(i -> order1 - 1 + i, orderS - 1)...)
    ec3 = ntuple(i -> i, order1 + orderS - 2)
    v1S = einsum(EinCode((ec1, ecS), ec3), (AbstractArray{T1}(v1), AbstractArray{TS}(S)))
    newc += order2
    ecv1S = (ntuple(i -> i, order1 + orderS - 3)..., newc)
    ec2 = (newc, ntuple(i -> order1 + orderS - 3 + i, order2 - 1)...)
    ec3 = ntuple(i -> i, newc - 4)
    return einsum(EinCode((ecv1S, ec2), ec3), (v1S, AbstractArray{T2}(v2)))
end

function qcontract(
    t1::AbstractArray{T1,order1},
    t2::AbstractArray{T2,order2},
) where {T1,T2,order1,order2}
    newc = order1 + order2
    ec1 = (ntuple(i -> i, order1 - 4)..., newc, newc + 1, newc + 2, newc + 3)
    ec2 = (newc, newc + 1, newc + 2, newc + 3, ntuple(i -> order1 - 4 + i, order2 - 4)...)
    ec3 = ntuple(i -> i, order1 + order2 - 8)
    return einsum(EinCode((ec1, ec2), ec3), (AbstractArray{T1}(t1), AbstractArray{T2}(t2)))
end

qcontract(t1::AbstractArray{T1,4}, t2::AbstractArray{T2,4}) where {T1,T2} =
    dot(AbstractArray{T1}(t1), AbstractArray{T2}(t2))

function Tensors.otimesu(
    t1::AbstractArray{T1,order1},
    t2::AbstractArray{T2,order2},
) where {T1,T2,order1,order2}
    ec1 = (ntuple(i -> i, order1 - 1)..., order1 + 1)
    ec2 = (order1, ntuple(i -> order1 + 1 + i, order2 - 1)...)
    ec3 = ntuple(i -> i, order1 + order2)
    return einsum(EinCode((ec1, ec2), ec3), (AbstractArray{T1}(t1), AbstractArray{T2}(t2)))
end

function Tensors.otimesl(
    t1::AbstractArray{T1,order1},
    t2::AbstractArray{T2,order2},
) where {T1,T2,order1,order2}
    ec1 = (ntuple(i -> i, order1 - 1)..., order1 + 2)
    ec2 = (order1, order1 + 1, ntuple(i -> order1 + 2 + i, order2 - 2)...)
    ec3 = ntuple(i -> i, order1 + order2)
    return einsum(EinCode((ec1, ec2), ec3), (AbstractArray{T1}(t1), AbstractArray{T2}(t2)))
end

otimesul(t1::AbstractArray{T1}, t2::AbstractArray{T2}) where {T1,T2} =
    (otimesu(t1, t2) + otimesl(t1, t2)) / promote_type(T1, T2)(2)

function sotimes(
    t1::AbstractArray{T1,order1},
    t2::AbstractArray{T2,order2},
) where {T1,T2,order1,order2}
    ec1 = ntuple(i -> i, order1)
    ec2 = ntuple(i -> order1 + i, order2)
    ec3 = ntuple(i -> i, order1 + order2)
    t3 = einsum(EinCode((ec1, ec2), ec3), (AbstractArray{T1}(t1), AbstractArray{T2}(t2)))
    ec1 = (ntuple(i -> i, order1 - 1)..., order1 + 1)
    ec2 = (order1, ntuple(i -> order1 + 1 + i, order2 - 1)...)
    ec3 = ntuple(i -> i, order1 + order2)
    t4 = einsum(EinCode((ec1, ec2), ec3), (AbstractArray{T1}(t1), AbstractArray{T2}(t2)))
    return (t3 + t4) / promote_type(T1, T2)(2)
end

@inline function sotimes(S1::Vec{dim}, S2::Vec{dim}) where {dim}
    return Tensor{2,dim}(@inline function (i, j)
        @inbounds (S1[i] * S2[j] + S1[j] * S2[i]) / 2
    end)
end

@inline function sotimes(S1::SecondOrderTensor{dim}, S2::SecondOrderTensor{dim}) where {dim}
    TensorType = Tensors.getreturntype(
        otimes,
        Tensors.get_base(typeof(S1)),
        Tensors.get_base(typeof(S2)),
    )
    TensorType(@inline function (i, j, k, l)
        @inbounds (S1[i, j] * S2[k, l] + S1[i, k] * S2[j, l]) / 2
    end)
end

Tensors.otimes(α::Number, t::AbstractArray) = α * t
Tensors.otimes(t::AbstractArray, α::Number) = α * t

sotimes(α::Number, t::AbstractArray) = α * t
sotimes(t::AbstractArray, α::Number) = α * t



const ⊙ = qcontract
const ⊠ = otimesu
const ⊠ˢ = otimesul
const ⊗ˢ = sotimes

const sboxtimes = otimesul

