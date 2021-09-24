δkron(T::Type{<:Number}, i::Integer, j::Integer) = i == j ? one(T) : zero(T)

struct Id2{dim,T<:Number} <: AbstractMatrix{T} end
@pure Base.size(::Id2{dim}) where {dim} = (dim, dim)
Base.getindex(::Id2{dim,T}, i::Integer, j::Integer) where {dim,T} = δkron(T, i, j)
function Base.replace_in_print_matrix(::Id2, i::Integer, j::Integer, s::AbstractString)
    i == j ? s : Base.replace_with_centered_mark(s)
end

struct Isotropic2{dim,T<:Number} <: AbstractMatrix{T}
    λ::T
    Isotropic2{dim}(λ::T) where {dim,T} = λ == one(T) ? Id2{dim,T}() : new{dim,T}(λ)
end
@pure Base.size(::Isotropic2{dim}) where {dim} = (dim, dim)
Base.getindex(A::Isotropic2{dim,T}, i::Integer, j::Integer) where {dim,T} =
    i == j ? A.λ : zero(T)
function Base.replace_in_print_matrix(
    ::Isotropic2,
    i::Integer,
    j::Integer,
    s::AbstractString,
)
    i == j ? s : Base.replace_with_centered_mark(s)
end

const AllIsotropic2{dim,T} = Union{Id2{dim,T},Isotropic2{dim,T}}

λ(A::Isotropic2) = A.λ
λ(::Id2{dim,T}) where {dim,T} = one(T)

@inline Base.:*(α::Number, A::AllIsotropic2{dim}) where {dim} = Isotropic2{dim}(α * λ(A))
@inline Base.:*(A::AllIsotropic2{dim}, α::Number) where {dim} = Isotropic2{dim}(λ(A) * α)
@inline Base.:/(A::AllIsotropic2{dim}, α::Number) where {dim} = Isotropic2{dim}(λ(A) / α)
for OP in (:+, :-, :*)
    @eval @inline Base.$OP(A1::AllIsotropic2{dim}, A2::AllIsotropic2{dim}) where {dim} =
        Isotropic2{dim}($OP(λ(A1), λ(A2)))
    @eval @inline Base.$OP(A1::AllIsotropic2{dim}, A2::UniformScaling) where {dim} =
        Isotropic2{dim}($OP(λ(A1), A2.λ))
    @eval @inline Base.$OP(A1::UniformScaling, A2::AllIsotropic2{dim}) where {dim} =
        Isotropic2{dim}($OP(A1.λ, λ(A2)))
end
for OP in (:(==), :(<=), :(>=), :(<), :(>))
    @eval @inline Base.$OP(A1::AllIsotropic2{dim}, A2::AllIsotropic2{dim}) where {dim} =
        $OP(λ(A1), λ(A2))
end
@inline Base.inv(A::AllIsotropic2{dim,T}) where {dim,T} = Isotropic2{dim}(one(T) / λ(A))


struct Id4{dim,T<:Number} <: AbstractArray{T,4} end
@pure Base.size(::Id4{dim}) where {dim} = (dim, dim, dim, dim)
Base.getindex(::Id4{dim,T}, i::Integer, j::Integer, k::Integer, l::Integer) where {dim,T} =
    δkron(T, i, k) * δkron(T, j, l)
Base.Matrix(::Id4{dim,T}) where {dim,T} = Id2{dim * dim,T}()

fI4(T::Type{<:Number}, i::Integer, j::Integer, k::Integer, l::Integer) =
    (δkron(T, i, k) * δkron(T, j, l) + δkron(T, i, l) * δkron(T, j, k)) / 2
struct I4{dim,T<:Number} <: AbstractArray{T,4} end
@pure Base.size(::I4{dim}) where {dim} = (dim, dim, dim, dim)
Base.getindex(::I4{dim,T}, i::Integer, j::Integer, k::Integer, l::Integer) where {dim,T} =
    fI4(T, i, j, k, l)


fJ4(T::Type{<:Number}, i::Integer, j::Integer, k::Integer, l::Integer) =
    δkron(T, i, j) * δkron(T, k, l)

struct J4{dim,T<:Number} <: AbstractArray{T,4} end
@pure Base.size(::J4{dim}) where {dim} = (dim, dim, dim, dim)
Base.getindex(::J4{dim,T}, i::Integer, j::Integer, k::Integer, l::Integer) where {dim,T} =
    fJ4(T, i, j, k, l) / dim

struct K4{dim,T<:Number} <: AbstractArray{T,4} end
@pure Base.size(::K4{dim}) where {dim} = (dim, dim, dim, dim)
Base.getindex(::K4{dim,T}, i::Integer, j::Integer, k::Integer, l::Integer) where {dim,T} =
    fI4(T, i, j, k, l) - fJ4(T, i, j, k, l) / dim


struct Isotropic4{dim,T<:Number} <: AbstractArray{T,4}
    aJ::T
    aK::T
    function Isotropic4{dim}(aJ::T, aK::T) where {dim,T}
        if aJ == one(T) && aK == one(T)
            return I4{dim,T}()
        elseif aJ == one(T) && aK == zero(T)
            return J4{dim,T}()
        elseif aJ == zero(T) && aK == one(T)
            return K4{dim,T}()
        else
            return new{dim,T}(aJ, aK)
        end
    end
end
@pure Base.size(::Isotropic4{dim}) where {dim} = (dim, dim, dim, dim)
Base.getindex(
    A::Isotropic4{dim,T},
    i::Integer,
    j::Integer,
    k::Integer,
    l::Integer,
) where {dim,T} = (A.aJ - A.aK) * fJ4(T, i, j, k, l) / dim + A.aK * fI4(T, i, j, k, l)

const AllIsotropic4{dim,T} = Union{I4{dim,T},J4{dim,T},K4{dim,T},Isotropic4{dim,T}}
const AllIsotropic{dim,T} = Union{AllIsotropic2{dim,T},AllIsotropic4{dim,T}}

aJ(A::Isotropic4) = A.aJ
aK(A::Isotropic4) = A.aK
aJ(::I4{dim,T}) where {dim,T} = one(T)
aK(::I4{dim,T}) where {dim,T} = one(T)
aJ(::J4{dim,T}) where {dim,T} = one(T)
aK(::J4{dim,T}) where {dim,T} = zero(T)
aJ(::K4{dim,T}) where {dim,T} = zero(T)
aK(::K4{dim,T}) where {dim,T} = one(T)

@inline Base.:*(α::Number, A::AllIsotropic4{dim}) where {dim} =
    Isotropic4{dim}(α * aJ(A), α * aK(A))
@inline Base.:*(A::AllIsotropic4{dim}, α::Number) where {dim} =
    Isotropic4{dim}(aJ(A) * α, aK(A) * α)
@inline Base.:/(A::AllIsotropic4{dim}, α::Number) where {dim} =
    Isotropic4{dim}(aJ(A) / α, aK(A) / α)
for OP in (:+, :-, :*)
    @eval @inline Base.$OP(A1::AllIsotropic4{dim}, A2::AllIsotropic4{dim}) where {dim} =
        Isotropic4{dim}($OP(aJ(A1), aJ(A2)), $OP(aK(A1), aK(A2)))
    @eval @inline Base.$OP(A1::AllIsotropic4{dim}, A2::UniformScaling) where {dim} =
        Isotropic4{dim}($OP(aJ(A1), A2.λ), $OP(aK(A1), A2.λ))
    @eval @inline Base.$OP(A1::UniformScaling, A2::AllIsotropic4{dim}) where {dim} =
        Isotropic4{dim}($OP(A1.λ, aJ(A2)), $OP(A1.λ, aK(A2)))
end
for OP in (:(==), :(<=), :(>=), :(<), :(>))
    @eval @inline Base.$OP(A1::AllIsotropic4{dim}, A2::AllIsotropic4{dim}) where {dim} =
        $OP(aJ(A1), aJ(A2)) && $OP(aK(A1), aK(A2))
end
@inline Base.inv(A::AllIsotropic4{dim,T}) where {dim,T} =
    Isotropic4{dim}(one(T) / aJ(A), one(T) / aK(A))


@inline Base.literal_pow(::typeof(^), A::AllIsotropic4, ::Val{-1}) = inv(A)
@inline Base.literal_pow(::typeof(^), A::AllIsotropic4, ::Val{0}) = one(A)
@inline Base.literal_pow(::typeof(^), A::AllIsotropic4, ::AllIsotropic{1}) = A
@inline Base.literal_pow(::typeof(^), A::AllIsotropic4{dim,T}, ::Val{p}) where {dim,T,p} =
    Isotropic4{dim}(aJ(A)^(p), aK(A)^(p))


@inline Base.literal_pow(::typeof(^), A::AllIsotropic2, ::Val{-1}) = inv(A)
@inline Base.literal_pow(::typeof(^), A::AllIsotropic2, ::Val{0}) = one(A)
@inline Base.literal_pow(::typeof(^), A::AllIsotropic2, ::AllIsotropic{1}) = A
@inline Base.literal_pow(::typeof(^), A::AllIsotropic2{dim,T}, ::Val{p}) where {dim,T,p} =
    Isotropic2{dim}(λ(A)^(p))

@inline Base.transpose(A::AllIsotropic{dim}) where {dim} = A
@inline Base.adjoint(A::AllIsotropic{dim}) where {dim} = A

@pure order(::AllIsotropic2) = 2
@pure order(::AllIsotropic4) = 4


function Base.display(A::AllIsotropic4{dim,T}) where {dim,T}
    aj = aJ(A)
    ak = aK(A)
    # res = aj != zero(T) ? "($(aj)) 𝕁" : ""
    # res *= ak != zero(T) ? " + ($(ak)) 𝕂" : ""
    # print(res)
    print("($(aj)) 𝕁 + ($(ak)) 𝕂")
end

isidentity(a::AbstractMatrix{T}) where {T} = a ≈ Id2{size(a, 1),T}()
isidentity(a::AbstractMatrix{Sym}) = a == Id2{size(a, 1),Sym}()

isdiagonal(a::AbstractMatrix{T}) where {T} = norm(a - Diagonal(a)) <= eps(T)
isdiagonal(a::AbstractMatrix{Sym}) = isdiag(a)

simplifyif(x) = x
simplifyif(x::Sym) = simplify(x)
simplifyif(m::Matrix{Sym}) = simplify.(m)
simplifyif(m::Symmetric{Sym}) = Symmetric(simplify.(m))

for OP in (:(simplify), :(factor), :(subs))
    @eval SymPy.$OP(A::AllIsotropic2{dim,Sym}, args...; kwargs...) where {dim} =
        Isotropic2{dim}($OP(λ(A), args...; kwargs...))
    @eval SymPy.$OP(A::AllIsotropic4{dim,Sym}, args...; kwargs...) where {dim} =
        Isotropic4{dim}($OP(aJ(A), args...; kwargs...), $OP(aK(A), args...; kwargs...))
end

for OP in (:(trigsimp), :(expand_trig))
    @eval $OP(A::AllIsotropic2{dim,Sym}, args...; kwargs...) where {dim} =
        Isotropic2{dim}(sympy.$OP(λ(A), args...; kwargs...))
    @eval $OP(A::AllIsotropic4{dim,Sym}, args...; kwargs...) where {dim} = Isotropic4{dim}(
        sympy.$OP(aJ(A), args...; kwargs...),
        sympy.$OP(aK(A), args...; kwargs...),
    )
end

"""
    KM(v::AllIsotropic{dim}; kwargs...)

Kelvin-Mandel vector or matrix representation
"""
KM(A::AllIsotropic{dim}) where {dim} = tomandel(SymmetricTensor{order(A),dim}(A))

@inline Base.one(::AllIsotropic2{dim,T}) where {dim,T} = Id2{dim,T}()
@inline Base.one(::AllIsotropic4{dim,T}) where {dim,T} = I4{dim,T}()


Tensors.otimes(A::AllIsotropic2{dim}, B::AllIsotropic2{dim}) where {dim} =
    Isotropic4{dim}(dim * λ(A) * λ(B), zero(eltype(A)))

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

Tensors.dotdot(
    v1::AbstractVector,
    S::AllIsotropic2{dim},
    v2::AbstractVector,
) where {dim} = λ(S) * v1 ⋅ v2


Tensors.dotdot(
    v1::AbstractVector,
    S::AllIsotropic4{dim},
    v2::AbstractVector,
) where {dim} = (aJ(S) - aK(S)) * (v1 ⊗ v2) / dim + aK(S) * (v2 ⊗ v1 + v1 ⋅ v2 * I) / 2

Tensors.dotdot(
    a1::AbstractMatrix,
    S::AllIsotropic4{dim},
    a2::AbstractMatrix,
) where {dim} = (aJ(S) - aK(S)) * tr(a1) * tr(a2) / dim + aK(S) * a1 ⊡ a2

qcontract(A::AllIsotropic4{dim}, B::AllIsotropic4{dim}) where {dim} =
    aJ(A) * aJ(B) + 5 * aK(A) * aK(B)
