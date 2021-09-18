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

Base.:*(α::Number, A::AllIsotropic2{dim}) where {dim} = Isotropic2{dim}(α * λ(A))
Base.:*(A::AllIsotropic2{dim}, α::Number) where {dim} = Isotropic2{dim}(λ(A) * α)
Base.:/(A::AllIsotropic2{dim}, α::Number) where {dim} = Isotropic2{dim}(λ(A) / α)
for OP in (:+, :-, :*)
    @eval Base.$OP(A1::AllIsotropic2{dim}, A2::AllIsotropic2{dim}) where {dim} =
        Isotropic2{dim}($OP(λ(A1), λ(A2)))
end
Base.inv(A::AllIsotropic2{dim,T}) where {dim,T} = Isotropic2{dim}(one(T) / λ(A))
Base.transpose(A::AllIsotropic2{dim}) where {dim} = A
Base.adjoint(A::AllIsotropic2{dim}) where {dim} = A


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

aJ(A::Isotropic4) = A.aJ
aK(A::Isotropic4) = A.aK
aJ(::Id4{dim,T}) where {dim,T} = one(T)
aK(::Id4{dim,T}) where {dim,T} = one(T)
aJ(::J4{dim,T}) where {dim,T} = one(T)
aK(::J4{dim,T}) where {dim,T} = zero(T)
aJ(::K4{dim,T}) where {dim,T} = zero(T)
aK(::K4{dim,T}) where {dim,T} = one(T)

Base.:*(α::Number, A::AllIsotropic4{dim}) where {dim} =
    Isotropic4{dim}(α * aJ(A), α * aK(A))
Base.:*(A::AllIsotropic4{dim}, α::Number) where {dim} =
    Isotropic4{dim}(aJ(A) * α, aK(A) * α)
Base.:/(A::AllIsotropic4{dim}, α::Number) where {dim} =
    Isotropic4{dim}(aJ(A) / α, aK(A) / α)
for OP in (:+, :-, :*)
    @eval Base.$OP(A1::AllIsotropic4{dim}, A2::AllIsotropic4{dim}) where {dim} =
        Isotropic4{dim}($OP(aJ(A1), aJ(A2)), $OP(aK(A1), aK(A2)))
end
Base.inv(A::AllIsotropic4{dim,T}) where {dim,T} =
    Isotropic4{dim}(one(T) / aJ(A), one(T) / aK(A))
Base.transpose(A::AllIsotropic4{dim}) where {dim} = A
Base.adjoint(A::AllIsotropic4{dim}) where {dim} = A




isidentity(a::AbstractMatrix{T}) where {T} = a ≈ Id2{size(a, 1),T}()
isidentity(a::AbstractMatrix{Sym}) = a == Id2{size(a, 1),Sym}()

isdiagonal(a::AbstractMatrix{T}) where {T} = norm(a - Diagonal(a)) <= eps(T)
isdiagonal(a::AbstractMatrix{Sym}) = isdiag(a)

simplifyif(x) = x
simplifyif(x::Sym) = simplify(x)
simplifyif(m::Matrix{Sym}) = simplify.(m)
simplifyif(m::Symmetric{Sym}) = Symmetric(simplify.(m))

