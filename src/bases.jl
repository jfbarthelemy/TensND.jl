abstract type AbstractBasis{dim,T<:Number} <: AbstractMatrix{T} end

@pure Base.size(::AbstractBasis{dim}) where {dim} = (dim, dim)
Base.getindex(b::AbstractBasis, i::Integer, j::Integer) = getindex(vecbasis(b, :cov), i, j)
@pure Base.eltype(::AbstractBasis{dim,T}) where {dim,T} = T
@pure getdim(::AbstractBasis{dim}) where {dim} = dim


"""
    Basis(v::AbstractMatrix{T}, ::Val{:cov})
    Basis{dim, T<:Number}()
    Basis(θ::T<:Number, ϕ::T<:Number, ψ::T<:Number)

Basis built from a square matrix `v` where columns correspond either to
- primal vectors ie `eᵢ=v[:,i]` if `var=:cov` as by default
- dual vectors ie `eⁱ=v[:,i]` if `var=:cont`.

Basis without any argument refers to the canonical basis (`CanonicalBasis`) in `Rᵈⁱᵐ` (by default `dim=3` and `T=Sym`)

Basis can also be built from Euler angles (`RotatedBasis`) `θ` in 2D and `(θ, ϕ, ψ)` in 3D

The attributes of this object can be obtained by
- `vecbasis(ℬ, :cov)`: square matrix defining the primal basis `eᵢ=e[:,i]`
- `vecbasis(ℬ, :cont)`: square matrix defining the dual basis `eⁱ=E[:,i]`
- `metric(ℬ, :cov)`: square matrix defining the covariant components of the metric tensor `gᵢⱼ=eᵢ⋅eⱼ=g[i,j]`
- `metric(ℬ, :cont)`: square matrix defining the contravariant components of the metric tensor `gⁱʲ=eⁱ⋅eʲ=G[i,j]`

# Examples
```julia
julia> v = Sym[1 0 0; 0 1 0; 0 1 1] ; ℬ = Basis(v)
Basis{3, Sym}
# basis: 3×3 Tensor{2, 3, Sym, 9}:
 1  0  0
 0  1  0
 0  1  1
# dual basis: 3×3 Tensor{2, 3, Sym, 9}:
 1  0   0
 0  1  -1
 0  0   1
# covariant metric tensor: 3×3 SymmetricTensor{2, 3, Sym, 6}:
 1  0  0
 0  2  1
 0  1  1
# contravariant metric tensor: 3×3 SymmetricTensor{2, 3, Sym, 6}:
 1   0   0
 0   1  -1
 0  -1   2

julia> θ, ϕ, ψ = symbols("θ, ϕ, ψ", real = true) ; ℬʳ = Basis(θ, ϕ, ψ) ; display(vecbasis(ℬʳ, :cov))
3×3 Tensor{2, 3, Sym, 9}:
 -sin(ψ)⋅sin(ϕ) + cos(θ)⋅cos(ψ)⋅cos(ϕ)  -sin(ψ)⋅cos(θ)⋅cos(ϕ) - sin(ϕ)⋅cos(ψ)  sin(θ)⋅cos(ϕ)
  sin(ψ)⋅cos(ϕ) + sin(ϕ)⋅cos(θ)⋅cos(ψ)  -sin(ψ)⋅sin(ϕ)⋅cos(θ) + cos(ψ)⋅cos(ϕ)  sin(θ)⋅sin(ϕ)
                        -sin(θ)⋅cos(ψ)                          sin(θ)⋅sin(ψ)         cos(θ)
```
"""
struct Basis{dim,T} <: AbstractBasis{dim,T}
    e::Matrix{T} # Primal basis `eᵢ=e[:,i]`
    E::Matrix{T} # Dual basis `eⁱ=E[:,i]`
    g::Symmetric{T,Matrix{T}} # Metric tensor `gᵢⱼ=eᵢ⋅eⱼ=g[i,j]`
    G::Symmetric{T,Matrix{T}} # Inverse of the metric tensor `gⁱʲ=eⁱ⋅eʲ=G[i,j]`
    function Basis(
        e::AbstractMatrix{T},
        E::AbstractMatrix{T},
        g::AbstractMatrix{T},
        G::AbstractMatrix{T},
    ) where {T}
        dim = size(e, 1)
        @assert dim == size(e, 2) "v should be a square matrix"
        if isidentity(e)
            return CanonicalBasis{dim,T}()
        elseif isidentity(g)
            return RotatedBasis(e)
        elseif isdiag(g)
            χ = sqrt.(diag(g))
            return OrthogonalBasis(RotatedBasis(e .* inv.(χ)'), χ)
        else
            e = Matrix(e)
            g = Symmetric(g)
            G = Symmetric(G)
            E = Matrix(E)
            new{dim,T}(e, E, g, G)
        end
    end
    function Basis(e::AbstractMatrix{T}, ::Val{:cov}) where {T}
        dim = size(e, 1)
        @assert dim == size(e, 2) "v should be a square matrix"
        if isidentity(e)
            return CanonicalBasis{dim,T}()
        else
            e = Matrix(e)
            g = simplifyif(Symmetric(e'e))
            if isidentity(g)
                return RotatedBasis(e)
            elseif isdiagonal(g)
                χ = sqrt.(diag(g))
                return OrthogonalBasis(RotatedBasis(e .* inv.(χ)'), χ)
            else
                if T == Sym
                    G = simplifyif(Symmetric(inv(Matrix(g))))
                else
                    G = simplifyif(inv(g))
                end
                E = simplifyif(e * G')
                new{dim,T}(e, E, g, G)
            end
        end
    end
    function Basis(E::AbstractMatrix{T}, ::Val{:cont}) where {T}
        dim = size(E, 1)
        @assert dim == size(E, 2) "v should be a square matrix"
        if isidentity(E)
            return CanonicalBasis{dim,T}()
        else
            E = Matrix(E)
            G = simplifyif(Symmetric(E'E))
            if isidentity(G)
                return RotatedBasis(E)
            elseif isdiagonal(G)
                uχ = inv.(sqrt.(diag(G)))
                return OrthogonalBasis(RotatedBasis(E .* uχ'), uχ)
            else
                if T == Sym
                    g = simplifyif(Symmetric(inv(Matrix(G))))
                else
                    g = simplifyif(inv(G))
                end
                e = simplifyif(E * g')
                new{dim,T}(e, E, g, G)
            end
        end
    end
    Basis(v::AbstractMatrix{T}, var) where {T} = Basis(v, Val(var))
    Basis(v::AbstractMatrix{T}) where {T} = Basis(v, :cov)
    Basis(θ::T1, ϕ::T2, ψ::T3 = 0) where {T1,T2,T3} = RotatedBasis(θ, ϕ, ψ)
    Basis(θ::T) where {T} = RotatedBasis(θ)
    Basis{dim,T}() where {dim,T} = CanonicalBasis{dim,T}()
    Basis() = CanonicalBasis()
end

"""
    CanonicalBasis{dim, T}

Canonical basis of dimension `dim` (default: 3) and type `T` (default: Sym)

The attributes of this object can be obtained by
- `vecbasis(ℬ, :cov)`: square matrix defining the primal basis `eᵢ=e[:,i]=δᵢⱼ`
- `vecbasis(ℬ, :cont)`: square matrix defining the dual basis `eⁱ=E[:,i]=δᵢⱼ`
- `metric(ℬ, :cov)`: square matrix defining the covariant components of the metric tensor `gᵢⱼ=eᵢ⋅eⱼ=g[i,j]=δᵢⱼ`
- `metric(ℬ, :cont)`: square matrix defining the contravariant components of the metric tensor `gⁱʲ=eⁱ⋅eʲ=G[i,j]=δᵢⱼ`

# Examples
```julia
julia> ℬ = CanonicalBasis()
CanonicalBasis{3, Sym}
# basis: 3×3 TensND.LazyIdentity{3, Sym}:
 1  0  0
 0  1  0
 0  0  1
# dual basis: 3×3 TensND.LazyIdentity{3, Sym}:
 1  0  0
 0  1  0
 0  0  1
# covariant metric tensor: 3×3 TensND.LazyIdentity{3, Sym}:
 1  0  0
 0  1  0
 0  0  1
# contravariant metric tensor: 3×3 TensND.LazyIdentity{3, Sym}:
 1  0  0
 0  1  0
 0  0  1

julia> ℬ₂ = CanonicalBasis{2, Float64}()
CanonicalBasis{2, Float64}
# basis: 2×2 TensND.LazyIdentity{2, Float64}:
 1.0  0.0
 0.0  1.0
# dual basis: 2×2 TensND.LazyIdentity{2, Float64}:
 1.0  0.0
 0.0  1.0
# covariant metric tensor: 2×2 TensND.LazyIdentity{2, Float64}:
 1.0  0.0
 0.0  1.0
# contravariant metric tensor: 2×2 TensND.LazyIdentity{2, Float64}:
 1.0  0.0
 0.0  1.0
```
"""
struct CanonicalBasis{dim,T} <: AbstractBasis{dim,T} end
CanonicalBasis() = CanonicalBasis{3,Sym}()


"""
    RotatedBasis(θ::T<:Number, ϕ::T<:Number, ψ::T<:Number)
    RotatedBasis(θ::T<:Number)

Orthonormal basis of dimension `dim` (default: 3) and type `T` (default: Sym) built from Euler angles `θ` in 2D and `(θ, ϕ, ψ)` in 3D

# Examples
```julia
julia> θ, ϕ, ψ = symbols("θ, ϕ, ψ", real = true) ; ℬʳ = RotatedBasis(θ, ϕ, ψ) ; display(vecbasis(ℬʳ, :cov))
3×3 Tensor{2, 3, Sym, 9}:
 -sin(ψ)⋅sin(ϕ) + cos(θ)⋅cos(ψ)⋅cos(ϕ)  -sin(ψ)⋅cos(θ)⋅cos(ϕ) - sin(ϕ)⋅cos(ψ)  sin(θ)⋅cos(ϕ)
  sin(ψ)⋅cos(ϕ) + sin(ϕ)⋅cos(θ)⋅cos(ψ)  -sin(ψ)⋅sin(ϕ)⋅cos(θ) + cos(ψ)⋅cos(ϕ)  sin(θ)⋅sin(ϕ)
                        -sin(θ)⋅cos(ψ)                          sin(θ)⋅sin(ψ)         cos(θ)
```
"""
struct RotatedBasis{dim,T} <: AbstractBasis{dim,T}
    e::Matrix{T} # Primal basis `eᵢ=e[:,i]`
    E::Matrix{T} # Dual basis `eⁱ=E[:,i]`
    angles::NamedTuple
    function RotatedBasis(R::AbstractMatrix{T}) where {T<:Number}
        dim = size(R, 1)
        if isidentity(R)
            return CanonicalBasis{dim,T}()
        else
            e = E = Matrix(R)
            return new{dim,T}(e, E, angles(R))
        end
    end
    function RotatedBasis(θ::T1, ϕ::T2, ψ::T3 = 0) where {T1<:Number,T2<:Number,T3<:Number}
        T = promote_type(T1, T2, T3)
        dim = 3
        R = RotZYZ(ϕ, θ, ψ)
        if isidentity(R)
            return CanonicalBasis{dim,T}()
        else
            e = E = Matrix(R)
            return new{dim,T}(e, E, angles(R))
        end
    end
    function RotatedBasis(θ::T) where {T<:Number}
        dim = 2
        cθ = cos(θ)
        sθ = sin(θ)
        e = E = [cθ -sθ; sθ cθ]
        if isidentity(e)
            return CanonicalBasis{dim,T}()
        else
            return new{dim,T}(e, E, angles(e))
        end
    end
    function RotatedBasis(θ::Sym)
        dim = 2
        cθ = cos(θ)
        sθ = sin(θ)
        e = E = [cθ -sθ; sθ cθ]
        if isidentity(e)
            return CanonicalBasis{dim,Sym}()
        else
            return new{dim,Sym}(e, E, (θ = θ,))
        end
    end
end

const OrthonormalBasis{dim,T} = Union{CanonicalBasis{dim,T}, RotatedBasis{dim,T}}

struct OrthogonalBasis{dim,T} <: AbstractBasis{dim,T}
    parent::OrthonormalBasis{dim,T}
    λ::Vector{T}
    e::Matrix{T}
    E::Matrix{T}
    g::Diagonal{T,Vector{T}}
    G::Diagonal{T,Vector{T}}
    function OrthogonalBasis(parent::OrthonormalBasis{dim,T}, λ::Vector) where {dim,T}
        λ = T.(λ)
        if λ == one.(λ)
            return parent
        else
            e = [λ[j] * parent[i,j] for i ∈ 1:dim, j ∈ 1:dim]
            E = [parent[i,j] / λ[j] for i ∈ 1:dim, j ∈ 1:dim]
            return new{dim,T}(parent, λ, e, E, Diagonal(λ.^2), Diagonal(inv.(λ).^2))
        end
    end
end

@inline CylindricalBasis(θ) = RotatedBasis(0, θ, 0)

@inline SphericalBasis(θ, ϕ) = RotatedBasis(θ, ϕ, 0)

const AllOrthogonalBasis{dim,T} = Union{OrthonormalBasis{dim,T}, OrthogonalBasis{dim,T}}

angles(M::AbstractMatrix{T}, ::Val{2}) where {T} =
    (θ = atan(M[2, 1] - M[1, 2], M[1, 1] + M[2, 2]),)
function angles(M::AbstractMatrix{T}, ::Val{3}) where {T}
    R = RotZYZ(M)
    return (θ = R.theta2, ϕ = R.theta1, ψ = R.theta3)
end

"""
    angles(M::AbstractMatrix{T})

Determines the Euler angles corresponding to the input matrix supposed to be a rotation matrix or at least a similarity

# Examples
```julia
julia> θ, ϕ, ψ = symbols("θ, ϕ, ψ", real = true) ; ℬʳ = RotatedBasis(θ, ϕ, ψ) ; display(vecbasis(ℬʳ, :cov))
3×3 Tensor{2, 3, Sym, 9}:
 -sin(ψ)⋅sin(ϕ) + cos(θ)⋅cos(ψ)⋅cos(ϕ)  -sin(ψ)⋅cos(θ)⋅cos(ϕ) - sin(ϕ)⋅cos(ψ)  sin(θ)⋅cos(ϕ)
  sin(ψ)⋅cos(ϕ) + sin(ϕ)⋅cos(θ)⋅cos(ψ)  -sin(ψ)⋅sin(ϕ)⋅cos(θ) + cos(ψ)⋅cos(ϕ)  sin(θ)⋅sin(ϕ)
                        -sin(θ)⋅cos(ψ)                          sin(θ)⋅sin(ψ)         cos(θ)

julia> angles(ℬʳ)
(θ = θ, ϕ = ϕ, ψ = ψ)
```
"""
angles(M::AbstractMatrix{T}) where {T} = angles(M, Val(size(M, 1)))
angles(ℬ::RotatedBasis) = ℬ.angles

angles(v::AbstractVector{T}, ::Val{2}) where {T} = (θ = atan(v[2], v[1]),)
angles(v::AbstractVector{T}, ::Val{3}) where {T} =
    (θ = atan(√(v[1]^2 + v[2]^2), v[3]), ϕ = atan(v[2], v[1]))
angles(v::AbstractVector{T}) where {T} = angles(v, Val(size(v, 1)))


invvar(::Val{:cov}) = :cont
invvar(::Val{:cont}) = :cov
invvar(var) = invvar(Val(var))


"""
    vecbasis(ℬ::AbstractBasis, var = :cov)

Returns the primal (if `var = :cov`) or dual (if `var = :cont`) basis
"""
vecbasis(ℬ::AbstractBasis, ::Val{:cov}) = ℬ.e
vecbasis(ℬ::AbstractBasis, ::Val{:cont}) = ℬ.E
vecbasis(::CanonicalBasis{dim,T}, ::Val{:cov}) where {dim,T} = Id2{dim,T}()
vecbasis(::CanonicalBasis{dim,T}, ::Val{:cont}) where {dim,T} = Id2{dim,T}()

vecbasis(ℬ::AbstractBasis, var) = vecbasis(ℬ, Val(var))
vecbasis(ℬ::AbstractBasis) = vecbasis(ℬ, :cov)
vecbasis(ℬ::AbstractBasis, i::Integer, j::Integer, var = :cov) = vecbasis(ℬ, Val(var))[i, j]


"""
    metric(ℬ::AbstractBasis, var = :cov)

Returns the covariant (if `var = :cov`) or contravariant (if `var = :cont`) metric matrix
"""
metric(ℬ::AbstractBasis, ::Val{:cov}) = ℬ.g
metric(ℬ::AbstractBasis, ::Val{:cont}) = ℬ.G
metric(::OrthonormalBasis{dim,T}, ::Val{:cov}) where {dim,T} = Id2{dim,T}()
metric(::OrthonormalBasis{dim,T}, ::Val{:cont}) where {dim,T} = Id2{dim,T}()

metric(ℬ::AbstractBasis, var) = metric(ℬ, Val(var))
metric(ℬ::AbstractBasis) = metric(ℬ, :cov)
metric(ℬ::AbstractBasis, i::Integer, j::Integer, var = :cov) = metric(ℬ, Val(var))[i, j]

"""
    normalize(ℬ::AbstractBasis, var = cov)

Builds a basis after normalization of column vectors of input matrix `v` where columns define either
- primal vectors ie `eᵢ=v[:,i]/norm(v[:,i])` if `var = :cov` as by default
- dual vector ie `eⁱ=v[:,i]/norm(v[:,i])` if `var = :cont`.
"""
function LinearAlgebra.normalize(ℬ::AbstractBasis, var = :cov)
    w = copy(vecbasis(ℬ, var))
    for i = 1:size(w, 2)
        w[:, i] /= norm(w[:, i])
    end
    return Basis(w, var)
end

"""
    isorthogonal(ℬ::AbstractBasis)

Checks whether the basis `ℬ` is orthogonal
"""
isorthogonal(ℬ::AbstractBasis) = isdiagonal(metric(ℬ))

isorthogonal(::OrthonormalBasis) = true

isorthogonal(::OrthogonalBasis) = true

"""
    isorthonormal(ℬ::AbstractBasis)

Checks whether the basis `ℬ` is orthonormal
"""
isorthonormal(ℬ::AbstractBasis) = isidentity(metric(ℬ))

isorthonormal(::OrthonormalBasis) = true

#####################
# Display Functions #
#####################
for OP in (:show, :print, :display)
    @eval begin
        function Base.$OP(ℬ::AbstractBasis)
            $OP(typeof(ℬ))
            print("# basis: ")
            $OP(vecbasis(ℬ, :cov))
            print("# dual basis: ")
            $OP(vecbasis(ℬ, :cont))
            print("# covariant metric tensor: ")
            $OP(metric(ℬ, :cov))
            print("# contravariant metric tensor: ")
            $OP(metric(ℬ, :cont))
        end
    end
end
