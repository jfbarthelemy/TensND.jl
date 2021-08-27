abstract type AbstractBasis{dim,T<:Number} <: AbstractArray{T,2} end
abstract type OrthonormalBasis{dim,T<:Number} <: AbstractBasis{dim,T} end

@pure Base.size(::AbstractBasis{dim}) where {dim} = (dim, dim)

Base.getindex(b::AbstractBasis, i::Int, j::Int) = getindex(b.e, i, j)


"""
    Basis(v::AbstractArray{T,2}, ::Val{:cov})
    Basis{dim, T<:Number}()
    Basis(θ::T<:Number, ϕ::T<:Number, ψ::T<:Number)

Basis built from a square matrix `v` where columns correspond either to
- primal vectors ie `eᵢ=v[:,i]` if `var=:cov` as by default
- dual vectors ie `eⁱ=v[:,i]` if `var=:cont`.

Basis without any argument refers to the canonical basis (`CanonicalBasis`) in `Rᵈⁱᵐ` (by default `dim=3` and `T=Sym`)

Basis can also be built from Euler angles (`RotatedBasis`) `θ` in 2D and `(θ, ϕ, ψ)` in 3D

The attributes of this object are
- `Basis.e`: square matrix defining the primal basis `eᵢ=e[:,i]`
- `Basis.E`: square matrix defining the dual basis `eⁱ=E[:,i]`
- `Basis.g`: square matrix defining the covariant components of the metric tensor `gᵢⱼ=eᵢ⋅eⱼ=g[i,j]`
- `Basis.G`: square matrix defining the contravariant components of the metric tensor `gⁱʲ=eⁱ⋅eʲ=G[i,j]`

# Examples
```jldoctest
julia> v = Sym[1 0 0; 0 1 0; 0 1 1] ; b = Basis(v)
3×3 Basis{3, Sym}:
 1  0  0
 0  1  0
 0  1  1

julia> v = Sym[1 0 0; 0 1 0; 0 1 1] ; b = Basis(v, :cont)
3×3 Basis{3, Sym}:
 1  0   0
 0  1  -1
 0  0   1
```

```julia
julia> θ, ϕ, ψ = symbols("θ, ϕ, ψ", real = true) ; b = Basis(θ, ϕ, ψ) ; display(b.e)
3×3 Tensor{2, 3, Sym, 9}:
 -sin(ψ)⋅sin(ϕ) + cos(θ)⋅cos(ψ)⋅cos(ϕ)  -sin(ψ)⋅cos(θ)⋅cos(ϕ) - sin(ϕ)⋅cos(ψ)  sin(θ)⋅cos(ϕ)
  sin(ψ)⋅cos(ϕ) + sin(ϕ)⋅cos(θ)⋅cos(ψ)  -sin(ψ)⋅sin(ϕ)⋅cos(θ) + cos(ψ)⋅cos(ϕ)  sin(θ)⋅sin(ϕ)
                        -sin(θ)⋅cos(ψ)                          sin(θ)⋅sin(ψ)         cos(θ)
```
"""
struct Basis{dim,T} <: AbstractBasis{dim,T}
    e::AbstractArray{T,2} # Primal basis `eᵢ=e[:,i]`
    E::AbstractArray{T,2} # Dual basis `eⁱ=E[:,i]`
    g::AbstractArray{T,2} # Metric tensor `gᵢⱼ=eᵢ⋅eⱼ=g[i,j]`
    G::AbstractArray{T,2} # Inverse of the metric tensor `gⁱʲ=eⁱ⋅eʲ=G[i,j]`
    function Basis(
        e::AbstractArray{T,2},
        E::AbstractArray{T,2},
        g::AbstractArray{T,2},
        G::AbstractArray{T,2},
    ) where {T}
        dim = size(v, 1)
        @assert dim == size(v, 2) "v should be a square matrix"
        e = Tensor{2,dim}(e)
        g = SymmetricTensor{2,dim}(g)
        G = SymmetricTensor{2,dim}(G)
        E = Tensor{2,dim}(E)
        new{dim,T}(e, E, g, G)
    end
    function Basis(v::AbstractArray{T,2}, ::Val{:cov}) where {T}
        dim = size(v, 1)
        @assert dim == size(v, 2) "v should be a square matrix"
        e = Tensor{2,dim}(v)
        g = SymmetricTensor{2,dim}(e' ⋅ e)
        G = inv(g)
        E = e ⋅ G'
        new{dim,T}(e, E, g, G)
    end
    function Basis(v::AbstractArray{T,2}, ::Val{:cont}) where {T}
        dim = size(v, 1)
        @assert dim == size(v, 2) "v should be a square matrix"
        E = Tensor{2,dim}(v)
        G = SymmetricTensor{2,dim}(E' ⋅ E)
        g = inv(G)
        e = E ⋅ g'
        new{dim,T}(e, E, g, G)
    end
    Basis(v::AbstractArray{T,2}, var) where {T} = Basis(v, Val(var))
    Basis(v::AbstractArray{T,2}) where {T} = Basis(v, :cov)
    Basis(θ::T, ϕ::T, ψ::T) where {T} = RotatedBasis(θ, ϕ, ψ)
    Basis(θ::T) where {T} = RotatedBasis(θ)
    Basis{dim,T}() where {dim,T} = CanonicalBasis{dim,T}()
    Basis() = CanonicalBasis()
end

"""
    CanonicalBasis{dim, T}

Canonical basis of dimension `dim` (default: 3) and type `T` (default: Sym)

The attributes of this object are
- `Basis.e`: identity matrix defining the primal basis `e[i,j]=δᵢⱼ`
- `Basis.E`: identity matrix defining the dual basis `g[i,j]=δᵢⱼ`
- `Basis.g`: identity matrix defining the covariant components of the metric tensor `g[i,j]=δᵢⱼ`
- `Basis.G`: identity matrix defining the contravariant components of the metric tensor `G[i,j]=δᵢⱼ`

# Examples
```jldoctest
julia> b = CanonicalBasis()
3×3 CanonicalBasis{3, Sym}:
 1  0  0
 0  1  0
 0  0  1

julia> b = CanonicalBasis{2, Float64}()
2×2 CanonicalBasis{2, Float64}:
 1.0  0.0
 0.0  1.0
```
"""
struct CanonicalBasis{dim,T} <: OrthonormalBasis{dim,T}
    e::AbstractArray{T,2} # Primal basis `eᵢ=e[:,i]`
    E::AbstractArray{T,2} # Dual basis `eⁱ=E[:,i]`
    g::AbstractArray{T,2} # Metric tensor `gᵢⱼ=eᵢ⋅eⱼ=g[i,j]`
    G::AbstractArray{T,2} # Inverse of the metric tensor `gⁱʲ=eⁱ⋅eʲ=G[i,j]`   
    function CanonicalBasis{dim,T}() where {dim,T}
        e = E = one(Tensor{2,dim,T})
        g = G = one(SymmetricTensor{2,dim,T})
        new{dim,T}(e, E, g, G)
    end
    CanonicalBasis() = CanonicalBasis{3,Sym}()
end

"""
    RotatedBasis(θ::T<:Number, ϕ::T<:Number, ψ::T<:Number)
    RotatedBasis(θ::T<:Number)

Orthonormal basis of dimension `dim` (default: 3) and type `T` (default: Sym) built from Euler angles `θ` in 2D and `(θ, ϕ, ψ)` in 3D

# Examples
```julia
julia> θ, ϕ, ψ = symbols("θ, ϕ, ψ", real = true) ; b = RotatedBasis(θ, ϕ, ψ) ; display(b.e)
3×3 Tensor{2, 3, Sym, 9}:
 -sin(ψ)⋅sin(ϕ) + cos(θ)⋅cos(ψ)⋅cos(ϕ)  -sin(ψ)⋅cos(θ)⋅cos(ϕ) - sin(ϕ)⋅cos(ψ)  sin(θ)⋅cos(ϕ)
  sin(ψ)⋅cos(ϕ) + sin(ϕ)⋅cos(θ)⋅cos(ψ)  -sin(ψ)⋅sin(ϕ)⋅cos(θ) + cos(ψ)⋅cos(ϕ)  sin(θ)⋅sin(ϕ)
                        -sin(θ)⋅cos(ψ)                          sin(θ)⋅sin(ψ)         cos(θ)
```
"""
struct RotatedBasis{dim,T} <: OrthonormalBasis{dim,T}
    e::AbstractArray{T,2} # Primal basis `eᵢ=e[:,i]`
    E::AbstractArray{T,2} # Dual basis `eⁱ=E[:,i]`
    g::AbstractArray{T,2} # Metric tensor `gᵢⱼ=eᵢ⋅eⱼ=g[i,j]`
    G::AbstractArray{T,2} # Inverse of the metric tensor `gⁱʲ=eⁱ⋅eʲ=G[i,j]`   
    function RotatedBasis(θ::T, ϕ::T, ψ::T) where {T<:Number}
        dim = 3
        e = E = Tensor{2,dim,T}(RotZYZ(ϕ, θ, ψ))
        g = G = one(SymmetricTensor{2,dim,T})
        new{dim,T}(e, E, g, G)
    end
    function RotatedBasis(θ::T) where {T<:Number}
        dim = 2
        e = E = Tensor{2,dim,T}((cos(θ), sin(θ), -sin(θ), cos(θ)))
        g = G = one(SymmetricTensor{2,dim,T})
        new{dim,T}(e, E, g, G)
    end
end

@pure Base.eltype(::AbstractBasis{dim,T}) where {dim,T} = T

getdim(::AbstractBasis{dim,T}) where {dim,T} = dim

"""
    vecbasis(b::AbstractBasis, var = :cov)

Returns the primal (if `var = :cov`) or primal (if `var = :cont`) basis
"""
vecbasis(b::AbstractBasis, ::Val{:cov}) = b.e
vecbasis(b::AbstractBasis, ::Val{:cont}) = b.E
vecbasis(b::AbstractBasis, var) = vecbasis(b, Val(var))
vecbasis(b::AbstractBasis) = vecbasis(b, :cov)

invvar(::Val{:cov}) = :cont
invvar(::Val{:cont}) = :cov
invvar(var) = invvar(Val(var))

"""
    metric(b::AbstractBasis, var = :cov)

Returns the covariant (if `var = :cov`) or contravariant (if `var = :cont`) metric matrix
"""
metric(b::AbstractBasis, ::Val{:cov}) = b.g
metric(b::AbstractBasis, ::Val{:cont}) = b.G
metric(b::AbstractBasis, var) = metric(b, Val(var))
metric(b::AbstractBasis) = metric(b, :cov)

"""
    normal_basis(v::AbstractArray{T,2}, var = :cov) where {T}

Builds a basis after normalization of column vectors of input matrix `v` where columns define either
- primal vectors ie `eᵢ=v[:,i]/norm(v[:,i])` if `var = :cov` as by default
- dual vector ie `eⁱ=v[:,i]/norm(v[:,i])` if `var = :cont`.
"""
function normal_basis(v::AbstractArray{T,2}, var = :cov) where {T}
    w = copy(v)
    for i = 1:size(w, 2)
        w[:, i] /= norm(w[:, i])
    end
    return Basis(w, var)
end

"""
    normalize(b::AbstractBasis, var = cov)

Builds a normalized basis from the input basis `b` by calling `normal_basis`
"""
LinearAlgebra.normalize(b::AbstractBasis, var = :cov) = normal_basis(vecbasis(b, var), var)

"""
    isorthogonal(b::AbstractBasis)

Checks whether the basis `b` is orthogonal
"""
function isorthogonal(b::AbstractBasis)
    ortho = true
    next = iterate(b.g)
    T = eltype(b)
    while ortho && next !== nothing
        (gij, state) = next
        i = state[end][1]
        j = state[end][2]
        ortho = i == j || gij == T(0)
        next = iterate(b.g, state)
    end
    return ortho
end

#####################
# Display Functions #
#####################
for OP in (:show, :print)
    @eval begin
        function Base.$OP(b::AbstractBasis)
            println("basis:")
            $OP(b.e)
            println("\ndual basis:")
            $OP(b.E)
            println("\ncovariant metric tensor:")
            $OP(b.g)
            println("\ncontravariant metric tensor:")
            $OP(b.G)
        end
    end
end
