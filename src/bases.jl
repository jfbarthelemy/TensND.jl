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
```julia
julia> v = Sym[1 0 0; 0 1 0; 0 1 1] ; b = Basis(v)
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
```julia
julia> b = CanonicalBasis()
CanonicalBasis{3, SymPy.Sym}
# basis: 3×3 Tensors.Tensor{2, 3, SymPy.Sym, 9}:
 1  0  0
 0  1  0
 0  0  1
# dual basis: 3×3 Tensors.Tensor{2, 3, SymPy.Sym, 9}:
 1  0  0
 0  1  0
 0  0  1
# covariant metric tensor: 3×3 Tensors.SymmetricTensor{2, 3, SymPy.Sym, 6}:
 1  0  0
 0  1  0
 0  0  1
# contravariant metric tensor: 3×3 Tensors.SymmetricTensor{2, 3, SymPy.Sym, 6}:
 1  0  0
 0  1  0
 0  0  1

julia> b = CanonicalBasis{2, Float64}()
CanonicalBasis{2, Float64}
# basis: 2×2 Tensors.Tensor{2, 2, Float64, 4}:
 1.0  0.0
 0.0  1.0
# dual basis: 2×2 Tensors.Tensor{2, 2, Float64, 4}:
 1.0  0.0
 0.0  1.0
# covariant metric tensor: 2×2 Tensors.SymmetricTensor{2, 2, Float64, 3}:
 1.0  0.0
 0.0  1.0
# contravariant metric tensor: 2×2 Tensors.SymmetricTensor{2, 2, Float64, 3}:
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
    angles::NamedTuple
    function RotatedBasis(θ::T, ϕ::T, ψ::T) where {T<:Number}
        dim = 3
        R = RotZYZ(ϕ, θ, ψ)
        e = E = Tensor{2,dim,T}(R)
        g = G = one(SymmetricTensor{2,dim,T})
        new{dim,T}(e, E, g, G, angles(R))
    end
    function RotatedBasis(θ::T) where {T<:Number}
        dim = 2
        e = E = Tensor{2,dim,T}((cos(θ), sin(θ), -sin(θ), cos(θ)))
        g = G = one(SymmetricTensor{2,dim,T})
        new{dim,T}(e, E, g, G, angles(e))
    end
    function RotatedBasis(θ::Sym)
        dim = 2
        e = E = Tensor{2,dim,Sym}((cos(θ), sin(θ), -sin(θ), cos(θ)))
        g = G = one(SymmetricTensor{2,dim,Sym})
        new{dim,Sym}(e, E, g, G, (θ = θ,))
    end
end


angles(M::AbstractArray{T,2}, ::Val{2}) where {T} = (θ = atan(M[2,1] - M[1,2], M[1,1] + M[2,2]),)
function angles(M::AbstractArray{T,2}, ::Val{3}) where {T}
    R = RotZYZ(M)
    return (θ = R.theta2, ϕ = R.theta1, ψ = R.theta3)
end
"""
    angles(M::AbstractArray{T,2})

Determines the Euler angles corresponding to the input matrix supposed to be a rotation matrix or at least a similarity

# Examples
```julia
julia> θ, ϕ, ψ = symbols("θ, ϕ, ψ", real = true) ; b = RotatedBasis(θ, ϕ, ψ) ; display(b.e)
3×3 Tensor{2, 3, Sym, 9}:
 -sin(ψ)⋅sin(ϕ) + cos(θ)⋅cos(ψ)⋅cos(ϕ)  -sin(ψ)⋅cos(θ)⋅cos(ϕ) - sin(ϕ)⋅cos(ψ)  sin(θ)⋅cos(ϕ)
  sin(ψ)⋅cos(ϕ) + sin(ϕ)⋅cos(θ)⋅cos(ψ)  -sin(ψ)⋅sin(ϕ)⋅cos(θ) + cos(ψ)⋅cos(ϕ)  sin(θ)⋅sin(ϕ)
                        -sin(θ)⋅cos(ψ)                          sin(θ)⋅sin(ψ)         cos(θ)

julia> angles(b)
(θ = θ, ϕ = ϕ, ψ = ψ)
```
"""
angles(M::AbstractArray{T,2}) where {T} = angles(M, Val(size(M)[1]))
angles(b::RotatedBasis) = b.angles

angles(v::AbstractArray{T,1}, ::Val{2}) where {T} = (θ = atan(v[2], v[1]))
angles(v::AbstractArray{T,1}, ::Val{3}) where {T} = (θ = atan(√(v[1]^2+v[2]^2), v[3]), ϕ = atan(v[2], v[1]))
angles(v::AbstractArray{T,1}) where {T} = angles(v, Val(size(v)[1]))


@pure Base.eltype(::AbstractBasis{dim,T}) where {dim,T} = T

@pure getdim(::AbstractBasis{dim}) where {dim} = dim

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
function isorthogonal(b::AbstractBasis{dim,T}) where {dim,T}
    ortho = true
    next = iterate(b.g)
    while ortho && next !== nothing
        (gij, state) = next
        i = state[end][1]
        j = state[end][2]
        ortho = i == j || gij ≈ T(0)
        next = iterate(b.g, state)
    end
    return ortho
end

function isorthogonal(b::AbstractBasis{dim,Sym}) where {dim}
    ortho = true
    next = iterate(b.g)
    while ortho && next !== nothing
        (gij, state) = next
        i = state[end][1]
        j = state[end][2]
        ortho = i == j || factor(simplify(gij)) == Sym(0)
        next = iterate(b.g, state)
    end
    return ortho
end

isorthogonal(b::OrthonormalBasis) = true

"""
    isorthonormal(b::AbstractBasis)

Checks whether the basis `b` is orthonormal
"""
function isorthonormal(b::AbstractBasis{dim,T}) where {dim,T}
    ortho = true
    next = iterate(b.g)
    while ortho && next !== nothing
        (gij, state) = next
        i = state[end][1]
        j = state[end][2]
        ortho = i == j ? gij ≈ T(1) : gij ≈ T(0)
        next = iterate(b.g, state)
    end
    return ortho
end

function isorthonormal(b::AbstractBasis{dim,Sym}) where {dim}
    ortho = true
    next = iterate(b.g)
    while ortho && next !== nothing
        (gij, state) = next
        i = state[end][1]
        j = state[end][2]
        ortho = i == j ? factor(simplify(gij)) == Sym(1) : factor(simplify(gij)) == Sym(0)
        next = iterate(b.g, state)
    end
    return ortho
end

isorthonormal(b::OrthonormalBasis) = true

#####################
# Display Functions #
#####################
for OP in (:show, :print, :display)
    @eval begin
        function Base.$OP(b::AbstractBasis)
            $OP(typeof(b))
            print("# basis: ")
            $OP(b.e)
            print("# dual basis: ")
            $OP(b.E)
            print("# covariant metric tensor: ")
            $OP(b.g)
            print("# contravariant metric tensor: ")
            $OP(b.G)
        end
    end
end
