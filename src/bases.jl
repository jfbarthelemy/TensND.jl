abstract type AbstractBasis{dim, T <: Number} end


"""
    Basis{dim, T<:Number}(v, var = :cov)

Basis built from a square matrix `v` where columns correspond either to
- primal vectors ie `eᵢ=v[:,i]` if `var = :cov` as by default
- dual vectors ie `eⁱ=v[:,i]` if `var = :cont`.

The attributes of this object are
- `Basis.e`: square matrix defining the primal basis `eᵢ=e[:,i]`
- `Basis.E`: square matrix defining the dual basis `eⁱ=E[:,i]`
- `Basis.g`: square matrix defining the covariant components of the metric tensor `gᵢⱼ=eᵢ⋅eⱼ=g[i,j]`
- `Basis.G`: square matrix defining the contravariant components of the metric tensor `gⁱʲ=eⁱ⋅eʲ=G[i,j]`

# Examples
```jldoctest
julia> using LinearAlgebra, SymPy

julia> v = Sym[1 0 0; 0 1 0; 0 1 1]
3×3 Matrix{Sym}:
 1  0  0
 0  1  0
 0  1  1

julia> b = Basis(v)
Basis{3, Sym}(Sym[1 0 0; 0 1 0; 0 1 1], Sym[1 0 0; 0 1 -1; 0 0 1], Sym[1 0 0; 0 2 1; 0 1 1], Sym[1 0 0; 0 1 -1; 0 -1 2])
```
"""
struct Basis{dim, T} <: AbstractBasis{dim, T}
    e # Primal basis `eᵢ=e[:,i]`
    E # Dual basis `eⁱ=E[:,i]`
    g # Metric tensor `gᵢⱼ=eᵢ⋅eⱼ=g[i,j]`
    G # Inverse of the metric tensor `gⁱʲ=eⁱ⋅eʲ=G[i,j]`   
    function Basis(v::Array{T,2}, ::Val{:cov}) where {T}
        dim = size(v,1)
        @assert dim==size(v,2) "v should be a square matrix"
        e = v
        g = [simplify(v[:, i]'v[:, j]) for i = 1:dim, j = 1:dim]
        G = inv(g)
        E = transpose(G * transpose(v))
        g = Symmetric(g) ; G = Symmetric(G)
        new{dim, T}(e, E, g, G)
    end
    function Basis(v::Array{T,2}, ::Val{:cont}) where {T}
        dim = size(v,1)
        @assert dim==size(v,2) "v should be a square matrix"
        E = v
        G = [simplify(v[:, i]'v[:, j]) for i = 1:dim, j = 1:dim]
        g = inv(G)
        e = transpose(g * transpose(v))
        g = Symmetric(g) ; G = Symmetric(G)
        new{dim, T}(e, E, g, G)
    end
    Basis(v::Array{T,2}, var) where {T} = Basis(v, Val(var))
    Basis(v::Array{T,2}) where {T} = Basis(v, :cov)
end

"""
    CanonicalBasis{dim, T}()

Canonical basis of dimension `dim` (default: 3) and type `T` (default: Sym)

The attributes of this object are
- `Basis.e`: identity matrix defining the primal basis `e[i,j]=δᵢⱼ`
- `Basis.E`: identity matrix defining the dual basis `g[i,j]=δᵢⱼ`
- `Basis.g`: identity matrix defining the covariant components of the metric tensor `g[i,j]=δᵢⱼ`
- `Basis.G`: identity matrix defining the contravariant components of the metric tensor `G[i,j]=δᵢⱼ`

# Examples
```jldoctest
julia> using LinearAlgebra, SymPy

julia> b = CanonicalBasis()
CanonicalBasis{3, Sym}(Sym[1 0 0; 0 1 0; 0 0 1], Sym[1 0 0; 0 1 0; 0 0 1], Sym[1 0 0; 0 1 0; 0 0 1], Sym[1 0 0; 0 1 0; 0 0 1])

julia> b = CanonicalBasis{2, Float64}()
CanonicalBasis{2, Float64}([1.0 0.0; 0.0 1.0], [1.0 0.0; 0.0 1.0], [1.0 0.0; 0.0 1.0], [1.0 0.0; 0.0 1.0])
```
"""
struct CanonicalBasis{dim, T} <: AbstractBasis{dim, T}
    e # Primal basis `eᵢ=e[:,i]`
    E # Dual basis `eⁱ=E[:,i]`
    g # Metric tensor `gᵢⱼ=eᵢ⋅eⱼ=g[i,j]`
    G # Inverse of the metric tensor `gⁱʲ=eⁱ⋅eʲ=G[i,j]`   
    function CanonicalBasis{dim, T}() where {dim, T}
        e = E = Matrix(one(T)*I, dim, dim)
        g = G = Symmetric(Matrix(one(T)*I, dim, dim))
        new{dim, T}(e, E, g, G)
    end
    CanonicalBasis() = CanonicalBasis{3, Sym}()
end


Base.eltype(::AbstractBasis{dim, T}) where {dim,T} = T

"""
    basis(b::AbstractBasis, var = :cov)

Returns the primal (if `var = :cov`) or primal (if `var = :cont`) basis
"""
basis(b::AbstractBasis, ::Val{:cov}) = b.e
basis(b::AbstractBasis, ::Val{:cont}) = b.E
basis(b::AbstractBasis, var) = basis(b, Val(var))
basis(b::AbstractBasis) = basis(b, :cov)

"""
    metric(b::AbstractBasis, var = :cov)

Returns the covariant (if `var = :cov`) or contravariant (if `var = :cont`) metric matrix
"""
metric(b::AbstractBasis, ::Val{:cov}) = b.g
metric(b::AbstractBasis, ::Val{:cont}) = b.G
metric(b::AbstractBasis, var) = metric(b, Val(var))
metric(b::AbstractBasis) = metric(b, :cov)

"""
    normal_basis(v::Array{T,2}, var = :cov) where {T}

Builds a basis after normalization of column vectors of input matrix `v` where columns define either
- primal vectors ie `eᵢ=v[:,i]/norm(v[:,i])` if `var = :cov` as by default
- dual vector ie `eⁱ=v[:,i]/norm(v[:,i])` if `var = :cont`.
"""
function normal_basis(v::Array{T,2}, var = :cov) where {T}
    w = copy(v)
    for i = 1:size(w,2)
        w[:,i] /= norm(w[:,i])
    end
    return Basis(w, var)
end

"""
    normalize(b::AbstractBasis, var = cov)

Builds a normalized basis from the input basis `b` by calling `normal_basis`
"""
LinearAlgebra.normalize(b::AbstractBasis, var = cov) = normal_basis(basis(b, var), var)

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
        i = state[end][1] ; j = state[end][2]
        ortho = i == j || gij == T(0)
        next = iterate(b.g, state)
    end
    return ortho
end

