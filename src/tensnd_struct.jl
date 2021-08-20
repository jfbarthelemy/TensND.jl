abstract type AbstractTensnd{order,dim,T<:Number} <: AbstractArray{T,order} end

Base.size(t::AbstractTensnd) = size(t.data)

Base.getindex(t::AbstractTensnd, ind...) = getindex(t.data, ind...)

@pure Base.eltype(::AbstractTensnd{order,dim,T}) where {order,dim,T} = T
getdim(::AbstractTensnd{order,dim,T}) where {order,dim,T} = dim
getorder(::AbstractTensnd{order,dim,T}) where {order,dim,T} = order


"""
    Tensnd{order,dim,T<:Number}

Tensor type of any order defined by
- a multiarray of components (of any type heriting from `AbstractArray`, e.g. `Tensor` or `SymmetricTensor`)
- a basis of `AbstractBasis` type
- a tuple of variances (covariant `:cov` or contravariant `:cont`) of length equal to the `order` of the tensor

# Examples
```jldoctest
julia> using SymPy

julia> v = Sym[1 0 0; 0 1 0; 0 1 1] ; b = Basis(v)
3×3 Basis{3, Sym}:
 1  0  0
 0  1  0
 0  1  1

julia> T = Tensnd(b.g,(:cov,:cov),b)
3×3 Tensnd{2, 3, Sym}:
 1  0  0
 0  2  1
 0  1  1

julia> components(T,(:cont,:cov),b)
3×3 Matrix{Sym}:
 1  0  0
 0  1  0
 0  0  1
```

"""
struct Tensnd{order,dim,T} <: AbstractTensnd{order,dim,T}
    data::AbstractArray{T,order}
    var::NTuple{order,Symbol}
    basis::AbstractBasis{dim,T}
    Tensnd(
        data::AbstractArray{T,order},
        var::NTuple{order,Symbol} = ntuple(_ -> :cont, order),
        basis::AbstractBasis{dim,T} = CanonicalBasis{3,T}(),
    ) where {order,dim,T} = new{order,dim,T}(data, var, basis)
    Tensnd(
        data::AbstractTensor{order,dim,T},
        var::NTuple{order,Symbol} = ntuple(_ -> :cont, order),
        basis::AbstractBasis{dim,T} = CanonicalBasis{dim,T}(),
    ) where {order,dim,T} = new{order,dim,T}(data, var, basis)
end

##############################
# Utility/Accessor Functions #
##############################
get_data(t::AbstractTensnd) = t.data

get_var(t::AbstractTensnd) = t.var

get_basis(t::AbstractTensnd) = t.basis

#####################
# Display Functions #
#####################
for OP in (:show, :print)
    @eval begin
        Base.$OP(U::FourthOrderTensor) = $OP(tomandel(U))

        function Base.$OP(t::AbstractTensnd)
            println("data: $(typeof(t.data))")
            $OP(t.data)
            println("\nvar:")
            $OP(t.var)
            println("\nbasis: $(typeof(t.basis))")
            $OP(t.basis.e)
        end
    end
end

########################
# Component extraction #
########################

"""
    components(::Tensnd{order,dim,T}, ::NTuple{order,Symbol})
    components(::Tensnd{order,dim,T},::NTuple{order,Symbol},::AbstractBasis{dim,T})

Extracts the components of a tensor for new variances and/or in a new basis

# Examples
```jldoctest
julia> using SymPy

julia> v = Sym[0 1 1; 1 0 1; 1 1 0] ; b = Basis(v)
3×3 Basis{3, Sym}:
 0  1  1
 1  0  1
 1  1  0

julia> V = Tensor{1,3}(i->symbols("v\$i",real=true))
3-element Vec{3, Sym}:
 v₁
 v₂
 v₃

julia> TV = Tensnd(V)
3-element Tensnd{1, 3, Sym}:
 v₁
 v₂
 v₃

julia> components(TV, (:cont,), b)
3-element Vector{Sym}:
 -v1/2 + v2/2 + v3/2
  v1/2 - v2/2 + v3/2
  v1/2 + v2/2 - v3/2

julia> components(TV, (:cov,), b)
3-element Vector{Sym}:
 v₂ + v₃
 v₁ + v₃
 v₁ + v₂

julia> components(TV, (:cov,), normal_basis(b))
3-element Vector{Sym}:
 sqrt(2)*(v2 + v3)/2
 sqrt(2)*(v1 + v3)/2
 sqrt(2)*(v1 + v2)/2

julia> T = Tensor{2,3}((i,j)->symbols("t\$i\$j",real=true))
3×3 Tensor{2, 3, Sym, 9}:
 t₁₁  t₁₂  t₁₃
 t₂₁  t₂₂  t₂₃
 t₃₁  t₃₂  t₃₃

julia> TT = Tensnd(T)
3×3 Tensnd{2, 3, Sym}:
 t₁₁  t₁₂  t₁₃
 t₂₁  t₂₂  t₂₃
 t₃₁  t₃₂  t₃₃

julia> components(TT, (:cov,:cov), b)
3×3 Matrix{Sym}:
 t₂₂ + t₂₃ + t₃₂ + t₃₃  t₂₁ + t₂₃ + t₃₁ + t₃₃  t₂₁ + t₂₂ + t₃₁ + t₃₂
 t₁₂ + t₁₃ + t₃₂ + t₃₃  t₁₁ + t₁₃ + t₃₁ + t₃₃  t₁₁ + t₁₂ + t₃₁ + t₃₂
 t₁₂ + t₁₃ + t₂₂ + t₂₃  t₁₁ + t₁₃ + t₂₁ + t₂₃  t₁₁ + t₁₂ + t₂₁ + t₂₂

julia> components(TT, (:cont,:cov), b)
3×3 Matrix{Sym}:
 -t12/2 - t13/2 + t22/2 + t23/2 + t32/2 + t33/2  -t11/2 - t13/2 + t21/2 + t23/2 + t31/2 + t33/2  -t11/2 - t12/2 + t21/2 + t22/2 + t31/2 + t32/2
  t12/2 + t13/2 - t22/2 - t23/2 + t32/2 + t33/2   t11/2 + t13/2 - t21/2 - t23/2 + t31/2 + t33/2   t11/2 + t12/2 - t21/2 - t22/2 + t31/2 + t32/2
  t12/2 + t13/2 + t22/2 + t23/2 - t32/2 - t33/2   t11/2 + t13/2 + t21/2 + t23/2 - t31/2 - t33/2   t11/2 + t12/2 + t21/2 + t22/2 - t31/2 - t32/2
```
"""
function components(t::Tensnd{order,dim,T}, var::NTuple{order,Symbol}) where {order,dim,T}
    if var == t.var
        return t.data
    else
        m = Array(t.data)
        ec1 = ntuple(i -> i, order)
        newcp = order + 1
        for i ∈ 1:order
            if t.var[i] ≠ var[i]
                g_or_G = metric(t.basis, var[i])
                ec2 = (i, newcp)
                ec3 = ntuple(j -> j ≠ i ? j : newcp, order)
                m = simplify(einsum(EinCode((ec1, ec2), ec3), (m, g_or_G)))
            end
        end
        return m
    end
end

function components(
    t::Tensnd{order,dim,T},
    var::NTuple{order,Symbol},
    basis::AbstractBasis{dim,T},
) where {order,dim,T}
    if basis == t.basis
        return components(t, var)
    else
        bb = Dict()
        for v1 ∈ (:cov, :cont), v2 ∈ (:cov, :cont)
            if v1 ∈ t.var && v2 ∈ var
                bb[v1, v2] = simplify.(vecbasis(t.basis, invvar(v1))' ⋅ vecbasis(basis, v2))
            end
        end
        m = Array(t.data)
        coef = zeros(T, ntuple(_ -> dim, 2 * order)...)
        for tind ∈ CartesianIndices(m), ind ∈ CartesianIndices(m)
            coef[Tuple(tind)..., Tuple(ind)...] =
                simplify(prod([bb[t.var[i], var[i]][tind[i], ind[i]] for i ∈ 1:order]))
        end
        ec1 = ntuple(i -> i, order)
        ec2 = ntuple(i -> i, 2 * order)
        ec3 = ntuple(i -> i + order, order)
        m = simplify.(einsum(EinCode((ec1, ec2), ec3), (m, coef)))
        return m
    end
end
