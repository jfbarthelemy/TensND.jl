abstract type AbstractTensnd{order,dim,T<:Number} <: AbstractArray{T,order} end

Base.size(t::AbstractTensnd) = size(t.data)

Base.getindex(t::AbstractTensnd, ind...) = getindex(t.data, ind...)


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

```
TODO: to complete

"""
struct Tensnd{order,dim,T} <: AbstractTensnd{order,dim,T}
    data::AbstractArray{T,order}
    basis::AbstractBasis{dim,T}
    var::NTuple{order,Symbol}
    Tensnd(
        data::AbstractArray{T,order},
        basis::AbstractBasis{dim,T} = CanonicalBasis{3,T}(),
        var::NTuple{order,Symbol} = ntuple(_ -> :cont, order),
    ) where {order,dim,T} = new{order,dim,T}(data, basis, var)
    Tensnd(
        data::AbstractTensor{order,dim,T},
        basis::AbstractBasis{dim,T} = CanonicalBasis{dim,T}(),
        var::NTuple{order,Symbol} = ntuple(_ -> :cont, order),
    ) where {order,dim,T} = new{order,dim,T}(data, basis, var)
end

##############################
# Utility/Accessor Functions #
##############################
get_data(t::AbstractTensnd) = t.data

get_basis(t::AbstractTensnd) = t.basis

get_var(t::AbstractTensnd) = t.var

#####################
# Display Functions #
#####################
for OP in (:show, :print, :display)
    @eval begin
        Base.$OP(U::FourthOrderTensor) = $OP(tomandel(U))

        function Base.$OP(t::Tensnd)
            println("data: $(typeof(t.data))")
            $OP(t.data)
            println("\nvar:")
            $OP(t.var)
            println("\nbasis: $(typeof(t.basis))")
            $OP(t.basis)
        end
    end
end

########################
# Component extraction #
########################

"""
    components(::Tensnd{order,dim,T}, ::NTuple{order,Symbol})

TODO: to complete
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
                m = einsum(EinCode((ec1, ec2), ec3), (m, g_or_G))
            end
        end
        return m
    end
end
