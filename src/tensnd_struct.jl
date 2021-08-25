abstract type AbstractTensnd{
    order,
    dim,
    TA<:Number,
    TB<:Number,
    A<:AbstractArray,
    B<:AbstractBasis,
} <: AbstractArray{TA,order} end

Base.size(t::AbstractTensnd) = size(t.data)

Base.getindex(t::AbstractTensnd, ind...) = getindex(t.data, ind...)

@pure Base.eltype(::AbstractTensnd{order,dim,TA,TB,A,B}) where {order,dim,TA,TB,A,B} = TA
getdim(::AbstractTensnd{order,dim,TA,TB,A,B}) where {order,dim,TA,TB,A,B} = dim
getorder(::AbstractTensnd{order,dim,TA,TB,A,B}) where {order,dim,TA,TB,A,B} = order
getdatatype(::AbstractTensnd{order,dim,TA,TB,A,B}) where {order,dim,TA,TB,A,B} = A
getbasistype(::AbstractTensnd{order,dim,TA,TB,A,B}) where {order,dim,TA,TB,A,B} = B


"""
    Tensnd{order,dim,TA<:Number,TB<:Number,A<:AbstractArray,B<:AbstractBasis}

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
struct Tensnd{order,dim,TA,TB,A,B} <: AbstractTensnd{order,dim,TA,TB,A,B}
    data::A
    var::NTuple{order,Symbol}
    basis::B
    function Tensnd(
        data::AbstractArray{TA,order},
        var::NTuple{order,Symbol} = ntuple(_ -> :cont, order),
        basis::AbstractBasis{dim,TB} = Basis{3,TA}(),
    ) where {order,dim,TA,TB}
        newdata = tensor_or_array(data, Val(dim))
        new{order,dim,TA,TB,typeof(newdata),typeof(basis)}(newdata, var, basis)
    end
    function Tensnd(
        data::AbstractTensor{order,dim,TA},
        var::NTuple{order,Symbol} = ntuple(_ -> :cont, order),
        basis::AbstractBasis{dim,TB} = Basis{dim,TA}(),
        ) where {order,dim,TA,TB}
        newdata = tensor_or_array(data, Val(dim))
        new{order,dim,TA,TB,typeof(newdata),typeof(basis)}(newdata, var, basis)
    end
end

# This function aims at storing the table of components in the `Tensor` type whenever possible
for dim ∈ (1, 2, 3)
    @eval tensor_or_array(tab::Array{T,1}, ::Val{$dim}) where {T} = Vec{$dim}(tab)
end
for order ∈ (2, 4), dim ∈ (1, 2, 3)
    @eval function tensor_or_array(tab::Array{T,$order}, ::Val{$dim}) where {T}
        newtab = Tensor{$order,$dim}(tab)
        if Tensors.issymmetric(newtab)
            newtab = convert(SymmetricTensor{$order,$dim}, newtab)
        end
        if T == Sym
            newtab = sympy.trigsimp.(newtab)
        end
        return newtab
    end
end
tensor_or_array(tab::Tensors.AllTensors, ::Val{dim}) where {dim} = tab
tensor_or_array(tab::AbstractArray, ::Val{dim}) where {dim} = tab

##############################
# Utility/Accessor Functions #
##############################
getdata(t::AbstractTensnd) = t.data

getvar(t::AbstractTensnd) = t.var

getbasis(t::AbstractTensnd) = t.basis

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

# Base.display(U::FourthOrderTensor) = display(tomandel(U))


########################
# Component extraction #
########################

"""
    components(::Tensnd{order,dim,T}, ::NTuple{order,Symbol})
    components(::Tensnd{order,dim,T}, ::NTuple{order,Symbol}, ::AbstractBasis{dim,T})

Extracts the components of a tensor for new variances and/or in a new basis

# Examples
```julia
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

julia> TV = Tensnd(V) # TV = Tensnd(V, (:cont,), CanonicalBasis())
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
function components(
    t::Tensnd{order,dim,T},
    var::NTuple{order,Symbol},
) where {order,dim,T<:Number}
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
                if T == Sym
                    m = sympy.trigsimp.(m)
                end
            end
        end
        return m
    end
end

function components(
    t::Tensnd{order,dim,T},
    var::NTuple{order,Symbol},
    basis::AbstractBasis{dim,T2},
) where {order,dim,T<:Number,T2<:Number}
    if basis == t.basis
        return components(t, var)
    else
        bb = Dict()
        for v1 ∈ (:cov, :cont), v2 ∈ (:cov, :cont)
            if v1 ∈ t.var && v2 ∈ var
                bb[v1, v2] = vecbasis(t.basis, invvar(v1))' ⋅ vecbasis(basis, v2)
            end
        end
        m = Array(t.data)
        ec1 = ntuple(i -> i, order)
        newcp = order + 1
        for i ∈ 1:order
            c = bb[t.var[i], var[i]]
            if c ≠ I
                ec2 = (i, newcp)
                ec3 = ntuple(j -> j ≠ i ? j : newcp, order)
                m = einsum(EinCode((ec1, ec2), ec3), (m, c))
                if T == Sym
                    m = sympy.trigsimp.(m)
                end
            end
        end
        return m
    end
end

##############
# Operations #
##############

function same_basis(
    t1::Tensnd{order1,dim,T1},
    t2::Tensnd{order2,dim,T2},
) where {order1,order2,dim,T1,T2}
    if t1.basis == t2.basis
        return t1, t2
    else
        newdata = components(t2, t2.var, t1.basis)
        t3 = Tensnd(newdata, t2.var, t1.basis)
        return t1, t3
    end
end

function same_basis_same_var(
    t1::Tensnd{order,dim,T1},
    t2::Tensnd{order,dim,T2},
) where {order,dim,T1,T2}
    if t1.basis == t2.basis && t1.var == t2.var
        return t1, t2
    else
        newdata = components(t2, t1.var, t1.basis)
        t3 = Tensnd(newdata, t1.var, t1.basis)
        return t1, t3
    end
end

for OP in (:(==), :(!=))
    @eval function Base.$OP(
        t1::Tensnd{order,dim,T1},
        t2::Tensnd{order,dim,T2},
    ) where {order,dim,T1,T2}
        nt1, nt2 = same_basis_same_var(t1, t2)
        return $OP(nt1.data, nt2.data)
    end
end


function Base.:+(t1::Tensnd{order,dim,T1}, t2::Tensnd{order,dim,T2}) where {order,dim,T1,T2}
    nt1, nt2 = same_basis_same_var(t1, t2)
    return Tensnd(nt1.data + nt2.data, nt1.var, nt1.basis)
end

function Base.:-(t1::Tensnd{order,dim,T1}, t2::Tensnd{order,dim,T2}) where {order,dim,T1,T2}
    nt1, nt2 = same_basis_same_var(t1, t2)
    return Tensnd(nt1.data - nt2.data, nt1.var, nt1.basis)
end

function Base.:*(α::T1, t::Tensnd{order,dim,T2}) where {order,dim,T1<:Number,T2}
    return Tensnd(α * t.data, t.var, t.basis)
end

function Base.:*(t::Tensnd{order,dim,T2}, α::T1) where {order,dim,T1<:Number,T2}
    return Tensnd(α * t.data, t.var, t.basis)
end

function Base.:/(t::Tensnd{order,dim,T2}, α::T1) where {order,dim,T1<:Number,T2}
    return Tensnd(t.data / α, t.var, t.basis)
end

function Base.inv(t::Tensnd{order,dim,T}) where {order,dim,T<:Number}
    var = ntuple(i -> invvar(t.var[i]), order)
    return Tensnd(inv(t.data), var, t.basis)
end

Tensors.tomandel(t::Tensnd{4,dim,T}) where {dim,T<:Number} = tomandel(t.data)

function Tensors.tomandel(t::Tensnd{4,dim,T}, b::AbstractBasis{dim,T}) where {dim,T<:Number}
    if t.basis == b
        return tomandel(t)
    else
        newt = tensor_or_array(components(t, t.var, b), Val(dim))
        return tomandel(newt)
    end
end

const KM = tomandel

# frommandel(TT::Type{<:SymmetricTensor}, v::AbstractVecOrMat{T}; kwargs...)




