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
@pure getdim(::AbstractTensnd{order,dim,TA,TB,A,B}) where {order,dim,TA,TB,A,B} = dim
@pure getorder(::AbstractTensnd{order,dim,TA,TB,A,B}) where {order,dim,TA,TB,A,B} = order
@pure getdatatype(::AbstractTensnd{order,dim,TA,TB,A,B}) where {order,dim,TA,TB,A,B} = A
@pure getbasistype(::AbstractTensnd{order,dim,TA,TB,A,B}) where {order,dim,TA,TB,A,B} = B


"""
    Tensnd{order,dim,TA<:Number,TB<:Number,A<:AbstractArray,B<:AbstractBasis}

Tensor type of any order defined by
- a multiarray of components (of any type heriting from `AbstractArray`, e.g. `Tensor` or `SymmetricTensor`)
- a basis of `AbstractBasis` type
- a tuple of variances (covariant `:cov` or contravariant `:cont`) of length equal to the `order` of the tensor

# Examples
```jldoctest
julia> v = Sym[1 0 0; 0 1 0; 0 1 1] ; b = Basis(v)
3×3 Basis{3, Sym}:
 1  0  0
 0  1  0
 0  1  1

julia> T = Tensnd(b.g,(:cov,:cov),b)
3×3 Tensnd{2, 3, Sym, Sym, SymmetricTensor{2, 3, Sym, 6}, Basis{3, Sym}}:
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
        newdata = tensor_or_array(data)
        new{order,dim,TA,TB,typeof(newdata),typeof(basis)}(newdata, var, basis)
    end
    function Tensnd(
        data::AbstractTensor{order,dim,TA},
        var::NTuple{order,Symbol} = ntuple(_ -> :cont, order),
        basis::AbstractBasis{dim,TB} = Basis{dim,TA}(),
    ) where {order,dim,TA,TB}
        newdata = tensor_or_array(data)
        new{order,dim,TA,TB,typeof(newdata),typeof(basis)}(newdata, var, basis)
    end
end

# This function aims at storing the table of components in the `Tensor` type whenever possible
tensor_or_array(tab::Array{T,1}) where {T} = Vec{size(tab)[1]}(tab)
for order ∈ (2, 4)
    @eval function tensor_or_array(tab::Array{T,$order}) where {T}
        dim = size(tab)[1]
        newtab = Tensor{$order,dim}(tab)
        if Tensors.issymmetric(newtab)
            newtab = convert(SymmetricTensor{$order,dim}, newtab)
        end
        if T == Sym
            newtab = Tensors.get_base(typeof(newtab))(sympy.trigsimp.(newtab))
        end
        return newtab
    end
end
tensor_or_array(tab::Tensors.AllTensors) = tab
tensor_or_array(tab::AbstractArray) = tab

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
```jldoctest
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
3-element Tensnd{1, 3, Sym, Sym, Vec{3, Sym}, CanonicalBasis{3, Sym}}:
 v₁
 v₂
 v₃

julia> factor.(components(TV, (:cont,), b))
3-element Vector{Sym}:
 -(v1 - v2 - v3)/2
  (v1 - v2 + v3)/2
  (v1 + v2 - v3)/2

julia> components(TV, (:cov,), b)
3-element Vector{Sym}:
 v₂ + v₃
 v₁ + v₃
 v₁ + v₂

julia> simplify.(components(TV, (:cov,), normal_basis(b)))
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
3×3 Tensnd{2, 3, Sym, Sym, Tensor{2, 3, Sym, 9}, CanonicalBasis{3, Sym}}:
 t₁₁  t₁₂  t₁₃
 t₂₁  t₂₂  t₂₃
 t₃₁  t₃₂  t₃₃

julia> components(TT, (:cov,:cov), b)
3×3 Matrix{Sym}:
 t₂₂ + t₂₃ + t₃₂ + t₃₃  t₂₁ + t₂₃ + t₃₁ + t₃₃  t₂₁ + t₂₂ + t₃₁ + t₃₂
 t₁₂ + t₁₃ + t₃₂ + t₃₃  t₁₁ + t₁₃ + t₃₁ + t₃₃  t₁₁ + t₁₂ + t₃₁ + t₃₂
 t₁₂ + t₁₃ + t₂₂ + t₂₃  t₁₁ + t₁₃ + t₂₁ + t₂₃  t₁₁ + t₁₂ + t₂₁ + t₂₂

julia> factor.(components(TT, (:cont,:cov), b))
3×3 Matrix{Sym}:
 -(t12 + t13 - t22 - t23 - t32 - t33)/2  …  -(t11 + t12 - t21 - t22 - t31 - t32)/2
  (t12 + t13 - t22 - t23 + t32 + t33)/2      (t11 + t12 - t21 - t22 + t31 + t32)/2
  (t12 + t13 + t22 + t23 - t32 - t33)/2      (t11 + t12 + t21 + t22 - t31 - t32)/2
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

components(
    t::Tensnd{order,dim,TA,TB,A,B},
    ::NTuple{order,Symbol},
) where {order,dim,TA<:Number,TB<:Number,A,B<:OrthonormalBasis} = t.data

components(
    t::Tensnd{order,dim,TA,TB,A,B},
) where {order,dim,TA<:Number,TB<:Number,A,B<:OrthonormalBasis} = t.data


function components(
    t::Tensnd{order,dim,T},
    var::NTuple{order,Symbol},
    basis::AbstractBasis{dim},
) where {order,dim,T}
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
    t1::Tensnd{order1,dim},
    t2::Tensnd{order2,dim},
) where {order1,order2,dim}
    if t1.basis == t2.basis
        return t1, t2
    else
        newdata = components(t2, t2.var, t1.basis)
        t3 = Tensnd(newdata, t2.var, t1.basis)
        return t1, t3
    end
end

function same_basis_same_var(t1::Tensnd{order,dim}, t2::Tensnd{order,dim}) where {order,dim}
    if t1.basis == t2.basis && t1.var == t2.var
        return t1, t2
    else
        newdata = components(t2, t1.var, t1.basis)
        t3 = Tensnd(newdata, t1.var, t1.basis)
        return t1, t3
    end
end

for OP in (:(==), :(!=))
    @eval function Base.$OP(t1::Tensnd{order,dim}, t2::Tensnd{order,dim}) where {order,dim}
        nt1, nt2 = same_basis_same_var(t1, t2)
        return $OP(nt1.data, nt2.data)
    end
end

for OP in (:+, :-)
    @eval function Base.$OP(t1::Tensnd{order,dim}, t2::Tensnd{order,dim}) where {order,dim}
        nt1, nt2 = same_basis_same_var(t1, t2)
        return Tensnd($OP(nt1.data, nt2.data), nt1.var, nt1.basis)
    end
end

Base.:*(α::Number, t::Tensnd) = Tensnd(α * t.data, t.var, t.basis)
Base.:*(t::Tensnd, α::Number) = Tensnd(α * t.data, t.var, t.basis)
Base.:/(t::Tensnd, α::Number) = Tensnd(t.data / α, t.var, t.basis)

Base.inv(t::Tensnd{2}) = Tensnd(inv(t.data), (invvar(t.var[2]), invvar(t.var[1])), t.basis)
Base.inv(t::Tensnd{4}) = Tensnd(
    inv(t.data),
    (invvar(t.var[3]), invvar(t.var[4]), invvar(t.var[1]), invvar(t.var[2])),
    t.basis,
)

KM(t::Tensors.AllTensors; kwargs...) = tomandel(t; kwargs...)
KM(t::Tensnd; kwargs...) = tomandel(t.data; kwargs...)

function KM(t::Tensnd{order,dim}, b::AbstractBasis{dim}; kwargs...) where {order,dim}
    if t.basis == b
        return KM(t; kwargs...)
    else
        newt = tensor_or_array(components(t, t.var, b))
        return tomandel(newt; kwargs...)
    end
end

KM(t::AbstractArray; kwargs...) = KM(Tensnd(t); kwargs...)
KM(t::AbstractArray, b::AbstractBasis; kwargs...) = KM(Tensnd(t), b; kwargs...)

const select_type_KM = Dict(
    (6, 6) => SymmetricTensor{4,3},
    (9, 9) => Tensor{4,3},
    (3, 3) => SymmetricTensor{4,2},
    (4, 4) => Tensor{4,2},
    (6,) => SymmetricTensor{2,3},
    (9,) => Tensor{2,3},
    (3,) => SymmetricTensor{2,2},
    (4,) => Tensor{2,2},
)
invKM(TT::Type{<:Tensors.AllTensors}, v::AbstractVecOrMat; kwargs...) =
    Tensnd(frommandel(TT, v; kwargs...))
invKM(v::AbstractVecOrMat; kwargs...) = invKM(select_type_KM[size(v)], v; kwargs...)

function Tensors.otimes(
    t1::AbstractArray{T1,order1},
    t2::AbstractArray{T2,order2},
) where {T1,T2,order1,order2}
    ec1 = ntuple(i -> i, order1)
    ec2 = ntuple(i -> order1 + i, order2)
    ec3 = ntuple(i -> i, order1 + order2)
    return einsum(EinCode((ec1, ec2), ec3), (AbstractArray{T1}(t1), AbstractArray{T2}(t2)))
end

function Tensors.otimes(
    t1::Tensnd{order1,dim},
    t2::Tensnd{order2,dim},
) where {order1,order2,dim}
    nt1, nt2 = same_basis(t1, t2)
    data = otimes(nt1.data, nt2.data)
    var = (nt1.var..., nt2.var...)
    return Tensnd(data, var, nt1.basis)
end

function scontract(
    t1::AbstractArray{T1,order1},
    t2::AbstractArray{T2,order2},
) where {T1,T2,order1,order2}
    ec1 = ntuple(i -> i, order1)
    ec2 = ntuple(i -> order1 - 1 + i, order2)
    ec3 = (ec1[begin:end-1]..., ec2[begin+1:end]...)
    return einsum(EinCode((ec1, ec2), ec3), (AbstractArray{T1}(t1), AbstractArray{T2}(t2)))
end

scontract(t1::AbstractArray{T1,1}, t2::AbstractArray{T2,1}) where {T1,T2} =
    dot(AbstractArray{T1}(t1), AbstractArray{T2}(t2))

for TT1 ∈ (Vec, SecondOrderTensor), TT2 ∈ (Vec, SecondOrderTensor)
    @eval scontract(S1::$TT1, S2::$TT2) = dot(S1, S2)
end


function LinearAlgebra.dot(
    t1::Tensnd{order1,dim},
    t2::Tensnd{order2,dim},
) where {order1,order2,dim}
    nt1, nt2 = same_basis(t1, t2)
    var = (invvar(nt1.var[end]), nt2.var[begin+1:end]...)
    nt2 = Tensnd(components(nt2, var, nt2.basis), var, nt2.basis)
    data = scontract(nt1.data, nt2.data)
    var = (nt1.var[begin:end-1]..., nt2.var[begin+1:end]...)
    return Tensnd(data, var, nt1.basis)
end

function LinearAlgebra.dot(t1::Tensnd{1,dim}, t2::Tensnd{1,dim}) where {dim}
    nt1, nt2 = same_basis(t1, t2)
    var = (invvar(nt1.var[end]), nt2.var[begin+1:end]...)
    nt2 = Tensnd(components(nt2, var, nt2.basis), var, nt2.basis)
    return scontract(nt1.data, nt2.data)
end

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

function Tensors.dcontract(
    t1::Tensnd{order1,dim},
    t2::Tensnd{order2,dim},
) where {order1,order2,dim}
    nt1, nt2 = same_basis(t1, t2)
    var = (invvar(nt1.var[end-1]), invvar(nt1.var[end]), nt2.var[begin+2:end]...)
    nt2 = Tensnd(components(nt2, var, nt2.basis), var, nt2.basis)
    data = Tensors.dcontract(nt1.data, nt2.data)
    var = (nt1.var[begin:end-2]..., nt2.var[begin+2:end]...)
    return Tensnd(data, var, nt1.basis)
end

function Tensors.dcontract(t1::Tensnd{2,dim}, t2::Tensnd{2,dim}) where {dim}
    nt1, nt2 = same_basis(t1, t2)
    var = (invvar(nt1.var[end-1]), invvar(nt1.var[end]), nt2.var[begin+2:end]...)
    nt2 = Tensnd(components(nt2, var, nt2.basis), var, nt2.basis)
    return Tensors.dcontract(nt1.data, nt2.data)
end

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

function Tensors.dotdot(v1::Tensnd{order1,dim}, S::Tensnd{orderS,dim}, v2::Tensnd{order2,dim}) where {order1,orderS,order2,dim}
    nS, nv1 = same_basis(S, v1)
    nS, nv2 = same_basis(S, v2)
    var = (invvar(nS.var[begin]),)
    nv1 = Tensnd(components(nv1, var, nv1.basis), var, nv1.basis)
    var = (invvar(nS.var[end]),)
    nv2 = Tensnd(components(nv2, var, nv2.basis), var, nv2.basis)
    data = dotdot(nv1.data, nS.data, nv2.data)
    var = (nS.var[begin+1], nS.var[end-1])
    return Tensnd(data, var, nS.basis)
end
