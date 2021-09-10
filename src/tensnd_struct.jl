abstract type AbstractTensnd{order,dim,T<:Number,A<:AbstractArray} <: AbstractArray{T,order} end
abstract type TensndOrthonormal{order,dim,T<:Number,A<:AbstractArray} <:
              AbstractTensnd{order,dim,T,A} end

Base.size(t::AbstractTensnd) = size(getdata(t))

Base.getindex(t::AbstractTensnd, ind...) = getindex(getdata(t), ind...)

@pure Base.eltype(::AbstractTensnd{order,dim,T,A}) where {order,dim,T,A} = T
@pure getdim(::AbstractTensnd{order,dim,T,A}) where {order,dim,T,A} = dim
@pure getorder(::AbstractTensnd{order,dim,T,A}) where {order,dim,T,A} = order
@pure getdatatype(::AbstractTensnd{order,dim,T,A}) where {order,dim,T,A} = A

"""
    Tensnd{order,dim,T,A<:AbstractArray,B<:AbstractBasis}

Tensor type of any order defined by
- a multiarray of components (of any type heriting from `AbstractArray`, e.g. `Tensor` or `SymmetricTensor`)
- a basis of `AbstractBasis` type
- a tuple of variances (covariant `:cov` or contravariant `:cont`) of length equal to the `order` of the tensor

# Examples
```julia
julia> ℬ = Basis(Sym[1 0 0; 0 1 0; 0 1 1]) ;

julia> T = Tensnd(metric(ℬ,:cov),ℬ,(:cov,:cov))
Tensnd{2, 3, Sym, SymmetricTensor{2, 3, Sym, 6}}
# data: 3×3 SymmetricTensor{2, 3, Sym, 6}:
 1  0  0
 0  2  1
 0  1  1
# basis: 3×3 Tensor{2, 3, Sym, 9}:
 1  0  0
 0  1  0
 0  1  1
# var: (:cov, :cov)

julia> components(T,(:cont,:cov),b)
3×3 Matrix{Sym}:
 1  0  0
 0  1  0
 0  0  1
```
"""
struct Tensnd{order,dim,T,A} <: AbstractTensnd{order,dim,T,A}
    data::A
    basis::Basis
    var::NTuple{order,Symbol}
    function Tensnd(
        data::AbstractArray{T,order},
        ℬ::Basis{dim},
        var::NTuple{order,Symbol} = ntuple(_ -> :cont, order),
    ) where {order,dim,T}
        newdata = tensor_or_array(data)
        new{order,dim,T,typeof(newdata)}(newdata, ℬ, var)
    end
    Tensnd(data::AbstractArray, ℬ::RotatedBasis, args...) = TensndRotated(data, ℬ)
    Tensnd(data::AbstractArray, ::CanonicalBasis, args...) = TensndCanonical(data)
    Tensnd(data::AbstractArray, args...) = TensndCanonical(data)
    Tensnd(data::AbstractArray, var::NTuple, ℬ::AbstractBasis) = Tensnd(data, ℬ, var)
end

struct TensndRotated{order,dim,T,A} <: TensndOrthonormal{order,dim,T,A}
    data::A
    basis::RotatedBasis
    function TensndRotated(
        data::AbstractArray{T,order},
        ℬ::RotatedBasis{dim},
    ) where {order,dim,T}
        newdata = tensor_or_array(data)
        new{order,dim,T,typeof(newdata)}(newdata, ℬ)
    end
end

struct TensndCanonical{order,dim,T,A} <: TensndOrthonormal{order,dim,T,A}
    data::A
    function TensndCanonical(data::AbstractArray{T,order}) where {order,T}
        newdata = tensor_or_array(data)
        new{order,size(newdata)[1],T,typeof(newdata)}(newdata)
    end
end

# This function aims at storing the table of components in the `Tensor` type whenever possible
tensor_or_array(tab::AbstractArray{T,1}) where {T} = Vec{size(tab)[1]}(tab)
for order ∈ (2, 4)
    @eval function tensor_or_array(tab::AbstractArray{T,$order}) where {T}
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
getbasis(t::AbstractTensnd) = t.basis

getvar(t::Tensnd) = t.var
getvar(t::Tensnd, i::Int) = getvar(t)[i]

getvar(::TensndOrthonormal{order}) where {order} = ntuple(_ -> :cont, order)
getvar(::TensndOrthonormal, i::Int) = :cont

getbasis(::TensndCanonical{order,dim,T}) where {order,dim,T} = CanonicalBasis{dim,T}()


#####################
# Display Functions #
#####################
for OP in (:show, :print, :display)
    @eval begin
        Base.$OP(U::FourthOrderTensor) = $OP(tomandel(U))

        function Base.$OP(t::AbstractTensnd)
            $OP(typeof(t))
            print("# data: ")
            $OP(getdata(t))
            print("# basis: ")
            $OP(vecbasis(getbasis(t)))
            print("# var: ")
            $OP(getvar(t))
        end
    end
end

# Base.display(U::FourthOrderTensor) = display(tomandel(U))


########################
# Component extraction #
########################

"""
    components(t::AbstractTensnd{order,dim,T},ℬ::AbstractBasis{dim},var::NTuple{order,Symbol})
    components(t::AbstractTensnd{order,dim,T},ℬ::AbstractBasis{dim})
    components(t::AbstractTensnd{order,dim,T},var::NTuple{order,Symbol})

Extracts the components of a tensor for new variances and/or in a new basis

# Examples
```julia
julia> ℬ = Basis(Sym[0 1 1; 1 0 1; 1 1 0]) ;

julia> TV = Tensnd(Tensor{1,3}(i->symbols("v\$i",real=true)))
TensND.TensndCanonical{1, 3, Sym, Vec{3, Sym}}
# data: 3-element Vec{3, Sym}:
 v₁
 v₂
 v₃
# basis: 3×3 TensND.LazyIdentity{3, Sym}:
 1  0  0
 0  1  0
 0  0  1
# var: (:cont,)

julia> factor.(components(TV, ℬ, (:cont,)))
3-element Vector{Sym}:
 -(v1 - v2 - v3)/2
  (v1 - v2 + v3)/2
  (v1 + v2 - v3)/2

julia> components(TV, ℬ, (:cov,))
3-element Vector{Sym}:
 v₂ + v₃
 v₁ + v₃
 v₁ + v₂

julia> simplify.(components(TV, normalize(ℬ), (:cov,)))
3-element Vector{Sym}:
 sqrt(2)*(v2 + v3)/2
 sqrt(2)*(v1 + v3)/2
 sqrt(2)*(v1 + v2)/2

julia> TT = Tensnd(Tensor{2,3}((i,j)->symbols("t\$i\$j",real=true)))
TensND.TensndCanonical{2, 3, Sym, Tensor{2, 3, Sym, 9}}
# data: 3×3 Tensor{2, 3, Sym, 9}:
 t₁₁  t₁₂  t₁₃
 t₂₁  t₂₂  t₂₃
 t₃₁  t₃₂  t₃₃
# basis: 3×3 TensND.LazyIdentity{3, Sym}:
 1  0  0
 0  1  0
 0  0  1
# var: (:cont, :cont)

julia> components(TT, ℬ, (:cov,:cov))
3×3 Matrix{Sym}:
 t₂₂ + t₂₃ + t₃₂ + t₃₃  t₂₁ + t₂₃ + t₃₁ + t₃₃  t₂₁ + t₂₂ + t₃₁ + t₃₂
 t₁₂ + t₁₃ + t₃₂ + t₃₃  t₁₁ + t₁₃ + t₃₁ + t₃₃  t₁₁ + t₁₂ + t₃₁ + t₃₂
 t₁₂ + t₁₃ + t₂₂ + t₂₃  t₁₁ + t₁₃ + t₂₁ + t₂₃  t₁₁ + t₁₂ + t₂₁ + t₂₂

julia> factor.(components(TT, ℬ, (:cont,:cov)))
3×3 Matrix{Sym}:
 -(t12 + t13 - t22 - t23 - t32 - t33)/2  …  -(t11 + t12 - t21 - t22 - t31 - t32)/2
  (t12 + t13 - t22 - t23 + t32 + t33)/2      (t11 + t12 - t21 - t22 + t31 + t32)/2
  (t12 + t13 + t22 + t23 - t32 - t33)/2      (t11 + t12 + t21 + t22 - t31 - t32)/2
```
"""
components(t::AbstractTensnd) = getdata(t)

components(t::TensndOrthonormal, ::NTuple) = getdata(t)

function components(
    t::Tensnd{order,dim,T},
    var::NTuple{order,Symbol},
) where {order,dim,T<:Number}
    if var == getvar(t)
        return getdata(t)
    else
        m = Array(getdata(t))
        ec1 = ntuple(i -> i, order)
        newcp = order + 1
        for i ∈ 1:order
            if getvar(t, i) ≠ var[i]
                g_or_G = metric(getbasis(t), var[i])
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
    t::AbstractTensnd{order,dim,T},
    ℬ::AbstractBasis{dim},
    var::NTuple{order,Symbol},
) where {order,dim,T}
    if ℬ == getbasis(t)
        return components(t, var)
    else
        bb = Dict()
        for v1 ∈ (:cov, :cont), v2 ∈ (:cov, :cont)
            if v1 ∈ getvar(t) && v2 ∈ var
                bb[v1, v2] =
                    Tensor{2,3}(vecbasis(getbasis(t), invvar(v1)))' ⋅
                    Tensor{2,3}(vecbasis(ℬ, v2))
            end
        end
        m = Array(getdata(t))
        ec1 = ntuple(i -> i, order)
        newcp = order + 1
        for i ∈ 1:order
            c = bb[getvar(t, i), var[i]]
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

components(t::AbstractTensnd{order,dim,T}, ℬ::AbstractBasis{dim}) where {order,dim,T} = components(t, ℬ, getvar(t))

components(t::AbstractTensnd{order,dim,T}, ℬ::OrthonormalBasis{dim}) where {order,dim,T} = components(t, ℬ, ntuple(_ -> :cont, order))

function components(
    t::TensndOrthonormal{order,dim,T},
    ℬ::OrthonormalBasis{dim},
) where {order,dim,T}
    if ℬ == getbasis(t)
        return getdata(t)
    else
        bb = Tensor{2,3}(vecbasis(getbasis(t)))' ⋅ Tensor{2,3}(vecbasis(ℬ))
        m = Array(getdata(t))
        ec1 = ntuple(i -> i, order)
        newcp = order + 1
        for i ∈ 1:order
            if bb ≠ I
                ec2 = (i, newcp)
                ec3 = ntuple(j -> j ≠ i ? j : newcp, order)
                m = einsum(EinCode((ec1, ec2), ec3), (m, bb))
                if T == Sym
                    m = sympy.trigsimp.(m)
                end
            end
        end
        return m
    end
end

components(t::TensndOrthonormal{order,dim,T}, basis::OrthonormalBasis{dim}, ::NTuple{order,Symbol}) where {order,dim,T} = components(t, basis)

"""
    components_canon(t::AbstractTensnd)

Extracts the components of a tensor in the canonical basis
"""
components_canon(t::AbstractTensnd) =
    components(t, CanonicalBasis{getdim(t),eltype(t)}(), getvar(t))

components_canon(t::TensndOrthonormal) =
    components(t, CanonicalBasis{getdim(t),eltype(t)}())

"""
    change_tens(t::AbstractTensnd{order,dim,T},ℬ::AbstractBasis{dim},var::NTuple{order,Symbol})
    change_tens(t::AbstractTensnd{order,dim,T},ℬ::AbstractBasis{dim})
    change_tens(t::AbstractTensnd{order,dim,T},var::NTuple{order,Symbol})

Rewrites the same tensor with components corresponding to new variances and/or to a new basis

```julia
julia> ℬ = Basis(Sym[0 1 1; 1 0 1; 1 1 0]) ;

julia> TV = Tensnd(Tensor{1,3}(i->symbols("v\$i",real=true)))
TensND.TensndCanonical{1, 3, Sym, Vec{3, Sym}}
# data: 3-element Vec{3, Sym}:
 v₁
 v₂
 v₃
# basis: 3×3 TensND.LazyIdentity{3, Sym}:
 1  0  0
 0  1  0
 0  0  1
# var: (:cont,)

julia> factor.(components(TV, ℬ, (:cont,)))
3-element Vector{Sym}:
 -(v1 - v2 - v3)/2
  (v1 - v2 + v3)/2
  (v1 + v2 - v3)/2

julia> ℬ₀ = Basis(Sym[0 1 1; 1 0 1; 1 1 1]) ;

julia> TV0 = change_tens(TV, ℬ₀)
Tensnd{1, 3, Sym, Vec{3, Sym}}
# data: 3-element Vec{3, Sym}:
     -v₁ + v₃
     -v₂ + v₃
 v₁ + v₂ - v₃
# basis: 3×3 Tensor{2, 3, Sym, 9}:
 0  1  1
 1  0  1
 1  1  1
# var: (:cont,)
```
"""
function change_tens(t::AbstractTensnd, ℬ::AbstractBasis, newvar::NTuple)
    if ℬ == getbasis(t) && newvar == getvar(t)
        return t
    else
        return Tensnd(components(t, ℬ, newvar), ℬ, newvar)
    end
end

function change_tens(t::AbstractTensnd, newbasis::AbstractBasis)
    if newbasis == getbasis(t)
        return t
    else
        return Tensnd(components(t, newbasis, getvar(t)), newbasis, getvar(t))
    end
end

function change_tens(t::AbstractTensnd, newvar::NTuple)
    if newvar == getvar(t)
        return t
    else
        return Tensnd(components(t, getbasis(t), newvar), getbasis(t), newvar)
    end
end

"""
    change_tens_canon(t::AbstractTensnd{order,dim,T},var::NTuple{order,Symbol})

Rewrites the same tensor with components corresponding to the canonical basis

```julia
julia> ℬ = Basis(Sym[0 1 1; 1 0 1; 1 1 0]) ;

julia> TV = Tensnd(Tensor{1,3}(i->symbols("v\$i",real=true)), ℬ)
Tensnd{1, 3, Sym, Vec{3, Sym}}
# data: 3-element Vec{3, Sym}:
 v₁
 v₂
 v₃
# basis: 3×3 Tensor{2, 3, Sym, 9}:
 0  1  1
 1  0  1
 1  1  1
# var: (:cont,)

julia> TV0 = change_tens_canon(TV)
TensND.TensndCanonical{1, 3, Sym, Vec{3, Sym}}
# data: 3-element Vec{3, Sym}:
      v₂ + v₃
      v₁ + v₃
 v₁ + v₂ + v₃
# basis: 3×3 TensND.LazyIdentity{3, Sym}:
 1  0  0
 0  1  0
 0  0  1
# var: (:cont,)
```
"""
change_tens_canon(t::AbstractTensnd) = change_tens(t, CanonicalBasis{getdim(t),eltype(t)}())


##############
# Operations #
##############


same_basis(
    t1::AbstractTensnd{order1,dim},
    t2::AbstractTensnd{order2,dim},
) where {order1,order2,dim} = t1, change_tens(t2, getbasis(t1))

same_basis_same_var(
    t1::AbstractTensnd{order1,dim},
    t2::AbstractTensnd{order2,dim},
) where {order1,order2,dim} = t1, change_tens(t2, getbasis(t1), getvar(t1))


for OP in (:(==), :(!=))
    @eval function Base.$OP(
        t1::AbstractTensnd{order,dim},
        t2::AbstractTensnd{order,dim},
    ) where {order,dim}
        nt1, nt2 = same_basis_same_var(t1, t2)
        return $OP(getdata(nt1), getdata(nt2))
    end
end

for OP in (:+, :-)
    @eval function Base.$OP(
        t1::AbstractTensnd{order,dim},
        t2::AbstractTensnd{order,dim},
    ) where {order,dim}
        nt1, nt2 = same_basis_same_var(t1, t2)
        return Tensnd($OP(getdata(nt1), getdata(nt2)), getvar(nt1), getbasis(nt1))
    end
end

Base.:*(α::Number, t::AbstractTensnd) = Tensnd(α * getdata(t), getbasis(t), getvar(t))
Base.:*(t::AbstractTensnd, α::Number) = Tensnd(α * getdata(t), getbasis(t), getvar(t))
Base.:/(t::AbstractTensnd, α::Number) = Tensnd(getdata(t) / α, getbasis(t), getvar(t))

Base.inv(t::AbstractTensnd{2}) =
    Tensnd(inv(getdata(t)), getbasis(t), (invvar(getvar(t, 2)), invvar(getvar(t, 1))))
Base.inv(t::AbstractTensnd{4}) = Tensnd(
    inv(getdata(t)),
    getbasis(t),
    (
        invvar(getvar(t, 3)),
        invvar(getvar(t, 4)),
        invvar(getvar(t, 1)),
        invvar(getvar(t, 2)),
    ),
)

"""
    KM(t::AbstractTensnd{order,dim}; kwargs...)
    KM(t::AbstractTensnd{order,dim}, var::NTuple{order,Symbol}, b::AbstractBasis{dim}; kwargs...)

Writes the components of a second or fourth order tensor in Kelvin-Mandel notation

# Examples
```julia
julia> σ = Tensnd(SymmetricTensor{2,3}((i, j) -> symbols("σ\$i\$j", real = true))) ;

julia> KM(σ)
6-element Vector{Sym}:
         σ11
         σ22
         σ33
      √2⋅σ32
      √2⋅σ31
      √2⋅σ21

julia> C = Tensnd(SymmetricTensor{4,3}((i, j, k, l) -> symbols("C\$i\$j\$k\$l", real = true))) ;

julia> KM(C)
6×6 Matrix{Sym}:
         C₁₁₁₁     C₁₁₂₂     C₁₁₃₃  √2⋅C₁₁₃₂  √2⋅C₁₁₃₁  √2⋅C₁₁₂₁
         C₂₂₁₁     C₂₂₂₂     C₂₂₃₃  √2⋅C₂₂₃₂  √2⋅C₂₂₃₁  √2⋅C₂₂₂₁
         C₃₃₁₁     C₃₃₂₂     C₃₃₃₃  √2⋅C₃₃₃₂  √2⋅C₃₃₃₁  √2⋅C₃₃₂₁
      √2⋅C₃₂₁₁  √2⋅C₃₂₂₂  √2⋅C₃₂₃₃   2⋅C₃₂₃₂   2⋅C₃₂₃₁   2⋅C₃₂₂₁
      √2⋅C₃₁₁₁  √2⋅C₃₁₂₂  √2⋅C₃₁₃₃   2⋅C₃₁₃₂   2⋅C₃₁₃₁   2⋅C₃₁₂₁
      √2⋅C₂₁₁₁  √2⋅C₂₁₂₂  √2⋅C₂₁₃₃   2⋅C₂₁₃₂   2⋅C₂₁₃₁   2⋅C₂₁₂₁
```
"""
KM(t::Tensors.AllTensors; kwargs...) = tomandel(t; kwargs...)
KM(t::AbstractTensnd; kwargs...) = tomandel(getdata(t); kwargs...)

KM(
    t::AbstractTensnd{order,dim},
    b::AbstractBasis{dim},
    var::NTuple{order,Symbol},
    kwargs...,
) where {order,dim} = tomandel(tensor_or_array(components(t, b, var)); kwargs...)

KM(t::AbstractTensnd{order,dim}, b::AbstractBasis{dim}; kwargs...) where {order,dim} =
    tomandel(tensor_or_array(components(t, b)); kwargs...)


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


"""
    invKM(v::AbstractVecOrMat; kwargs...)

Defines a tensor from a Kelvin-Mandel vector or matrix representation
"""
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

"""
    otimes(t1::AbstractTensnd{order1,dim}, t2::AbstractTensnd{order2,dim})

Defines a tensor product between two tensors

`(aⁱeᵢ) ⊗ (bʲeⱼ) = aⁱbʲ eᵢ⊗eⱼ`
"""
function Tensors.otimes(
    t1::AbstractTensnd{order1,dim},
    t2::AbstractTensnd{order2,dim},
) where {order1,order2,dim}
    nt1, nt2 = same_basis(t1, t2)
    data = otimes(getdata(nt1), getdata(nt2))
    var = (getvar(nt1)..., getvar(nt2)...)
    return Tensnd(data, getbasis(nt1), var)
end

function Tensors.otimes(
    t1::TensndOrthonormal{order1,dim},
    t2::TensndOrthonormal{order2,dim},
) where {order1,order2,dim}
    nt1, nt2 = same_basis(t1, t2)
    data = otimes(getdata(nt1), getdata(nt2))
    return Tensnd(data, getbasis(nt1))
end

Tensors.otimes(α::Number, t::AbstractTensnd) = α * t
Tensors.otimes(t::AbstractTensnd, α::Number) = α * t

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

"""
    dot(t1::AbstractTensnd{order1,dim}, t2::AbstractTensnd{order2,dim})

Defines a contracted product between two tensors

`a ⋅ b = aⁱbⱼ`
"""
function LinearAlgebra.dot(
    t1::AbstractTensnd{order1,dim},
    t2::AbstractTensnd{order2,dim},
) where {order1,order2,dim}
    nt1, nt2 = same_basis(t1, t2)
    var = (invvar(getvar(nt1)[end]), getvar(nt2)[begin+1:end]...)
    nt2 = change_tens(nt2, getbasis(nt2), var)
    data = scontract(getdata(nt1), getdata(nt2))
    var = (getvar(nt1)[begin:end-1]..., getvar(nt2)[begin+1:end]...)
    return Tensnd(data, getbasis(nt1), var)
end

function LinearAlgebra.dot(
    t1::TensndOrthonormal{order1,dim},
    t2::TensndOrthonormal{order2,dim},
) where {order1,order2,dim}
    nt1, nt2 = same_basis(t1, t2)
    nt2 = change_tens(nt2, getbasis(nt2))
    data = scontract(getdata(nt1), getdata(nt2))
    return Tensnd(data, getbasis(nt1))
end

function LinearAlgebra.dot(t1::AbstractTensnd{1,dim}, t2::AbstractTensnd{1,dim}) where {dim}
    nt1, nt2 = same_basis(t1, t2)
    var = (invvar(getvar(nt1)[end]), getvar(nt2)[begin+1:end]...)
    nt2 = change_tens(nt2, getbasis(nt2), var)
    return scontract(getdata(nt1), getdata(nt2))
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

"""
    dcontract(t1::AbstractTensnd{order1,dim}, t2::AbstractTensnd{order2,dim})

Defines a double contracted product between two tensors

`𝛔 ⊡ 𝛆 = σⁱʲεᵢⱼ`
`𝛔 = ℂ ⊡ 𝛆`

# Examples
```julia
julia> 𝛆 = Tensnd(SymmetricTensor{2,3}((i, j) -> symbols("ε\$i\$j", real = true))) ;

julia> k, μ = symbols("k μ", real =true) ;

julia> ℂ = 3k * t𝕁() + 2μ * t𝕂() ;

julia> 𝛔 = ℂ ⊡ 𝛆
Tensnd{2, 3, Sym, Sym, SymmetricTensor{2, 3, Sym, 6}, CanonicalBasis{3, Sym}}
# data: 3×3 SymmetricTensor{2, 3, Sym, 6}:
 ε11*(k + 4*μ/3) + ε22*(k - 2*μ/3) + ε33*(k - 2*μ/3)                                              2⋅ε21⋅μ                                              2⋅ε31⋅μ
                                             2⋅ε21⋅μ  ε11*(k - 2*μ/3) + ε22*(k + 4*μ/3) + ε33*(k - 2*μ/3)                                              2⋅ε32⋅μ
                                             2⋅ε31⋅μ                                              2⋅ε32⋅μ  ε11*(k - 2*μ/3) + ε22*(k - 2*μ/3) + ε33*(k + 4*μ/3)
# var: (:cont, :cont)
# basis: 3×3 Tensor{2, 3, Sym, 9}:
 1  0  0
 0  1  0
 0  0  1
```
"""
function Tensors.dcontract(
    t1::AbstractTensnd{order1,dim},
    t2::AbstractTensnd{order2,dim},
) where {order1,order2,dim}
    nt1, nt2 = same_basis(t1, t2)
    var =
        (invvar(getvar(nt1)[end-1]), invvar(getvar(nt1)[end]), getvar(nt2)[begin+2:end]...)
    nt2 = change_tens(nt2, getbasis(nt2), var)
    data = Tensors.dcontract(getdata(nt1), getdata(nt2))
    var = (getvar(nt1)[begin:end-2]..., getvar(nt2)[begin+2:end]...)
    return Tensnd(data, getbasis(nt1), var)
end

function Tensors.dcontract(
    t1::TensndOrthonormal{order1,dim},
    t2::TensndOrthonormal{order2,dim},
) where {order1,order2,dim}
    nt1, nt2 = same_basis(t1, t2)
    nt2 = change_tens(nt2, getbasis(nt2))
    data = Tensors.dcontract(getdata(nt1), getdata(nt2))
    return Tensnd(data, getbasis(nt1))
end

function Tensors.dcontract(t1::AbstractTensnd{2,dim}, t2::AbstractTensnd{2,dim}) where {dim}
    nt1, nt2 = same_basis(t1, t2)
    var =
        (invvar(getvar(nt1)[end-1]), invvar(getvar(nt1)[end]), getvar(nt2)[begin+2:end]...)
    nt2 = change_tens(nt2, getbasis(nt2), var)
    return Tensors.dcontract(getdata(nt1), getdata(nt2))
end

function Tensors.dcontract(t1::TensndOrthonormal{2,dim}, t2::TensndOrthonormal{2,dim}) where {dim}
    nt1, nt2 = same_basis(t1, t2)
    return Tensors.dcontract(getdata(nt1), getdata(nt2))
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

"""
    dotdot(v1::AbstractTensnd{order1,dim}, S::AbstractTensnd{orderS,dim}, v2::AbstractTensnd{order2,dim})

Defines a bilinear operator `𝐯₁⋅𝕊⋅𝐯₂`

# Examples
```julia
julia> n = Tensnd(Sym[0, 0, 1]) ;

julia> k, μ = symbols("k μ", real =true) ;

julia> ℂ = 3k * t𝕁() + 2μ * t𝕂() ;

julia> dotdot(n,ℂ,n) # Acoustic tensor
3×3 Tensnd{2, 3, Sym, Sym, Tensor{2, 3, Sym, 9}, CanonicalBasis{3, Sym}}:
 μ  0          0
 0  μ          0
 0  0  k + 4*μ/3
```
"""
function Tensors.dotdot(
    v1::AbstractTensnd{order1,dim},
    S::AbstractTensnd{orderS,dim},
    v2::AbstractTensnd{order2,dim},
) where {order1,orderS,order2,dim}
    nS, nv1 = same_basis(S, v1)
    nS, nv2 = same_basis(S, v2)
    var = (invvar(getvar(nS)[begin]),)
    nv1 = change_tens(nv1, getbasis(nv1), var)
    var = (invvar(getvar(nS)[end]),)
    nv2 = change_tens(nv2, getbasis(nv2), var)
    data = dotdot(nv1.data, nS.data, nv2.data)
    var = (getvar(nS)[begin+1], getvar(nS)[end-1])
    return Tensnd(data, getbasis(nS), var)
end

function Tensors.dotdot(
    v1::TensndOrthonormal{order1,dim},
    S::TensndOrthonormal{orderS,dim},
    v2::TensndOrthonormal{order2,dim},
) where {order1,orderS,order2,dim}
    nS, nv1 = same_basis(S, v1)
    nS, nv2 = same_basis(S, v2)
    data = dotdot(nv1.data, nS.data, nv2.data)
    return Tensnd(data, getbasis(nS))
end

"""
    dcontract(t1::AbstractTensnd{order1,dim}, t2::AbstractTensnd{order2,dim})

Defines a quadruple contracted product between two tensors

`𝔸 ⊙ 𝔹 = AᵢⱼₖₗBⁱʲᵏˡ`

# Examples
```julia
julia> 𝕀 = t𝕀(Sym) ; 𝕁 = t𝕁(Sym) ; 𝕂 = t𝕂(Sym) ;

julia> 𝕀 ⊙ 𝕀
6

julia> 𝕁 ⊙ 𝕀
1

julia> 𝕂 ⊙ 𝕀
5

julia> 𝕂 ⊙ 𝕁
0
```
"""
function qcontract(
    t1::AbstractArray{T1,order1},
    t2::AbstractArray{T2,order2},
) where {T1,T2,order1,order2}
    newc = order1 + order2
    ec1 = (ntuple(i -> i, order1 - 4)..., newc, newc + 1, newc + 2, newc + 3)
    ec2 = (newc, newc + 1, newc + 2, newc + 3, ntuple(i -> order1 - 4 + i, order2 - 4)...)
    ec3 = ntuple(i -> i, order1 + order2 - 8)
    return einsum(EinCode((ec1, ec2), ec3), (AbstractArray{T1}(t1), AbstractArray{T2}(t2)))
end

qcontract(t1::AbstractArray{T1,4}, t2::AbstractArray{T2,4}) where {T1,T2} =
    dot(AbstractArray{T1}(t1), AbstractArray{T2}(t2))

function qcontract(
    t1::AbstractTensnd{order1,dim},
    t2::AbstractTensnd{order2,dim},
) where {order1,order2,dim}
    nt1, nt2 = same_basis(t1, t2)
    var = (
        invvar(getvar(nt1)[end-3]),
        invvar(getvar(nt1)[end-2]),
        invvar(getvar(nt1)[end-1]),
        invvar(getvar(nt1)[end]),
        getvar(nt2)[begin+4:end]...,
    )
    nt2 = change_tens(nt2, getbasis(nt2), var)
    data = qcontract(getdata(nt1), getdata(nt2))
    var = (getvar(nt1)[begin:end-4]..., getvar(nt2)[begin+4:end]...)
    return Tensnd(data, getbasis(nt1), var)
end

function qcontract(
    t1::TensndOrthonormal{order1,dim},
    t2::TensndOrthonormal{order2,dim},
) where {order1,order2,dim}
    nt1, nt2 = same_basis(t1, t2)
    nt2 = change_tens(nt2, getbasis(nt2))
    data = qcontract(getdata(nt1), getdata(nt2))
    return Tensnd(data, getbasis(nt1))
end
    
function qcontract(t1::AbstractTensnd{4,dim}, t2::AbstractTensnd{4,dim}) where {dim}
    nt1, nt2 = same_basis(t1, t2)
    var = (
        invvar(getvar(nt1)[end-3]),
        invvar(getvar(nt1)[end-2]),
        invvar(getvar(nt1)[end-1]),
        invvar(getvar(nt1)[end]),
        getvar(nt2)[begin+4:end]...,
    )
    nt2 = change_tens(nt2, getbasis(nt2), var)
    return qcontract(getdata(nt1), getdata(nt2))
end

function qcontract(t1::TensndOrthonormal{4,dim}, t2::TensndOrthonormal{4,dim}) where {dim}
    nt1, nt2 = same_basis(t1, t2)
    return qcontract(getdata(nt1), getdata(nt2))
end

function Tensors.otimesu(
    t1::AbstractArray{T1,order1},
    t2::AbstractArray{T2,order2},
) where {T1,T2,order1,order2}
    ec1 = (ntuple(i -> i, order1 - 1)..., order1 + 1)
    ec2 = (order1, ntuple(i -> order1 + 1 + i, order2 - 1)...)
    ec3 = ntuple(i -> i, order1 + order2)
    return einsum(EinCode((ec1, ec2), ec3), (AbstractArray{T1}(t1), AbstractArray{T2}(t2)))
end

"""
    otimesu(t1::AbstractTensnd{order1,dim}, t2::AbstractTensnd{order2,dim})

Defines a special tensor product between two tensors of at least second order

`(𝐚 ⊠ 𝐛) ⊡ 𝐩 = 𝐚⋅𝐩⋅𝐛 = aⁱᵏbʲˡpₖₗ eᵢ⊗eⱼ`
"""
function Tensors.otimesu(
    t1::AbstractTensnd{order1,dim},
    t2::AbstractTensnd{order2,dim},
) where {order1,order2,dim}
    nt1, nt2 = same_basis(t1, t2)
    data = otimesu(getdata(nt1), getdata(nt2))
    var = (
        getvar(nt1)[begin:end-1]...,
        getvar(nt2)[begin],
        getvar(nt1)[end],
        getvar(nt2)[begin+1:end]...,
    )
    return Tensnd(data, getbasis(nt1), var)
end

function Tensors.otimesu(
    t1::TensndOrthonormal{order1,dim},
    t2::TensndOrthonormal{order2,dim},
) where {order1,order2,dim}
    nt1, nt2 = same_basis(t1, t2)
    data = otimesu(getdata(nt1), getdata(nt2))
    return Tensnd(data, getbasis(nt1))
end

function Tensors.otimesl(
    t1::AbstractArray{T1,order1},
    t2::AbstractArray{T2,order2},
) where {T1,T2,order1,order2}
    ec1 = (ntuple(i -> i, order1 - 1)..., order1 + 2)
    ec2 = (order1, order1 + 1, ntuple(i -> order1 + 2 + i, order2 - 2)...)
    ec3 = ntuple(i -> i, order1 + order2)
    return einsum(EinCode((ec1, ec2), ec3), (AbstractArray{T1}(t1), AbstractArray{T2}(t2)))
end

function Tensors.otimesl(
    t1::AbstractTensnd{order1,dim},
    t2::AbstractTensnd{order2,dim},
) where {order1,order2,dim}
    nt1, nt2 = same_basis(t1, t2)
    data = otimesl(getdata(nt1), getdata(nt2))
    var = (
        getvar(nt1)[begin:end-1]...,
        getvar(nt2)[begin+1],
        getvar(nt1)[end],
        getvar(nt2)[begin],
        getvar(nt2)[begin+2:end]...,
    )
    return Tensnd(data, getbasis(nt1), var)
end

function Tensors.otimesl(
    t1::TensndOrthonormal{order1,dim},
    t2::TensndOrthonormal{order2,dim},
) where {order1,order2,dim}
    nt1, nt2 = same_basis(t1, t2)
    data = otimesl(getdata(nt1), getdata(nt2))
    return Tensnd(data, getbasis(nt1))
end

otimesul(t1::AbstractArray{T1}, t2::AbstractArray{T2}) where {T1,T2} =
    (otimesu(t1, t2) + otimesl(t1, t2)) / promote_type(T1, T2)(2)

otimesul(S1::SecondOrderTensor{dim}, S2::SecondOrderTensor{dim}) where {dim} =
    symmetric(otimesu(S1, S2))

"""
    otimesul(t1::AbstractTensnd{order1,dim}, t2::AbstractTensnd{order2,dim})

Defines a special tensor product between two tensors of at least second order

`(𝐚 ⊠ˢ 𝐛) ⊡ 𝐩 = (𝐚 ⊠ 𝐛) ⊡ (𝐩 + ᵗ𝐩)/2  = 1/2(aⁱᵏbʲˡ+aⁱˡbʲᵏ) pₖₗ eᵢ⊗eⱼ`
"""
function otimesul(
    t1::AbstractTensnd{order1,dim},
    t2::AbstractTensnd{order2,dim},
) where {order1,order2,dim}
    nt1, nt2 = same_basis(t1, t2)
    var = (getvar(nt1)[end-1], getvar(nt1)[end], getvar(nt2)[begin+2:end]...)
    nt2 = change_tens(nt2, getbasis(nt2), var)
    data = otimesul(getdata(nt1), getdata(nt2))
    var = (getvar(nt1)..., getvar(nt2)...)
    return Tensnd(data, getbasis(nt1), var)
end

function otimesul(
    t1::TensndOrthonormal{order1,dim},
    t2::TensndOrthonormal{order2,dim},
) where {order1,order2,dim}
    nt1, nt2 = same_basis(t1, t2)
    nt2 = change_tens(nt2, getbasis(nt2))
    data = otimesul(getdata(nt1), getdata(nt2))
    return Tensnd(data, getbasis(nt1))
end

@inline function sotimes(
    t1::AbstractArray{T1,order1},
    t2::AbstractArray{T2,order2},
) where {T1,T2,order1,order2}
    ec1 = ntuple(i -> i, order1)
    ec2 = ntuple(i -> order1 + i, order2)
    ec3 = ntuple(i -> i, order1 + order2)
    t3 = einsum(EinCode((ec1, ec2), ec3), (AbstractArray{T1}(t1), AbstractArray{T2}(t2)))
    ec1 = (ntuple(i -> i, order1 - 1)..., order1 + 1)
    ec2 = (order1, ntuple(i -> order1 + 1 + i, order2 - 1)...)
    ec3 = ntuple(i -> i, order1 + order2)
    t4 = einsum(EinCode((ec1, ec2), ec3), (AbstractArray{T1}(t1), AbstractArray{T2}(t2)))
    return (t3 + t4) / promote_type(T1, T2)(2)
end

@inline function sotimes(S1::Vec{dim}, S2::Vec{dim}) where {dim}
    return Tensor{2,dim}(@inline function (i, j)
        @inbounds (S1[i] * S2[j] + S1[j] * S2[i]) / 2
    end)
end

@inline function sotimes(S1::SecondOrderTensor{dim}, S2::SecondOrderTensor{dim}) where {dim}
    TensorType = Tensors.getreturntype(
        otimes,
        Tensors.get_base(typeof(S1)),
        Tensors.get_base(typeof(S2)),
    )
    TensorType(@inline function (i, j, k, l)
        @inbounds (S1[i, j] * S2[k, l] + S1[i, k] * S2[j, l]) / 2
    end)
end

sotimes(α::Number, t::AbstractTensnd) = α * t
sotimes(t::AbstractTensnd, α::Number) = α * t



"""
    sotimes(t1::AbstractTensnd{order1,dim}, t2::AbstractTensnd{order2,dim})

Defines a symmetric tensor product between two tensors

`(aⁱeᵢ) ⊗ˢ (bʲeⱼ) = 1/2(aⁱbʲ + aʲbⁱ) eᵢ⊗eⱼ`
"""
function sotimes(
    t1::AbstractTensnd{order1,dim},
    t2::AbstractTensnd{order2,dim},
) where {order1,order2,dim}
    nt1, nt2 = same_basis(t1, t2)
    var = (getvar(nt1)[end], getvar(nt2)[begin+1:end]...)
    nt2 = change_tens(nt2, getbasis(nt2), var)
    data = sotimes(getdata(nt1), getdata(nt2))
    var = (getvar(nt1)..., getvar(nt2)...)
    return Tensnd(data, getbasis(nt1), var)
end

function sotimes(
    t1::TensndOrthonormal{order1,dim},
    t2::TensndOrthonormal{order2,dim},
) where {order1,order2,dim}
    nt1, nt2 = same_basis(t1, t2)
    data = sotimes(getdata(nt1), getdata(nt2))
    return Tensnd(data, getbasis(nt1))
end

Base.transpose(t::AbstractTensnd{order,dim,T,<:SecondOrderTensor}) where {order,dim,T} =
    Tensnd(transpose(getdata(t)), getbasis(t), (getvar(t)[2], getvar(t)[1]))

Base.transpose(t::TensndOrthonormal{order,dim,T,<:SecondOrderTensor}) where {order,dim,T} =
    Tensnd(transpose(getdata(t)), getbasis(t))

Base.transpose(t::AbstractTensnd{order,dim,T,<:FourthOrderTensor}) where {order,dim,T} =
    Tensnd(
        Tensors.transpose(getdata(t)),
        getbasis(t),
        (getvar(t)[2], getvar(t)[1], getvar(t)[4], getvar(t)[3]),
    )

Base.transpose(t::TensndOrthonormal{order,dim,T,<:FourthOrderTensor}) where {order,dim,T} =
    Tensnd(
        Tensors.transpose(getdata(t)),
        getbasis(t),
    )

Tensors.majortranspose(
        t::AbstractTensnd{order,dim,T,<:FourthOrderTensor},
    ) where {order,dim,T} = Tensnd(
        majortranspose(getdata(t)),
        getbasis(t),
        (getvar(t)[3], getvar(t)[4], getvar(t)[1], getvar(t)[2]),
    )
    
Tensors.majortranspose(
    t::TensndOrthonormal{order,dim,T,<:FourthOrderTensor},
) where {order,dim,T} = Tensnd(
    majortranspose(getdata(t)),
    getbasis(t),
)

Tensors.minortranspose(
    t::AbstractTensnd{order,dim,T,<:FourthOrderTensor},
) where {order,dim,T} = Tensnd(
    minortranspose(getdata(t)),
    getbasis(t),
    (getvar(t)[2], getvar(t)[1], getvar(t)[4], getvar(t)[3]),
)

Tensors.minortranspose(
    t::TensndOrthonormal{order,dim,T,<:FourthOrderTensor},
) where {order,dim,T} = Tensnd(
    minortranspose(getdata(t)),
    getbasis(t),
)

const ⊙ = qcontract
const ⊠ = otimesu
const ⊠ˢ = otimesul
const ⊗ˢ = sotimes

const sboxtimes = otimesul

# const ⊗̅ = otimesu
# const ⊗̲ = otimesl
# const ⊗̲̅ = otimesul
