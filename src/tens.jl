abstract type AbstractTens{order,dim,T<:Number} <: AbstractArray{T,order} end

@pure getorder(::AbstractTens{order,dim,T}) where {order,dim,T} = order
@pure getdim(::AbstractTens{order,dim,T}) where {order,dim,T} = dim
@pure Base.eltype(::Type{AbstractTens{order,dim,T}}) where {order,dim,T} = T


"""
    Tens{order,dim,T,A<:AbstractArray}

Tensor type of any order defined by
- a multidata of components (of any type heriting from `AbstractArray`, e.g. `Tensor` or `SymmetricTensor`)
- a basis of `AbstractBasis` type
- a tuple of variances (covariant `:cov` or contravariant `:cont`) of length equal to the `order` of the tensor

# Examples
```julia
julia> ℬ = Basis(Sym[1 0 0; 0 1 0; 0 1 1]) ;

julia> T = Tens(metric(ℬ,:cov),ℬ,(:cov,:cov))
Tens{2, 3, Sym, SymmetricTensor{2, 3, Sym, 6}}
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
struct Tens{order,dim,T,A<:AbstractArray} <: AbstractTens{order,dim,T}
    data::A
    basis::Basis
    var::NTuple{order,Symbol}
    function Tens(
        data::AbstractArray{T,order},
        basis::Basis{dim},
        var::NTuple{order,Symbol} = ntuple(_ -> :cont, Val(order)),
    ) where {order,dim,T}
        newdata = tensor_or_array(data)
        new{order,dim,T,typeof(newdata)}(newdata, basis, var)
    end
    Tens(data::AbstractArray, basis::RotatedBasis, args...) = TensRotated(data, basis)
    Tens(data::AbstractArray, basis::OrthogonalBasis, args...) =
        TensOrthogonal(data, basis, args...)
    Tens(
        data::AbstractArray{T},
        basis::OrthonormalBasis = CanonicalBasis{size(data)[1],T}(),
        args...,
    ) where {T} = TensOrthonormal(data, basis)
    Tens(data::AbstractArray, var::NTuple, basis::AbstractBasis = CanonicalBasis{size(data,1),eltype(data)}()) = Tens(data, basis, var)
    Tens(data::AbstractArray{T,0}, args...) where {T} = T(data[1])
    Tens(data::T, args...) where {T} = data
end

proj_tens(sym::Symbol, A::AbstractArray) = proj_tens(Val(sym), A)

for order ∈ (2, 4)
    @eval function best_sym_tens(
        t::AbstractTens{$order,dim,T},
        args...;
        proj = (:ISO,),
        ε = 1.e-6,
    ) where {dim,T}
        basis = relevant_OrthonormalBasis(getbasis(t))
        newt = change_tens(t, basis)
        for sym ∈ proj
            (projt, d, drel) = proj_tens(sym, newt)
            if d == zero(T) || drel < ε
                return projt, d, drel, sym
            end
        end
        return newt, zero(T), zero(T), :ANISO
    end
end

struct TensRotated{order,dim,T,A<:AbstractArray} <: AbstractTens{order,dim,T}
    data::A
    basis::RotatedBasis
    function TensRotated(
        data::AbstractArray{T,order},
        basis::RotatedBasis{dim},
    ) where {order,dim,T}
        newdata = tensor_or_array(data)
        new{order,dim,T,typeof(newdata)}(newdata, basis)
    end
end

struct TensCanonical{order,dim,T,A<:AbstractArray} <: AbstractTens{order,dim,T}
    data::A
    function TensCanonical(data::AbstractArray{T,order}) where {order,T}
        newdata = tensor_or_array(data)
        new{order,size(data, 1),T,typeof(newdata)}(newdata)
    end
end

struct TensOrthogonal{order,dim,T,A<:AbstractArray} <: AbstractTens{order,dim,T}
    data::A
    basis::OrthogonalBasis
    var::NTuple{order,Symbol}
    function TensOrthogonal(
        data::AbstractArray{T,order},
        basis::OrthogonalBasis{dim},
        var::NTuple{order,Symbol} = ntuple(_ -> :cont, Val(order)),
    ) where {order,dim,T}
        newdata = tensor_or_array(data)
        new{order,dim,T,typeof(newdata)}(newdata, basis, var)
    end
end

const TensOrthonormal{order,dim,T,A} =
    Union{TensRotated{order,dim,T,A},TensCanonical{order,dim,T,A}}
const AllTensOrthogonal{order,dim,T,A} = Union{TensOrthonormal{order,dim,T,A}, TensOrthogonal{order,dim,T,A}}
const TensVar{order,dim,T,A} = Union{Tens{order,dim,T,A},TensOrthogonal{order,dim,T,A}}
const TensBasis{order,dim,T,A} =
    Union{Tens{order,dim,T,A},TensRotated{order,dim,T,A},TensOrthogonal{order,dim,T,A}}
const TensArray{order,dim,T,A} = Union{
    Tens{order,dim,T,A},
    TensRotated{order,dim,T,A},
    TensCanonical{order,dim,T,A},
    TensOrthogonal{order,dim,T,A},
}

TensOrthonormal(data::AbstractArray, basis::RotatedBasis) = TensRotated(data, basis)
TensOrthonormal(data::AbstractArray, ::CanonicalBasis) = TensCanonical(data)

TensOrthonormal(t::TensOrthonormal) = t
function TensOrthonormal(t::TensOrthogonal{order,dim,T}) where {order,dim,T}
    m = Array(getarray(t))
    ℬ = getbasis(t)
    onℬ = relevant_OrthonormalBasis(ℬ)
    Λ = Dict(:cov => inv.(ℬ.λ), :cont => ℬ.λ)
    for ind ∈ CartesianIndices(m)
        m[ind] *= prod([Λ[getvar(t, i)][ind[i]] for i ∈ 1:order])
    end
    return Tens(m, onℬ)
end

@inline Base.size(t::TensArray) = size(getarray(t))
@inline Base.getindex(t::TensArray, ind...) = getindex(getarray(t), ind...)
@pure datatype(::TensArray{order,dim,T,A}) where {order,dim,T,A} = A

@inline Base.zero(t::AbstractTens) = Tens(zero.(getarray(t)), getbasis(t), getvar(t))

# This function aims at storing the table of components in the `Tensor` type whenever possible
tensor_or_array(tab::AbstractArray{T,1}) where {T} = Vec{size(tab, 1)}(tab)
for order ∈ (2, 4)
    @eval function tensor_or_array(tab::AbstractArray{T,$order}) where {T}
        dim = size(tab, 1)
        newtab = Tensor{$order,dim}(tab)
        if Tensors.issymmetric(newtab)
            newtab = convert(SymmetricTensor{$order,dim}, newtab)
        end
        return newtab
    end
    @eval tensor_or_array(tab::Tensor{$order,dim}) where {dim} = Tensors.issymmetric(tab) ? convert(SymmetricTensor{$order,dim}, tab) : tab
end
tensor_or_array(tab::Tensors.AllTensors) = tab
tensor_or_array(tab::AbstractArray) = tab

##############################
# Utility/Accessor Functions #
##############################

getarray(t::TensArray) = t.data
getbasis(t::TensBasis) = t.basis
getbasis(::TensCanonical{order,dim,T}) where {order,dim,T} = CanonicalBasis{dim,T}()
getvar(::TensOrthonormal{order}) where {order} = ntuple(_ -> :cov, Val(order))
getvar(::TensOrthonormal, i::Integer) = :cov
getvar(t::TensVar) = t.var
getvar(t::TensVar, i::Integer) = t.var[i]


#####################
# Display Functions #
#####################
for OP in (:show, :print, :display)
    @eval begin
        Base.$OP(U::FourthOrderTensor) = $OP(tomandel(U))

        function Base.$OP(t::AbstractTens)
            $OP(typeof(t))
            print("→ array: ")
            $OP(getarray(t))
            print("→ basis: ")
            $OP(vecbasis(getbasis(t)))
            print("→ var: ")
            $OP(getvar(t))
        end
        function Base.$OP(t::TensOrthonormal)
            $OP(typeof(t))
            print("→ array: ")
            $OP(getarray(t))
            print("→ basis: ")
            $OP(vecbasis(getbasis(t)))
        end

        # Base.$OP(t::AbstractTens{order,dim,T}; vec = '𝐞', coords = ntuple(i -> i, dim)) where {order,dim,T} = intrinsic(t; vec= vec, coords = coords)        
    end
end

intrinsic(t::T) where {T} = println(t)

function intrinsic(t::AbstractTens{order,dim,T}; vec = '𝐞', coords = ntuple(i -> i, dim)) where {order,dim,T}
    ind = CartesianIndices(t)
    ℬ = getbasis(t)
    firstprint = true
    s = ""
    for i ∈ ind
        x = t[i]
        if x ≠ zero(T)
            if !firstprint
                s *= " + "
            end
            if x ≠ one(T)
                s *= "(" * string(x) * ")"
            end
            j = Tuple(i)
            for k ∈ 1:order
                s *= strvecbasis(ℬ, coords[j[k]], invvar(getvar(t,k)); vec = vec)
                if k < order
                    s *= "⊗"
                end
            end
            firstprint = false
        end
    end
    if length(s)>0
        println(s)
    else
        println(0)
    end
end



########################
# Component extraction #
########################

"""
    components(t::AbstractTens{order,dim,T},ℬ::AbstractBasis{dim},var::NTuple{order,Symbol})
    components(t::AbstractTens{order,dim,T},ℬ::AbstractBasis{dim})
    components(t::AbstractTens{order,dim,T},var::NTuple{order,Symbol})

Extract the components of a tensor for new variances and/or in a new basis

# Examples
```julia
julia> ℬ = Basis(Sym[0 1 1; 1 0 1; 1 1 0]) ;

julia> TV = Tens(Tensor{1,3}(i->symbols("v\$i",real=true)))
TensND.TensCanonical{1, 3, Sym, Vec{3, Sym}}
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

julia> TT = Tens(Tensor{2,3}((i,j)->symbols("t\$i\$j",real=true)))
TensND.TensCanonical{2, 3, Sym, Tensor{2, 3, Sym, 9}}
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
components(t::AbstractTens) = getarray(t)

components(t::TensOrthonormal, ::NTuple) = getarray(t)

function components(
    t::TensOrthogonal{order,dim,T},
    var::NTuple{order,Symbol},
) where {order,dim,T}
    if var == getvar(t)
        return getarray(t)
    else
        m = Array(getarray(t))
        ℬ = getbasis(t)
        g_or_G = ntuple(i -> getvar(t, i) == var[i] ? I : metric(ℬ, var[i]), order)
        for ind ∈ CartesianIndices(m)
            m[ind] *= prod([g_or_G[i][ind[i], ind[i]] for i ∈ 1:order])
        end
        return m
    end
end

for B in (AbstractBasis, OrthogonalBasis, OrthonormalBasis)
    @eval function components(
        t::TensOrthogonal{order,dim,T},
        ℬ::$B{dim},
        var::NTuple{order,Symbol},
    ) where {order,dim,T}
        if ℬ == getbasis(t)
            return components(t, var)
        else
            return components(TensOrthonormal(t), ℬ, var)
        end
    end
end

function components(t::Tens{order,dim,T}, var::NTuple{order,Symbol}) where {order,dim,T}
    if var == getvar(t)
        return getarray(t)
    else
        m = Array(getarray(t))
        ec1 = ntuple(i -> i, Val(order))
        newcp = order + 1
        for i ∈ 1:order
            if getvar(t, i) ≠ var[i]
                g_or_G = metric(getbasis(t), var[i])
                ec2 = (i, newcp)
                ec3 = ntuple(j -> j ≠ i ? j : newcp, Val(order))
                m = einsum(EinCode((ec1, ec2), ec3), (m, g_or_G))
            end
        end
        return m
    end
end

function components(
    t::AbstractTens{order,dim,T},
    ℬ::AbstractBasis{dim},
    var::NTuple{order,Symbol},
) where {order,dim,T}
    if ℬ == getbasis(t)
        return components(t, var)
    else
        bb = Dict()
        for v1 ∈ (:cov, :cont), v2 ∈ (:cov, :cont)
            if v1 ∈ getvar(t) && v2 ∈ var
                bb[v1, v2] = vecbasis(getbasis(t), invvar(v1))' * vecbasis(ℬ, v2)
            end
        end
        m = Array(getarray(t))
        ec1 = ntuple(i -> i, Val(order))
        newcp = order + 1
        for i ∈ 1:order
            c = bb[getvar(t, i), var[i]]
            if c ≠ I
                ec2 = (i, newcp)
                ec3 = ntuple(j -> j ≠ i ? j : newcp, Val(order))
                m = einsum(EinCode((ec1, ec2), ec3), (m, c))
            end
        end
        return m
    end
end

function components(
    t::AbstractTens{order,dim,T},
    ℬ::OrthogonalBasis{dim},
    var::NTuple{order,Symbol},
) where {order,dim,T}
    if ℬ == getbasis(t)
        return components(t, var)
    else
        m = Array(components(t, relevant_OrthonormalBasis(ℬ)))
        Λ = Dict(:cont => inv.(ℬ.λ), :cov => ℬ.λ)
        for ind ∈ CartesianIndices(m)
            m[ind] *= prod([Λ[var[i]][ind[i]] for i ∈ 1:order])
        end
        return m
    end
end

components(t::AbstractTens{order,dim,T}, ℬ::AbstractBasis{dim}) where {order,dim,T} =
    components(t, ℬ, getvar(t))

components(t::AbstractTens{order,dim,T}, ℬ::OrthonormalBasis{dim}) where {order,dim,T} =
    components(t, ℬ, ntuple(_ -> :cont, Val(order)))

function components(
    t::TensOrthonormal{order,dim,T},
    ℬ::OrthonormalBasis{dim},
) where {order,dim,T}
    if ℬ == getbasis(t)
        return getarray(t)
    else
        bb = vecbasis(getbasis(t))' * vecbasis(ℬ)
        m = Array(getarray(t))
        ec1 = ntuple(i -> i, Val(order))
        newcp = order + 1
        for i ∈ 1:order
            if bb ≠ I
                ec2 = (i, newcp)
                ec3 = ntuple(j -> j ≠ i ? j : newcp, Val(order))
                m = einsum(EinCode((ec1, ec2), ec3), (m, bb))
            end
        end
        return m
    end
end

components(
    t::TensOrthonormal{order,dim,T},
    basis::OrthonormalBasis{dim},
    ::NTuple{order,Symbol},
) where {order,dim,T} = components(t, basis)

"""
    components_canon(t::AbstractTens)

Extract the components of a tensor in the canonical basis
"""
components_canon(t::AbstractTens) =
    components(t, CanonicalBasis{getdim(t),eltype(t)}(), getvar(t))

components_canon(t::TensOrthonormal) = components(t, CanonicalBasis{getdim(t),eltype(t)}())

"""
    change_tens(t::AbstractTens{order,dim,T},ℬ::AbstractBasis{dim},var::NTuple{order,Symbol})
    change_tens(t::AbstractTens{order,dim,T},ℬ::AbstractBasis{dim})
    change_tens(t::AbstractTens{order,dim,T},var::NTuple{order,Symbol})

Rewrite the same tensor with components corresponding to new variances and/or to a new basis

```julia
julia> ℬ = Basis(Sym[0 1 1; 1 0 1; 1 1 0]) ;

julia> TV = Tens(Tensor{1,3}(i->symbols("v\$i",real=true)))
TensND.TensCanonical{1, 3, Sym, Vec{3, Sym}}
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
Tens{1, 3, Sym, Vec{3, Sym}}
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
function change_tens(t::AbstractTens, ℬ::AbstractBasis, newvar::NTuple)
    if ℬ == getbasis(t) && newvar == getvar(t)
        return t
    else
        return Tens(components(t, ℬ, newvar), ℬ, newvar)
    end
end

function change_tens(t::AbstractTens, newbasis::AbstractBasis)
    if newbasis == getbasis(t)
        return t
    else
        return Tens(components(t, newbasis, getvar(t)), newbasis, getvar(t))
    end
end

function change_tens(t::AbstractTens, newvar::NTuple)
    if newvar == getvar(t)
        return t
    else
        return Tens(components(t, getbasis(t), newvar), getbasis(t), newvar)
    end
end

"""
    change_tens_canon(t::AbstractTens{order,dim,T},var::NTuple{order,Symbol})

Rewrite the same tensor with components corresponding to the canonical basis

```julia
julia> ℬ = Basis(Sym[0 1 1; 1 0 1; 1 1 0]) ;

julia> TV = Tens(Tensor{1,3}(i->symbols("v\$i",real=true)), ℬ)
Tens{1, 3, Sym, Vec{3, Sym}}
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
TensND.TensCanonical{1, 3, Sym, Vec{3, Sym}}
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
change_tens_canon(t::AbstractTens) = change_tens(t, CanonicalBasis{getdim(t),eltype(t)}())


for OP in (:(diff),)
    @eval SymPy.$OP(t::AbstractTens{order,dim,Sym}, args...; kwargs...) where {order,dim} =
        change_tens(Tens($OP(components_canon(t), args...; kwargs...)), getbasis(t), getvar(t))
    @eval SymPy.$OP(t::AbstractArray{Sym}, args...; kwargs...) = $OP.(t, args...; kwargs...)
    @eval SymPy.$OP(t::Tensors.AllTensors{dim,Sym}, args...; kwargs...) where {dim} =
        Tensors.get_base(typeof(t))($OP.(Tensors.get_data(t), args...; kwargs...))
end

for OP in (:(simplify), :(factor), :(subs))
    @eval SymPy.$OP(t::AbstractTens{order,dim,Sym}, args...; kwargs...) where {order,dim} =
        Tens(SymPy.$OP(getarray(t), args...; kwargs...), SymPy.$OP(getbasis(t), args...; kwargs...), getvar(t))
    @eval SymPy.$OP(t::AbstractArray{Sym}, args...; kwargs...) = SymPy.$OP.(t, args...; kwargs...)
    @eval SymPy.$OP(t::Tensors.AllTensors{dim,Sym}, args...; kwargs...) where {dim} =
        Tensors.get_base(typeof(t))(SymPy.$OP.(Tensors.get_data(t), args...; kwargs...))
end

for OP in (:(trigsimp), :(expand_trig))
    @eval $OP(t::AbstractTens{order,dim}, args...; kwargs...) where {order,dim} =
        Tens($OP(getarray(t); kwargs...), $OP(getbasis(t), args...; kwargs...), getvar(t))
    @eval $OP(t::AbstractArray{Sym}, args...; kwargs...) = sympy.$OP.(t, args...; kwargs...)
    @eval $OP(t::Tensors.AllTensors{dim,Sym}, args...; kwargs...) where {dim} =
        Tensors.get_base(typeof(t))(sympy.$OP.(Tensors.get_data(t), args...; kwargs...))
    @eval $OP(t::Sym, args...; kwargs...) = sympy.$OP.(t, args...; kwargs...)
end


# SymPy.simplify(t::AbstractTens{order,dim,Sym}; kwargs...) where {order,dim} = Tens(simplify(data(t); kwargs...), basis(t), getvar(t))
# SymPy.simplify(t::AbstractArray{Sym}; kwargs...) = simplify.(t; kwargs...)
# SymPy.simplify(t::Tensors.AllTensors{dim, Sym}; kwargs...) where {dim} = Tensors.get_base(typeof(t))(simplify.(Tensors.get_data(t); kwargs...))

# sympy.trigsimp(t::AbstractTens{order,dim,Sym}; kwargs...) where {order,dim} = Tens(trigsimp(data(t); kwargs...), basis(t), getvar(t))
# sympy.trigsimp(t::AbstractArray{Sym}; kwargs...) = trigsimp.(t; kwargs...)
# sympy.trigsimp(t::Tensors.AllTensors{dim, Sym}; kwargs...) where {dim} = Tensors.get_base(typeof(t))(trigsimp.(Tensors.get_data(t); kwargs...))

# SymPy.factor(t::AbstractTens{order,dim,Sym}; kwargs...) where {order,dim} = Tens(factor(data(t); kwargs...), basis(t), getvar(t))
# SymPy.factor(t::AbstractArray{Sym}; kwargs...) = factor.(t; kwargs...)
# SymPy.factor(t::Tensors.AllTensors{dim, Sym}; kwargs...) where {dim} = Tensors.get_base(typeof(t))(factor.(Tensors.get_data(t); kwargs...))

# SymPy.subs(t::AbstractTens{order,dim,Sym}, d...; kwargs...) where {order,dim} = Tens(subs(data(t), d...; kwargs...), basis(t), getvar(t))
# SymPy.subs(t::AbstractArray{Sym}, d...; kwargs...) = subs.(t, d...; kwargs...)
# SymPy.subs(t::Tensors.AllTensors{dim, Sym}, d...; kwargs...) where {dim} = Tensors.get_base(typeof(t))(subs.(Tensors.get_data(t), d...; kwargs...))


##############
# Operations #
##############

choose_best_basis(ℬ::AbstractBasis, ::AbstractBasis) = ℬ

choose_best_basis(ℬ::OrthonormalBasis, ::AbstractBasis) = ℬ
choose_best_basis(ℬ::OrthogonalBasis, ::AbstractBasis) = ℬ
choose_best_basis(::AbstractBasis, ℬ::OrthonormalBasis) = ℬ
choose_best_basis(::AbstractBasis, ℬ::OrthogonalBasis) = ℬ

choose_best_basis(::CanonicalBasis, ℬ::OrthonormalBasis) = ℬ
choose_best_basis(::CanonicalBasis, ℬ::OrthogonalBasis) = ℬ
choose_best_basis(ℬ::OrthonormalBasis, ::CanonicalBasis) = ℬ
choose_best_basis(ℬ::OrthogonalBasis, ::CanonicalBasis) = ℬ

choose_best_basis(::OrthonormalBasis, ℬ::OrthogonalBasis) = ℬ
choose_best_basis(ℬ::OrthogonalBasis, ::OrthonormalBasis) = ℬ
choose_best_basis(ℬ::CanonicalBasis, ::CanonicalBasis) = ℬ
choose_best_basis(ℬ::OrthonormalBasis, ::OrthonormalBasis) = ℬ
choose_best_basis(ℬ::OrthogonalBasis, ::OrthogonalBasis) = ℬ

function same_basis(
    t1::AbstractTens{order1,dim},
    t2::AbstractTens{order2,dim},
) where {order1,order2,dim}
    ℬ = choose_best_basis(getbasis(t1), getbasis(t2))
    return change_tens(t1, ℬ), change_tens(t2, ℬ)
end

function same_basis_same_var(
    t1::AbstractTens{order1,dim},
    t2::AbstractTens{order2,dim},
) where {order1,order2,dim}
    ℬ = choose_best_basis(getbasis(t1), getbasis(t2))
    return change_tens(t1, ℬ, getvar(t1)), change_tens(t2, ℬ, getvar(t1))
end

# same_basis(
#     t1::AbstractTens{order1,dim},
#     t2::AbstractTens{order2,dim},
# ) where {order1,order2,dim} = t1, change_tens(t2, basis(t1))

# same_basis_same_var(
#     t1::AbstractTens{order1,dim},
#     t2::AbstractTens{order2,dim},
# ) where {order1,order2,dim} = t1, change_tens(t2, basis(t1), getvar(t1))


for OP in (:(==), :(!=))
    @eval @inline function Base.$OP(
        t1::AbstractTens{order,dim},
        t2::AbstractTens{order,dim},
    ) where {order,dim}
        nt1, nt2 = same_basis_same_var(t1, t2)
        return $OP(getarray(nt1), getarray(nt2))
    end
end

for OP in (:+, :-)
    @eval @inline function Base.$OP(
        t1::AbstractTens{order,dim},
        t2::AbstractTens{order,dim},
    ) where {order,dim}
        nt1, nt2 = same_basis_same_var(t1, t2)
        return Tens($OP(getarray(nt1), getarray(nt2)), getbasis(nt1), getvar(nt1))
    end
    @eval @inline function Base.$OP(
        t1::AllTensOrthogonal{order,dim,T},
        t2::UniformScaling{T},
    ) where {order,dim,T<:SymPy.SymbolicObject}
        nt1 = TensOrthonormal(t1)
        return Tens($OP(getarray(nt1), t2), getbasis(nt1), getvar(nt1))
    end
    @eval @inline function Base.$OP(
        t1::UniformScaling{T},
        t2::AllTensOrthogonal{order,dim,T},
    ) where {order,dim,T<:SymPy.SymbolicObject}
        nt2 = TensOrthonormal(t2)
        return Tens($OP(t1, getarray(nt2)), getbasis(nt2), getvar(nt2))
    end
end

@inline Base.:-(t::AbstractTens) = Tens(.-(getarray(t)), getbasis(t), getvar(t))
@inline Base.:*(α::Number, t::AbstractTens) = Tens(α * getarray(t), getbasis(t), getvar(t))
@inline Base.:*(t::AbstractTens, α::Number) = Tens(α * getarray(t), getbasis(t), getvar(t))
@inline Base.:/(t::AbstractTens, α::Number) = Tens(getarray(t) / α, getbasis(t), getvar(t))

@inline Base.inv(t::AbstractTens{2}) =
    Tens(inv(getarray(t)), getbasis(t), (invvar(getvar(t, 2)), invvar(getvar(t, 1))))
@inline Base.inv(t::AbstractTens{4}) = Tens(
    inv(getarray(t)),
    getbasis(t),
    (
        invvar(getvar(t, 3)),
        invvar(getvar(t, 4)),
        invvar(getvar(t, 1)),
        invvar(getvar(t, 2)),
    ),
)

"""
    KM(t::AbstractTens{order,dim}; kwargs...)
    KM(t::AbstractTens{order,dim}, var::NTuple{order,Symbol}, b::AbstractBasis{dim}; kwargs...)

Write the components of a second or fourth order tensor in Kelvin-Mandel notation

# Examples
```julia
julia> σ = Tens(SymmetricTensor{2,3}((i, j) -> symbols("σ\$i\$j", real = true))) ;

julia> KM(σ)
6-element Vector{Sym}:
         σ11
         σ22
         σ33
      √2⋅σ32
      √2⋅σ31
      √2⋅σ21

julia> C = Tens(SymmetricTensor{4,3}((i, j, k, l) -> symbols("C\$i\$j\$k\$l", real = true))) ;

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
KM(t::AbstractTens; kwargs...) = tomandel(getarray(t); kwargs...)

KM(
    t::AbstractTens{order,dim},
    b::AbstractBasis{dim},
    var::NTuple{order,Symbol},
    kwargs...,
) where {order,dim} = tomandel(tensor_or_array(components(t, b, var)); kwargs...)

KM(t::AbstractTens{order,dim}, b::AbstractBasis{dim}; kwargs...) where {order,dim} =
    tomandel(tensor_or_array(components(t, b)); kwargs...)


KM(t::AbstractArray; kwargs...) = KM(Tens(t); kwargs...)
KM(t::AbstractArray, b::AbstractBasis; kwargs...) = KM(Tens(t), b; kwargs...)

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

Define a tensor from a Kelvin-Mandel vector or matrix representation
"""
invKM(TT::Type{<:Tensors.AllTensors}, v::AbstractVecOrMat; kwargs...) =
    Tens(frommandel(TT, v; kwargs...))
invKM(TT::Type{<:Tensors.AllTensors}, v::AbstractVecOrMat, b::AbstractBasis; kwargs...) =
    Tens(frommandel(TT, v; kwargs...), b)
invKM(v::AbstractVecOrMat; kwargs...) = invKM(select_type_KM[size(v)], v; kwargs...)
invKM(v::AbstractVecOrMat, b::AbstractBasis; kwargs...) = invKM(select_type_KM[size(v)], v, b; kwargs...)


"""
    otimes(t1::AbstractTens{order1,dim}, t2::AbstractTens{order2,dim})

Define a tensor product between two tensors

`(aⁱeᵢ) ⊗ (bʲeⱼ) = aⁱbʲ eᵢ⊗eⱼ`
"""
function Tensors.otimes(
    t1::AbstractTens{order1,dim},
    t2::AbstractTens{order2,dim},
) where {order1,order2,dim}
    if t1 == t2
        return otimes(t1)
    else
        nt1, nt2 = same_basis(t1, t2)
        data = otimes(getarray(nt1), getarray(nt2))
        var = (getvar(nt1)..., getvar(nt2)...)
        return Tens(data, getbasis(nt1), var)
    end
end

function Tensors.otimes(
    t1::TensOrthonormal{order1,dim},
    t2::TensOrthonormal{order2,dim},
) where {order1,order2,dim}
    if t1 == t2
        return otimes(t1)
    else
        nt1, nt2 = same_basis(t1, t2)
        data = otimes(getarray(nt1), getarray(nt2))
        return Tens(data, getbasis(nt1))
    end
end

Tensors.otimes(v::AbstractTens{1,dim}) where {dim} =
    Tens(otimes(getarray(v)), getbasis(v), (getvar(v)..., getvar(v)...))

Tensors.otimes(v::TensOrthonormal{1,dim}) where {dim} =
    Tens(otimes(getarray(v)), getbasis(v))

@inline function Tensors.otimes(S::SymmetricTensor{2,dim}) where {dim}
    return SymmetricTensor{4,dim}(@inline function (i, j, k, l)
        @inbounds S[i, j] * S[k, l]
    end)
end

Tensors.otimes(t::AbstractTens{2,dim}) where {dim} =
    Tens(otimes(getarray(t)), getbasis(t), (getvar(t)..., getvar(t)...))

Tensors.otimes(t::TensOrthonormal{2,dim}) where {dim} =
    Tens(otimes(getarray(t)), getbasis(t))


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
    dot(t1::AbstractTens{order1,dim}, t2::AbstractTens{order2,dim})

Define a contracted product between two tensors

`a ⋅ b = aⁱbⱼ`
"""
function LinearAlgebra.dot(
    t1::AbstractTens{order1,dim},
    t2::AbstractTens{order2,dim},
) where {order1,order2,dim}
    nt1, nt2 = same_basis(t1, t2)
    var = (invvar(getvar(nt1)[end]), getvar(nt2)[begin+1:end]...)
    nt2 = change_tens(nt2, getbasis(nt2), var)
    data = scontract(getarray(nt1), getarray(nt2))
    var = (getvar(nt1)[begin:end-1]..., getvar(nt2)[begin+1:end]...)
    return Tens(data, getbasis(nt1), var)
end

function LinearAlgebra.dot(
    t1::TensOrthonormal{order1,dim},
    t2::TensOrthonormal{order2,dim},
) where {order1,order2,dim}
    nt1, nt2 = same_basis(t1, t2)
    data = scontract(getarray(nt1), getarray(nt2))
    return Tens(data, getbasis(nt1))
end

function LinearAlgebra.dot(t1::AbstractTens{1,dim}, t2::AbstractTens{1,dim}) where {dim}
    nt1, nt2 = same_basis(t1, t2)
    var = (invvar(getvar(nt1)[end]), getvar(nt2)[begin+1:end]...)
    nt2 = change_tens(nt2, getbasis(nt2), var)
    return scontract(getarray(nt1), getarray(nt2))
end

function LinearAlgebra.dot(
    t1::TensOrthonormal{1,dim},
    t2::TensOrthonormal{1,dim},
) where {dim}
    nt1, nt2 = same_basis(t1, t2)
    return scontract(getarray(nt1), getarray(nt2))
end

LinearAlgebra.dot(v::TensOrthonormal{1,dim}) where {dim} = dot(getarray(v))

LinearAlgebra.dot(t::TensOrthonormal{2,dim}) where {dim} = dot(getarray(t))

LinearAlgebra.norm(u::AbstractTens{1,dim}) where {dim} = √(dot(u, u))

LinearAlgebra.norm(t::AbstractTens{2,dim}) where {dim} = √(dot(t, t))

"""
    contract(t::AbstractTens{order,dim}, i::Integer, j::Integer)

Calculate the tensor obtained after contraction with respect to the indices `i` and `j`
"""
function contract(t::AbstractTens{order,dim}, i::Integer, j::Integer) where {order,dim}
    var = ntuple(k -> k == j ? invvar(getvar(t, i)) : getvar(t, k), order)
    nt = change_tens(t, getbasis(t), var)
    data = contract(getarray(nt), i, j)
    m = min(i, j)
    M = max(i, j)
    var = (getvar(nt)[1:m-1]..., getvar(nt)[m+1:M-1]..., getvar(nt)[M+1:order]...)
    return Tens(data, getbasis(nt), var)
end

function contract(t::AbstractTens{2,dim}, i::Integer, j::Integer) where {dim}
    var = ntuple(k -> k == j ? invvar(getvar(t, i)) : getvar(t, k), 2)
    nt = change_tens(t, getbasis(t), var)
    return contract(getarray(nt), i, j)
end

contract(t::TensOrthonormal{order,dim}, i::Integer, j::Integer) where {order,dim} =
    Tens(contract(getarray(t), i, j), getbasis(t))

contract(t::TensOrthonormal{2,dim}, i::Integer, j::Integer) where {dim} =
    contract(getarray(t), i, j)

LinearAlgebra.tr(t::AbstractTens{2}) = contract(t, 1, 2)

"""
    dcontract(t1::AbstractTens{order1,dim}, t2::AbstractTens{order2,dim})

Define a double contracted product between two tensors

`𝛔 ⊡ 𝛆 = σⁱʲεᵢⱼ`
`𝛔 = ℂ ⊡ 𝛆`

# Examples
```julia
julia> 𝛆 = Tens(SymmetricTensor{2,3}((i, j) -> symbols("ε\$i\$j", real = true))) ;

julia> k, μ = symbols("k μ", real =true) ;

julia> ℂ = 3k * t𝕁() + 2μ * t𝕂() ;

julia> 𝛔 = ℂ ⊡ 𝛆
Tens{2, 3, Sym, Sym, SymmetricTensor{2, 3, Sym, 6}, CanonicalBasis{3, Sym}}
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
    t1::AbstractTens{order1,dim},
    t2::AbstractTens{order2,dim},
) where {order1,order2,dim}
    nt1, nt2 = same_basis(t1, t2)
    var =
        (invvar(getvar(nt1)[end-1]), invvar(getvar(nt1)[end]), getvar(nt2)[begin+2:end]...)
    nt2 = change_tens(nt2, getbasis(nt2), var)
    data = Tensors.dcontract(getarray(nt1), getarray(nt2))
    var = (getvar(nt1)[begin:end-2]..., getvar(nt2)[begin+2:end]...)
    return Tens(data, getbasis(nt1), var)
end

function Tensors.dcontract(
    t1::TensOrthonormal{order1,dim},
    t2::TensOrthonormal{order2,dim},
) where {order1,order2,dim}
    nt1, nt2 = same_basis(t1, t2)
    data = Tensors.dcontract(getarray(nt1), getarray(nt2))
    return Tens(data, getbasis(nt1))
end

function Tensors.dcontract(t1::AbstractTens{2,dim}, t2::AbstractTens{2,dim}) where {dim}
    nt1, nt2 = same_basis(t1, t2)
    var =
        (invvar(getvar(nt1)[end-1]), invvar(getvar(nt1)[end]), getvar(nt2)[begin+2:end]...)
    nt2 = change_tens(nt2, getbasis(nt2), var)
    return Tensors.dcontract(getarray(nt1), getarray(nt2))
end

function Tensors.dcontract(
    t1::TensOrthonormal{2,dim},
    t2::TensOrthonormal{2,dim},
) where {dim}
    nt1, nt2 = same_basis(t1, t2)
    return Tensors.dcontract(getarray(nt1), getarray(nt2))
end


"""
    dotdot(v1::AbstractTens{order1,dim}, S::AbstractTens{orderS,dim}, v2::AbstractTens{order2,dim})

Define a bilinear operator `𝐯₁⋅𝕊⋅𝐯₂`

# Examples
```julia
julia> n = Tens(Sym[0, 0, 1]) ;

julia> k, μ = symbols("k μ", real =true) ;

julia> ℂ = 3k * t𝕁() + 2μ * t𝕂() ;

julia> dotdot(n,ℂ,n) # Acoustic tensor
3×3 Tens{2, 3, Sym, Sym, Tensor{2, 3, Sym, 9}, CanonicalBasis{3, Sym}}:
 μ  0          0
 0  μ          0
 0  0  k + 4*μ/3
```
"""
function Tensors.dotdot(
    v1::AbstractTens{order1,dim},
    S::AbstractTens{orderS,dim},
    v2::AbstractTens{order2,dim},
) where {order1,orderS,order2,dim}
    nS, nv1 = same_basis(S, v1)
    nS, nv2 = same_basis(S, v2)
    var = (invvar(getvar(nS)[begin]),)
    nv1 = change_tens(nv1, getbasis(nv1), var)
    var = (invvar(getvar(nS)[end]),)
    nv2 = change_tens(nv2, getbasis(nv2), var)
    data = dotdot(nv1.data, nS.data, nv2.data)
    var = (getvar(nS)[begin+1], getvar(nS)[end-1])
    return Tens(data, getbasis(nS), var)
end

function Tensors.dotdot(
    v1::TensOrthonormal{order1,dim},
    S::TensOrthonormal{orderS,dim},
    v2::TensOrthonormal{order2,dim},
) where {order1,orderS,order2,dim}
    nS, nv1 = same_basis(S, v1)
    nS, nv2 = same_basis(S, v2)
    data = dotdot(nv1.data, nS.data, nv2.data)
    return Tens(data, getbasis(nS))
end

"""
    qcontract(t1::AbstractTens{order1,dim}, t2::AbstractTens{order2,dim})

Define a quadruple contracted product between two tensors

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
    t1::AbstractTens{order1,dim},
    t2::AbstractTens{order2,dim},
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
    data = qcontract(getarray(nt1), getarray(nt2))
    var = (getvar(nt1)[begin:end-4]..., getvar(nt2)[begin+4:end]...)
    return Tens(data, getbasis(nt1), var)
end

function qcontract(
    t1::TensOrthonormal{order1,dim},
    t2::TensOrthonormal{order2,dim},
) where {order1,order2,dim}
    nt1, nt2 = same_basis(t1, t2)
    data = qcontract(getarray(nt1), getarray(nt2))
    return Tens(data, getbasis(nt1))
end

function qcontract(t1::AbstractTens{4,dim}, t2::AbstractTens{4,dim}) where {dim}
    nt1, nt2 = same_basis(t1, t2)
    var = (
        invvar(getvar(nt1)[end-3]),
        invvar(getvar(nt1)[end-2]),
        invvar(getvar(nt1)[end-1]),
        invvar(getvar(nt1)[end]),
        getvar(nt2)[begin+4:end]...,
    )
    nt2 = change_tens(nt2, getbasis(nt2), var)
    return qcontract(getarray(nt1), getarray(nt2))
end

function qcontract(t1::TensOrthonormal{4,dim}, t2::TensOrthonormal{4,dim}) where {dim}
    nt1, nt2 = same_basis(t1, t2)
    return qcontract(getarray(nt1), getarray(nt2))
end

"""
    otimesu(t1::AbstractTens{order1,dim}, t2::AbstractTens{order2,dim})

Define a special tensor product between two tensors of at least second order

`(𝐚 ⊠ 𝐛) ⊡ 𝐩 = 𝐚⋅𝐩⋅𝐛 = aⁱᵏbʲˡpₖₗ eᵢ⊗eⱼ`
"""
function Tensors.otimesu(
    t1::AbstractTens{order1,dim},
    t2::AbstractTens{order2,dim},
) where {order1,order2,dim}
    nt1, nt2 = same_basis(t1, t2)
    data = otimesu(getarray(nt1), getarray(nt2))
    var = (
        getvar(nt1)[begin:end-1]...,
        getvar(nt2)[begin],
        getvar(nt1)[end],
        getvar(nt2)[begin+1:end]...,
    )
    return Tens(data, getbasis(nt1), var)
end

function Tensors.otimesu(
    t1::TensOrthonormal{order1,dim},
    t2::TensOrthonormal{order2,dim},
) where {order1,order2,dim}
    nt1, nt2 = same_basis(t1, t2)
    data = otimesu(getarray(nt1), getarray(nt2))
    return Tens(data, getbasis(nt1))
end

function Tensors.otimesl(
    t1::AbstractTens{order1,dim},
    t2::AbstractTens{order2,dim},
) where {order1,order2,dim}
    nt1, nt2 = same_basis(t1, t2)
    data = otimesl(getarray(nt1), getarray(nt2))
    var = (
        getvar(nt1)[begin:end-1]...,
        getvar(nt2)[begin+1],
        getvar(nt1)[end],
        getvar(nt2)[begin],
        getvar(nt2)[begin+2:end]...,
    )
    return Tens(data, getbasis(nt1), var)
end

function Tensors.otimesl(
    t1::TensOrthonormal{order1,dim},
    t2::TensOrthonormal{order2,dim},
) where {order1,order2,dim}
    nt1, nt2 = same_basis(t1, t2)
    data = otimesl(getarray(nt1), getarray(nt2))
    return Tens(data, getbasis(nt1))
end

otimesul(S1::SecondOrderTensor{dim}, S2::SecondOrderTensor{dim}) where {dim} =
    symmetric(otimesu(S1, S2))

"""
    otimesul(t1::AbstractTens{order1,dim}, t2::AbstractTens{order2,dim})

Define a special tensor product between two tensors of at least second order

`(𝐚 ⊠ˢ 𝐛) ⊡ 𝐩 = (𝐚 ⊠ 𝐛) ⊡ (𝐩 + ᵗ𝐩)/2  = 1/2(aⁱᵏbʲˡ+aⁱˡbʲᵏ) pₖₗ eᵢ⊗eⱼ`
"""
function otimesul(
    t1::AbstractTens{order1,dim},
    t2::AbstractTens{order2,dim},
) where {order1,order2,dim}
    nt1, nt2 = same_basis(t1, t2)
    var = (getvar(nt1)[end-1], getvar(nt1)[end], getvar(nt2)[begin+2:end]...)
    nt2 = change_tens(nt2, getbasis(nt2), var)
    data = otimesul(getarray(nt1), getarray(nt2))
    var = (getvar(nt1)..., getvar(nt2)...)
    return Tens(data, getbasis(nt1), var)
end

function otimesul(
    t1::TensOrthonormal{order1,dim},
    t2::TensOrthonormal{order2,dim},
) where {order1,order2,dim}
    nt1, nt2 = same_basis(t1, t2)
    data = otimesul(getarray(nt1), getarray(nt2))
    return Tens(data, getbasis(nt1))
end



"""
    sotimes(t1::AbstractTens{order1,dim}, t2::AbstractTens{order2,dim})

Define a symmetric tensor product between two tensors

`(aⁱeᵢ) ⊗ˢ (bʲeⱼ) = 1/2(aⁱbʲ + aʲbⁱ) eᵢ⊗eⱼ`
"""
function sotimes(
    t1::AbstractTens{order1,dim},
    t2::AbstractTens{order2,dim},
) where {order1,order2,dim}
    nt1, nt2 = same_basis(t1, t2)
    var = (getvar(nt1)[end], getvar(nt2)[begin+1:end]...)
    nt2 = change_tens(nt2, getbasis(nt2), var)
    data = sotimes(getarray(nt1), getarray(nt2))
    var = (getvar(nt1)..., getvar(nt2)...)
    return Tens(data, getbasis(nt1), var)
end

function sotimes(
    t1::AbstractTens{1,dim},
    t2::AbstractTens{1,dim},
) where {dim}
    if t1 == t2
        return otimes(t1)
    else
        nt1, nt2 = same_basis(t1, t2)
        var = (getvar(nt1)[end], getvar(nt2)[begin+1:end]...)
        nt2 = change_tens(nt2, getbasis(nt2), var)
        data = sotimes(getarray(nt1), getarray(nt2))
        var = (getvar(nt1)..., getvar(nt2)...)
        return Tens(data, getbasis(nt1), var)
    end
end

function sotimes(
    t1::TensOrthonormal{order1,dim},
    t2::TensOrthonormal{order2,dim},
) where {order1,order2,dim}
    nt1, nt2 = same_basis(t1, t2)
    data = sotimes(getarray(nt1), getarray(nt2))
    return Tens(data, getbasis(nt1))
end

function sotimes(
    t1::TensOrthonormal{1,dim},
    t2::TensOrthonormal{1,dim},
) where {dim}
    if t1 == t2
        return otimes(t1)
    else
        nt1, nt2 = same_basis(t1, t2)
        data = sotimes(getarray(nt1), getarray(nt2))
        return Tens(data, getbasis(nt1))
    end
end

Base.transpose(t::TensArray{order,dim,T,<:SecondOrderTensor}) where {order,dim,T} =
    Tens(transpose(getarray(t)), getbasis(t), (getvar(t)[2], getvar(t)[1]))

Base.transpose(t::TensOrthonormal{order,dim,T,<:SecondOrderTensor}) where {order,dim,T} =
    Tens(transpose(getarray(t)), getbasis(t))

Base.transpose(t::TensArray{order,dim,T,<:FourthOrderTensor}) where {order,dim,T} = Tens(
    Tensors.transpose(getarray(t)),
    getbasis(t),
    (getvar(t)[2], getvar(t)[1], getvar(t)[4], getvar(t)[3]),
)

Base.transpose(t::TensOrthonormal{order,dim,T,<:FourthOrderTensor}) where {order,dim,T} =
    Tens(Tensors.transpose(getarray(t)), getbasis(t))

Tensors.majortranspose(t::TensArray{order,dim,T,<:FourthOrderTensor}) where {order,dim,T} =
    Tens(
        majortranspose(getarray(t)),
        getbasis(t),
        (getvar(t)[3], getvar(t)[4], getvar(t)[1], getvar(t)[2]),
    )

Tensors.majortranspose(
    t::TensOrthonormal{order,dim,T,<:FourthOrderTensor},
) where {order,dim,T} = Tens(majortranspose(getarray(t)), getbasis(t))

Tensors.minortranspose(t::TensArray{order,dim,T,<:FourthOrderTensor}) where {order,dim,T} =
    Tens(
        minortranspose(getarray(t)),
        getbasis(t),
        (getvar(t)[2], getvar(t)[1], getvar(t)[4], getvar(t)[3]),
    )

Tensors.minortranspose(
    t::TensOrthonormal{order,dim,T,<:FourthOrderTensor},
) where {order,dim,T} = Tens(minortranspose(getarray(t)), getbasis(t))

function tensbasis(ℬ::AbstractBasis, i::Integer, ::Val{:cov})
    T = eltype(ℬ)
    t = [T(Int(j == i)) for j ∈ 1:getdim(ℬ)]
    return Tens(t, ℬ, (:cont,))
end
function tensbasis(ℬ::AbstractBasis, i::Integer, ::Val{:cont})
    T = eltype(ℬ)
    t = [T(Int(j == i)) for j ∈ 1:getdim(ℬ)]
    return Tens(t, ℬ, (:cov,))
end
tensbasis(ℬ::AbstractBasis, i::Integer, var = :cov) = tensbasis(ℬ, i, Val(var))
tensbasis(ℬ::AbstractBasis, var = :cov) = ntuple(i -> tensbasis(ℬ, i, Val(var)), getdim(ℬ))

export AbstractTens, Tens
export proj_tens, best_sym_tens
export getorder, arraytype, getdata, getarray, getbasis, getvar
export intrinsic
export components, components_canon, change_tens, change_tens_canon
export trigsimp, expand_trig
export KM, invKM
export getbasis, getvar
export tensbasis