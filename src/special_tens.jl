"""
    LeviCivita(T::Type{<:Number} = Sym)

Builds an Array{T,3} of Levi-Civita Symbol `ϵᵢⱼₖ = (i-j) (j-k) (k-i) / 2`

# Examples
```julia
julia> ε = LeviCivita(Sym)
3×3×3 Array{Sym, 3}:
[:, :, 1] =
 0   0  0
 0   0  1
 0  -1  0

[:, :, 2] =
 0  0  -1
 0  0   0
 1  0   0

[:, :, 3] =
  0  1  0
 -1  0  0
  0  0  0
``` 
"""
LeviCivita(T::Type{<:Number} = Sym) = [T(T((i - j) * (j - k) * (k - i)) / T(2)) for i = 1:3, j = 1:3, k = 1:3]


"""
    tensId2(T::Type{<:Number} = Sym, dim = 3)
    t𝟏(T::Type{<:Number} = Sym, dim = 3)

Identity tensor of second order `𝟏ᵢⱼ = δᵢⱼ = 1 if i=j otherwise 0`

# Examples
```julia
julia> 𝟏 = t𝟏() ; KM(𝟏)
6-element Vector{Sym}:
 1
 1
 1
 0
 0
 0

julia> 𝟏.data
3×3 SymmetricTensor{2, 3, Sym, 6}:
 1  0  0
 0  1  0
 0  0  1
```  
"""
tensId2(T::Type{<:Number} = Sym, dim = 3) =
    Tensnd(one(SymmetricTensor{2,dim,T}), CanonicalBasis{dim,T}())

"""
    tensId4(T::Type{<:Number} = Sym, dim = 3)
    t𝟙(T::Type{<:Number} = Sym, dim = 3)

Identity tensor of fourth order  `𝟙 = 𝟏 ⊠ 𝟏` i.e. `(𝟙)ᵢⱼₖₗ = δᵢₖδⱼₗ`

# Examples
```julia
julia> 𝟙 = t𝟙() ; KM(𝟙)
9×9 Matrix{Sym}:
 1  0  0  0  0  0  0  0  0
 0  1  0  0  0  0  0  0  0
 0  0  1  0  0  0  0  0  0
 0  0  0  1  0  0  0  0  0
 0  0  0  0  1  0  0  0  0
 0  0  0  0  0  1  0  0  0
 0  0  0  0  0  0  1  0  0
 0  0  0  0  0  0  0  1  0
 0  0  0  0  0  0  0  0  1
``` 
"""
tensId4(T::Type{<:Number} = Sym, dim = 3) =
    Tensnd(one(Tensor{4,dim,T}), CanonicalBasis{dim,T}())

"""
    tensId4s(T::Type{<:Number} = Sym, dim = 3)
    t𝕀(T::Type{<:Number} = Sym, dim = 3)

Symmetric identity tensor of fourth order  `𝕀 = 𝟏 ⊠ˢ 𝟏` i.e. `(𝕀)ᵢⱼₖₗ = (δᵢₖδⱼₗ+δᵢₗδⱼₖ)/2`

# Examples
```julia
julia> 𝕀 = t𝕀() ; KM(𝕀)
6×6 Matrix{Sym}:
 1  0  0  0  0  0
 0  1  0  0  0  0
 0  0  1  0  0  0
 0  0  0  1  0  0
 0  0  0  0  1  0
 0  0  0  0  0  1
``` 
"""
tensId4s(T::Type{<:Number} = Sym, dim = 3) =
    Tensnd(one(SymmetricTensor{4,dim,T}), CanonicalBasis{dim,T}())

"""
    tensJ4(T::Type{<:Number} = Sym, dim = 3)
    t𝕁(T::Type{<:Number} = Sym, dim = 3)

Spherical projector of fourth order  `𝕁 = (𝟏 ⊗ 𝟏) / dim` i.e. `(𝕁)ᵢⱼₖₗ = δᵢⱼδₖₗ/dim`

# Examples
```julia
julia> 𝕁 = t𝕁() ; KM(𝕁)
6×6 Matrix{Sym}:
 1/3  1/3  1/3  0  0  0
 1/3  1/3  1/3  0  0  0
 1/3  1/3  1/3  0  0  0
   0    0    0  0  0  0
   0    0    0  0  0  0
   0    0    0  0  0  0
``` 
"""
function tensJ4(T::Type{<:Number} = Sym, dim = 3)
    δ = one(SymmetricTensor{2,dim,T})
    return Tensnd(δ ⊗ δ / dim, CanonicalBasis{dim,T}())
end

"""
    tensK4(T::Type{<:Number} = Sym, dim = 3)
    t𝕂(T::Type{<:Number} = Sym, dim = 3)

Deviatoric projector of fourth order  `𝕂 = 𝕀 - 𝕁` i.e. `(𝕂)ᵢⱼₖₗ = (δᵢₖδⱼₗ+δᵢₗδⱼₖ)/2 - δᵢⱼδₖₗ/dim`

# Examples
```julia
julia> 𝕂 = t𝕂() ; KM(𝕂)
6×6 Matrix{Sym}:
  2/3  -1/3  -1/3  0  0  0
 -1/3   2/3  -1/3  0  0  0
 -1/3  -1/3   2/3  0  0  0
    0     0     0  1  0  0
    0     0     0  0  1  0
    0     0     0  0  0  1
``` 
"""
tensK4(T::Type{<:Number} = Sym, dim = 3) = tensId4s(T, dim) - tensJ4(T, dim)


"""
    𝐞(i::Int, dim::Int = 3, T::Type{<:Number} = Sym)

Vector of the canonical basis

# Examples
```julia
julia> 𝐞(1)
Tensnd{1, 3, Sym, Sym, Vec{3, Sym}, CanonicalBasis{3, Sym}}
# data: 3-element Vec{3, Sym}:
 1
 0
 0
# var: (:cont,)
# basis: 3×3 Tensor{2, 3, Sym, 9}:
 1  0  0
 0  1  0
 0  0  1
``` 
"""
𝐞(i::Int, dim::Int = 3, T::Type{<:Number} = Sym) =
    Tensnd(Vec{dim}(j -> j == i ? one(T) : zero(T)))

"""
    𝐞ᵖ(i::Int, θ::T = zero(Sym); canonical = false)

Vector of the polar basis

# Examples
```julia
julia> θ = symbols("θ", real = true) ;

julia> 𝐞ᵖ(1, θ)
Tensnd{1, 2, Sym, Sym, Vec{2, Sym}, RotatedBasis{2, Sym}}
# data: 2-element Vec{2, Sym}:
 1
 0
# var: (:cont,)
# basis: 2×2 Tensor{2, 2, Sym, 4}:
 cos(θ)  -sin(θ)
 sin(θ)   cos(θ)
``` 
"""
𝐞ᵖ(::Val{1}, θ::T = zero(Sym); canonical = false) where {T<:Number} =
    canonical ? Tensnd(Vec{2}([cos(θ), sin(θ)])) :
    Tensnd(Vec{2}([one(T), zero(T)]), Basis(θ))
𝐞ᵖ(::Val{2}, θ::T = zero(Sym); canonical = false) where {T<:Number} =
    canonical ? Tensnd(Vec{2}([-sin(θ), cos(θ)])) :
    Tensnd(Vec{2}([zero(T), one(T)]), Basis(θ))

"""
    𝐞ᶜ(i::Int, θ::T = zero(Sym); canonical = false)

Vector of the cylindrical basis

# Examples
```julia
julia> θ = symbols("θ", real = true) ;

julia> 𝐞ᶜ(1, θ)
Tensnd{1, 3, Sym, Sym, Vec{3, Sym}, RotatedBasis{3, Sym}}
# data: 3-element Vec{3, Sym}:
 1
 0
 0
# var: (:cont,)
# basis: 3×3 Tensor{2, 3, Sym, 9}:
 cos(θ)  -sin(θ)  0
 sin(θ)   cos(θ)  0
      0        0  1
``` 
"""
𝐞ᶜ(::Val{1}, θ::T = zero(Sym); canonical = false) where {T<:Number} =
    canonical ? Tensnd(Vec{3}([cos(θ), sin(θ), zero(T)])) :
    Tensnd(Vec{3}([one(T), zero(T), zero(T)]), CylindricalBasis(θ))
𝐞ᶜ(::Val{2}, θ::T = zero(Sym); canonical = false) where {T<:Number} =
    canonical ? Tensnd(Vec{3}([-sin(θ), cos(θ), zero(T)])) :
    Tensnd(Vec{3}([zero(T), one(T), zero(T)]), CylindricalBasis(θ))
𝐞ᶜ(::Val{3}, θ::T = zero(Sym); canonical = false) where {T<:Number} =
    canonical ? Tensnd(Vec{3}([zero(T), zero(T), one(T)])) :
    Tensnd(Vec{3}([zero(T), zero(T), one(T)]), CylindricalBasis(θ))

"""
    𝐞ˢ(i::Int, θ::T = zero(Sym), ϕ::T = zero(Sym), ψ::T = zero(Sym); canonical = false)

Vector of the basis rotated with the 3 Euler angles `θ, ϕ, ψ` (spherical if `ψ=0`)

# Examples
```julia
julia> θ, ϕ, ψ = symbols("θ, ϕ, ψ", real = true) ;

Tensnd{1, 3, Sym, Sym, Vec{3, Sym}, RotatedBasis{3, Sym}}
# data: 3-element Vec{3, Sym}:
 1
 0
 0
# var: (:cont,)
# basis: 3×3 Tensor{2, 3, Sym, 9}:
 -sin(ψ)⋅sin(ϕ) + cos(θ)⋅cos(ψ)⋅cos(ϕ)  -sin(ψ)⋅cos(θ)⋅cos(ϕ) - sin(ϕ)⋅cos(ψ)  sin(θ)⋅cos(ϕ)
  sin(ψ)⋅cos(ϕ) + sin(ϕ)⋅cos(θ)⋅cos(ψ)  -sin(ψ)⋅sin(ϕ)⋅cos(θ) + cos(ψ)⋅cos(ϕ)  sin(θ)⋅sin(ϕ)
                        -sin(θ)⋅cos(ψ)                          sin(θ)⋅sin(ψ)         cos(θ)
``` 
"""
function 𝐞ˢ(
    ::Val{1},
    θ::T1 = 0,
    ϕ::T2 = 0,
    ψ::T3 = 0;
    canonical = false,
) where {T1<:Number,T2<:Number,T3<:Number}
    if canonical
        return Tensnd(
            Vec{3}([
                -sin(ψ) * sin(ϕ) + cos(θ) * cos(ψ) * cos(ϕ),
                sin(ψ) * cos(ϕ) + sin(ϕ) * cos(θ) * cos(ψ),
                -sin(θ) * cos(ψ),
            ]),
        )
    else
        T = promote_type(T1, T2, T3)
        return Tensnd(Vec{3}([one(T), zero(T), zero(T)]), Basis(θ, ϕ, ψ))
    end
end
function 𝐞ˢ(
    ::Val{2},
    θ::T1 = 0,
    ϕ::T2 = 0,
    ψ::T3 = 0;
    canonical = false,
) where {T1<:Number,T2<:Number,T3<:Number}
    if canonical
        return Tensnd(
            Vec{3}([
                -sin(ψ) * cos(θ) * cos(ϕ) - sin(ϕ) * cos(ψ),
                -sin(ψ) * sin(ϕ) * cos(θ) + cos(ψ) * cos(ϕ),
                sin(θ) * sin(ψ),
            ]),
        )
    else
        T = promote_type(T1, T2, T3)
        return Tensnd(Vec{3}([zero(T), one(T), zero(T)]), Basis(θ, ϕ, ψ))
    end
end
function 𝐞ˢ(
    ::Val{3},
    θ::T1 = 0,
    ϕ::T2 = 0,
    ψ::T3 = 0;
    canonical = false,
) where {T1<:Number,T2<:Number,T3<:Number}
    if canonical
        return Tensnd(Vec{3}([sin(θ) * cos(ϕ), sin(θ) * sin(ϕ), cos(θ)]))
    else
        T = promote_type(T1, T2, T3)
        return Tensnd(Vec{3}([zero(T), zero(T), one(T)]), Basis(θ, ϕ, ψ))
    end
end


for eb in (:𝐞ᵖ, :𝐞ᶜ, :𝐞ˢ)
    @eval $eb(i::Int, args...; kwargs...) = $eb(Val(i), args...; kwargs...)
end


"""
    init_isotropic(T::Type{<:Number} = Sym)

Returns the isotropic tensors

# Examples
```julia
julia> 𝟏, 𝟙, 𝕀, 𝕁, 𝕂 = init_isotropic() ;
``` 
"""
init_isotropic(T::Type{<:Number} = Sym) = t𝟏(T), t𝟙(T), t𝕀(T), t𝕁(T), t𝕂(T)


"""
    rot3(θ, ϕ = 0, ψ = 0)

Returns a rotation matrix with respect to the 3 Euler angles `θ, ϕ, ψ`

# Examples
```julia
julia> cθ, cϕ, cψ, sθ, sϕ, sψ = symbols("cθ cϕ cψ sθ sϕ sψ", real = true) ;

julia> d = Dict(cos(θ) => cθ, cos(ϕ) => cϕ, cos(ψ) => cψ, sin(θ) => sθ, sin(ϕ) => sϕ, sin(ψ) => sψ) ;

julia> subs.(rot3(θ, ϕ, ψ),d...)
3×3 StaticArrays.SMatrix{3, 3, Sym, 9} with indices SOneTo(3)×SOneTo(3):
 cθ⋅cψ⋅cϕ - sψ⋅sϕ  -cθ⋅cϕ⋅sψ - cψ⋅sϕ  cϕ⋅sθ
 cθ⋅cψ⋅sϕ + cϕ⋅sψ  -cθ⋅sψ⋅sϕ + cψ⋅cϕ  sθ⋅sϕ
           -cψ⋅sθ              sθ⋅sψ     cθ
```
"""
rot3(θ, ϕ = 0, ψ = 0) = RotZYZ(ϕ, θ, ψ)

"""
    rot2(θ)

Returns a 2D rotation matrix with respect to the angle `θ`

# Examples
```julia
julia> rot2(θ)
2×2 Tensor{2, 2, Sym, 4}:
 cos(θ)  -sin(θ)
 sin(θ)   cos(θ)
```
"""
rot2(θ) = Tensor{2,2}((cos(θ), sin(θ), -sin(θ), cos(θ)))


"""
    rot6(θ, ϕ = 0, ψ = 0)

Returns a rotation matrix with respect to the 3 Euler angles `θ, ϕ, ψ`

# Examples
```julia
julia> cθ, cϕ, cψ, sθ, sϕ, sψ = symbols("cθ cϕ cψ sθ sϕ sψ", real = true) ;

julia> d = Dict(cos(θ) => cθ, cos(ϕ) => cϕ, cos(ψ) => cψ, sin(θ) => sθ, sin(ϕ) => sϕ, sin(ψ) => sψ) ;

julia> R = Tensnd(subs.(rot3(θ, ϕ, ψ),d...))
TensND.TensndCanonical{2, 3, Sym, Tensor{2, 3, Sym, 9}}
# data: 3×3 Tensor{2, 3, Sym, 9}:
 cθ⋅cψ⋅cϕ - sψ⋅sϕ  -cθ⋅cϕ⋅sψ - cψ⋅sϕ  cϕ⋅sθ
 cθ⋅cψ⋅sϕ + cϕ⋅sψ  -cθ⋅sψ⋅sϕ + cψ⋅cϕ  sθ⋅sϕ
           -cψ⋅sθ              sθ⋅sψ     cθ
# var: (:cont, :cont)
# basis: 3×3 TensND.LazyIdentity{3, Sym}:
 1  0  0
 0  1  0
 0  0  1

julia> RR = R ⊠ˢ R
TensND.TensndCanonical{4, 3, Sym, SymmetricTensor{4, 3, Sym, 36}}
# data: 6×6 Matrix{Sym}:
                          (cθ*cψ*cϕ - sψ*sϕ)^2                            (-cθ*cϕ*sψ - cψ*sϕ)^2           cϕ^2*sθ^2                      √2⋅cϕ⋅sθ⋅(-cθ⋅cϕ⋅sψ - cψ⋅sϕ)                     √2⋅cϕ⋅sθ⋅(cθ⋅cψ⋅cϕ - sψ⋅sϕ)                                   √2⋅(cθ⋅cψ⋅cϕ - sψ⋅sϕ)⋅(-cθ⋅cϕ⋅sψ - cψ⋅sϕ)
                          (cθ*cψ*sϕ + cϕ*sψ)^2                            (-cθ*sψ*sϕ + cψ*cϕ)^2           sθ^2*sϕ^2                      √2⋅sθ⋅sϕ⋅(-cθ⋅sψ⋅sϕ + cψ⋅cϕ)                     √2⋅sθ⋅sϕ⋅(cθ⋅cψ⋅sϕ + cϕ⋅sψ)                                   √2⋅(cθ⋅cψ⋅sϕ + cϕ⋅sψ)⋅(-cθ⋅sψ⋅sϕ + cψ⋅cϕ)
                                     cψ^2*sθ^2                                        sθ^2*sψ^2                cθ^2                                       √2⋅cθ⋅sθ⋅sψ                                    -√2⋅cθ⋅cψ⋅sθ                                                              -sqrt(2)*cψ*sθ^2*sψ
             -√2⋅cψ⋅sθ⋅(cθ⋅cψ⋅sϕ + cϕ⋅sψ)                √2⋅sθ⋅sψ⋅(-cθ⋅sψ⋅sϕ + cψ⋅cϕ)    √2⋅cθ⋅sθ⋅sϕ                    cθ*(-cθ*sψ*sϕ + cψ*cϕ) + sθ^2*sψ*sϕ                   cθ*(cθ*cψ*sϕ + cϕ*sψ) - cψ*sθ^2*sϕ                            -cψ⋅sθ⋅(-cθ⋅sψ⋅sϕ + cψ⋅cϕ) + sθ⋅sψ⋅(cθ⋅cψ⋅sϕ + cϕ⋅sψ)
             -√2⋅cψ⋅sθ⋅(cθ⋅cψ⋅cϕ - sψ⋅sϕ)                √2⋅sθ⋅sψ⋅(-cθ⋅cϕ⋅sψ - cψ⋅sϕ)    √2⋅cθ⋅cϕ⋅sθ                    cθ*(-cθ*cϕ*sψ - cψ*sϕ) + cϕ*sθ^2*sψ                   cθ*(cθ*cψ*cϕ - sψ*sϕ) - cψ*cϕ*sθ^2                            -cψ⋅sθ⋅(-cθ⋅cϕ⋅sψ - cψ⋅sϕ) + sθ⋅sψ⋅(cθ⋅cψ⋅cϕ - sψ⋅sϕ)
 √2⋅(cθ⋅cψ⋅cϕ - sψ⋅sϕ)⋅(cθ⋅cψ⋅sϕ + cϕ⋅sψ)  √2⋅(-cθ⋅cϕ⋅sψ - cψ⋅sϕ)⋅(-cθ⋅sψ⋅sϕ + cψ⋅cϕ)  sqrt(2)*cϕ*sθ^2*sϕ  cϕ⋅sθ⋅(-cθ⋅sψ⋅sϕ + cψ⋅cϕ) + sθ⋅sϕ⋅(-cθ⋅cϕ⋅sψ - cψ⋅sϕ)  cϕ⋅sθ⋅(cθ⋅cψ⋅sϕ + cϕ⋅sψ) + sθ⋅sϕ⋅(cθ⋅cψ⋅cϕ - sψ⋅sϕ)  (cθ*cψ*cϕ - sψ*sϕ)*(-cθ*sψ*sϕ + cψ*cϕ) + (cθ*cψ*sϕ + cϕ*sψ)*(-cθ*cϕ*sψ - cψ*sϕ)
# var: (:cont, :cont, :cont, :cont)
# basis: 3×3 TensND.LazyIdentity{3, Sym}:
 1  0  0
 0  1  0
 0  0  1

julia> R6 = invKM(subs.(KM(rot6(θ, ϕ, ψ)),d...))
TensND.TensndCanonical{4, 3, Sym, SymmetricTensor{4, 3, Sym, 36}}
# data: 6×6 Matrix{Sym}:
                          (cθ*cψ*cϕ - sψ*sϕ)^2                            (-cθ*cϕ*sψ - cψ*sϕ)^2           cϕ^2*sθ^2                      √2⋅cϕ⋅sθ⋅(-cθ⋅cϕ⋅sψ - cψ⋅sϕ)                     √2⋅cϕ⋅sθ⋅(cθ⋅cψ⋅cϕ - sψ⋅sϕ)                                   √2⋅(cθ⋅cψ⋅cϕ - sψ⋅sϕ)⋅(-cθ⋅cϕ⋅sψ - cψ⋅sϕ)
                          (cθ*cψ*sϕ + cϕ*sψ)^2                            (-cθ*sψ*sϕ + cψ*cϕ)^2           sθ^2*sϕ^2                      √2⋅sθ⋅sϕ⋅(-cθ⋅sψ⋅sϕ + cψ⋅cϕ)                     √2⋅sθ⋅sϕ⋅(cθ⋅cψ⋅sϕ + cϕ⋅sψ)                                   √2⋅(cθ⋅cψ⋅sϕ + cϕ⋅sψ)⋅(-cθ⋅sψ⋅sϕ + cψ⋅cϕ)
                                     cψ^2*sθ^2                                        sθ^2*sψ^2                cθ^2                                       √2⋅cθ⋅sθ⋅sψ                                    -√2⋅cθ⋅cψ⋅sθ                                                              -sqrt(2)*cψ*sθ^2*sψ
             -√2⋅cψ⋅sθ⋅(cθ⋅cψ⋅sϕ + cϕ⋅sψ)                √2⋅sθ⋅sψ⋅(-cθ⋅sψ⋅sϕ + cψ⋅cϕ)    √2⋅cθ⋅sθ⋅sϕ                    cθ*(-cθ*sψ*sϕ + cψ*cϕ) + sθ^2*sψ*sϕ                   cθ*(cθ*cψ*sϕ + cϕ*sψ) - cψ*sθ^2*sϕ                            -cψ⋅sθ⋅(-cθ⋅sψ⋅sϕ + cψ⋅cϕ) + sθ⋅sψ⋅(cθ⋅cψ⋅sϕ + cϕ⋅sψ)
             -√2⋅cψ⋅sθ⋅(cθ⋅cψ⋅cϕ - sψ⋅sϕ)                √2⋅sθ⋅sψ⋅(-cθ⋅cϕ⋅sψ - cψ⋅sϕ)    √2⋅cθ⋅cϕ⋅sθ                    cθ*(-cθ*cϕ*sψ - cψ*sϕ) + cϕ*sθ^2*sψ                   cθ*(cθ*cψ*cϕ - sψ*sϕ) - cψ*cϕ*sθ^2                            -cψ⋅sθ⋅(-cθ⋅cϕ⋅sψ - cψ⋅sϕ) + sθ⋅sψ⋅(cθ⋅cψ⋅cϕ - sψ⋅sϕ)
 √2⋅(cθ⋅cψ⋅cϕ - sψ⋅sϕ)⋅(cθ⋅cψ⋅sϕ + cϕ⋅sψ)  √2⋅(-cθ⋅cϕ⋅sψ - cψ⋅sϕ)⋅(-cθ⋅sψ⋅sϕ + cψ⋅cϕ)  sqrt(2)*cϕ*sθ^2*sϕ  cϕ⋅sθ⋅(-cθ⋅sψ⋅sϕ + cψ⋅cϕ) + sθ⋅sϕ⋅(-cθ⋅cϕ⋅sψ - cψ⋅sϕ)  cϕ⋅sθ⋅(cθ⋅cψ⋅sϕ + cϕ⋅sψ) + sθ⋅sϕ⋅(cθ⋅cψ⋅cϕ - sψ⋅sϕ)  (cθ*cψ*cϕ - sψ*sϕ)*(-cθ*sψ*sϕ + cψ*cϕ) + (cθ*cψ*sϕ + cϕ*sψ)*(-cθ*cϕ*sψ - cψ*sϕ)
# var: (:cont, :cont, :cont, :cont)
# basis: 3×3 TensND.LazyIdentity{3, Sym}:
 1  0  0
 0  1  0
 0  0  1

julia> R6 == RR
true
```
"""
function rot6(θ, ϕ = 0, ψ = 0)
    R = Tensnd(rot3(θ, ϕ, ψ))
    return sboxtimes(R, R)
end




const t𝟏 = tensId2
const t𝟙 = tensId4
const t𝕀 = tensId4s
const t𝕁 = tensJ4
const t𝕂 = tensK4
