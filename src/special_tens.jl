"""
    fϵ(T, i::Int, j::Int, k::Int)
    fϵ(i::Int, j::Int, k::Int) = fϵ(Int, i::Int, j::Int, k::Int)

Function giving Levi-Civita symbol `ϵᵢⱼₖ = (i-j) (j-k) (k-i) / 2`
"""
fϵ(i::Int, j::Int, k::Int, ::Type{<:T} = Int) where {T} =
    T(T((i - j) * (j - k) * (k - i)) / T(2))

"""
    ϵ[i,j,k]

Levi-Civita symbol `ϵᵢⱼₖ=(i-j)(j-k)(k-i)/2`
"""
const ϵ = [fϵ(i, j, k) for i = 1:3, j = 1:3, k = 1:3]


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
    Tensnd(one(SymmetricTensor{2,dim,T}), (:cont, :cont), CanonicalBasis{dim,T}())

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
    Tensnd(one(Tensor{4,dim,T}), (:cont, :cont, :cont, :cont), CanonicalBasis{dim,T}())

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
tensId4s(T::Type{<:Number} = Sym, dim = 3) = Tensnd(
    one(SymmetricTensor{4,dim,T}),
    (:cont, :cont, :cont, :cont),
    CanonicalBasis{dim,T}(),
)

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
    return Tensnd(δ ⊗ δ / dim, (:cont, :cont, :cont, :cont), CanonicalBasis{dim,T}())
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
    𝐞p(i::Int, θ::T = zero(Sym))

Vector of the polar basis

# Examples
```julia
julia> θ = symbols("θ", real = true) ;

julia> 𝐞p(1, θ)
Tensnd{1, 2, Sym, Sym, Vec{2, Sym}, CanonicalBasis{2, Sym}}
# data: 2-element Vec{2, Sym}:
 cos(θ)
 sin(θ)
# var: (:cont,)
# basis: 2×2 Tensor{2, 2, Sym, 4}:
 1  0
 0  1
``` 
"""
𝐞p(::Val{1}, θ::T = zero(Sym)) where {T<:Number} = Tensnd(Vec{2}([cos(θ), sin(θ)]))
𝐞p(::Val{2}, θ::T = zero(Sym)) where {T<:Number} = Tensnd(Vec{2}([-sin(θ), cos(θ)]))

𝐞c(::Val{1}, θ::T = zero(Sym)) where {T<:Number} = Tensnd(Vec{3}([cos(θ), sin(θ), zero(T)]))
𝐞c(::Val{2}, θ::T = zero(Sym)) where {T<:Number} = Tensnd(Vec{3}([-sin(θ), cos(θ), zero(T)]))
𝐞c(::Val{3}, θ::T = zero(Sym)) where {T<:Number} = Tensnd(Vec{3}([zero(T), zero(T), one(T)]))

𝐞s(::Val{1}, θ::T = zero(Sym), ϕ::T = zero(Sym), ψ::T = zero(Sym)) where {T<:Number} =
    Tensnd(
        Vec{3}([
            -sin(ψ) ⋅ sin(ϕ) + cos(θ) ⋅ cos(ψ) ⋅ cos(ϕ),
            sin(ψ) ⋅ cos(ϕ) + sin(ϕ) ⋅ cos(θ) ⋅ cos(ψ),
            -sin(θ) ⋅ cos(ψ),
        ]),
    )

𝐞s(::Val{2}, θ::T = zero(Sym), ϕ::T = zero(Sym), ψ::T = zero(Sym)) where {T<:Number} =
    Tensnd(
        Vec{3}([
            -sin(ψ) ⋅ cos(θ) ⋅ cos(ϕ) - sin(ϕ) ⋅ cos(ψ),
            -sin(ψ) ⋅ sin(ϕ) ⋅ cos(θ) + cos(ψ) ⋅ cos(ϕ),
            sin(θ) ⋅ sin(ψ),
        ]),
    )

𝐞s(::Val{3}, θ::T = zero(Sym), ϕ::T = zero(Sym), ψ::T = zero(Sym)) where {T<:Number} =
    Tensnd(Vec{3}([sin(θ) ⋅ cos(ϕ), sin(θ) ⋅ sin(ϕ), cos(θ)]))


for eb in (:𝐞p, :𝐞c, :𝐞s)
    @eval $eb(i::Int, args...) = $eb(Val(i), args...)
end

for eb in (:𝐞, :𝐞p, :𝐞c, :𝐞s)
    @eval begin 
        $(Symbol(eb,eb))(i::Int, j::Int, args...) = $eb(i, args...) ⊗ $eb(j, args...)
        $(Symbol(eb,eb,"s"))(i::Int, j::Int, args...) = $eb(i, args...) ⊗ˢ $eb(j, args...)
    end
end
    

    
        




const t𝟏 = tensId2
const t𝟙 = tensId4
const t𝕀 = tensId4s
const t𝕁 = tensJ4
const t𝕂 = tensK4
