"""
    fϵ(T, i::Int, j::Int, k::Int)
    fϵ(i::Int, j::Int, k::Int) = fϵ(Int, i::Int, j::Int, k::Int)

Function giving Levi-Civita symbol `ϵᵢⱼₖ = (i-j) (j-k) (k-i) / 2`
"""
fϵ(i::Int, j::Int, k::Int, ::Type{<:T} = Int) where {T} = T(T((i-j)*(j-k)*(k-i))/T(2))

"""
    ϵ[i,j,k]

Levi-Civita symbol `ϵᵢⱼₖ=(i-j)(j-k)(k-i)/2`
"""
const ϵ = [fϵ(i,j,k) for i=1:3, j=1:3, k=1:3]



tensId2(T::Type{<:Number} = Sym, dim = 3) = Tensnd(one(SymmetricTensor{2, dim, T}), (:cont, :cont), CanonicalBasis{dim,T}())

tensId4(T::Type{<:Number} = Sym, dim = 3) = Tensnd(one(Tensor{4, dim, T}), (:cont, :cont, :cont, :cont), CanonicalBasis{dim,T}())

tensId4s(T::Type{<:Number} = Sym, dim = 3) = Tensnd(one(SymmetricTensor{4, dim, T}), (:cont, :cont, :cont, :cont), CanonicalBasis{dim,T}())

function tensJ4(T::Type{<:Number} = Sym, dim = 3)
    δ = one(SymmetricTensor{2, dim, T})
    return Tensnd(δ⊗δ/dim, (:cont, :cont, :cont, :cont), CanonicalBasis{dim,T}())
end

tensK4(T::Type{<:Number} = Sym, dim = 3) = tensId4s(T, dim) - tensJ4(T, dim)

const 𝟏 = tensId2
const 𝟙 = tensId4
const 𝕀 = tensId4s
const 𝕁 = tensJ4
const 𝕂 = tensK4
