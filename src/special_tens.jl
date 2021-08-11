"""
    fϵ(T, i::Int, j::Int, k::Int)
    fϵ(i::Int, j::Int, k::Int) = fϵ(Int, i::Int, j::Int, k::Int)

Function giving Levi-Civita symbol `ϵᵢⱼₖ=(i-j)(j-k)(k-i)/2`
"""
fϵ(i::Int, j::Int, k::Int, ::Type{<:T} = Int) where {T} = T(T((i-j)*(j-k)*(k-i))/T(2))

"""
    ϵ[i,j,k]

Levi-Civita symbol `ϵᵢⱼₖ=(i-j)(j-k)(k-i)/2`
"""
global const ϵ = [fϵ(i,j,k) for i=1:3, j=1:3, k=1:3]
