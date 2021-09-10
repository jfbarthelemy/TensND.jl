abstract type AbstractCoorSystem{dim,T<:Number} <: Any end


"""
    ∂(t::AbstractTensnd{order,dim,Sym,A},xᵢ::Sym)

Returns the derivative of the tensor `t` with respect to the variable `x_i`

# Examples
```julia

julia> θ, ϕ, ℬˢ, 𝐞ᶿ, 𝐞ᵠ, 𝐞ʳ = init_spherical(symbols("θ ϕ", real = true)...) ;

julia> ∂(𝐞ʳ, ϕ) == sin(θ) * 𝐞ᵠ
true

julia> ∂(𝐞ʳ ⊗ 𝐞ʳ,θ)
TensND.TensndRotated{2, 3, Sym, SymmetricTensor{2, 3, Sym, 6}}
# data: 3×3 SymmetricTensor{2, 3, Sym, 6}:
 0  0  1
 0  0  0
 1  0  0
# basis: 3×3 Tensor{2, 3, Sym, 9}:
 cos(θ)⋅cos(ϕ)  -sin(ϕ)  sin(θ)⋅cos(ϕ)
 sin(ϕ)⋅cos(θ)   cos(ϕ)  sin(θ)⋅sin(ϕ)
       -sin(θ)        0         cos(θ)
# var: (:cont, :cont)
```
"""
∂(t::AbstractTensnd{order,dim,Sym,A},xᵢ::Sym) where {order,dim,A} =
    change_tens(Tensnd(diff.(components_canon(t), xᵢ)), getbasis(t), getvar(t))

