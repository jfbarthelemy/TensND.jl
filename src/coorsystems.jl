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

∂(t::Sym,xᵢ::Sym) = diff(t, xᵢ)


struct CoorSystemSym{dim} <: AbstractCoorSystem{dim,Sym}
    OM::AbstractTensnd{1,dim,Sym}
    x::AbstractVector{Sym}
    basis::AbstractBasis{dim,Sym}
    bnorm::AbstractBasis{dim,Sym}
    aᵢ::NTuple{dim,AbstractTensnd}
    aⁱ::NTuple{dim,AbstractTensnd}
    eᵢ::NTuple{dim,AbstractTensnd}
    function CoorSystemSym(OM::AbstractTensnd{1,dim,Sym}, x::AbstractVector{Sym} ; simp::Dict = Dict()) where {dim}
        var = getvar(OM)
        ℬ = getbasis(OM)
        aᵢ = ntuple(i -> ∂(OM,x[i]), dim)
        e = Tensor{2,dim}(hcat(components.(aᵢ)...))
        # g = SymmetricTensor{2,dim}(simplify.(e' ⋅ e))
        # G = inv(g)
        # E = e ⋅ G'
        E = inv(e)'
        aⁱ = ntuple(i -> Tensnd(E[:,i], ℬ, invvar.(var)), dim)
        basis = Basis(simplify.(subs.(simplify.(hcat(components_canon.(aᵢ)...)), simp...)))
        eᵢ = ntuple(i -> Tensnd(simplify.(subs.(simplify.(aᵢ[i] / norm(aᵢ[i])), simp...)), ℬ, invvar.(var)), dim)
        bnorm = Basis(simplify.(subs.(simplify.(hcat(components_canon.(eᵢ)...)), simp...)))
        new{dim}(OM,x,basis,bnorm,aᵢ,aⁱ,eᵢ)
    end
end