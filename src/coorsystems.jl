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
∂(t::AbstractTensnd{order,dim,Sym,A}, xᵢ::Sym) where {order,dim,A} =
    change_tens(Tensnd(diff.(components_canon(t), xᵢ)), getbasis(t), getvar(t))

∂(t::Sym, xᵢ::Sym) = diff(t, xᵢ)

"""
    init_canonical(coords = symbols("x y z", real = true))

Returns the coordinates, base vectors and basis of the canonical basis

# Examples
```julia
julia> coords, vectors, ℬ = init_canonical() ; x, y, z = coords ; 𝐞₁, 𝐞₂, 𝐞₃ = vectors ;
``` 
"""
init_canonical(coords = symbols("x y z", real = true)) = Tuple(coords),
ntuple(i -> 𝐞(i, length(coords), eltype(coords)), length(coords)),
CanonicalBasis{length(coords),eltype(coords)}()

init_canonical(::Val{3}) = init_canonical(symbols("x y z", real = true))
init_canonical(::Val{2}) = init_canonical(symbols("x y", real = true))
init_canonical(dim::Int) = init_canonical(Val(dim))



"""
    init_polar(θ ; canonical = false)

Returns the coordinates, base vectors and basis of the polar basis

# Examples
```julia
julia> coords, vectors, ℬᵖ = init_polar() ; r, θ = coords ; 𝐞ʳ, 𝐞ᶿ = vectors ;
``` 
"""
init_polar(
    coords = (symbols("r", positive = true), symbols("θ", real = true));
    canonical = false,
) = Tuple(coords), ntuple(i -> 𝐞ᵖ(i, coords[2]; canonical = canonical), 2), Basis(coords[2])

"""
    init_cylindrical(coords = (symbols("r", positive = true), symbols("θ", real = true), symbols("z", real = true)); canonical = false)

Returns the coordinates, base vectors and basis of the cylindrical basis

# Examples
```julia
julia> coords, vectors, ℬᶜ = init_cylindrical() ; r, θ, z = coords ; 𝐞ʳ, 𝐞ᶿ, 𝐞ᶻ = vectors ;
``` 
"""
init_cylindrical(
    coords = (
        symbols("r", positive = true),
        symbols("θ", real = true),
        symbols("z", real = true),
    );
    canonical = false,
) = Tuple(coords),
ntuple(i -> 𝐞ᶜ(i, coords[2]; canonical = canonical), 3),
CylindricalBasis(coords[2])



"""
    init_spherical(coords = (symbols("θ", real = true), symbols("ϕ", real = true), symbols("r", positive = true)); canonical = false)

Returns the coordinates, base vectors and basis of the spherical basis.
Take care that the order of the 3 vectors is `𝐞ᶿ, 𝐞ᵠ, 𝐞ʳ` so that
the basis coincides with the canonical one when the angles are null and in consistency
the coordinates are ordered as `θ, ϕ, r`.

# Examples
```julia
julia> coords, vectors, ℬˢ = init_spherical() ; θ, ϕ, r = coords ; 𝐞ᶿ, 𝐞ᵠ, 𝐞ʳ  = vectors ;
``` 
"""
init_spherical(
    coords = (
        symbols("θ", real = true),
        symbols("ϕ", real = true),
        symbols("r", positive = true),
    );
    canonical = false,
) = Tuple(coords),
ntuple(i -> 𝐞ˢ(i, coords[1:2]...; canonical = canonical), 3),
SphericalBasis(coords[1:2]...)


"""
    init_rotated(coords = symbols("θ ϕ ψ", real = true); canonical = false)

Returns the angles, base vectors and basis of the rotated basis.
Note that here the coordinates are angles and do not represent a valid parametrization of `ℝ³`

# Examples
```julia
julia> angles, vectors, ℬʳ = init_rotated() ; θ, ϕ, ψ = angles ; 𝐞ᶿ, 𝐞ᵠ, 𝐞ʳ = vectors ;
```
"""
init_rotated(angles = symbols("θ ϕ ψ", real = true); canonical = false) =
    Tuple(angles), ntuple(i -> 𝐞ˢ(i, angles...; canonical = canonical), 3), Basis(angles...)


struct CoorSystemSym{dim} <: AbstractCoorSystem{dim,Sym}
    OM::AbstractTensnd{1,dim,Sym}
    coords::NTuple{dim,Sym}
    basis::AbstractBasis{dim,Sym}
    bnorm::AbstractBasis{dim,Sym}
    aᵢ::NTuple{dim,AbstractTensnd}
    aⁱ::NTuple{dim,AbstractTensnd}
    eᵢ::NTuple{dim,AbstractTensnd}
    eⁱ::NTuple{dim,AbstractTensnd}
    function CoorSystemSym(
        OM::AbstractTensnd{1,dim,Sym},
        coords::NTuple{dim,Sym};
        simp::Dict = Dict(),
    ) where {dim}
        sd = length(simp) > 0 ? x -> simplify(subs(simplify(x), simp...)) : x -> simplify(x)
        var = getvar(OM)
        ℬ = getbasis(OM)
        aᵢ = ntuple(i -> ∂(OM, coords[i]), dim)
        e = Tensor{2,dim}(sd.(hcat(components.(aᵢ)...)))
        # g = SymmetricTensor{2,dim}(simplify.(e' ⋅ e))
        # G = inv(g)
        # E = e ⋅ G'
        E = sd.(inv(e)')
        aⁱ = ntuple(i -> Tensnd(E[:, i], ℬ, invvar.(var)), dim)
        basis = Basis(sd.(hcat(components_canon.(aᵢ)...)))
        eᵢ = ntuple(i -> Tensnd(sd.(aᵢ[i] / norm(aᵢ[i])), ℬ, var), dim)
        bnorm = Basis(sd.(hcat(components_canon.(eᵢ)...)))
        eⁱ = ntuple(i -> Tensnd(sd.(aⁱ[i] / norm(aⁱ[i])), ℬ, invvar.(var)), dim)
        new{dim}(OM, coords, basis, bnorm, aᵢ, aⁱ, eᵢ, eⁱ)
    end
end
