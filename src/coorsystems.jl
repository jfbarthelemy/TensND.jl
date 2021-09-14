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
∂(t::AbstractTensnd{order,dim,Sym,A}, xᵢ...) where {order,dim,A} =
    change_tens(Tensnd(diff.(components_canon(t), xᵢ...)), getbasis(t), getvar(t))

∂(t::Sym, xᵢ...) = diff(t, xᵢ...)



struct CoorSystemSym{dim} <: AbstractCoorSystem{dim,Sym}
    OM::AbstractTensnd{1,dim,Sym}
    coords::NTuple{dim,Sym}
    bnorm::AbstractBasis{dim,Sym}
    aᵢ::NTuple{dim,AbstractTensnd}
    χᵢ::NTuple{dim}
    aⁱ::NTuple{dim,AbstractTensnd}
    eᵢ::NTuple{dim,AbstractTensnd}
    rules::Dict
    tmp_var::Dict
    to_coords::Dict
    function CoorSystemSym(
        OM::AbstractTensnd{1,dim,Sym},
        coords::NTuple{dim,Sym};
        rules::Dict = Dict(),
        tmp_var::Dict = Dict(),
        to_coords::Dict = Dict(),
    ) where {dim}
        simp(t) =
            length(rules) > 0 ? tenssimp(tenssubs(tenssimp(t), rules...)) : tenssimp(t)
        chvar(t, d) = length(d) > 0 ? tenssubs(t, d...) : t
        OMc = chvar(OM, to_coords)
        aᵢ = ntuple(i -> simp(chvar(∂(OMc, coords[i]), tmp_var)), dim)
        χᵢ = ntuple(i -> simp(norm(aᵢ[i])), dim)
        eᵢ = ntuple(i -> simp(aᵢ[i] / χᵢ[i]), dim)
        χᵢ = ntuple(i -> simp(chvar(χᵢ[i], to_coords)), dim)
        eᵢ = ntuple(i -> simp(chvar(eᵢ[i], to_coords)), dim)
        bnorm = Basis(tenssimp(hcat(components_canon.(eᵢ)...)))
        eᵢ = ntuple(
            i -> Tensnd(Vec{dim}(j -> j == i ? one(Sym) : zero(Sym)), bnorm, (:cov,)),
            dim,
        )
        aᵢ = ntuple(
            i -> Tensnd(Vec{dim}(j -> j == i ? χᵢ[i] : zero(Sym)), bnorm, (:cov,)),
            dim,
        )
        aⁱ = ntuple(
            i -> Tensnd(Vec{dim}(j -> j == i ? 1 / χᵢ[i] : zero(Sym)), bnorm, (:cont,)),
            dim,
        )
        new{dim}(OMc, coords, bnorm, aᵢ, χᵢ, aⁱ, eᵢ, rules, tmp_var, to_coords)
    end
end

with_tmp_var(CS::CoorSystemSym, t) = length(CS.tmp_var) > 0 ? tenssubs(t, CS.tmp_var...) : t
only_coords(CS::CoorSystemSym, t) = length(CS.to_coords) > 0 ? tenssubs(t, CS.to_coords...) : t

getcoords(CS::CoorSystemSym) = CS.coords
getcoords(CS::CoorSystemSym, i::Int) = getcoords(CS)[i]

getOM(CS::CoorSystemSym) = CS.OM

getbnorm(CS::CoorSystemSym) = CS.bnorm

natvec(CS::CoorSystemSym, ::Val{:cov}) = CS.aᵢ
natvec(CS::CoorSystemSym, ::Val{:cont}) = CS.aⁱ
natvec(CS::CoorSystemSym, var = :cov) = natvec(CS, Val(var))
natvec(CS::CoorSystemSym, i::Int, var = :cov) = natvec(CS, var)[i]

unitvec(CS::CoorSystemSym) = CS.eᵢ
unitvec(CS::CoorSystemSym, i::Int) = unitvec(CS)[i]



GRAD(
    T::Union{Sym,AbstractTensnd{order,dim,Sym}},
    CS::CoorSystemSym{dim},
) where {order,dim} = tenssimp(sum([∂(only_coords(CS,T), getcoords(CS, i)) ⊗ natvec(CS, i, :cont) for i = 1:dim]))

SYMGRAD(
    T::Union{Sym,AbstractTensnd{order,dim,Sym}},
    CS::CoorSystemSym{dim},
) where {order,dim} = tenssimp(sum([∂(only_coords(CS,T), getcoords(CS, i)) ⊗ˢ natvec(CS, i, :cont) for i = 1:dim]))

DIV(T::Union{AbstractTensnd{order,dim,Sym}}, CS::CoorSystemSym{dim}) where {order,dim} =
    tenssimp(sum([∂(only_coords(CS,T), getcoords(CS, i)) ⋅ natvec(CS, i, :cont) for i = 1:dim]))

LAPLACE(
    T::Union{Sym,AbstractTensnd{order,dim,Sym}},
    CS::CoorSystemSym{dim},
) where {order,dim} = DIV(GRAD(T, CS), CS)

HESS(
    T::Union{Sym,AbstractTensnd{order,dim,Sym}},
    CS::CoorSystemSym{dim},
) where {order,dim} = GRAD(GRAD(T, CS), CS)




"""
    init_cartesian(coords = symbols("x y z", real = true))

Returns the coordinates, unit vectors and basis of the cartesian basis

# Examples
```julia
julia> coords, vectors, ℬ = init_cartesian() ; x, y, z = coords ; 𝐞₁, 𝐞₂, 𝐞₃ = vectors ;
``` 
"""
init_cartesian(coords = symbols("x y z", real = true)) = Tuple(coords),
ntuple(i -> 𝐞(i, length(coords), eltype(coords)), length(coords)),
CanonicalBasis{length(coords),eltype(coords)}()

init_cartesian(::Val{3}) = init_cartesian(symbols("x y z", real = true))
init_cartesian(::Val{2}) = init_cartesian(symbols("x y", real = true))
init_cartesian(dim::Int) = init_cartesian(Val(dim))

"""
    CS_cartesian(coords = symbols("x y z", real = true))

Returns the cartesian coordinate system, coordinates, unit vectors and basis

# Examples
```julia
julia> CScar, 𝐗, 𝐄, ℬ = CS_cartesian() ;

julia> 𝛔 = Tensnd(SymmetricTensor{2,3}((i, j) -> SymFunction("σ\$i\$j", real = true)(𝐗...))) ;

julia> DIV(𝛔, CScar)
TensND.TensndCanonical{1, 3, Sym, Vec{3, Sym}}
# data: 3-element Vec{3, Sym}:
 Derivative(σ11(x, y, z), x) + Derivative(σ21(x, y, z), y) + Derivative(σ31(x, y, z), z)
 Derivative(σ21(x, y, z), x) + Derivative(σ22(x, y, z), y) + Derivative(σ32(x, y, z), z)
 Derivative(σ31(x, y, z), x) + Derivative(σ32(x, y, z), y) + Derivative(σ33(x, y, z), z)
# basis: 3×3 TensND.LazyIdentity{3, Sym}:
 1  0  0
 0  1  0
 0  0  1
# var: (:cont,)
``` 
"""
function CS_cartesian(coords = symbols("x y z", real = true))
    (x, y, z), (𝐞₁, 𝐞₂, 𝐞₃), ℬ = init_cartesian(coords)
    OM = x * 𝐞₁ + y * 𝐞₂ + z * 𝐞₃
    CS = CoorSystemSym(OM, coords)
    return CS, (x, y, z), (𝐞₁, 𝐞₂, 𝐞₃), ℬ
end


"""
    init_polar(coords = (symbols("r", positive = true), symbols("θ", real = true)); canonical = false)

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
    CS_polar(coords = (symbols("r", positive = true), symbols("θ", real = true)); canonical = false)

Returns the polar coordinate system, coordinates, unit vectors and basis

# Examples
```julia
julia> Polar, (r, θ), (𝐞ʳ, 𝐞ᶿ), ℬᵖ = CS_polar() ;

julia> f = SymFunction("f", real = true)(r, θ) ;

julia> LAPLACE(f, Polar)
                               2
                              ∂
                             ───(f(r, θ))
                               2
               ∂             ∂θ
  2            ──(f(r, θ)) + ────────────
 ∂             ∂r                 r
───(f(r, θ)) + ──────────────────────────
  2                        r
∂r
``` 
"""
function CS_polar(
    coords = (symbols("r", positive = true), symbols("θ", real = true));
    canonical = false,
)
    (r, θ), (𝐞ʳ, 𝐞ᶿ), ℬᵖ = init_polar(coords, canonical = canonical)
    OM = r * 𝐞ʳ
    CS = CoorSystemSym(OM, coords)
    return CS, (r, θ), (𝐞ʳ, 𝐞ᶿ), ℬᵖ
end

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
    CS_cylindrical(coords = (symbols("r", positive = true), symbols("θ", real = true), symbols("z", real = true)); canonical = false)

Returns the cylindrical coordinate system, coordinates, unit vectors and basis

# Examples
```julia
julia> Cylindrical, rθz, (𝐞ʳ, 𝐞ᶿ, 𝐞ᶻ), ℬᶜ = CS_cylindrical() ;

julia> 𝐯 = Tensnd(Vec{3}(i -> SymFunction("v\$(rθz[i])", real = true)(rθz...)), ℬᶜ) ;

julia> DIV(𝐯, Cylindrical)
                                                  ∂
                                    vr(r, θ, z) + ──(vθ(r, θ, z))
∂                 ∂                               ∂θ
──(vr(r, θ, z)) + ──(vz(r, θ, z)) + ─────────────────────────────
∂r                ∂z                              r
``` 
"""
function CS_cylindrical(
    coords = (
        symbols("r", positive = true),
        symbols("θ", real = true),
        symbols("z", real = true),
    );
    canonical = false,
)
    (r, θ, z), (𝐞ʳ, 𝐞ᶿ, 𝐞ᶻ), ℬᶜ = init_cylindrical(coords, canonical = canonical)
    OM = r * 𝐞ʳ + z * 𝐞ᶻ
    CS = CoorSystemSym(OM, coords)
    return CS, (r, θ, z), (𝐞ʳ, 𝐞ᶿ, 𝐞ᶻ), ℬᶜ
end



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
    CS_spherical(coords = (symbols("θ", real = true), symbols("ϕ", real = true), symbols("r", positive = true)); canonical = false)

Returns the spherical coordinate system, coordinates, unit vectors and basis

# Examples
```julia
julia> Spherical, (θ, ϕ, r), (𝐞ᶿ, 𝐞ᵠ, 𝐞ʳ), ℬˢ = CS_spherical() ;

julia> for σⁱʲ ∈ ("σʳʳ", "σᶿᶿ", "σᵠᵠ") @eval \$(Symbol(σⁱʲ)) = SymFunction(\$σⁱʲ, real = true)(\$r) end ;

julia> 𝛔 = σʳʳ * 𝐞ʳ ⊗ 𝐞ʳ + σᶿᶿ * 𝐞ᶿ ⊗ 𝐞ᶿ + σᵠᵠ * 𝐞ᵠ ⊗ 𝐞ᵠ ;

julia> div𝛔 = DIV(𝛔, Spherical)
TensND.TensndRotated{1, 3, Sym, Vec{3, Sym}}
# data: 3-element Vec{3, Sym}:
                              (-σᵠᵠ(r) + σᶿᶿ(r))*cos(θ)/(r*sin(θ))
                                                                 0
 Derivative(σʳʳ(r), r) + (σʳʳ(r) - σᵠᵠ(r))/r + (σʳʳ(r) - σᶿᶿ(r))/r
# basis: 3×3 Tensor{2, 3, Sym, 9}:
 cos(θ)⋅cos(ϕ)  -sin(ϕ)  sin(θ)⋅cos(ϕ)
 sin(ϕ)⋅cos(θ)   cos(ϕ)  sin(θ)⋅sin(ϕ)
       -sin(θ)        0         cos(θ)
# var: (:cont,)

julia> div𝛔 ⋅ 𝐞ʳ
d            σʳʳ(r) - σᵠᵠ(r)   σʳʳ(r) - σᶿᶿ(r)
──(σʳʳ(r)) + ─────────────── + ───────────────
dr                  r                 r
``` 
"""
function CS_spherical(
    coords = (
        symbols("θ", real = true),
        symbols("ϕ", real = true),
        symbols("r", positive = true),
    );
    canonical = false,
)
    (θ, ϕ, r), (𝐞ᶿ, 𝐞ᵠ, 𝐞ʳ), ℬˢ = init_spherical(coords, canonical = canonical)
    OM = r * 𝐞ʳ
    rules = Dict(abs(sin(θ)) => sin(θ))
    CS = CoorSystemSym(OM, coords; rules = rules)
    return CS, (θ, ϕ, r), (𝐞ᶿ, 𝐞ᵠ, 𝐞ʳ), ℬˢ
end



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
