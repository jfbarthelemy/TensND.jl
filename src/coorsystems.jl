abstract type AbstractCoorSystem{dim,T<:Number} <: Any end


"""
    CoorSystemSym(OM::AbstractTens{1,dim,Sym},coords::NTuple{dim,Sym},bnorm::AbstractBasis{dim,Sym},χᵢ::NTuple{dim},
                  tmp_coords::NTuple = (),params::NTuple = ();rules::Dict = Dict(),tmp_var::Dict = Dict(),to_coords::Dict = Dict()) where {dim}
    CoorSystemSym(OM::AbstractTens{1,dim,Sym},coords::NTuple{dim,Sym},
                  tmp_coords::NTuple = (),params::NTuple = ();rules::Dict = Dict(),tmp_var::Dict = Dict(),to_coords::Dict = Dict()) where {dim}

Defines a new coordinate system either from
1. the position vector `OM`, the coordinates `coords`, the basis of unit vectors (`𝐞ᵢ`) `bnorm` and the Lamé coefficients `χᵢ`

    In this case the natural basis is formed by the vectors `𝐚ᵢ = χᵢ 𝐞ᵢ` directly calculated from the input data.

1. or the position vector `OM` and the coordinates `coords`

    In this case the natural basis is formed by the vectors `𝐚ᵢ = ∂ᵢOM` i.e. by the derivative of the position vector with respect to the `iᵗʰ` coordinate

Optional parameters can be provided:
- `tmp_coords` contains temporary variables depending on coordinates (in order to allow symbolic simplifications)
- `params` contains possible parameters involved in `OM`
- `rules` contains a `Dict` with substitution rules to facilitate the simplification of formulas
- `tmp_var` contains a `Dict` with substitution of coordinates by temporary variables
- `to_coords` indicates how to eliminate the temporary variables to come back to the actual coordinates before derivation for Examples

# Examples
```julia
julia> ϕ, p = symbols("ϕ p", real = true) ;

julia> p̄, q, q̄, c = symbols("p̄ q q̄ c", positive = true) ;

julia> coords = (ϕ, p, q) ; tmp_coords = (p̄, q̄) ; params = (c,) ;

julia> OM = Tens(c * [p̄ * q̄ * cos(ϕ), p̄ * q̄ * sin(ϕ), p * q]) ;

julia> Spheroidal = CoorSystemSym(OM, coords, tmp_coords, params; tmp_var = Dict(1-p^2 => p̄^2, q^2-1 => q̄^2), to_coords = Dict(p̄ => √(1-p^2), q̄ => √(q^2-1))) ;
```
"""
struct CoorSystemSym{dim,VEC,BNORM,BNAT} <: AbstractCoorSystem{dim,Sym}
    OM::VEC
    coords::NTuple{dim,Sym}
    normalized_basis::BNORM
    natural_basis::BNAT
    aᵢ::NTuple{dim}
    χᵢ::NTuple{dim}
    aⁱ::NTuple{dim}
    eᵢ::NTuple{dim}
    Γ::Array{Sym,3}
    tmp_coords::NTuple
    params::NTuple
    rules::Dict
    tmp_var::Dict
    to_coords::Dict
    function CoorSystemSym(
        OM::VEC,
        coords::NTuple{dim,Sym},
        normalized_basis::AbstractBasis{dim,Sym},
        χᵢ::NTuple{dim},
        tmp_coords::NTuple = (),
        params::NTuple = ();
        rules::Dict = Dict(),
        tmp_var::Dict = Dict(),
        to_coords::Dict = Dict(),
    ) where {VEC,dim}
        simp(t) = length(rules) > 0 ? simplify(subs(simplify(t), rules...)) : simplify(t)
        eᵢ = ntuple(
            i -> Tens(
                Vec{dim}(j -> j == i ? one(Sym) : zero(Sym)),
                normalized_basis,
                (:cov,),
            ),
            dim,
        )
        aᵢ = ntuple(
            i -> Tens(Vec{dim}(j -> j == i ? χᵢ[i] : zero(Sym)), normalized_basis, (:cov,)),
            dim,
        )
        aⁱ = ntuple(
            i -> Tens(
                Vec{dim}(j -> j == i ? inv(χᵢ[i]) : zero(Sym)),
                normalized_basis,
                (:cont,),
            ),
            dim,
        )
        Γ = compute_Christoffel(
            coords,
            χᵢ,
            metric(normalized_basis, :cov),
            metric(normalized_basis, :cont),
        )
        natural_basis = Basis(normalized_basis, χᵢ)
        new{dim,typeof(OM),typeof(normalized_basis),typeof(natural_basis)}(
            OM,
            coords,
            normalized_basis,
            natural_basis,
            aᵢ,
            χᵢ,
            aⁱ,
            eᵢ,
            Γ,
            tmp_coords,
            params,
            rules,
            tmp_var,
            to_coords,
        )
    end
    function CoorSystemSym(
        OM::VEC,
        coords::NTuple{dim,Sym},
        tmp_coords::NTuple = (),
        params::NTuple = ();
        rules::Dict = Dict(),
        tmp_var::Dict = Dict(),
        to_coords::Dict = Dict(),
    ) where {VEC,dim}
        simp(t) = length(rules) > 0 ? simplify(subs(simplify(t), rules...)) : simplify(t)
        chvar(t, d) = length(d) > 0 ? subs(t, d...) : t
        OMc = chvar(OM, to_coords)
        aᵢ = ntuple(i -> simp(chvar(∂(OMc, coords[i]), tmp_var)), dim)
        χᵢ = ntuple(i -> simp(norm(aᵢ[i])), dim)
        eᵢ = ntuple(i -> simp(aᵢ[i] / χᵢ[i]), dim)
        χᵢ = ntuple(i -> simp(chvar(χᵢ[i], to_coords)), dim)
        eᵢ = ntuple(i -> simp(chvar(eᵢ[i], to_coords)), dim)
        normalized_basis = Basis(simplify(hcat(components_canon.(eᵢ)...)))
        eᵢ = ntuple(
            i -> Tens(
                Vec{dim}(j -> j == i ? one(Sym) : zero(Sym)),
                normalized_basis,
                (:cov,),
            ),
            dim,
        )
        aᵢ = ntuple(
            i -> Tens(Vec{dim}(j -> j == i ? χᵢ[i] : zero(Sym)), normalized_basis, (:cov,)),
            dim,
        )
        aⁱ = ntuple(
            i -> Tens(
                Vec{dim}(j -> j == i ? inv(χᵢ[i]) : zero(Sym)),
                normalized_basis,
                (:cont,),
            ),
            dim,
        )
        Γ = compute_Christoffel(
            coords,
            χᵢ,
            metric(normalized_basis, :cov),
            metric(normalized_basis, :cont),
        )
        natural_basis = Basis(normalized_basis, χᵢ)
        new{dim,typeof(OM),typeof(normalized_basis),typeof(natural_basis)}(
            OMc,
            coords,
            normalized_basis,
            natural_basis,
            aᵢ,
            χᵢ,
            aⁱ,
            eᵢ,
            Γ,
            tmp_coords,
            params,
            rules,
            tmp_var,
            to_coords,
        )
    end
end

with_tmp_var(CS::CoorSystemSym, t) = length(CS.tmp_var) > 0 ? subs(t, CS.tmp_var...) : t
only_coords(CS::CoorSystemSym, t) = length(CS.to_coords) > 0 ? subs(t, CS.to_coords...) : t

getcoords(CS::CoorSystemSym) = CS.coords
getcoords(CS::CoorSystemSym, i::Integer) = getcoords(CS)[i]

getOM(CS::CoorSystemSym) = CS.OM

get_normalized_basis(CS::CoorSystemSym) = CS.normalized_basis
get_natural_basis(CS::CoorSystemSym) = CS.natural_basis

getLame(CS::CoorSystemSym) = CS.χᵢ
getChristoffel(CS::CoorSystemSym) = CS.Γ

natvec(CS::CoorSystemSym, ::Val{:cov}) = CS.aᵢ
natvec(CS::CoorSystemSym, ::Val{:cont}) = CS.aⁱ
natvec(CS::CoorSystemSym, var = :cov) = natvec(CS, Val(var))
natvec(CS::CoorSystemSym, i::Integer, var = :cov) = natvec(CS, var)[i]

unitvec(CS::CoorSystemSym) = CS.eᵢ
unitvec(CS::CoorSystemSym, i::Integer) = unitvec(CS)[i]


function compute_Christoffel(coords, χ, γ, invγ)
    dim = length(χ)
    gᵢⱼ = [γ[i, j] * χ[i] * χ[j] for i ∈ 1:dim, j ∈ 1:dim]
    gⁱʲ = [invγ[i, j] / (χ[i] * χ[j]) for i ∈ 1:dim, j ∈ 1:dim]
    ∂g = [SymPy.diff(gᵢⱼ[i, j], coords[k]) for i ∈ 1:dim, j ∈ 1:dim, k ∈ 1:dim]
    Γᵢⱼₖ =
        [(∂g[i, k, j] + ∂g[j, k, i] - ∂g[i, j, k]) / 2 for i ∈ 1:dim, j ∈ 1:dim, k ∈ 1:dim]
    return ein"ijl,lk->ijk"(Γᵢⱼₖ, gⁱʲ)
end

"""
    ∂(t::AbstractTens{order,dim,Sym,A},xᵢ::Sym)

Returns the derivative of the tensor `t` with respect to the variable `x_i`

# Examples
```julia

julia> θ, ϕ, ℬˢ, 𝐞ᶿ, 𝐞ᵠ, 𝐞ʳ = init_spherical(symbols("θ ϕ", real = true)...) ;

julia> ∂(𝐞ʳ, ϕ) == sin(θ) * 𝐞ᵠ
true

julia> ∂(𝐞ʳ ⊗ 𝐞ʳ,θ)
Tens.TensRotated{2, 3, Sym, SymmetricTensor{2, 3, Sym, 6}}
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
∂(t::AbstractTens{order,dim,Sym}, xᵢ...) where {order,dim} =
    change_tens(Tens(SymPy.diff(components_canon(t), xᵢ...)), getbasis(t), getvar(t))

∂(t::Sym, xᵢ...) = SymPy.diff(t, xᵢ...)

function ∂(
    t::AbstractTens{order,dim,Sym},
    i::Integer,
    CS::CoorSystemSym{dim},
) where {order,dim}
    t = only_coords(CS, t)
    ℬ = get_natural_basis(CS)
    var = ntuple(_ -> :cont, order)
    t = Array(components(t, ℬ, var))
    Γ = getChristoffel(CS)
    data = diff.(t, getcoords(CS, i))
    for o ∈ 1:order
        ec1 = ntuple(j -> j == o ? order + 1 : j, order)
        ec2 = (order + 1, o)
        ec3 = ntuple(j -> j, order)
        data += einsum(EinCode((ec1, ec2), ec3), (t, view(Γ, i, :, :)))
    end
    return change_tens(Tens(data, ℬ, var), get_normalized_basis(CS), var)
end

∂(t::Sym, i::Integer, CS::CoorSystemSym{dim}) where {dim} =
    SymPy.diff(only_coords(CS, t), getcoords(CS, i))

function ∂(
    t::AbstractTens{order,dim,Sym},
    x::Sym,
    CS::CoorSystemSym{dim},
) where {order,dim}
    ind = findfirst(i -> i == x, getcoords(CS))
    return ind === nothing ? zero(t) : ∂(t, ind, CS)
end

function ∂(
    t::Sym,
    x::Sym,
    CS::CoorSystemSym{dim},
) where {order,dim}
    ind = findfirst(i -> i == x, getcoords(CS))
    return ind === nothing ? zero(t) : ∂(t, ind, CS)
end

"""
    GRAD(T::Union{Sym,AbstractTens{order,dim,Sym}},CS::CoorSystemSym{dim}) where {order,dim}

Calculates the gradient of `T` with respect to the coordinate system `CS`
"""
GRAD(T::Union{Sym,AbstractTens{order,dim,Sym}}, CS::CoorSystemSym{dim}) where {order,dim} =
    sum([∂(T, i, CS) ⊗ natvec(CS, i, :cont) for i = 1:dim])


"""
    SYMGRAD(T::Union{Sym,AbstractTens{order,dim,Sym}},CS::CoorSystemSym{dim}) where {order,dim}

Calculates the symmetrized gradient of `T` with respect to the coordinate system `CS`
"""
SYMGRAD(
    T::Union{Sym,AbstractTens{order,dim,Sym}},
    CS::CoorSystemSym{dim},
) where {order,dim} = sum([∂(T, i, CS) ⊗ˢ natvec(CS, i, :cont) for i = 1:dim])


"""
    DIV(T::AbstractTens{order,dim,Sym},CS::CoorSystemSym{dim}) where {order,dim}

Calculates the divergence  of `T` with respect to the coordinate system `CS`
"""
DIV(T::AbstractTens{order,dim,Sym}, CS::CoorSystemSym{dim}) where {order,dim} =
    sum([∂(T, i, CS) ⋅ natvec(CS, i, :cont) for i = 1:dim])



"""
    LAPLACE(T::Union{Sym,AbstractTens{order,dim,Sym}},CS::CoorSystemSym{dim}) where {order,dim}

Calculates the Laplace operator of `T` with respect to the coordinate system `CS`
"""
LAPLACE(
    T::Union{Sym,AbstractTens{order,dim,Sym}},
    CS::CoorSystemSym{dim},
) where {order,dim} = DIV(GRAD(T, CS), CS)

"""
    HESS(T::Union{Sym,AbstractTens{order,dim,Sym}},CS::CoorSystemSym{dim}) where {order,dim}

Calculates the Hessian of `T` with respect to the coordinate system `CS`
"""
HESS(T::Union{Sym,AbstractTens{order,dim,Sym}}, CS::CoorSystemSym{dim}) where {order,dim} =
    GRAD(GRAD(T, CS), CS)

"""
    coorsys_cartesian(coords = symbols("x y z", real = true))

Returns the cartesian coordinate system

# Examples
```julia
julia> Cartesian = coorsys_cartesian() ; 𝐗 = getcoords(Cartesian) ; 𝐄 = unitvec(Cartesian) ; ℬ = getbasis(Cartesian)

julia> 𝛔 = Tens(SymmetricTensor{2,3}((i, j) -> SymFunction("σ\$i\$j", real = true)(𝐗...))) ;

julia> DIV(𝛔, CScar)
Tens.TensCanonical{1, 3, Sym, Vec{3, Sym}}
# data: 3-element Vec{3, Sym}:
 Derivative(σ11(x, y, z), x) + Derivative(σ21(x, y, z), y) + Derivative(σ31(x, y, z), z)
 Derivative(σ21(x, y, z), x) + Derivative(σ22(x, y, z), y) + Derivative(σ32(x, y, z), z)
 Derivative(σ31(x, y, z), x) + Derivative(σ32(x, y, z), y) + Derivative(σ33(x, y, z), z)
# basis: 3×3 Tens.LazyIdentity{3, Sym}:
 1  0  0
 0  1  0
 0  0  1
# var: (:cont,)
``` 
"""
function coorsys_cartesian(coords = symbols("x y z", real = true))
    dim = length(coords)
    𝐗, 𝐄, ℬ = init_cartesian(coords)
    OM = sum([𝐗[i] * 𝐄[i] for i = 1:dim])
    χᵢ = ntuple(_ -> one(Sym), dim)
    return CoorSystemSym(OM, coords, ℬ, χᵢ)
end

"""
    coorsys_polar(coords = (symbols("r", positive = true), symbols("θ", real = true)); canonical = false)

Returns the polar coordinate system

# Examples
```julia
julia> Polar = coorsys_polar() ; r, θ = getcoords(Polar) ; 𝐞ʳ, 𝐞ᶿ = unitvec(Polar) ; ℬᵖ = getbasis(Polar)

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
function coorsys_polar(
    coords = (symbols("r", positive = true), symbols("θ", real = true));
    canonical = false,
)
    (r, θ), (𝐞ʳ, 𝐞ᶿ), ℬᵖ = init_polar(coords, canonical = canonical)
    OM = r * 𝐞ʳ
    return CoorSystemSym(OM, coords, ℬᵖ, (one(Sym), r))
end

"""
    coorsys_cylindrical(coords = (symbols("r", positive = true), symbols("θ", real = true), symbols("z", real = true)); canonical = false)

Returns the cylindrical coordinate system

# Examples
```julia
julia> Cylindrical = coorsys_cylindrical() ; rθz = getcoords(Cylindrical) ; 𝐞ʳ, 𝐞ᶿ, 𝐞ᶻ = unitvec(Cylindrical) ; ℬᶜ = getbasis(Cylindrical)

julia> 𝐯 = Tens(Vec{3}(i -> SymFunction("v\$(rθz[i])", real = true)(rθz...)), ℬᶜ) ;

julia> DIV(𝐯, Cylindrical)
                                                  ∂
                                    vr(r, θ, z) + ──(vθ(r, θ, z))
∂                 ∂                               ∂θ
──(vr(r, θ, z)) + ──(vz(r, θ, z)) + ─────────────────────────────
∂r                ∂z                              r
``` 
"""
function coorsys_cylindrical(
    coords = (
        symbols("r", positive = true),
        symbols("θ", real = true),
        symbols("z", real = true),
    );
    canonical = false,
)
    (r, θ, z), (𝐞ʳ, 𝐞ᶿ, 𝐞ᶻ), ℬᶜ = init_cylindrical(coords, canonical = canonical)
    OM = r * 𝐞ʳ + z * 𝐞ᶻ
    return CoorSystemSym(OM, coords, ℬᶜ, (one(Sym), r, one(Sym)))
end

"""
    coorsys_spherical(coords = (symbols("θ", real = true), symbols("ϕ", real = true), symbols("r", positive = true)); canonical = false)

Returns the spherical coordinate system

# Examples
```julia
julia> Spherical = coorsys_spherical() ; θ, ϕ, r = getcoords(Spherical) ; 𝐞ᶿ, 𝐞ᵠ, 𝐞ʳ = unitvec(Spherical) ; ℬˢ = getbasis(Spherical)

julia> for σⁱʲ ∈ ("σʳʳ", "σᶿᶿ", "σᵠᵠ") @eval \$(Symbol(σⁱʲ)) = SymFunction(\$σⁱʲ, real = true)(\$r) end ;

julia> 𝛔 = σʳʳ * 𝐞ʳ ⊗ 𝐞ʳ + σᶿᶿ * 𝐞ᶿ ⊗ 𝐞ᶿ + σᵠᵠ * 𝐞ᵠ ⊗ 𝐞ᵠ ;

julia> div𝛔 = DIV(𝛔, Spherical)
Tens.TensRotated{1, 3, Sym, Vec{3, Sym}}
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
function coorsys_spherical(
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
    return CoorSystemSym(OM, coords, ℬˢ, (r, r * sin(θ), one(Sym)); rules = rules)
end

"""
    coorsys_spheroidal(coords = (symbols("ϕ", real = true),symbols("p", real = true),symbols("q", positive = true),),
                            c = symbols("c", positive = true),tmp_coords = (symbols("p̄ q̄", positive = true)...,),)

Returns the spheroidal coordinate system

# Examples
```julia
julia> Spheroidal = coorsys_spheroidal() ; OM = getOM(Spheroidal)
Tens.TensCanonical{1, 3, Sym, Vec{3, Sym}}
# data: 3-element Vec{3, Sym}:
 c⋅p̄⋅q̄⋅cos(ϕ)
 c⋅p̄⋅q̄⋅sin(ϕ)
          c⋅p⋅q
# basis: 3×3 Tens.LazyIdentity{3, Sym}:
 1  0  0
 0  1  0
 0  0  1
# var: (:cont,)

julia> LAPLACE(OM[1]^2, Spheroidal)
2
``` 
"""
function coorsys_spheroidal(
    coords = (
        symbols("ϕ", real = true),
        symbols("p", real = true),
        symbols("q", positive = true),
    ),
    c = symbols("c", positive = true),
    tmp_coords = (symbols("p̄ q̄", positive = true)...,),
)
    ϕ, p, q = coords
    params = (c,)
    p̄, q̄ = tmp_coords
    OM = Tens(c * [p̄ * q̄ * cos(ϕ), p̄ * q̄ * sin(ϕ), p * q])
    # OM = Tens(c * [√(1 - p^2) * √(q^2 - 1) * cos(ϕ), √(1 - p^2) * √(q^2 - 1) * sin(ϕ), p * q])
    ℬ = RotatedBasis(
        Sym[
            -sin(ϕ) -p*sqrt(q^2 - 1)*cos(ϕ)/sqrt(q^2 - p^2) q*sqrt(1 - p^2)*cos(ϕ)/sqrt(q^2 - p^2)
            cos(ϕ) -p*sqrt(q^2 - 1)*sin(ϕ)/sqrt(q^2 - p^2) q*sqrt(1 - p^2)*sin(ϕ)/sqrt(q^2 - p^2)
            0 q*sqrt(1 - p^2)/sqrt(q^2 - p^2) p*sqrt(q^2 - 1)/sqrt(q^2 - p^2)
        ],
    )
    χᵢ = (
        c * sqrt(1 - p^2) * sqrt(q^2 - 1),
        c * sqrt(q^2 - p^2) / sqrt(1 - p^2),
        c * sqrt(q^2 - p^2) / sqrt(q^2 - 1),
    )
    tmp_var = Dict(1 - p^2 => p̄^2, q^2 - 1 => q̄^2)
    to_coords = Dict(p̄ => √(1 - p^2), q̄ => √(q^2 - 1))
    return CoorSystemSym(
        OM,
        coords,
        ℬ,
        χᵢ,
        tmp_coords,
        params;
        tmp_var = tmp_var,
        to_coords = to_coords,
    )
end

"""
    @set_coorsys CS
    @set_coorsys(CS)

Sets a coordinate system in order to avoid precising it in differential operators

# Examples
```julia
julia> Spherical = coorsys_spherical() ; θ, ϕ, r = getcoords(Spherical) ; 𝐞ᶿ, 𝐞ᵠ, 𝐞ʳ = unitvec(Spherical) ;

julia> @set_coorsys Spherical

julia> GRAD(𝐞ʳ)
TensND.TensRotated{2, 3, Sym, Tensor{2, 3, Sym, 9}}
→ data: 3×3 Tensor{2, 3, Sym, 9}:
 1/r    0  0
   0  1/r  0
   0    0  0
→ basis: 3×3 Matrix{Sym}:
 cos(θ)⋅cos(ϕ)  -sin(ϕ)  sin(θ)⋅cos(ϕ)
 sin(ϕ)⋅cos(θ)   cos(ϕ)  sin(θ)⋅sin(ϕ)
       -sin(θ)        0         cos(θ)

julia> DIV(𝐞ʳ ⊗ 𝐞ʳ)
TensND.TensRotated{1, 3, Sym, Vec{3, Sym}}
→ data: 3-element Vec{3, Sym}:
   0
   0
 2/r
→ basis: 3×3 Matrix{Sym}:
 cos(θ)⋅cos(ϕ)  -sin(ϕ)  sin(θ)⋅cos(ϕ)
 sin(ϕ)⋅cos(θ)   cos(ϕ)  sin(θ)⋅sin(ϕ)
       -sin(θ)        0         cos(θ)

julia> LAPLACE(1/r)
0
``` 
"""
macro set_coorsys(CS)
    m = @__MODULE__
    return quote
            $m.∂(t::AbstractTens{order,dim,Sym}, i::Integer) where {order,dim} = $m.∂(t, i, $(esc(CS)))
            $m.∂(t::AbstractTens{order,dim,Sym}, x::Sym) where {order,dim}  = $m.∂(t, x, $(esc(CS)))
            $m.∂(t::Sym, i::Integer) = $m.∂(t, i, $(esc(CS)))
            $m.∂(t::Sym, x::Sym) = $m.∂(t, x, $(esc(CS)))
            $m.GRAD(t::Union{Sym,AbstractTens}) = $m.GRAD(t, $(esc(CS)))
            $m.SYMGRAD(t::Union{Sym,AbstractTens}) = $m.SYMGRAD(t, $(esc(CS)))
            $m.DIV(t::AbstractTens) = $m.DIV(t, $(esc(CS)))
            $m.LAPLACE(t::Union{Sym,AbstractTens}) = $m.LAPLACE(t, $(esc(CS)))
            $m.HESS(t::Union{Sym,AbstractTens}) = $m.HESS(t, $(esc(CS)))
        end
end

export ∂, CoorSystemSym, getChristoffel
export GRAD, SYMGRAD, DIV, LAPLACE, HESS
export get_normalized_basis, get_natural_basis, natvec, unitvec, getcoords, getOM
export coorsys_cartesian, coorsys_polar, coorsys_cylindrical, coorsys_spherical, coorsys_spheroidal
export @set_coorsys
