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
    𝐞(i::Integer, dim::Int = 3, T::Type{<:Number} = Sym)

Vector of the canonical basis

# Examples
```julia
julia> 𝐞(1)
Tens{1, 3, Sym, Sym, Vec{3, Sym}, CanonicalBasis{3, Sym}}
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
𝐞(::Val{i}, ::Val{dim} = Val(3), ::Val{T} = Val(Sym)) where {i, dim, T<:Number} =
    Tens(Vec{dim}(j -> j == i ? one(T) : zero(T)))

"""
    init_cartesian(coords = symbols("x y z", real = true))

Returns the coordinates, unit vectors and basis of the cartesian basis

# Examples
```julia
julia> coords, vectors, ℬ = init_cartesian() ; x, y, z = coords ; 𝐞₁, 𝐞₂, 𝐞₃ = vectors ;
``` 
"""
init_cartesian(coords = symbols("x y z", real = true)) = Tuple(coords),
ntuple(i -> 𝐞(Val(i), Val(length(coords)), Val(eltype(coords))), length(coords)),
CanonicalBasis{length(coords),eltype(coords)}()

init_cartesian(::Val{3}) = init_cartesian(symbols("x y z", real = true))
init_cartesian(::Val{2}) = init_cartesian(symbols("x y", real = true))
init_cartesian(dim::Integer) = init_cartesian(Val(dim))


"""
    𝐞ᵖ(i::Integer, θ::T = zero(Sym); canonical = false)

Vector of the polar basis

# Examples
```julia
julia> θ = symbols("θ", real = true) ;

julia> 𝐞ᵖ(1, θ)
Tens{1, 2, Sym, Sym, Vec{2, Sym}, RotatedBasis{2, Sym}}
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
    canonical ? Tens(Vec{2}([cos(θ), sin(θ)])) :
    Tens(Vec{2}([one(T), zero(T)]), Basis(θ))
𝐞ᵖ(::Val{2}, θ::T = zero(Sym); canonical = false) where {T<:Number} =
    canonical ? Tens(Vec{2}([-sin(θ), cos(θ)])) :
    Tens(Vec{2}([zero(T), one(T)]), Basis(θ))

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
) = Tuple(coords),
ntuple(i -> 𝐞ᵖ(Val(i), coords[2]; canonical = canonical), 2),
Basis(coords[2])

"""
    𝐞ᶜ(i::Integer, θ::T = zero(Sym); canonical = false)

Vector of the cylindrical basis

# Examples
```julia
julia> θ = symbols("θ", real = true) ;

julia> 𝐞ᶜ(1, θ)
Tens{1, 3, Sym, Sym, Vec{3, Sym}, RotatedBasis{3, Sym}}
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
    canonical ? Tens(Vec{3}([cos(θ), sin(θ), zero(T)])) :
    Tens(Vec{3}([one(T), zero(T), zero(T)]), CylindricalBasis(θ))
𝐞ᶜ(::Val{2}, θ::T = zero(Sym); canonical = false) where {T<:Number} =
    canonical ? Tens(Vec{3}([-sin(θ), cos(θ), zero(T)])) :
    Tens(Vec{3}([zero(T), one(T), zero(T)]), CylindricalBasis(θ))
𝐞ᶜ(::Val{3}, θ::T = zero(Sym); canonical = false) where {T<:Number} =
    canonical ? Tens(Vec{3}([zero(T), zero(T), one(T)])) :
    Tens(Vec{3}([zero(T), zero(T), one(T)]), CylindricalBasis(θ))

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
ntuple(i -> 𝐞ᶜ(Val(i), coords[2]; canonical = canonical), 3),
CylindricalBasis(coords[2])

"""
    𝐞ˢ(i::Integer, θ::T = zero(Sym), ϕ::T = zero(Sym), ψ::T = zero(Sym); canonical = false)

Vector of the basis rotated with the 3 Euler angles `θ, ϕ, ψ` (spherical if `ψ=0`)

# Examples
```julia
julia> θ, ϕ, ψ = symbols("θ, ϕ, ψ", real = true) ;

Tens{1, 3, Sym, Sym, Vec{3, Sym}, RotatedBasis{3, Sym}}
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
        return Tens(
            Vec{3}([
                -sin(ψ) * sin(ϕ) + cos(θ) * cos(ψ) * cos(ϕ),
                sin(ψ) * cos(ϕ) + sin(ϕ) * cos(θ) * cos(ψ),
                -sin(θ) * cos(ψ),
            ]),
        )
    else
        T = promote_type(T1, T2, T3)
        return Tens(Vec{3}([one(T), zero(T), zero(T)]), Basis(θ, ϕ, ψ))
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
        return Tens(
            Vec{3}([
                -sin(ψ) * cos(θ) * cos(ϕ) - sin(ϕ) * cos(ψ),
                -sin(ψ) * sin(ϕ) * cos(θ) + cos(ψ) * cos(ϕ),
                sin(θ) * sin(ψ),
            ]),
        )
    else
        T = promote_type(T1, T2, T3)
        return Tens(Vec{3}([zero(T), one(T), zero(T)]), Basis(θ, ϕ, ψ))
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
        return Tens(Vec{3}([sin(θ) * cos(ϕ), sin(θ) * sin(ϕ), cos(θ)]))
    else
        T = promote_type(T1, T2, T3)
        return Tens(Vec{3}([zero(T), zero(T), one(T)]), Basis(θ, ϕ, ψ))
    end
end

for eb in (:𝐞, :𝐞ᵖ, :𝐞ᶜ, :𝐞ˢ)
    @eval $eb(i::Integer, args...; kwargs...) = $eb(Val(i), args...; kwargs...)
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
ntuple(i -> 𝐞ˢ(Val(i), coords[1:2]...; canonical = canonical), 3),
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
init_rotated(angles = symbols("θ ϕ ψ", real = true); canonical = false) = Tuple(angles),
ntuple(i -> 𝐞ˢ(Val(i), angles...; canonical = canonical), 3),
Basis(angles...)

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

julia> R = Tens(subs.(rot3(θ, ϕ, ψ),d...))
Tens.TensCanonical{2, 3, Sym, Tensor{2, 3, Sym, 9}}
# data: 3×3 Tensor{2, 3, Sym, 9}:
 cθ⋅cψ⋅cϕ - sψ⋅sϕ  -cθ⋅cϕ⋅sψ - cψ⋅sϕ  cϕ⋅sθ
 cθ⋅cψ⋅sϕ + cϕ⋅sψ  -cθ⋅sψ⋅sϕ + cψ⋅cϕ  sθ⋅sϕ
           -cψ⋅sθ              sθ⋅sψ     cθ
# var: (:cont, :cont)
# basis: 3×3 Tens.LazyIdentity{3, Sym}:
 1  0  0
 0  1  0
 0  0  1

julia> RR = R ⊠ˢ R
Tens.TensCanonical{4, 3, Sym, SymmetricTensor{4, 3, Sym, 36}}
# data: 6×6 Matrix{Sym}:
                          (cθ*cψ*cϕ - sψ*sϕ)^2                            (-cθ*cϕ*sψ - cψ*sϕ)^2           cϕ^2*sθ^2                      √2⋅cϕ⋅sθ⋅(-cθ⋅cϕ⋅sψ - cψ⋅sϕ)                     √2⋅cϕ⋅sθ⋅(cθ⋅cψ⋅cϕ - sψ⋅sϕ)                                   √2⋅(cθ⋅cψ⋅cϕ - sψ⋅sϕ)⋅(-cθ⋅cϕ⋅sψ - cψ⋅sϕ)
                          (cθ*cψ*sϕ + cϕ*sψ)^2                            (-cθ*sψ*sϕ + cψ*cϕ)^2           sθ^2*sϕ^2                      √2⋅sθ⋅sϕ⋅(-cθ⋅sψ⋅sϕ + cψ⋅cϕ)                     √2⋅sθ⋅sϕ⋅(cθ⋅cψ⋅sϕ + cϕ⋅sψ)                                   √2⋅(cθ⋅cψ⋅sϕ + cϕ⋅sψ)⋅(-cθ⋅sψ⋅sϕ + cψ⋅cϕ)
                                     cψ^2*sθ^2                                        sθ^2*sψ^2                cθ^2                                       √2⋅cθ⋅sθ⋅sψ                                    -√2⋅cθ⋅cψ⋅sθ                                                              -sqrt(2)*cψ*sθ^2*sψ
             -√2⋅cψ⋅sθ⋅(cθ⋅cψ⋅sϕ + cϕ⋅sψ)                √2⋅sθ⋅sψ⋅(-cθ⋅sψ⋅sϕ + cψ⋅cϕ)    √2⋅cθ⋅sθ⋅sϕ                    cθ*(-cθ*sψ*sϕ + cψ*cϕ) + sθ^2*sψ*sϕ                   cθ*(cθ*cψ*sϕ + cϕ*sψ) - cψ*sθ^2*sϕ                            -cψ⋅sθ⋅(-cθ⋅sψ⋅sϕ + cψ⋅cϕ) + sθ⋅sψ⋅(cθ⋅cψ⋅sϕ + cϕ⋅sψ)
             -√2⋅cψ⋅sθ⋅(cθ⋅cψ⋅cϕ - sψ⋅sϕ)                √2⋅sθ⋅sψ⋅(-cθ⋅cϕ⋅sψ - cψ⋅sϕ)    √2⋅cθ⋅cϕ⋅sθ                    cθ*(-cθ*cϕ*sψ - cψ*sϕ) + cϕ*sθ^2*sψ                   cθ*(cθ*cψ*cϕ - sψ*sϕ) - cψ*cϕ*sθ^2                            -cψ⋅sθ⋅(-cθ⋅cϕ⋅sψ - cψ⋅sϕ) + sθ⋅sψ⋅(cθ⋅cψ⋅cϕ - sψ⋅sϕ)
 √2⋅(cθ⋅cψ⋅cϕ - sψ⋅sϕ)⋅(cθ⋅cψ⋅sϕ + cϕ⋅sψ)  √2⋅(-cθ⋅cϕ⋅sψ - cψ⋅sϕ)⋅(-cθ⋅sψ⋅sϕ + cψ⋅cϕ)  sqrt(2)*cϕ*sθ^2*sϕ  cϕ⋅sθ⋅(-cθ⋅sψ⋅sϕ + cψ⋅cϕ) + sθ⋅sϕ⋅(-cθ⋅cϕ⋅sψ - cψ⋅sϕ)  cϕ⋅sθ⋅(cθ⋅cψ⋅sϕ + cϕ⋅sψ) + sθ⋅sϕ⋅(cθ⋅cψ⋅cϕ - sψ⋅sϕ)  (cθ*cψ*cϕ - sψ*sϕ)*(-cθ*sψ*sϕ + cψ*cϕ) + (cθ*cψ*sϕ + cϕ*sψ)*(-cθ*cϕ*sψ - cψ*sϕ)
# var: (:cont, :cont, :cont, :cont)
# basis: 3×3 Tens.LazyIdentity{3, Sym}:
 1  0  0
 0  1  0
 0  0  1

julia> R6 = invKM(subs.(KM(rot6(θ, ϕ, ψ)),d...))
Tens.TensCanonical{4, 3, Sym, SymmetricTensor{4, 3, Sym, 36}}
# data: 6×6 Matrix{Sym}:
                          (cθ*cψ*cϕ - sψ*sϕ)^2                            (-cθ*cϕ*sψ - cψ*sϕ)^2           cϕ^2*sθ^2                      √2⋅cϕ⋅sθ⋅(-cθ⋅cϕ⋅sψ - cψ⋅sϕ)                     √2⋅cϕ⋅sθ⋅(cθ⋅cψ⋅cϕ - sψ⋅sϕ)                                   √2⋅(cθ⋅cψ⋅cϕ - sψ⋅sϕ)⋅(-cθ⋅cϕ⋅sψ - cψ⋅sϕ)
                          (cθ*cψ*sϕ + cϕ*sψ)^2                            (-cθ*sψ*sϕ + cψ*cϕ)^2           sθ^2*sϕ^2                      √2⋅sθ⋅sϕ⋅(-cθ⋅sψ⋅sϕ + cψ⋅cϕ)                     √2⋅sθ⋅sϕ⋅(cθ⋅cψ⋅sϕ + cϕ⋅sψ)                                   √2⋅(cθ⋅cψ⋅sϕ + cϕ⋅sψ)⋅(-cθ⋅sψ⋅sϕ + cψ⋅cϕ)
                                     cψ^2*sθ^2                                        sθ^2*sψ^2                cθ^2                                       √2⋅cθ⋅sθ⋅sψ                                    -√2⋅cθ⋅cψ⋅sθ                                                              -sqrt(2)*cψ*sθ^2*sψ
             -√2⋅cψ⋅sθ⋅(cθ⋅cψ⋅sϕ + cϕ⋅sψ)                √2⋅sθ⋅sψ⋅(-cθ⋅sψ⋅sϕ + cψ⋅cϕ)    √2⋅cθ⋅sθ⋅sϕ                    cθ*(-cθ*sψ*sϕ + cψ*cϕ) + sθ^2*sψ*sϕ                   cθ*(cθ*cψ*sϕ + cϕ*sψ) - cψ*sθ^2*sϕ                            -cψ⋅sθ⋅(-cθ⋅sψ⋅sϕ + cψ⋅cϕ) + sθ⋅sψ⋅(cθ⋅cψ⋅sϕ + cϕ⋅sψ)
             -√2⋅cψ⋅sθ⋅(cθ⋅cψ⋅cϕ - sψ⋅sϕ)                √2⋅sθ⋅sψ⋅(-cθ⋅cϕ⋅sψ - cψ⋅sϕ)    √2⋅cθ⋅cϕ⋅sθ                    cθ*(-cθ*cϕ*sψ - cψ*sϕ) + cϕ*sθ^2*sψ                   cθ*(cθ*cψ*cϕ - sψ*sϕ) - cψ*cϕ*sθ^2                            -cψ⋅sθ⋅(-cθ⋅cϕ⋅sψ - cψ⋅sϕ) + sθ⋅sψ⋅(cθ⋅cψ⋅cϕ - sψ⋅sϕ)
 √2⋅(cθ⋅cψ⋅cϕ - sψ⋅sϕ)⋅(cθ⋅cψ⋅sϕ + cϕ⋅sψ)  √2⋅(-cθ⋅cϕ⋅sψ - cψ⋅sϕ)⋅(-cθ⋅sψ⋅sϕ + cψ⋅cϕ)  sqrt(2)*cde Liv Lehn ϕ*sθ^2*sϕ  cϕ⋅sθ⋅(-cθ⋅sψ⋅sϕ + cψ⋅cϕ) + sθ⋅sϕ⋅(-cθ⋅cϕ⋅sψ - cψ⋅sϕ)  cϕ⋅sθ⋅(cθ⋅cψ⋅sϕ + cϕ⋅sψ) + sθ⋅sϕ⋅(cθ⋅cψ⋅cϕ - sψ⋅sϕ)  (cθ*cψ*cϕ - sψ*sϕ)*(-cθ*sψ*sϕ + cψ*cϕ) + (cθ*cψ*sϕ + cϕ*sψ)*(-cθ*cϕ*sψ - cψ*sϕ)
# var: (:cont, :cont, :cont, :cont)
# basis: 3×3 Tens.LazyIdentity{3, Sym}:
 1  0  0
 0  1  0
 0  0  1

julia> R6 == RR
true
```
"""
function rot6(θ, ϕ = 0, ψ = 0)
    R = TensCanonical(rot3(θ, ϕ, ψ))
    return sboxtimes(R, R)
end

export LeviCivita
export 𝐞, 𝐞ᵖ, 𝐞ᶜ, 𝐞ˢ
export init_cartesian, init_polar, init_cylindrical, init_spherical, init_rotated
export rot2, rot3, rot6
