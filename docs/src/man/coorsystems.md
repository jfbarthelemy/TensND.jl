# Coordinate systems and differential operators

For the moment only symbolic coordinate systems are available. Their numerical counterpart will be later developed. 

```julia
julia> Polar = coorsys_polar() ; r, Î¸ = getcoords(Polar) ; ðÊ³, ðá¶¿ = unitvec(Polar) ;

julia> @set_coorsys Polar

julia> LAPLACE(SymFunction("f", real = true)(r, Î¸))
                               2
                              â
               â             âââ(f(r, Î¸))
  2            ââ(f(r, Î¸))     2
 â             âr            âÎ¸
âââ(f(r, Î¸)) + âââââââââââ + ââââââââââââ
  2                 r              2
âr                                r

julia> n = symbols("n", integer = true)
n

julia> simplify(HESS(r^n))
(n*r^(n - 2)*(n - 1))ðÊ³âðÊ³ + (n*r^(n - 2))ðá¶¿âðá¶¿
```

```julia
julia> Spherical = coorsys_spherical() ; Î¸, Ï, r = getcoords(Spherical) ; ðá¶¿, ðáµ , ðÊ³ = unitvec(Spherical) ;

julia> @set_coorsys Spherical

julia> getChristoffel(Spherical)
3Ã3Ã3 Array{Sym, 3}:
[:, :, 1] =
   0               0  1/r
   0  -sin(Î¸)âcos(Î¸)    0
 1/r               0    0

[:, :, 2] =
             0  cos(Î¸)/sin(Î¸)    0
 cos(Î¸)/sin(Î¸)              0  1/r
             0            1/r    0

[:, :, 3] =
 -r            0  0
  0  -r*sin(Î¸)^2  0
  0            0  0

julia> â¬Ë¢ = get_normalized_basis(Spherical)
RotatedBasis{3, Sym}
â basis: 3Ã3 Matrix{Sym}:
 cos(Î¸)âcos(Ï)  -sin(Ï)  sin(Î¸)âcos(Ï)
 sin(Ï)âcos(Î¸)   cos(Ï)  sin(Î¸)âsin(Ï)
       -sin(Î¸)        0         cos(Î¸)
â dual basis: 3Ã3 Matrix{Sym}:
 cos(Î¸)âcos(Ï)  -sin(Ï)  sin(Î¸)âcos(Ï)
 sin(Ï)âcos(Î¸)   cos(Ï)  sin(Î¸)âsin(Ï)
       -sin(Î¸)        0         cos(Î¸)
â covariant metric tensor: 3Ã3 TensND.Id2{3, Sym}:
 1  â  â
 â  1  â
 â  â  1
â contravariant metric tensor: 3Ã3 TensND.Id2{3, Sym}:
 1  â  â
 â  1  â
 â  â  1

julia> for Ïâ±Ê² â ("ÏÊ³Ê³", "Ïá¶¿á¶¿", "Ïáµ áµ ") @eval $(Symbol(Ïâ±Ê²)) = SymFunction($Ïâ±Ê², real = true)($r) end

julia> ð = ÏÊ³Ê³ * ðÊ³ â ðÊ³ + Ïá¶¿á¶¿ * ðá¶¿ â ðá¶¿ + Ïáµ áµ  * ðáµ  â ðáµ 
(Ïá¶¿á¶¿(r))ðá¶¿âðá¶¿ + (Ïáµ áµ (r))ðáµ âðáµ  + (ÏÊ³Ê³(r))ðÊ³âðÊ³

julia> divð = simplify(DIV(ð))
((-Ïáµ áµ (r) + Ïá¶¿á¶¿(r))/(r*tan(Î¸)))ðá¶¿ + ((r*Derivative(ÏÊ³Ê³(r), r) + 2*ÏÊ³Ê³(r) - Ïáµ áµ (r) - Ïá¶¿á¶¿(r))/r)ðÊ³
```
