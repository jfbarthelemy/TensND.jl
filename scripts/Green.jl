using TensND, LinearAlgebra, SymPy, Tensors, OMEinsum, Rotations
sympy.init_printing(use_unicode=true)

Polar = coorsys_polar() ; r, ฮธ = getcoords(Polar) ; ๐สณ, ๐แถฟ = unitvec(Polar)
@set_coorsys Polar
โฌหข = get_normalized_basis(Polar)
๐ฑ = getOM(Polar)
Cartesian = coorsys_cartesian(symbols("x y", real = true))
๐โ, ๐โ = unitvec(Cartesian)
xโ, xโ = getcoords(Cartesian)
๐, ๐, ๐ = ISO(Val(2),Val(Sym))
๐ = tensId2(Val(2),Val(Sym))

E, k, ฮผ = symbols("E k ฮผ", positive = true)
ฮฝ, ฮบ = symbols("ฮฝ ฮบ", real = true)
k = E / (3(1-2ฮฝ)) ; ฮผ = E / (2(1+ฮฝ))
ฮป = k -2ฮผ/3

๐ =simplify(1/(8 * PI * ฮผ * (1-ฮฝ)) * (๐สณ โ ๐สณ -(3-4ฮฝ) * ln(r) * ๐))
HG = -simplify(HESS(๐))
aHG = getarray(HG)
๐ = SymmetricTensor{4,2}((i,j,k,l)->(aHG[i,k,j,l]+aHG[j,k,i,l]+aHG[i,l,j,k]+aHG[j,l,i,k])/4)
โพ = simplify(Tens(๐,โฌหข))
โพโ = simplify(1/(8PI * ฮผ * (1-ฮฝ) * r^2) * (-2๐ +2(1-2ฮฝ)*๐ + 2(๐โ๐สณโ๐สณ + ๐สณโ๐สณโ๐) + 8ฮฝ*๐สณโหข๐โหข๐สณ -8๐สณโ๐สณโ๐สณโ๐สณ))
simplify(โพ-โพโ)

โ = 2ฮป * ๐ + 2ฮผ * ๐

๐ = simplify(โพ โก โ)
d = Dict(r => sqrt(xโ^2+xโ^2), sin(ฮธ) => xโ/sqrt(xโ^2+xโ^2), cos(ฮธ) => xโ/sqrt(xโ^2+xโ^2), ฮฝ => (3-ฮบ)/4)


Spherical = coorsys_spherical() ; ฮธ, ฯ, r = getcoords(Spherical) ; ๐แถฟ, ๐แต , ๐สณ = unitvec(Spherical) ;
โฌหข = get_normalized_basis(Spherical)
@set_coorsys Spherical
๐, ๐, ๐ = ISO(Val(3),Val(Sym))
๐ = tensId2(Val(3),Val(Sym))

E, k, ฮผ = symbols("E k ฮผ", positive = true)
ฮฝ = symbols("ฮฝ", real = true)
k = E / (3(1-2ฮฝ)) ; ฮผ = E / (2(1+ฮฝ))
ฮป = k -2ฮผ/3

๐ = 1/ (8PI * ฮผ * (3k+4ฮผ) * r) * ( (3k+7ฮผ) * ๐ + (3k+ฮผ) * ๐สณโ๐สณ)
๐โ = 1 / (16PI * ฮผ * (1-ฮฝ) * r) * ( (3-4ฮฝ) * ๐ + ๐สณโ๐สณ)
simplify(๐-๐โ)

HG = -simplify(HESS(๐))
aHG = getarray(HG)
๐ = SymmetricTensor{4,3}((i,j,k,l)->(aHG[i,k,j,l]+aHG[j,k,i,l]+aHG[i,l,j,k]+aHG[j,l,i,k])/4)
โพ = simplify(Tens(๐,โฌหข))
โพโ = simplify(1/(16PI * ฮผ * (1-ฮฝ) * r^3) * (-3๐ +2(1-2ฮฝ)*๐ + 3(๐โ๐สณโ๐สณ + ๐สณโ๐สณโ๐) + 12ฮฝ*๐สณโหข๐โหข๐สณ -15๐สณโ๐สณโ๐สณโ๐สณ))
simplify(โพ-โพโ)

Cartesian = coorsys_cartesian(symbols("x y z", real = true))
๐โ, ๐โ, ๐3 = unitvec(Cartesian)
F = symbols("F", real = true)
J = simplify(det(๐ + F*GRAD(๐โ๐โ)))
factor(simplify(subs(J, ฮธ => PI/2, ฯ => 0)))