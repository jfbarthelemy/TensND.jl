# Numerical differential operators

`CoorSystemNum` provides the same differential operators as `CoorSystemSym` but evaluated **pointwise at numerical coordinates** using automatic differentiation (ForwardDiff.jl).
This is useful when:

- the field to differentiate is defined by a numerical function (not a symbolic expression);
- a parametric study over many evaluation points is needed;
- the coordinate system is defined only implicitly (e.g. from a position vector function).

## Setup

Predefined factories return a `CoorSystemNum` ready to use:

```@example numcs
using TensND, LinearAlgebra

CS_polar  = coorsys_polar_num()       # (r, θ)
```

```@example numcs
CS_cyl    = coorsys_cylindrical_num() # (r, θ, z)
```

```@example numcs
CS_sph    = coorsys_spherical_num()   # (θ, ϕ, r) — same convention as CoorSystemSym
```

For a coordinate system defined by a position vector $\mathbf{OM}(x)$, use the generic constructor:

```@example numcs
OM_func = x -> [x[1]*cos(x[2]), x[1]*sin(x[2]), x[3]];  # cylindrical
CS_gen  = CoorSystemNum(OM_func, 3)
```

## Operator syntax

All operators take a **function** and return a **function**. Evaluation at a point requires an extra call:

```@example numsyntax
using TensND, LinearAlgebra
CS = coorsys_spherical_num();
f = x -> x[3]^2;      # f(θ,ϕ,r) = r²  in spherical coords
LAPLACE(f, CS)([π/4, π/3, 2.0])   # = n(n+1) r^(n-2) = 6 for n=2, r=2
```

The supported operators are:

| Operator | Input | Output |
| -------- | ----- | ------ |
| `GRAD(f, CS)` | scalar → vector, vector → matrix | Function |
| `SYMGRAD(f, CS)` | vector → symmetric matrix | Function |
| `DIV(f, CS)` | vector → scalar, matrix → vector | Function |
| `LAPLACE(f, CS)` | scalar → scalar | Function |
| `HESS(f, CS)` | scalar → matrix | Function |

All component arrays use the **physical (normalized) basis** matching the convention of `CoorSystemSym`.

Tensor results are returned as `AbstractTens` objects attached to the local normalized basis.

## Basis accessors

The same accessors as `CoorSystemSym` are available, evaluated pointwise:

```@example numbasis
using TensND, LinearAlgebra
CS = coorsys_spherical_num();
x₀ = [π/3, π/4, 2.0];   # (θ, ϕ, r)
Lame(CS, x₀)             # Lamé coefficients χ = (r, r·sinθ, 1)
```

```@example numbasis
normalized_basis(CS, x₀)
```

```@example numbasis
natvec(CS, x₀, 1, :cov)   # covariant natural vector aθ = χθ·eθ
```

```@example numbasis
natvec(CS, x₀, 1, :cont)  # contravariant aθ = eθ/χθ
```

```@example numbasis
unitvec(CS, x₀, 3)         # unit vector eʳ
```

## Basic scalar operators in polar coordinates

In polar coordinates $(r, \theta)$ with Lamé coefficients $(\chi_1, \chi_2) = (1, r)$, the classical result is

```math
\nabla^2 (r^n f(\theta)) = (n^2 - m^2)\, r^{n-2} f(\theta)
\qquad \text{for } f(\theta) = \sin(m\theta).
```

For $n = 2$, $m = 1$: $\nabla^2(r^2\sin\theta) = (4-1)\sin\theta = 3\sin\theta$.

```@example numpolar
using TensND
CS = coorsys_polar_num();
f = x -> x[1]^2 * sin(x[2]);   # r²sinθ
LAPLACE(f, CS)([2.5, 0.7])     # = 3·sin(0.7)
```

## Gradient and Laplacian in spherical coordinates

For $f(r) = r^n$ in spherical coordinates $(\theta, \phi, r)$:

```math
\nabla f = n r^{n-1}\, \mathbf{e}_r, \qquad \nabla^2 f = n(n+1)\, r^{n-2}.
```

For $f = 1/r$ (harmonic function): $\nabla f = -r^{-2}\mathbf{e}_r$, $\nabla^2 f = 0$.

```@example numsph
using TensND, LinearAlgebra
CS = coorsys_spherical_num();
f = x -> x[3]^(-1);              # 1/r
GRAD(f, CS)([π/3, π/4, 2.0])    # → -1/r²·eʳ, components (eθ, eϕ, eʳ)
```

```@example numsph
LAPLACE(f, CS)([π/3, π/4, 2.0]) # harmonic → 0
```

## Lamé problem — hollow sphere under internal pressure

A hollow isotropic elastic sphere with inner radius $a$ and outer radius $b$ is subjected to an internal pressure $p$. The exact (Lamé) solution for the displacement is

```math
u_r(r) = A\, r + \frac{B}{r^2},
\qquad
A = \frac{p\,a^3}{3\kappa(b^3-a^3)},
\quad
B = \frac{p\,a^3 b^3}{4\mu(b^3-a^3)},
```

and the non-zero strain components in the normalized spherical basis are

```math
\varepsilon_{rr} = A - \frac{2B}{r^3},
\qquad
\varepsilon_{\theta\theta} = \varepsilon_{\phi\phi} = A + \frac{B}{r^3}.
```

### Parameters

```@example numlame
using TensND, LinearAlgebra, Printf

CS = coorsys_spherical_num();   # coords: (θ, ϕ, r)
a, b   = 1.0, 3.0;              # inner / outer radius
p_i    = 1.0;                   # internal pressure
κ_val  = 2.0; μ_val = 1.0;     # bulk / shear modulus
λ_val  = κ_val - 2μ_val/3;
A = p_i * a^3 / (3κ_val * (b^3 - a^3))
```

```@example numlame
B = p_i * a^3 * b^3 / (4μ_val * (b^3 - a^3))
```

### Strain from SYMGRAD

The radial displacement field in the (θ, ϕ, r) normalized basis:

```@example numlame
u_func = x -> [0.0, 0.0, A*x[3] + B/x[3]^2];
ε_func = x -> SYMGRAD(u_func, CS)(x);
ε = ε_func([π/3, π/4, 2.0])
```

The numerical values match the analytical expressions:

```@example numlame
r₀ = 2.0;
A + B/r₀^3   # ε_θθ = ε_ϕϕ (exact)
```

```@example numlame
A - 2B/r₀^3  # ε_rr (exact)
```

### Stress and equilibrium

The isotropic constitutive law $\boldsymbol{\sigma} = \lambda\operatorname{tr}(\boldsymbol{\varepsilon})\mathbf{1} + 2\mu\,\boldsymbol{\varepsilon}$:

```@example numlame
σ_func = x -> begin
    ε    = Array(ε_func(x))
    tr_ε = sum(ε[i,i] for i in 1:3)
    [λ_val * tr_ε * (i==j ? 1.0 : 0.0) + 2μ_val * ε[i,j] for i in 1:3, j in 1:3]
end;
```

Verify equilibrium $\operatorname{div}\boldsymbol{\sigma} = \mathbf{0}$ and the boundary conditions $\sigma_{rr}(a) = -p$, $\sigma_{rr}(b) = 0$:

```@example numlame
norm(Array(DIV(σ_func, CS)([π/3, π/4, 1.5])))  # ≈ 0 (machine precision)
```

```@example numlame
σ_func([π/3, π/4, a])[3,3]   # σ_rr at inner radius ≈ -1
```

```@example numlame
σ_func([π/3, π/4, b])[3,3]   # σ_rr at outer radius ≈ 0
```

### Parametric study

```@example numlame
println("  r     ε_rr(num)   ε_rr(exact)  σ_rr(num)  σ_rr(exact)  |div σ|")
for r in [1.1, 1.5, 2.0, 2.5, 2.9]
    x    = [π/3, π/4, r]
    ε    = Array(ε_func(x))
    σ    = σ_func(x)
    divσ = norm(Array(DIV(σ_func, CS)(x)))
    @printf("  %.1f  %+.6f  %+.6f   %+.6f  %+.6f   %.2e\n",
            r, ε[3,3], A-2B/r^3, σ[3,3], 3κ_val*A-4μ_val*B/r^3, divσ)
end
```

## Generic constructor from a position vector

For coordinate systems not predefined, pass an `OM` function to `CoorSystemNum`. The Lamé coefficients and Christoffel symbols are computed automatically by nested automatic differentiation of the metric tensor $g_{ij} = \partial_i \mathbf{OM} \cdot \partial_j \mathbf{OM}$.

**Example — elliptic coordinates** $(μ, ν)$ in 2D:

```math
\mathbf{OM}(\mu, \nu) = a\begin{pmatrix}\cosh\mu\,\cos\nu \\ \sinh\mu\,\sin\nu\end{pmatrix},
\qquad
\chi_1 = \chi_2 = a\sqrt{\sinh^2\!\mu + \sin^2\!\nu}.
```

```@example numell
using TensND
a_ell  = 2.0;
OM_ell = x -> [a_ell * cosh(x[1]) * cos(x[2]),
               a_ell * sinh(x[1]) * sin(x[2])];
CS_ell = CoorSystemNum(OM_ell, 2);
Lame(CS_ell, [1.0, 0.8])    # χ₁ = χ₂ = a√(sinh²μ + sin²ν)
```

```@example numell
LAPLACE(x -> cosh(x[1])*cos(x[2]), CS_ell)([1.0, 0.8])  # harmonic → ≈ 0
```
