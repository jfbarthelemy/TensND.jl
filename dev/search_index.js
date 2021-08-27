var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = TensND","category":"page"},{"location":"#TensND","page":"Home","title":"TensND","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for TensND.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [TensND]","category":"page"},{"location":"#TensND.ϵ","page":"Home","title":"TensND.ϵ","text":"ϵ[i,j,k]\n\nLevi-Civita symbol ϵᵢⱼₖ=(i-j)(j-k)(k-i)/2\n\n\n\n\n\n","category":"constant"},{"location":"#TensND.Basis","page":"Home","title":"TensND.Basis","text":"Basis(v::AbstractArray{T,2}, ::Val{:cov})\nBasis{dim, T<:Number}()\nBasis(θ::T<:Number, ϕ::T<:Number, ψ::T<:Number)\n\nBasis built from a square matrix v where columns correspond either to\n\nprimal vectors ie eᵢ=v[:,i] if var=:cov as by default\ndual vectors ie eⁱ=v[:,i] if var=:cont.\n\nBasis without any argument refers to the canonical basis (CanonicalBasis) in Rᵈⁱᵐ (by default dim=3 and T=Sym)\n\nBasis can also be built from Euler angles (RotatedBasis) θ in 2D and (θ, ϕ, ψ) in 3D\n\nThe attributes of this object are\n\nBasis.e: square matrix defining the primal basis eᵢ=e[:,i]\nBasis.E: square matrix defining the dual basis eⁱ=E[:,i]\nBasis.g: square matrix defining the covariant components of the metric tensor gᵢⱼ=eᵢ⋅eⱼ=g[i,j]\nBasis.G: square matrix defining the contravariant components of the metric tensor gⁱʲ=eⁱ⋅eʲ=G[i,j]\n\nExamples\n\njulia> v = Sym[1 0 0; 0 1 0; 0 1 1] ; b = Basis(v)\n3×3 Basis{3, Sym}:\n 1  0  0\n 0  1  0\n 0  1  1\n\njulia> v = Sym[1 0 0; 0 1 0; 0 1 1] ; b = Basis(v, :cont)\n3×3 Basis{3, Sym}:\n 1  0   0\n 0  1  -1\n 0  0   1\n\njulia> θ, ϕ, ψ = symbols(\"θ, ϕ, ψ\", real = true) ; b = Basis(θ, ϕ, ψ) ; display(b.e)\n3×3 Tensor{2, 3, Sym, 9}:\n -sin(ψ)⋅sin(ϕ) + cos(θ)⋅cos(ψ)⋅cos(ϕ)  -sin(ψ)⋅cos(θ)⋅cos(ϕ) - sin(ϕ)⋅cos(ψ)  sin(θ)⋅cos(ϕ)\n  sin(ψ)⋅cos(ϕ) + sin(ϕ)⋅cos(θ)⋅cos(ψ)  -sin(ψ)⋅sin(ϕ)⋅cos(θ) + cos(ψ)⋅cos(ϕ)  sin(θ)⋅sin(ϕ)\n                        -sin(θ)⋅cos(ψ)                          sin(θ)⋅sin(ψ)         cos(θ)\n\n\n\n\n\n","category":"type"},{"location":"#TensND.CanonicalBasis","page":"Home","title":"TensND.CanonicalBasis","text":"CanonicalBasis{dim, T}\n\nCanonical basis of dimension dim (default: 3) and type T (default: Sym)\n\nThe attributes of this object are\n\nBasis.e: identity matrix defining the primal basis e[i,j]=δᵢⱼ\nBasis.E: identity matrix defining the dual basis g[i,j]=δᵢⱼ\nBasis.g: identity matrix defining the covariant components of the metric tensor g[i,j]=δᵢⱼ\nBasis.G: identity matrix defining the contravariant components of the metric tensor G[i,j]=δᵢⱼ\n\nExamples\n\njulia> b = CanonicalBasis()\n3×3 CanonicalBasis{3, Sym}:\n 1  0  0\n 0  1  0\n 0  0  1\n\njulia> b = CanonicalBasis{2, Float64}()\n2×2 CanonicalBasis{2, Float64}:\n 1.0  0.0\n 0.0  1.0\n\n\n\n\n\n","category":"type"},{"location":"#TensND.RotatedBasis","page":"Home","title":"TensND.RotatedBasis","text":"RotatedBasis(θ::T<:Number, ϕ::T<:Number, ψ::T<:Number)\nRotatedBasis(θ::T<:Number)\n\nOrthonormal basis of dimension dim (default: 3) and type T (default: Sym) built from Euler angles θ in 2D and (θ, ϕ, ψ) in 3D\n\nExamples\n\njulia> θ, ϕ, ψ = symbols(\"θ, ϕ, ψ\", real = true) ; b = RotatedBasis(θ, ϕ, ψ) ; display(b.e)\n3×3 Tensor{2, 3, Sym, 9}:\n -sin(ψ)⋅sin(ϕ) + cos(θ)⋅cos(ψ)⋅cos(ϕ)  -sin(ψ)⋅cos(θ)⋅cos(ϕ) - sin(ϕ)⋅cos(ψ)  sin(θ)⋅cos(ϕ)\n  sin(ψ)⋅cos(ϕ) + sin(ϕ)⋅cos(θ)⋅cos(ψ)  -sin(ψ)⋅sin(ϕ)⋅cos(θ) + cos(ψ)⋅cos(ϕ)  sin(θ)⋅sin(ϕ)\n                        -sin(θ)⋅cos(ψ)                          sin(θ)⋅sin(ψ)         cos(θ)\n\n\n\n\n\n","category":"type"},{"location":"#TensND.Tensnd","page":"Home","title":"TensND.Tensnd","text":"Tensnd{order,dim,TA<:Number,TB<:Number,A<:AbstractArray,B<:AbstractBasis}\n\nTensor type of any order defined by\n\na multiarray of components (of any type heriting from AbstractArray, e.g. Tensor or SymmetricTensor)\na basis of AbstractBasis type\na tuple of variances (covariant :cov or contravariant :cont) of length equal to the order of the tensor\n\nExamples\n\njulia> v = Sym[1 0 0; 0 1 0; 0 1 1] ; b = Basis(v)\n3×3 Basis{3, Sym}:\n 1  0  0\n 0  1  0\n 0  1  1\n\njulia> T = Tensnd(b.g,(:cov,:cov),b)\n3×3 Tensnd{2, 3, Sym, Sym, SymmetricTensor{2, 3, Sym, 6}, Basis{3, Sym}}:\n 1  0  0\n 0  2  1\n 0  1  1\n\njulia> components(T,(:cont,:cov),b)\n3×3 Matrix{Sym}:\n 1  0  0\n 0  1  0\n 0  0  1\n\n\n\n\n\n","category":"type"},{"location":"#LinearAlgebra.normalize","page":"Home","title":"LinearAlgebra.normalize","text":"normalize(b::AbstractBasis, var = cov)\n\nBuilds a normalized basis from the input basis b by calling normal_basis\n\n\n\n\n\n","category":"function"},{"location":"#TensND.components-Union{Tuple{T}, Tuple{dim}, Tuple{order}, Tuple{Tensnd{order, dim, T, TB, A, B} where {TB, A, B}, Tuple{Vararg{Symbol, order}}}} where {order, dim, T<:Number}","page":"Home","title":"TensND.components","text":"components(::Tensnd{order,dim,T}, ::NTuple{order,Symbol})\ncomponents(::Tensnd{order,dim,T}, ::NTuple{order,Symbol}, ::AbstractBasis{dim,T})\n\nExtracts the components of a tensor for new variances and/or in a new basis\n\nExamples\n\njulia> v = Sym[0 1 1; 1 0 1; 1 1 0] ; b = Basis(v)\n3×3 Basis{3, Sym}:\n 0  1  1\n 1  0  1\n 1  1  0\n\njulia> V = Tensor{1,3}(i->symbols(\"v$i\",real=true))\n3-element Vec{3, Sym}:\n v₁\n v₂\n v₃\n\njulia> TV = Tensnd(V) # TV = Tensnd(V, (:cont,), CanonicalBasis())\n3-element Tensnd{1, 3, Sym, Sym, Vec{3, Sym}, CanonicalBasis{3, Sym}}:\n v₁\n v₂\n v₃\n\njulia> factor.(components(TV, (:cont,), b))\n3-element Vector{Sym}:\n -(v1 - v2 - v3)/2\n  (v1 - v2 + v3)/2\n  (v1 + v2 - v3)/2\n\njulia> components(TV, (:cov,), b)\n3-element Vector{Sym}:\n v₂ + v₃\n v₁ + v₃\n v₁ + v₂\n\njulia> simplify.(components(TV, (:cov,), normal_basis(b)))\n3-element Vector{Sym}:\n sqrt(2)*(v2 + v3)/2\n sqrt(2)*(v1 + v3)/2\n sqrt(2)*(v1 + v2)/2\n\njulia> T = Tensor{2,3}((i,j)->symbols(\"t$i$j\",real=true))\n3×3 Tensor{2, 3, Sym, 9}:\n t₁₁  t₁₂  t₁₃\n t₂₁  t₂₂  t₂₃\n t₃₁  t₃₂  t₃₃\n\njulia> TT = Tensnd(T)\n3×3 Tensnd{2, 3, Sym, Sym, Tensor{2, 3, Sym, 9}, CanonicalBasis{3, Sym}}:\n t₁₁  t₁₂  t₁₃\n t₂₁  t₂₂  t₂₃\n t₃₁  t₃₂  t₃₃\n\njulia> components(TT, (:cov,:cov), b)\n3×3 Matrix{Sym}:\n t₂₂ + t₂₃ + t₃₂ + t₃₃  t₂₁ + t₂₃ + t₃₁ + t₃₃  t₂₁ + t₂₂ + t₃₁ + t₃₂\n t₁₂ + t₁₃ + t₃₂ + t₃₃  t₁₁ + t₁₃ + t₃₁ + t₃₃  t₁₁ + t₁₂ + t₃₁ + t₃₂\n t₁₂ + t₁₃ + t₂₂ + t₂₃  t₁₁ + t₁₃ + t₂₁ + t₂₃  t₁₁ + t₁₂ + t₂₁ + t₂₂\n\njulia> factor.(components(TT, (:cont,:cov), b))\n3×3 Matrix{Sym}:\n -(t12 + t13 - t22 - t23 - t32 - t33)/2  …  -(t11 + t12 - t21 - t22 - t31 - t32)/2\n  (t12 + t13 - t22 - t23 + t32 + t33)/2      (t11 + t12 - t21 - t22 + t31 + t32)/2\n  (t12 + t13 + t22 + t23 - t32 - t33)/2      (t11 + t12 + t21 + t22 - t31 - t32)/2\n\n\n\n\n\n","category":"method"},{"location":"#TensND.fϵ-Union{Tuple{Int64, Int64, Int64}, Tuple{T}, Tuple{Int64, Int64, Int64, Type{var\"#s12\"} where var\"#s12\"<:T}} where T","page":"Home","title":"TensND.fϵ","text":"fϵ(T, i::Int, j::Int, k::Int)\nfϵ(i::Int, j::Int, k::Int) = fϵ(Int, i::Int, j::Int, k::Int)\n\nFunction giving Levi-Civita symbol ϵᵢⱼₖ = (i-j) (j-k) (k-i) / 2\n\n\n\n\n\n","category":"method"},{"location":"#TensND.isorthogonal-Tuple{AbstractBasis}","page":"Home","title":"TensND.isorthogonal","text":"isorthogonal(b::AbstractBasis)\n\nChecks whether the basis b is orthogonal\n\n\n\n\n\n","category":"method"},{"location":"#TensND.metric-Tuple{AbstractBasis, Val{:cov}}","page":"Home","title":"TensND.metric","text":"metric(b::AbstractBasis, var = :cov)\n\nReturns the covariant (if var = :cov) or contravariant (if var = :cont) metric matrix\n\n\n\n\n\n","category":"method"},{"location":"#TensND.normal_basis-Union{Tuple{AbstractMatrix{T}}, Tuple{T}, Tuple{AbstractMatrix{T}, Any}} where T","page":"Home","title":"TensND.normal_basis","text":"normal_basis(v::AbstractArray{T,2}, var = :cov) where {T}\n\nBuilds a basis after normalization of column vectors of input matrix v where columns define either\n\nprimal vectors ie eᵢ=v[:,i]/norm(v[:,i]) if var = :cov as by default\ndual vector ie eⁱ=v[:,i]/norm(v[:,i]) if var = :cont.\n\n\n\n\n\n","category":"method"},{"location":"#TensND.vecbasis-Tuple{AbstractBasis, Val{:cov}}","page":"Home","title":"TensND.vecbasis","text":"vecbasis(b::AbstractBasis, var = :cov)\n\nReturns the primal (if var = :cov) or primal (if var = :cont) basis\n\n\n\n\n\n","category":"method"}]
}
