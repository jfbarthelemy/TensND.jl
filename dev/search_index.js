var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = TensND","category":"page"},{"location":"#TensND","page":"Home","title":"TensND","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for TensND.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [TensND]","category":"page"},{"location":"#TensND.Basis","page":"Home","title":"TensND.Basis","text":"Basis(v::AbstractArray{T,2}, ::Val{:cov})\nBasis{dim, T<:Number}()\nBasis(θ::T<:Number, ϕ::T<:Number, ψ::T<:Number)\n\nBasis built from a square matrix v where columns correspond either to\n\nprimal vectors ie eᵢ=v[:,i] if var=:cov as by default\ndual vectors ie eⁱ=v[:,i] if var=:cont.\n\nBasis without any argument refers to the canonical basis (CanonicalBasis) in Rᵈⁱᵐ (by default dim=3 and T=Sym)\n\nBasis can also be built from Euler angles (RotatedBasis) θ in 2D and (θ, ϕ, ψ) in 3D\n\nThe attributes of this object can be obtained by\n\nvecbasis(ℬ, :cov): square matrix defining the primal basis eᵢ=e[:,i]\nvecbasis(ℬ, :cont): square matrix defining the dual basis eⁱ=E[:,i]\nmetric(ℬ, :cov): square matrix defining the covariant components of the metric tensor gᵢⱼ=eᵢ⋅eⱼ=g[i,j]\nmetric(ℬ, :cont): square matrix defining the contravariant components of the metric tensor gⁱʲ=eⁱ⋅eʲ=G[i,j]\n\nExamples\n\njulia> v = Sym[1 0 0; 0 1 0; 0 1 1] ; ℬ = Basis(v)\nBasis{3, Sym}\n# basis: 3×3 Tensor{2, 3, Sym, 9}:\n 1  0  0\n 0  1  0\n 0  1  1\n# dual basis: 3×3 Tensor{2, 3, Sym, 9}:\n 1  0   0\n 0  1  -1\n 0  0   1\n# covariant metric tensor: 3×3 SymmetricTensor{2, 3, Sym, 6}:\n 1  0  0\n 0  2  1\n 0  1  1\n# contravariant metric tensor: 3×3 SymmetricTensor{2, 3, Sym, 6}:\n 1   0   0\n 0   1  -1\n 0  -1   2\n\njulia> θ, ϕ, ψ = symbols(\"θ, ϕ, ψ\", real = true) ; ℬʳ = Basis(θ, ϕ, ψ) ; display(vecbasis(ℬʳ, :cov))\n3×3 Tensor{2, 3, Sym, 9}:\n -sin(ψ)⋅sin(ϕ) + cos(θ)⋅cos(ψ)⋅cos(ϕ)  -sin(ψ)⋅cos(θ)⋅cos(ϕ) - sin(ϕ)⋅cos(ψ)  sin(θ)⋅cos(ϕ)\n  sin(ψ)⋅cos(ϕ) + sin(ϕ)⋅cos(θ)⋅cos(ψ)  -sin(ψ)⋅sin(ϕ)⋅cos(θ) + cos(ψ)⋅cos(ϕ)  sin(θ)⋅sin(ϕ)\n                        -sin(θ)⋅cos(ψ)                          sin(θ)⋅sin(ψ)         cos(θ)\n\n\n\n\n\n","category":"type"},{"location":"#TensND.CanonicalBasis","page":"Home","title":"TensND.CanonicalBasis","text":"CanonicalBasis{dim, T}\n\nCanonical basis of dimension dim (default: 3) and type T (default: Sym)\n\nThe attributes of this object can be obtained by\n\nvecbasis(ℬ, :cov): square matrix defining the primal basis eᵢ=e[:,i]=δᵢⱼ\nvecbasis(ℬ, :cont): square matrix defining the dual basis eⁱ=E[:,i]=δᵢⱼ\nmetric(ℬ, :cov): square matrix defining the covariant components of the metric tensor gᵢⱼ=eᵢ⋅eⱼ=g[i,j]=δᵢⱼ\nmetric(ℬ, :cont): square matrix defining the contravariant components of the metric tensor gⁱʲ=eⁱ⋅eʲ=G[i,j]=δᵢⱼ\n\nExamples\n\njulia> ℬ = CanonicalBasis()\nCanonicalBasis{3, Sym}\n# basis: 3×3 TensND.LazyIdentity{3, Sym}:\n 1  0  0\n 0  1  0\n 0  0  1\n# dual basis: 3×3 TensND.LazyIdentity{3, Sym}:\n 1  0  0\n 0  1  0\n 0  0  1\n# covariant metric tensor: 3×3 TensND.LazyIdentity{3, Sym}:\n 1  0  0\n 0  1  0\n 0  0  1\n# contravariant metric tensor: 3×3 TensND.LazyIdentity{3, Sym}:\n 1  0  0\n 0  1  0\n 0  0  1\n\njulia> ℬ₂ = CanonicalBasis{2, Float64}()\nCanonicalBasis{2, Float64}\n# basis: 2×2 TensND.LazyIdentity{2, Float64}:\n 1.0  0.0\n 0.0  1.0\n# dual basis: 2×2 TensND.LazyIdentity{2, Float64}:\n 1.0  0.0\n 0.0  1.0\n# covariant metric tensor: 2×2 TensND.LazyIdentity{2, Float64}:\n 1.0  0.0\n 0.0  1.0\n# contravariant metric tensor: 2×2 TensND.LazyIdentity{2, Float64}:\n 1.0  0.0\n 0.0  1.0\n\n\n\n\n\n","category":"type"},{"location":"#TensND.RotatedBasis","page":"Home","title":"TensND.RotatedBasis","text":"RotatedBasis(θ::T<:Number, ϕ::T<:Number, ψ::T<:Number)\nRotatedBasis(θ::T<:Number)\n\nOrthonormal basis of dimension dim (default: 3) and type T (default: Sym) built from Euler angles θ in 2D and (θ, ϕ, ψ) in 3D\n\nExamples\n\njulia> θ, ϕ, ψ = symbols(\"θ, ϕ, ψ\", real = true) ; ℬʳ = RotatedBasis(θ, ϕ, ψ) ; display(vecbasis(ℬʳ, :cov))\n3×3 Tensor{2, 3, Sym, 9}:\n -sin(ψ)⋅sin(ϕ) + cos(θ)⋅cos(ψ)⋅cos(ϕ)  -sin(ψ)⋅cos(θ)⋅cos(ϕ) - sin(ϕ)⋅cos(ψ)  sin(θ)⋅cos(ϕ)\n  sin(ψ)⋅cos(ϕ) + sin(ϕ)⋅cos(θ)⋅cos(ψ)  -sin(ψ)⋅sin(ϕ)⋅cos(θ) + cos(ψ)⋅cos(ϕ)  sin(θ)⋅sin(ϕ)\n                        -sin(θ)⋅cos(ψ)                          sin(θ)⋅sin(ψ)         cos(θ)\n\n\n\n\n\n","category":"type"},{"location":"#TensND.Tensnd","page":"Home","title":"TensND.Tensnd","text":"Tensnd{order,dim,T,A<:AbstractArray,B<:AbstractBasis}\n\nTensor type of any order defined by\n\na multiarray of components (of any type heriting from AbstractArray, e.g. Tensor or SymmetricTensor)\na basis of AbstractBasis type\na tuple of variances (covariant :cov or contravariant :cont) of length equal to the order of the tensor\n\nExamples\n\njulia> ℬ = Basis(Sym[1 0 0; 0 1 0; 0 1 1]) ;\n\njulia> T = Tensnd(metric(ℬ,:cov),ℬ,(:cov,:cov))\nTensnd{2, 3, Sym, SymmetricTensor{2, 3, Sym, 6}}\n# data: 3×3 SymmetricTensor{2, 3, Sym, 6}:\n 1  0  0\n 0  2  1\n 0  1  1\n# basis: 3×3 Tensor{2, 3, Sym, 9}:\n 1  0  0\n 0  1  0\n 0  1  1\n# var: (:cov, :cov)\n\njulia> components(T,(:cont,:cov),b)\n3×3 Matrix{Sym}:\n 1  0  0\n 0  1  0\n 0  0  1\n\n\n\n\n\n","category":"type"},{"location":"#LinearAlgebra.dot-Union{Tuple{dim}, Tuple{order2}, Tuple{order1}, Tuple{TensND.AbstractTensnd{order1, dim, T, A} where {T<:Number, A<:AbstractArray}, TensND.AbstractTensnd{order2, dim, T, A} where {T<:Number, A<:AbstractArray}}} where {order1, order2, dim}","page":"Home","title":"LinearAlgebra.dot","text":"dot(t1::AbstractTensnd{order1,dim}, t2::AbstractTensnd{order2,dim})\n\nDefines a contracted product between two tensors\n\na ⋅ b = aⁱbⱼ\n\n\n\n\n\n","category":"method"},{"location":"#LinearAlgebra.normalize","page":"Home","title":"LinearAlgebra.normalize","text":"normalize(ℬ::AbstractBasis, var = cov)\n\nBuilds a basis after normalization of column vectors of input matrix v where columns define either\n\nprimal vectors ie eᵢ=v[:,i]/norm(v[:,i]) if var = :cov as by default\ndual vector ie eⁱ=v[:,i]/norm(v[:,i]) if var = :cont.\n\n\n\n\n\n","category":"function"},{"location":"#TensND.KM-Tuple{Union{Tensors.Vec{dim, T}, Tensors.SymmetricTensor{2, dim, T, M} where M, Tensors.SymmetricTensor{4, dim, T, M} where M, Tensors.Tensor{2, dim, T, M} where M, Tensors.Tensor{4, dim, T, M} where M} where {dim, T}}","page":"Home","title":"TensND.KM","text":"KM(t::AbstractTensnd{order,dim}; kwargs...)\nKM(t::AbstractTensnd{order,dim}, var::NTuple{order,Symbol}, b::AbstractBasis{dim}; kwargs...)\n\nWrites the components of a second or fourth order tensor in Kelvin-Mandel notation\n\nExamples\n\njulia> σ = Tensnd(SymmetricTensor{2,3}((i, j) -> symbols(\"σ$i$j\", real = true))) ;\n\njulia> KM(σ)\n6-element Vector{Sym}:\n         σ11\n         σ22\n         σ33\n      √2⋅σ32\n      √2⋅σ31\n      √2⋅σ21\n\njulia> C = Tensnd(SymmetricTensor{4,3}((i, j, k, l) -> symbols(\"C$i$j$k$l\", real = true))) ;\n\njulia> KM(C)\n6×6 Matrix{Sym}:\n         C₁₁₁₁     C₁₁₂₂     C₁₁₃₃  √2⋅C₁₁₃₂  √2⋅C₁₁₃₁  √2⋅C₁₁₂₁\n         C₂₂₁₁     C₂₂₂₂     C₂₂₃₃  √2⋅C₂₂₃₂  √2⋅C₂₂₃₁  √2⋅C₂₂₂₁\n         C₃₃₁₁     C₃₃₂₂     C₃₃₃₃  √2⋅C₃₃₃₂  √2⋅C₃₃₃₁  √2⋅C₃₃₂₁\n      √2⋅C₃₂₁₁  √2⋅C₃₂₂₂  √2⋅C₃₂₃₃   2⋅C₃₂₃₂   2⋅C₃₂₃₁   2⋅C₃₂₂₁\n      √2⋅C₃₁₁₁  √2⋅C₃₁₂₂  √2⋅C₃₁₃₃   2⋅C₃₁₃₂   2⋅C₃₁₃₁   2⋅C₃₁₂₁\n      √2⋅C₂₁₁₁  √2⋅C₂₁₂₂  √2⋅C₂₁₃₃   2⋅C₂₁₃₂   2⋅C₂₁₃₁   2⋅C₂₁₂₁\n\n\n\n\n\n","category":"method"},{"location":"#TensND.LeviCivita","page":"Home","title":"TensND.LeviCivita","text":"LeviCivita(T::Type{<:Number} = Sym)\n\nBuilds an Array{T,3} of Levi-Civita Symbol ϵᵢⱼₖ = (i-j) (j-k) (k-i) / 2\n\nExamples\n\njulia> ε = LeviCivita(Sym)\n3×3×3 Array{Sym, 3}:\n[:, :, 1] =\n 0   0  0\n 0   0  1\n 0  -1  0\n\n[:, :, 2] =\n 0  0  -1\n 0  0   0\n 1  0   0\n\n[:, :, 3] =\n  0  1  0\n -1  0  0\n  0  0  0\n\n\n\n\n\n","category":"function"},{"location":"#TensND.angles-Union{Tuple{AbstractMatrix{T}}, Tuple{T}} where T","page":"Home","title":"TensND.angles","text":"angles(M::AbstractArray{T,2})\n\nDetermines the Euler angles corresponding to the input matrix supposed to be a rotation matrix or at least a similarity\n\nExamples\n\njulia> θ, ϕ, ψ = symbols(\"θ, ϕ, ψ\", real = true) ; ℬʳ = RotatedBasis(θ, ϕ, ψ) ; display(vecbasis(ℬʳ, :cov))\n3×3 Tensor{2, 3, Sym, 9}:\n -sin(ψ)⋅sin(ϕ) + cos(θ)⋅cos(ψ)⋅cos(ϕ)  -sin(ψ)⋅cos(θ)⋅cos(ϕ) - sin(ϕ)⋅cos(ψ)  sin(θ)⋅cos(ϕ)\n  sin(ψ)⋅cos(ϕ) + sin(ϕ)⋅cos(θ)⋅cos(ψ)  -sin(ψ)⋅sin(ϕ)⋅cos(θ) + cos(ψ)⋅cos(ϕ)  sin(θ)⋅sin(ϕ)\n                        -sin(θ)⋅cos(ψ)                          sin(θ)⋅sin(ψ)         cos(θ)\n\njulia> angles(ℬʳ)\n(θ = θ, ϕ = ϕ, ψ = ψ)\n\n\n\n\n\n","category":"method"},{"location":"#TensND.change_tens-Tuple{TensND.AbstractTensnd, AbstractBasis, Tuple{Vararg{T, N}} where {N, T}}","page":"Home","title":"TensND.change_tens","text":"change_tens(t::AbstractTensnd{order,dim,T},ℬ::AbstractBasis{dim},var::NTuple{order,Symbol})\nchange_tens(t::AbstractTensnd{order,dim,T},ℬ::AbstractBasis{dim})\nchange_tens(t::AbstractTensnd{order,dim,T},var::NTuple{order,Symbol})\n\nRewrites the same tensor with components corresponding to new variances and/or to a new basis\n\njulia> ℬ = Basis(Sym[0 1 1; 1 0 1; 1 1 0]) ;\n\njulia> TV = Tensnd(Tensor{1,3}(i->symbols(\"v$i\",real=true)))\nTensND.TensndCanonical{1, 3, Sym, Vec{3, Sym}}\n# data: 3-element Vec{3, Sym}:\n v₁\n v₂\n v₃\n# basis: 3×3 TensND.LazyIdentity{3, Sym}:\n 1  0  0\n 0  1  0\n 0  0  1\n# var: (:cont,)\n\njulia> factor.(components(TV, ℬ, (:cont,)))\n3-element Vector{Sym}:\n -(v1 - v2 - v3)/2\n  (v1 - v2 + v3)/2\n  (v1 + v2 - v3)/2\n\njulia> ℬ₀ = Basis(Sym[0 1 1; 1 0 1; 1 1 1]) ;\n\njulia> TV0 = change_tens(TV, ℬ₀)\nTensnd{1, 3, Sym, Vec{3, Sym}}\n# data: 3-element Vec{3, Sym}:\n     -v₁ + v₃\n     -v₂ + v₃\n v₁ + v₂ - v₃\n# basis: 3×3 Tensor{2, 3, Sym, 9}:\n 0  1  1\n 1  0  1\n 1  1  1\n# var: (:cont,)\n\n\n\n\n\n","category":"method"},{"location":"#TensND.change_tens_canon-Tuple{TensND.AbstractTensnd}","page":"Home","title":"TensND.change_tens_canon","text":"change_tens_canon(t::AbstractTensnd{order,dim,T},var::NTuple{order,Symbol})\n\nRewrites the same tensor with components corresponding to the canonical basis\n\njulia> ℬ = Basis(Sym[0 1 1; 1 0 1; 1 1 0]) ;\n\njulia> TV = Tensnd(Tensor{1,3}(i->symbols(\"v$i\",real=true)), ℬ)\nTensnd{1, 3, Sym, Vec{3, Sym}}\n# data: 3-element Vec{3, Sym}:\n v₁\n v₂\n v₃\n# basis: 3×3 Tensor{2, 3, Sym, 9}:\n 0  1  1\n 1  0  1\n 1  1  1\n# var: (:cont,)\n\njulia> TV0 = change_tens_canon(TV)\nTensND.TensndCanonical{1, 3, Sym, Vec{3, Sym}}\n# data: 3-element Vec{3, Sym}:\n      v₂ + v₃\n      v₁ + v₃\n v₁ + v₂ + v₃\n# basis: 3×3 TensND.LazyIdentity{3, Sym}:\n 1  0  0\n 0  1  0\n 0  0  1\n# var: (:cont,)\n\n\n\n\n\n","category":"method"},{"location":"#TensND.components-Tuple{TensND.AbstractTensnd}","page":"Home","title":"TensND.components","text":"components(t::AbstractTensnd{order,dim,T},ℬ::AbstractBasis{dim},var::NTuple{order,Symbol})\ncomponents(t::AbstractTensnd{order,dim,T},ℬ::AbstractBasis{dim})\ncomponents(t::AbstractTensnd{order,dim,T},var::NTuple{order,Symbol})\n\nExtracts the components of a tensor for new variances and/or in a new basis\n\nExamples\n\njulia> ℬ = Basis(Sym[0 1 1; 1 0 1; 1 1 0]) ;\n\njulia> TV = Tensnd(Tensor{1,3}(i->symbols(\"v$i\",real=true)))\nTensND.TensndCanonical{1, 3, Sym, Vec{3, Sym}}\n# data: 3-element Vec{3, Sym}:\n v₁\n v₂\n v₃\n# basis: 3×3 TensND.LazyIdentity{3, Sym}:\n 1  0  0\n 0  1  0\n 0  0  1\n# var: (:cont,)\n\njulia> factor.(components(TV, ℬ, (:cont,)))\n3-element Vector{Sym}:\n -(v1 - v2 - v3)/2\n  (v1 - v2 + v3)/2\n  (v1 + v2 - v3)/2\n\njulia> components(TV, ℬ, (:cov,))\n3-element Vector{Sym}:\n v₂ + v₃\n v₁ + v₃\n v₁ + v₂\n\njulia> simplify.(components(TV, normalize(ℬ), (:cov,)))\n3-element Vector{Sym}:\n sqrt(2)*(v2 + v3)/2\n sqrt(2)*(v1 + v3)/2\n sqrt(2)*(v1 + v2)/2\n\njulia> TT = Tensnd(Tensor{2,3}((i,j)->symbols(\"t$i$j\",real=true)))\nTensND.TensndCanonical{2, 3, Sym, Tensor{2, 3, Sym, 9}}\n# data: 3×3 Tensor{2, 3, Sym, 9}:\n t₁₁  t₁₂  t₁₃\n t₂₁  t₂₂  t₂₃\n t₃₁  t₃₂  t₃₃\n# basis: 3×3 TensND.LazyIdentity{3, Sym}:\n 1  0  0\n 0  1  0\n 0  0  1\n# var: (:cont, :cont)\n\njulia> components(TT, ℬ, (:cov,:cov))\n3×3 Matrix{Sym}:\n t₂₂ + t₂₃ + t₃₂ + t₃₃  t₂₁ + t₂₃ + t₃₁ + t₃₃  t₂₁ + t₂₂ + t₃₁ + t₃₂\n t₁₂ + t₁₃ + t₃₂ + t₃₃  t₁₁ + t₁₃ + t₃₁ + t₃₃  t₁₁ + t₁₂ + t₃₁ + t₃₂\n t₁₂ + t₁₃ + t₂₂ + t₂₃  t₁₁ + t₁₃ + t₂₁ + t₂₃  t₁₁ + t₁₂ + t₂₁ + t₂₂\n\njulia> factor.(components(TT, ℬ, (:cont,:cov)))\n3×3 Matrix{Sym}:\n -(t12 + t13 - t22 - t23 - t32 - t33)/2  …  -(t11 + t12 - t21 - t22 - t31 - t32)/2\n  (t12 + t13 - t22 - t23 + t32 + t33)/2      (t11 + t12 - t21 - t22 + t31 + t32)/2\n  (t12 + t13 + t22 + t23 - t32 - t33)/2      (t11 + t12 + t21 + t22 - t31 - t32)/2\n\n\n\n\n\n","category":"method"},{"location":"#TensND.components_canon-Tuple{TensND.AbstractTensnd}","page":"Home","title":"TensND.components_canon","text":"components_canon(t::AbstractTensnd)\n\nExtracts the components of a tensor in the canonical basis\n\n\n\n\n\n","category":"method"},{"location":"#TensND.init_canonical","page":"Home","title":"TensND.init_canonical","text":"init_canonical(T::Type{<:Number} = Sym)\n\nReturns the canonical basis and the 3 unit vectors\n\nExamples\n\njulia> ℬ, 𝐞₁, 𝐞₂, 𝐞₃ = init_canonical()\n(Sym[1 0 0; 0 1 0; 0 0 1], Sym[1, 0, 0], Sym[0, 1, 0], Sym[0, 0, 1])\n\n\n\n\n\n","category":"function"},{"location":"#TensND.init_cylindrical-Tuple{Any}","page":"Home","title":"TensND.init_cylindrical","text":"init_cylindrical(θ ; canonical = false)\n\nReturns the angle, the cylindrical basis and the 3 unit vectors\n\nExamples\n\njulia> θ, ℬᶜ, 𝐞ʳ, 𝐞ᶿ, 𝐞ᶻ = init_cylindrical(symbols(\"θ\", real = true)) ;\n\n\n\n\n\n","category":"method"},{"location":"#TensND.init_isotropic","page":"Home","title":"TensND.init_isotropic","text":"init_isotropic(T::Type{<:Number} = Sym)\n\nReturns the isotropic tensors\n\nExamples\n\njulia> 𝟏, 𝟙, 𝕀, 𝕁, 𝕂 = init_isotropic() ;\n\n\n\n\n\n","category":"function"},{"location":"#TensND.init_polar-Tuple{Any}","page":"Home","title":"TensND.init_polar","text":"init_polar(θ ; canonical = false)\n\nReturns the angle, the polar basis and the 2 unit vectors\n\nExamples\n\njulia> θ, ℬᵖ, 𝐞ʳ, 𝐞ᶿ = init_polar(symbols(\"θ\", real = true)) ;\n\n\n\n\n\n","category":"method"},{"location":"#TensND.init_rotated-Tuple{Any, Any, Any}","page":"Home","title":"TensND.init_rotated","text":"init_rotated(θ, ϕ, ψ; canonical = false)\n\nReturns the angles, the ratated basis and the 3 unit vectors\n\nExamples\n\njulia> θ, ϕ, ψ, ℬʳ, 𝐞ᶿ, 𝐞ᵠ, 𝐞ʳ = init_rotated(symbols(\"θ ϕ ψ\", real = true)...) ;\n\n\n\n\n\n","category":"method"},{"location":"#TensND.init_spherical-Tuple{Any, Any}","page":"Home","title":"TensND.init_spherical","text":"init_spherical(θ, ϕ; canonical = false)\n\nReturns the angles, the spherical basis and the 3 unit vectors. Take care that the order of the 3 vectors is 𝐞ᶿ, 𝐞ᵠ, 𝐞ʳ so that the basis coincides with the canonical one when the angles are null.\n\nExamples\n\njulia> θ, ϕ, ℬˢ, 𝐞ᶿ, 𝐞ᵠ, 𝐞ʳ = init_spherical(symbols(\"θ ϕ\", real = true)...) ;\n\n\n\n\n\n","category":"method"},{"location":"#TensND.invKM-Tuple{Type{var\"#s21\"} where var\"#s21\"<:(Union{Tensors.Vec{dim, T}, Tensors.SymmetricTensor{2, dim, T, M} where M, Tensors.SymmetricTensor{4, dim, T, M} where M, Tensors.Tensor{2, dim, T, M} where M, Tensors.Tensor{4, dim, T, M} where M} where {dim, T}), AbstractVecOrMat{T} where T}","page":"Home","title":"TensND.invKM","text":"invKM(v::AbstractVecOrMat; kwargs...)\n\nDefines a tensor from a Kelvin-Mandel vector or matrix representation\n\n\n\n\n\n","category":"method"},{"location":"#TensND.isorthogonal-Tuple{AbstractBasis}","page":"Home","title":"TensND.isorthogonal","text":"isorthogonal(ℬ::AbstractBasis)\n\nChecks whether the basis ℬ is orthogonal\n\n\n\n\n\n","category":"method"},{"location":"#TensND.isorthonormal-Tuple{AbstractBasis}","page":"Home","title":"TensND.isorthonormal","text":"isorthonormal(ℬ::AbstractBasis)\n\nChecks whether the basis ℬ is orthonormal\n\n\n\n\n\n","category":"method"},{"location":"#TensND.metric-Tuple{AbstractBasis, Val{:cov}}","page":"Home","title":"TensND.metric","text":"metric(ℬ::AbstractBasis, var = :cov)\n\nReturns the covariant (if var = :cov) or contravariant (if var = :cont) metric matrix\n\n\n\n\n\n","category":"method"},{"location":"#TensND.otimesul-Union{Tuple{dim}, Tuple{order2}, Tuple{order1}, Tuple{TensND.AbstractTensnd{order1, dim, T, A} where {T<:Number, A<:AbstractArray}, TensND.AbstractTensnd{order2, dim, T, A} where {T<:Number, A<:AbstractArray}}} where {order1, order2, dim}","page":"Home","title":"TensND.otimesul","text":"otimesul(t1::AbstractTensnd{order1,dim}, t2::AbstractTensnd{order2,dim})\n\nDefines a special tensor product between two tensors of at least second order\n\n(𝐚 ⊠ˢ 𝐛) ⊡ 𝐩 = (𝐚 ⊠ 𝐛) ⊡ (𝐩 + ᵗ𝐩)/2  = 1/2(aⁱᵏbʲˡ+aⁱˡbʲᵏ) pₖₗ eᵢ⊗eⱼ\n\n\n\n\n\n","category":"method"},{"location":"#TensND.qcontract-Union{Tuple{order2}, Tuple{order1}, Tuple{T2}, Tuple{T1}, Tuple{AbstractArray{T1, order1}, AbstractArray{T2, order2}}} where {T1, T2, order1, order2}","page":"Home","title":"TensND.qcontract","text":"dcontract(t1::AbstractTensnd{order1,dim}, t2::AbstractTensnd{order2,dim})\n\nDefines a quadruple contracted product between two tensors\n\n𝔸 ⊙ 𝔹 = AᵢⱼₖₗBⁱʲᵏˡ\n\nExamples\n\njulia> 𝕀 = t𝕀(Sym) ; 𝕁 = t𝕁(Sym) ; 𝕂 = t𝕂(Sym) ;\n\njulia> 𝕀 ⊙ 𝕀\n6\n\njulia> 𝕁 ⊙ 𝕀\n1\n\njulia> 𝕂 ⊙ 𝕀\n5\n\njulia> 𝕂 ⊙ 𝕁\n0\n\n\n\n\n\n","category":"method"},{"location":"#TensND.rot2-Tuple{Any}","page":"Home","title":"TensND.rot2","text":"rot2(θ)\n\nReturns a 2D rotation matrix with respect to the angle θ\n\nExamples\n\njulia> rot2(θ)\n2×2 Tensor{2, 2, Sym, 4}:\n cos(θ)  -sin(θ)\n sin(θ)   cos(θ)\n\n\n\n\n\n","category":"method"},{"location":"#TensND.rot3","page":"Home","title":"TensND.rot3","text":"rot3(θ, ϕ = 0, ψ = 0)\n\nReturns a rotation matrix with respect to the 3 Euler angles θ, ϕ, ψ\n\nExamples\n\njulia> cθ, cϕ, cψ, sθ, sϕ, sψ = symbols(\"cθ cϕ cψ sθ sϕ sψ\", real = true) ;\n\njulia> d = Dict(cos(θ) => cθ, cos(ϕ) => cϕ, cos(ψ) => cψ, sin(θ) => sθ, sin(ϕ) => sϕ, sin(ψ) => sψ) ;\n\njulia> subs.(rot3(θ, ϕ, ψ),d...)\n3×3 StaticArrays.SMatrix{3, 3, Sym, 9} with indices SOneTo(3)×SOneTo(3):\n cθ⋅cψ⋅cϕ - sψ⋅sϕ  -cθ⋅cϕ⋅sψ - cψ⋅sϕ  cϕ⋅sθ\n cθ⋅cψ⋅sϕ + cϕ⋅sψ  -cθ⋅sψ⋅sϕ + cψ⋅cϕ  sθ⋅sϕ\n           -cψ⋅sθ              sθ⋅sψ     cθ\n\n\n\n\n\n","category":"function"},{"location":"#TensND.rot6","page":"Home","title":"TensND.rot6","text":"rot6(θ, ϕ = 0, ψ = 0)\n\nReturns a rotation matrix with respect to the 3 Euler angles θ, ϕ, ψ\n\nExamples\n\njulia> cθ, cϕ, cψ, sθ, sϕ, sψ = symbols(\"cθ cϕ cψ sθ sϕ sψ\", real = true) ;\n\njulia> d = Dict(cos(θ) => cθ, cos(ϕ) => cϕ, cos(ψ) => cψ, sin(θ) => sθ, sin(ϕ) => sϕ, sin(ψ) => sψ) ;\n\njulia> R = Tensnd(subs.(rot3(θ, ϕ, ψ),d...))\nTensND.TensndCanonical{2, 3, Sym, Tensor{2, 3, Sym, 9}}\n# data: 3×3 Tensor{2, 3, Sym, 9}:\n cθ⋅cψ⋅cϕ - sψ⋅sϕ  -cθ⋅cϕ⋅sψ - cψ⋅sϕ  cϕ⋅sθ\n cθ⋅cψ⋅sϕ + cϕ⋅sψ  -cθ⋅sψ⋅sϕ + cψ⋅cϕ  sθ⋅sϕ\n           -cψ⋅sθ              sθ⋅sψ     cθ\n# var: (:cont, :cont)\n# basis: 3×3 TensND.LazyIdentity{3, Sym}:\n 1  0  0\n 0  1  0\n 0  0  1\n\njulia> RR = R ⊠ˢ R\nTensND.TensndCanonical{4, 3, Sym, SymmetricTensor{4, 3, Sym, 36}}\n# data: 6×6 Matrix{Sym}:\n                          (cθ*cψ*cϕ - sψ*sϕ)^2                            (-cθ*cϕ*sψ - cψ*sϕ)^2           cϕ^2*sθ^2                      √2⋅cϕ⋅sθ⋅(-cθ⋅cϕ⋅sψ - cψ⋅sϕ)                     √2⋅cϕ⋅sθ⋅(cθ⋅cψ⋅cϕ - sψ⋅sϕ)                                   √2⋅(cθ⋅cψ⋅cϕ - sψ⋅sϕ)⋅(-cθ⋅cϕ⋅sψ - cψ⋅sϕ)\n                          (cθ*cψ*sϕ + cϕ*sψ)^2                            (-cθ*sψ*sϕ + cψ*cϕ)^2           sθ^2*sϕ^2                      √2⋅sθ⋅sϕ⋅(-cθ⋅sψ⋅sϕ + cψ⋅cϕ)                     √2⋅sθ⋅sϕ⋅(cθ⋅cψ⋅sϕ + cϕ⋅sψ)                                   √2⋅(cθ⋅cψ⋅sϕ + cϕ⋅sψ)⋅(-cθ⋅sψ⋅sϕ + cψ⋅cϕ)\n                                     cψ^2*sθ^2                                        sθ^2*sψ^2                cθ^2                                       √2⋅cθ⋅sθ⋅sψ                                    -√2⋅cθ⋅cψ⋅sθ                                                              -sqrt(2)*cψ*sθ^2*sψ\n             -√2⋅cψ⋅sθ⋅(cθ⋅cψ⋅sϕ + cϕ⋅sψ)                √2⋅sθ⋅sψ⋅(-cθ⋅sψ⋅sϕ + cψ⋅cϕ)    √2⋅cθ⋅sθ⋅sϕ                    cθ*(-cθ*sψ*sϕ + cψ*cϕ) + sθ^2*sψ*sϕ                   cθ*(cθ*cψ*sϕ + cϕ*sψ) - cψ*sθ^2*sϕ                            -cψ⋅sθ⋅(-cθ⋅sψ⋅sϕ + cψ⋅cϕ) + sθ⋅sψ⋅(cθ⋅cψ⋅sϕ + cϕ⋅sψ)\n             -√2⋅cψ⋅sθ⋅(cθ⋅cψ⋅cϕ - sψ⋅sϕ)                √2⋅sθ⋅sψ⋅(-cθ⋅cϕ⋅sψ - cψ⋅sϕ)    √2⋅cθ⋅cϕ⋅sθ                    cθ*(-cθ*cϕ*sψ - cψ*sϕ) + cϕ*sθ^2*sψ                   cθ*(cθ*cψ*cϕ - sψ*sϕ) - cψ*cϕ*sθ^2                            -cψ⋅sθ⋅(-cθ⋅cϕ⋅sψ - cψ⋅sϕ) + sθ⋅sψ⋅(cθ⋅cψ⋅cϕ - sψ⋅sϕ)\n √2⋅(cθ⋅cψ⋅cϕ - sψ⋅sϕ)⋅(cθ⋅cψ⋅sϕ + cϕ⋅sψ)  √2⋅(-cθ⋅cϕ⋅sψ - cψ⋅sϕ)⋅(-cθ⋅sψ⋅sϕ + cψ⋅cϕ)  sqrt(2)*cϕ*sθ^2*sϕ  cϕ⋅sθ⋅(-cθ⋅sψ⋅sϕ + cψ⋅cϕ) + sθ⋅sϕ⋅(-cθ⋅cϕ⋅sψ - cψ⋅sϕ)  cϕ⋅sθ⋅(cθ⋅cψ⋅sϕ + cϕ⋅sψ) + sθ⋅sϕ⋅(cθ⋅cψ⋅cϕ - sψ⋅sϕ)  (cθ*cψ*cϕ - sψ*sϕ)*(-cθ*sψ*sϕ + cψ*cϕ) + (cθ*cψ*sϕ + cϕ*sψ)*(-cθ*cϕ*sψ - cψ*sϕ)\n# var: (:cont, :cont, :cont, :cont)\n# basis: 3×3 TensND.LazyIdentity{3, Sym}:\n 1  0  0\n 0  1  0\n 0  0  1\n\njulia> R6 = invKM(subs.(KM(rot6(θ, ϕ, ψ)),d...))\nTensND.TensndCanonical{4, 3, Sym, SymmetricTensor{4, 3, Sym, 36}}\n# data: 6×6 Matrix{Sym}:\n                          (cθ*cψ*cϕ - sψ*sϕ)^2                            (-cθ*cϕ*sψ - cψ*sϕ)^2           cϕ^2*sθ^2                      √2⋅cϕ⋅sθ⋅(-cθ⋅cϕ⋅sψ - cψ⋅sϕ)                     √2⋅cϕ⋅sθ⋅(cθ⋅cψ⋅cϕ - sψ⋅sϕ)                                   √2⋅(cθ⋅cψ⋅cϕ - sψ⋅sϕ)⋅(-cθ⋅cϕ⋅sψ - cψ⋅sϕ)\n                          (cθ*cψ*sϕ + cϕ*sψ)^2                            (-cθ*sψ*sϕ + cψ*cϕ)^2           sθ^2*sϕ^2                      √2⋅sθ⋅sϕ⋅(-cθ⋅sψ⋅sϕ + cψ⋅cϕ)                     √2⋅sθ⋅sϕ⋅(cθ⋅cψ⋅sϕ + cϕ⋅sψ)                                   √2⋅(cθ⋅cψ⋅sϕ + cϕ⋅sψ)⋅(-cθ⋅sψ⋅sϕ + cψ⋅cϕ)\n                                     cψ^2*sθ^2                                        sθ^2*sψ^2                cθ^2                                       √2⋅cθ⋅sθ⋅sψ                                    -√2⋅cθ⋅cψ⋅sθ                                                              -sqrt(2)*cψ*sθ^2*sψ\n             -√2⋅cψ⋅sθ⋅(cθ⋅cψ⋅sϕ + cϕ⋅sψ)                √2⋅sθ⋅sψ⋅(-cθ⋅sψ⋅sϕ + cψ⋅cϕ)    √2⋅cθ⋅sθ⋅sϕ                    cθ*(-cθ*sψ*sϕ + cψ*cϕ) + sθ^2*sψ*sϕ                   cθ*(cθ*cψ*sϕ + cϕ*sψ) - cψ*sθ^2*sϕ                            -cψ⋅sθ⋅(-cθ⋅sψ⋅sϕ + cψ⋅cϕ) + sθ⋅sψ⋅(cθ⋅cψ⋅sϕ + cϕ⋅sψ)\n             -√2⋅cψ⋅sθ⋅(cθ⋅cψ⋅cϕ - sψ⋅sϕ)                √2⋅sθ⋅sψ⋅(-cθ⋅cϕ⋅sψ - cψ⋅sϕ)    √2⋅cθ⋅cϕ⋅sθ                    cθ*(-cθ*cϕ*sψ - cψ*sϕ) + cϕ*sθ^2*sψ                   cθ*(cθ*cψ*cϕ - sψ*sϕ) - cψ*cϕ*sθ^2                            -cψ⋅sθ⋅(-cθ⋅cϕ⋅sψ - cψ⋅sϕ) + sθ⋅sψ⋅(cθ⋅cψ⋅cϕ - sψ⋅sϕ)\n √2⋅(cθ⋅cψ⋅cϕ - sψ⋅sϕ)⋅(cθ⋅cψ⋅sϕ + cϕ⋅sψ)  √2⋅(-cθ⋅cϕ⋅sψ - cψ⋅sϕ)⋅(-cθ⋅sψ⋅sϕ + cψ⋅cϕ)  sqrt(2)*cϕ*sθ^2*sϕ  cϕ⋅sθ⋅(-cθ⋅sψ⋅sϕ + cψ⋅cϕ) + sθ⋅sϕ⋅(-cθ⋅cϕ⋅sψ - cψ⋅sϕ)  cϕ⋅sθ⋅(cθ⋅cψ⋅sϕ + cϕ⋅sψ) + sθ⋅sϕ⋅(cθ⋅cψ⋅cϕ - sψ⋅sϕ)  (cθ*cψ*cϕ - sψ*sϕ)*(-cθ*sψ*sϕ + cψ*cϕ) + (cθ*cψ*sϕ + cϕ*sψ)*(-cθ*cϕ*sψ - cψ*sϕ)\n# var: (:cont, :cont, :cont, :cont)\n# basis: 3×3 TensND.LazyIdentity{3, Sym}:\n 1  0  0\n 0  1  0\n 0  0  1\n\njulia> R6 == RR\ntrue\n\n\n\n\n\n","category":"function"},{"location":"#TensND.sotimes-Union{Tuple{dim}, Tuple{order2}, Tuple{order1}, Tuple{TensND.AbstractTensnd{order1, dim, T, A} where {T<:Number, A<:AbstractArray}, TensND.AbstractTensnd{order2, dim, T, A} where {T<:Number, A<:AbstractArray}}} where {order1, order2, dim}","page":"Home","title":"TensND.sotimes","text":"sotimes(t1::AbstractTensnd{order1,dim}, t2::AbstractTensnd{order2,dim})\n\nDefines a symmetric tensor product between two tensors\n\n(aⁱeᵢ) ⊗ˢ (bʲeⱼ) = 1/2(aⁱbʲ + aʲbⁱ) eᵢ⊗eⱼ\n\n\n\n\n\n","category":"method"},{"location":"#TensND.tensId2","page":"Home","title":"TensND.tensId2","text":"tensId2(T::Type{<:Number} = Sym, dim = 3)\nt𝟏(T::Type{<:Number} = Sym, dim = 3)\n\nIdentity tensor of second order 𝟏ᵢⱼ = δᵢⱼ = 1 if i=j otherwise 0\n\nExamples\n\njulia> 𝟏 = t𝟏() ; KM(𝟏)\n6-element Vector{Sym}:\n 1\n 1\n 1\n 0\n 0\n 0\n\njulia> 𝟏.data\n3×3 SymmetricTensor{2, 3, Sym, 6}:\n 1  0  0\n 0  1  0\n 0  0  1\n\n\n\n\n\n","category":"function"},{"location":"#TensND.tensId4","page":"Home","title":"TensND.tensId4","text":"tensId4(T::Type{<:Number} = Sym, dim = 3)\nt𝟙(T::Type{<:Number} = Sym, dim = 3)\n\nIdentity tensor of fourth order  𝟙 = 𝟏 ⊠ 𝟏 i.e. (𝟙)ᵢⱼₖₗ = δᵢₖδⱼₗ\n\nExamples\n\njulia> 𝟙 = t𝟙() ; KM(𝟙)\n9×9 Matrix{Sym}:\n 1  0  0  0  0  0  0  0  0\n 0  1  0  0  0  0  0  0  0\n 0  0  1  0  0  0  0  0  0\n 0  0  0  1  0  0  0  0  0\n 0  0  0  0  1  0  0  0  0\n 0  0  0  0  0  1  0  0  0\n 0  0  0  0  0  0  1  0  0\n 0  0  0  0  0  0  0  1  0\n 0  0  0  0  0  0  0  0  1\n\n\n\n\n\n","category":"function"},{"location":"#TensND.tensId4s","page":"Home","title":"TensND.tensId4s","text":"tensId4s(T::Type{<:Number} = Sym, dim = 3)\nt𝕀(T::Type{<:Number} = Sym, dim = 3)\n\nSymmetric identity tensor of fourth order  𝕀 = 𝟏 ⊠ˢ 𝟏 i.e. (𝕀)ᵢⱼₖₗ = (δᵢₖδⱼₗ+δᵢₗδⱼₖ)/2\n\nExamples\n\njulia> 𝕀 = t𝕀() ; KM(𝕀)\n6×6 Matrix{Sym}:\n 1  0  0  0  0  0\n 0  1  0  0  0  0\n 0  0  1  0  0  0\n 0  0  0  1  0  0\n 0  0  0  0  1  0\n 0  0  0  0  0  1\n\n\n\n\n\n","category":"function"},{"location":"#TensND.tensJ4","page":"Home","title":"TensND.tensJ4","text":"tensJ4(T::Type{<:Number} = Sym, dim = 3)\nt𝕁(T::Type{<:Number} = Sym, dim = 3)\n\nSpherical projector of fourth order  𝕁 = (𝟏 ⊗ 𝟏) / dim i.e. (𝕁)ᵢⱼₖₗ = δᵢⱼδₖₗ/dim\n\nExamples\n\njulia> 𝕁 = t𝕁() ; KM(𝕁)\n6×6 Matrix{Sym}:\n 1/3  1/3  1/3  0  0  0\n 1/3  1/3  1/3  0  0  0\n 1/3  1/3  1/3  0  0  0\n   0    0    0  0  0  0\n   0    0    0  0  0  0\n   0    0    0  0  0  0\n\n\n\n\n\n","category":"function"},{"location":"#TensND.tensK4","page":"Home","title":"TensND.tensK4","text":"tensK4(T::Type{<:Number} = Sym, dim = 3)\nt𝕂(T::Type{<:Number} = Sym, dim = 3)\n\nDeviatoric projector of fourth order  𝕂 = 𝕀 - 𝕁 i.e. (𝕂)ᵢⱼₖₗ = (δᵢₖδⱼₗ+δᵢₗδⱼₖ)/2 - δᵢⱼδₖₗ/dim\n\nExamples\n\njulia> 𝕂 = t𝕂() ; KM(𝕂)\n6×6 Matrix{Sym}:\n  2/3  -1/3  -1/3  0  0  0\n -1/3   2/3  -1/3  0  0  0\n -1/3  -1/3   2/3  0  0  0\n    0     0     0  1  0  0\n    0     0     0  0  1  0\n    0     0     0  0  0  1\n\n\n\n\n\n","category":"function"},{"location":"#TensND.vecbasis-Tuple{AbstractBasis, Val{:cov}}","page":"Home","title":"TensND.vecbasis","text":"vecbasis(ℬ::AbstractBasis, var = :cov)\n\nReturns the primal (if var = :cov) or primal (if var = :cont) basis\n\n\n\n\n\n","category":"method"},{"location":"#TensND.∂-Union{Tuple{A}, Tuple{dim}, Tuple{order}, Tuple{TensND.AbstractTensnd{order, dim, Sym, A}, Sym}} where {order, dim, A}","page":"Home","title":"TensND.∂","text":"∂(t::AbstractTensnd{order,dim,Sym,A},xᵢ::Sym)\n\nReturns the derivative of the tensor t with respect to the variable x_i\n\nExamples\n\n\njulia> θ, ϕ, ℬˢ, 𝐞ᶿ, 𝐞ᵠ, 𝐞ʳ = init_spherical(symbols(\"θ ϕ\", real = true)...) ;\n\njulia> ∂(𝐞ʳ, ϕ) == sin(θ) * 𝐞ᵠ\ntrue\n\njulia> ∂(𝐞ʳ ⊗ 𝐞ʳ,θ)\nTensND.TensndRotated{2, 3, Sym, SymmetricTensor{2, 3, Sym, 6}}\n# data: 3×3 SymmetricTensor{2, 3, Sym, 6}:\n 0  0  1\n 0  0  0\n 1  0  0\n# basis: 3×3 Tensor{2, 3, Sym, 9}:\n cos(θ)⋅cos(ϕ)  -sin(ϕ)  sin(θ)⋅cos(ϕ)\n sin(ϕ)⋅cos(θ)   cos(ϕ)  sin(θ)⋅sin(ϕ)\n       -sin(θ)        0         cos(θ)\n# var: (:cont, :cont)\n\n\n\n\n\n","category":"method"},{"location":"#TensND.𝐞","page":"Home","title":"TensND.𝐞","text":"𝐞(i::Int, dim::Int = 3, T::Type{<:Number} = Sym)\n\nVector of the canonical basis\n\nExamples\n\njulia> 𝐞(1)\nTensnd{1, 3, Sym, Sym, Vec{3, Sym}, CanonicalBasis{3, Sym}}\n# data: 3-element Vec{3, Sym}:\n 1\n 0\n 0\n# var: (:cont,)\n# basis: 3×3 Tensor{2, 3, Sym, 9}:\n 1  0  0\n 0  1  0\n 0  0  1\n\n\n\n\n\n","category":"function"},{"location":"#TensND.𝐞ˢ-Union{Tuple{Val{1}}, Tuple{T3}, Tuple{T2}, Tuple{T1}, Tuple{Val{1}, T1}, Tuple{Val{1}, T1, T2}, Tuple{Val{1}, T1, T2, T3}} where {T1<:Number, T2<:Number, T3<:Number}","page":"Home","title":"TensND.𝐞ˢ","text":"𝐞ˢ(i::Int, θ::T = zero(Sym), ϕ::T = zero(Sym), ψ::T = zero(Sym); canonical = false)\n\nVector of the basis rotated with the 3 Euler angles θ, ϕ, ψ (spherical if ψ=0)\n\nExamples\n\njulia> θ, ϕ, ψ = symbols(\"θ, ϕ, ψ\", real = true) ;\n\nTensnd{1, 3, Sym, Sym, Vec{3, Sym}, RotatedBasis{3, Sym}}\n# data: 3-element Vec{3, Sym}:\n 1\n 0\n 0\n# var: (:cont,)\n# basis: 3×3 Tensor{2, 3, Sym, 9}:\n -sin(ψ)⋅sin(ϕ) + cos(θ)⋅cos(ψ)⋅cos(ϕ)  -sin(ψ)⋅cos(θ)⋅cos(ϕ) - sin(ϕ)⋅cos(ψ)  sin(θ)⋅cos(ϕ)\n  sin(ψ)⋅cos(ϕ) + sin(ϕ)⋅cos(θ)⋅cos(ψ)  -sin(ψ)⋅sin(ϕ)⋅cos(θ) + cos(ψ)⋅cos(ϕ)  sin(θ)⋅sin(ϕ)\n                        -sin(θ)⋅cos(ψ)                          sin(θ)⋅sin(ψ)         cos(θ)\n\n\n\n\n\n","category":"method"},{"location":"#TensND.𝐞ᵖ-Union{Tuple{Val{1}}, Tuple{T}, Tuple{Val{1}, T}} where T<:Number","page":"Home","title":"TensND.𝐞ᵖ","text":"𝐞ᵖ(i::Int, θ::T = zero(Sym); canonical = false)\n\nVector of the polar basis\n\nExamples\n\njulia> θ = symbols(\"θ\", real = true) ;\n\njulia> 𝐞ᵖ(1, θ)\nTensnd{1, 2, Sym, Sym, Vec{2, Sym}, RotatedBasis{2, Sym}}\n# data: 2-element Vec{2, Sym}:\n 1\n 0\n# var: (:cont,)\n# basis: 2×2 Tensor{2, 2, Sym, 4}:\n cos(θ)  -sin(θ)\n sin(θ)   cos(θ)\n\n\n\n\n\n","category":"method"},{"location":"#TensND.𝐞ᶜ-Union{Tuple{Val{1}}, Tuple{T}, Tuple{Val{1}, T}} where T<:Number","page":"Home","title":"TensND.𝐞ᶜ","text":"𝐞ᶜ(i::Int, θ::T = zero(Sym); canonical = false)\n\nVector of the cylindrical basis\n\nExamples\n\njulia> θ = symbols(\"θ\", real = true) ;\n\njulia> 𝐞ᶜ(1, θ)\nTensnd{1, 3, Sym, Sym, Vec{3, Sym}, RotatedBasis{3, Sym}}\n# data: 3-element Vec{3, Sym}:\n 1\n 0\n 0\n# var: (:cont,)\n# basis: 3×3 Tensor{2, 3, Sym, 9}:\n cos(θ)  -sin(θ)  0\n sin(θ)   cos(θ)  0\n      0        0  1\n\n\n\n\n\n","category":"method"},{"location":"#Tensors.dcontract-Union{Tuple{dim}, Tuple{order2}, Tuple{order1}, Tuple{TensND.AbstractTensnd{order1, dim, T, A} where {T<:Number, A<:AbstractArray}, TensND.AbstractTensnd{order2, dim, T, A} where {T<:Number, A<:AbstractArray}}} where {order1, order2, dim}","page":"Home","title":"Tensors.dcontract","text":"dcontract(t1::AbstractTensnd{order1,dim}, t2::AbstractTensnd{order2,dim})\n\nDefines a double contracted product between two tensors\n\n𝛔 ⊡ 𝛆 = σⁱʲεᵢⱼ 𝛔 = ℂ ⊡ 𝛆\n\nExamples\n\njulia> 𝛆 = Tensnd(SymmetricTensor{2,3}((i, j) -> symbols(\"ε$i$j\", real = true))) ;\n\njulia> k, μ = symbols(\"k μ\", real =true) ;\n\njulia> ℂ = 3k * t𝕁() + 2μ * t𝕂() ;\n\njulia> 𝛔 = ℂ ⊡ 𝛆\nTensnd{2, 3, Sym, Sym, SymmetricTensor{2, 3, Sym, 6}, CanonicalBasis{3, Sym}}\n# data: 3×3 SymmetricTensor{2, 3, Sym, 6}:\n ε11*(k + 4*μ/3) + ε22*(k - 2*μ/3) + ε33*(k - 2*μ/3)                                              2⋅ε21⋅μ                                              2⋅ε31⋅μ\n                                             2⋅ε21⋅μ  ε11*(k - 2*μ/3) + ε22*(k + 4*μ/3) + ε33*(k - 2*μ/3)                                              2⋅ε32⋅μ\n                                             2⋅ε31⋅μ                                              2⋅ε32⋅μ  ε11*(k - 2*μ/3) + ε22*(k - 2*μ/3) + ε33*(k + 4*μ/3)\n# var: (:cont, :cont)\n# basis: 3×3 Tensor{2, 3, Sym, 9}:\n 1  0  0\n 0  1  0\n 0  0  1\n\n\n\n\n\n","category":"method"},{"location":"#Tensors.dotdot-Union{Tuple{dim}, Tuple{order2}, Tuple{orderS}, Tuple{order1}, Tuple{TensND.AbstractTensnd{order1, dim, T, A} where {T<:Number, A<:AbstractArray}, TensND.AbstractTensnd{orderS, dim, T, A} where {T<:Number, A<:AbstractArray}, TensND.AbstractTensnd{order2, dim, T, A} where {T<:Number, A<:AbstractArray}}} where {order1, orderS, order2, dim}","page":"Home","title":"Tensors.dotdot","text":"dotdot(v1::AbstractTensnd{order1,dim}, S::AbstractTensnd{orderS,dim}, v2::AbstractTensnd{order2,dim})\n\nDefines a bilinear operator 𝐯₁⋅𝕊⋅𝐯₂\n\nExamples\n\njulia> n = Tensnd(Sym[0, 0, 1]) ;\n\njulia> k, μ = symbols(\"k μ\", real =true) ;\n\njulia> ℂ = 3k * t𝕁() + 2μ * t𝕂() ;\n\njulia> dotdot(n,ℂ,n) # Acoustic tensor\n3×3 Tensnd{2, 3, Sym, Sym, Tensor{2, 3, Sym, 9}, CanonicalBasis{3, Sym}}:\n μ  0          0\n 0  μ          0\n 0  0  k + 4*μ/3\n\n\n\n\n\n","category":"method"},{"location":"#Tensors.otimes-Union{Tuple{dim}, Tuple{order2}, Tuple{order1}, Tuple{TensND.AbstractTensnd{order1, dim, T, A} where {T<:Number, A<:AbstractArray}, TensND.AbstractTensnd{order2, dim, T, A} where {T<:Number, A<:AbstractArray}}} where {order1, order2, dim}","page":"Home","title":"Tensors.otimes","text":"otimes(t1::AbstractTensnd{order1,dim}, t2::AbstractTensnd{order2,dim})\n\nDefines a tensor product between two tensors\n\n(aⁱeᵢ) ⊗ (bʲeⱼ) = aⁱbʲ eᵢ⊗eⱼ\n\n\n\n\n\n","category":"method"},{"location":"#Tensors.otimesu-Union{Tuple{dim}, Tuple{order2}, Tuple{order1}, Tuple{TensND.AbstractTensnd{order1, dim, T, A} where {T<:Number, A<:AbstractArray}, TensND.AbstractTensnd{order2, dim, T, A} where {T<:Number, A<:AbstractArray}}} where {order1, order2, dim}","page":"Home","title":"Tensors.otimesu","text":"otimesu(t1::AbstractTensnd{order1,dim}, t2::AbstractTensnd{order2,dim})\n\nDefines a special tensor product between two tensors of at least second order\n\n(𝐚 ⊠ 𝐛) ⊡ 𝐩 = 𝐚⋅𝐩⋅𝐛 = aⁱᵏbʲˡpₖₗ eᵢ⊗eⱼ\n\n\n\n\n\n","category":"method"}]
}
