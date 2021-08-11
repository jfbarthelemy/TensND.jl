var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = TensND","category":"page"},{"location":"#TensND","page":"Home","title":"TensND","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for TensND.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [TensND]","category":"page"},{"location":"#TensND.ϵ","page":"Home","title":"TensND.ϵ","text":"ϵ[i,j,k]\n\nLevi-Civita symbol ϵᵢⱼₖ=(i-j)(j-k)(k-i)/2\n\n\n\n\n\n","category":"constant"},{"location":"#TensND.Basis","page":"Home","title":"TensND.Basis","text":"Basis{dim, T<:Number}(v, var = :cov)\n\nBasis defined from a square matrix v where columns define either\n\nprimal vectors ie eᵢ=v[:,i] if var = :cov as by default\ndual vector ie eⁱ=v[:,i] if var = :cont.\n\nThe attributes of this object are\n\nBasis.e: square matrix defining the primal basis eᵢ=e[:,i]\nBasis.E: square matrix defining the dual basis eⁱ=E[:,i]\nBasis.g: square matrix defining the covariant components of the metric tensor gᵢⱼ=eᵢ⋅eⱼ=g[i,j]\nBasis.G: square matrix defining the contravariant components of the metric tensor gⁱʲ=eⁱ⋅eʲ=G[i,j]\n\nExamples\n\njulia> using LinearAlgebra, SymPy\n\njulia> v = Sym[1 0 0; 0 1 0; 0 1 1]\n3×3 Matrix{Sym}:\n 1  0  0\n 0  1  0\n 0  1  1\n\njulia> b = Basis(v)\nBasis{3, Sym}(Sym[1 0 0; 0 1 0; 0 1 1], Sym[1 0 0; 0 1 -1; 0 0 1], Sym[1 0 0; 0 2 1; 0 1 1], Sym[1 0 0; 0 1 -1; 0 -1 2])\n\n\n\n\n\n","category":"type"},{"location":"#TensND.CanonicalBasis","page":"Home","title":"TensND.CanonicalBasis","text":"CanonicalBasis{dim, T}()\n\nCanonical basis of dimension dim (default: 3) and type T (default: Sym)\n\nThe attributes of this object are\n\nBasis.e: identity matrix defining the primal basis e[i,j]=δᵢⱼ\nBasis.E: identity matrix defining the dual basis g[i,j]=δᵢⱼ\nBasis.g: identity matrix defining the covariant components of the metric tensor g[i,j]=δᵢⱼ\nBasis.G: identity matrix defining the contravariant components of the metric tensor G[i,j]=δᵢⱼ\n\nExamples\n\njulia> using LinearAlgebra, SymPy\n\njulia> b = CanonicalBasis()\nCanonicalBasis{3, Sym}(Sym[1 0 0; 0 1 0; 0 0 1], Sym[1 0 0; 0 1 0; 0 0 1], Sym[1 0 0; 0 1 0; 0 0 1], Sym[1 0 0; 0 1 0; 0 0 1])\n\njulia> b = CanonicalBasis{2, Float64}()\nCanonicalBasis{2, Float64}([1.0 0.0; 0.0 1.0], [1.0 0.0; 0.0 1.0], [1.0 0.0; 0.0 1.0], [1.0 0.0; 0.0 1.0])\n\n\n\n\n\n","category":"type"},{"location":"#LinearAlgebra.normalize","page":"Home","title":"LinearAlgebra.normalize","text":"normalize(b::AbstractBasis, var = cov)\n\nBuilds a normalized basis from the input basis b by calling normal_basis\n\n\n\n\n\n","category":"function"},{"location":"#TensND.basis-Tuple{AbstractBasis, Val{:cov}}","page":"Home","title":"TensND.basis","text":"basis(b::AbstractBasis, var = :cov)\n\nReturns the primal (if var = :cov) or primal (if var = :cont) basis\n\n\n\n\n\n","category":"method"},{"location":"#TensND.fϵ-Union{Tuple{Int64, Int64, Int64}, Tuple{T}, Tuple{Int64, Int64, Int64, Type{var\"#s4\"} where var\"#s4\"<:T}} where T","page":"Home","title":"TensND.fϵ","text":"fϵ(T, i::Int, j::Int, k::Int)\nfϵ(i::Int, j::Int, k::Int) = fϵ(Int, i::Int, j::Int, k::Int)\n\nFunction giving Levi-Civita symbol ϵᵢⱼₖ=(i-j)(j-k)(k-i)/2\n\n\n\n\n\n","category":"method"},{"location":"#TensND.isorthogonal-Tuple{AbstractBasis}","page":"Home","title":"TensND.isorthogonal","text":"isorthogonal(b::AbstractBasis)\n\nChecks whether the basis b is orthogonal\n\n\n\n\n\n","category":"method"},{"location":"#TensND.metric-Tuple{AbstractBasis, Val{:cov}}","page":"Home","title":"TensND.metric","text":"metric(b::AbstractBasis, var = :cov)\n\nReturns the covariant (if var = :cov) or contravariant (if var = :cont) metric matrix\n\n\n\n\n\n","category":"method"},{"location":"#TensND.normal_basis-Union{Tuple{Matrix{T}}, Tuple{T}, Tuple{Matrix{T}, Any}} where T","page":"Home","title":"TensND.normal_basis","text":"normal_basis(v::Array{T,2}, var = :cov) where {T}\n\nBuilds a basis after normalization of column vectors of input matrix v where columns define either\n\nprimal vectors ie eᵢ=v[:,i]/norm(v[:,i]) if var = :cov as by default\ndual vector ie eⁱ=v[:,i]/norm(v[:,i]) if var = :cont.\n\n\n\n\n\n","category":"method"}]
}
