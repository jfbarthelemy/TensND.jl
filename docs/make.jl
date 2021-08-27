using Documenter, TensND
using LinearAlgebra, SymPy, Tensors, OMEinsum, Rotations

DocMeta.setdocmeta!(TensND, :DocTestSetup, :(using TensND, LinearAlgebra, SymPy, Tensors, OMEinsum, Rotations); recursive = true)

makedocs(;
    modules = [TensND],
    authors = "Jean-François Barthélémy",
    repo = "https://github.com/jfbarthelemy/TensND.jl/blob/{commit}{path}#{line}",
    sitename = "TensND.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://jfbarthelemy.github.io/TensND.jl",
        assets = String[],
    ),
    pages = ["Home" => "index.md"],
)

deploydocs(; repo = "github.com/jfbarthelemy/TensND.jl", devbranch = "main")
