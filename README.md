# TensND

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://jfbarthelemy.github.io/TensND.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jfbarthelemy.github.io/TensND.jl/dev)
[![Build Status](https://github.com/jfbarthelemy/TensND.jl/workflows/CI/badge.svg)](https://github.com/jfbarthelemy/TensND.jl/actions)

## Construction of the package

### Construction from [`PkgTemplates.jl`](https://github.com/invenia/PkgTemplates.jl)

The package [`TensND.jl`](https://github.com/jfbarthelemy/TensND.jl) has been created using [`PkgTemplates.jl`](https://github.com/invenia/PkgTemplates.jl):

```julia
julia> using PkgTemplates
julia> t=Template(interactive=true)
julia> t("TensND.jl")
```

Note that the chosen plugins are the default ones as well as **Documenter** and **GitHubActions**.

It is then necessary to change the environment to that of the present package `TensND.jl` and to add the following packages:

```julia
julia> cd("Path_To_TensND_folder")
julia> ]
(@v1.6) pkg> activate .
(TensND) pkg> add LinearAlgebra
(TensND) pkg> add OMEinsum
(TensND) pkg> add SymPy
(TensND) pkg> add Tensors
```

### Manual modification of `Project.toml`

In order to ensure that CompactHelper runs correctly, it is necessary to manually modify `Project.toml` by completing the `[compat]` section:

```toml
[compat]
OMEinsum = "0.4"
Tensors = "1.6"
SymPy = "1"
julia = "1.6"
```

### Adaptations related to the use of [`SymPy.jl`](https://github.com/JuliaPy/SymPy.jl)

The use of [`SymPy.jl`](https://github.com/JuliaPy/SymPy.jl) requires some adaptations of the configuration files of **GitHubActions** for *tests* and *documentation*.

- Concerning *tests*, two steps are needed:

    1. Create a file `install_dependencies.jl` (copied and modified from [install_dependencies.jl](https://github.com/tkf/IPython.jl/blob/master/test/install_dependencies.jl)) under `test` beside `runtests.jl` containing:

        ```julia
        Pkg = Base.require(Base.PkgId(Base.UUID(0x44cfe95a1eb252eab672e2afdf69b78f), "Pkg"))
        ENV["PYTHON"] = ""
        Pkg.build("PyCall")
        ```

    1. Add the following lines at the beginning of `runtests.jl`:

        ```julia
        if lowercase(get(ENV, "CI", "false")) == "true"
            include("install_dependencies.jl")
        end
        ```

- Concerning *documentation*, two actions are required:

    1. Add [`SymPy.jl`](https://github.com/JuliaPy/SymPy.jl) to the `Project.toml` file located in the folder `docs`

        ```julia
        julia> cd("Path_To_TensND_folder/docs")
        julia> ]
        (@v1.6) pkg> activate .
        (docs) pkg> add SymPy
        (docs) pkg> ⇐ # type Backspace
        julia> cd("..")
        julia> ]
        (docs) pkg> activate . # to activate the environment TensND
        ```

    1. Modify the file `CI.yml` located in `.github/workflows`.

       In `jobs: docs: steps:`, add the following lines

        ```yml
        - run: |
            julia --project=docs -e '
            using Pkg
            Pkg = Base.require(Base.PkgId(Base.UUID(0x44cfe95a1eb252eab672e2afdf69b78f), "Pkg"))
            ENV["PYTHON"] = ""
            Pkg.build("PyCall")'
        ```

        and complete with `using SymPy` in

        ```yml
        - run: |
            julia --project=docs -e '
                using Documenter: DocMeta, doctest
                using SymPy # **line to add**
                using TensND
                DocMeta.setdocmeta!(TensND, :DocTestSetup, :(using TensND); recursive=true)
                doctest(TensND)'
        ```
