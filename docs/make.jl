using Documenter
using DHMMs

makedocs(
    sitename = "DHMMs.jl",
    modules = [DHMMs],
    pages = [
        "Home" => "index.md",
        "API" => "api.md",
    ],
    warnonly = [:missing_docs],
)

deploydocs(
    repo = "github.com/mashu/DHMMs.jl.git",
)

