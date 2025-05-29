using LogicRecovery
using Documenter

DocMeta.setdocmeta!(LogicRecovery, :DocTestSetup, :(using LogicRecovery); recursive=true)

makedocs(;
    modules=[LogicRecovery],
    authors="Xiwei Pan",
    sitename="LogicRecovery.jl",
    format=Documenter.HTML(;
        canonical="https://isPANN.github.io/LogicRecovery.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/isPANN/LogicRecovery.jl",
    devbranch="main",
)
