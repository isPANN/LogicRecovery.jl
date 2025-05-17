using IsingLogicMachine
using Documenter

DocMeta.setdocmeta!(IsingLogicMachine, :DocTestSetup, :(using IsingLogicMachine); recursive=true)

makedocs(;
    modules=[IsingLogicMachine],
    authors="Xiwei Pan",
    sitename="IsingLogicMachine.jl",
    format=Documenter.HTML(;
        canonical="https://isPANN.github.io/IsingLogicMachine.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/isPANN/IsingLogicMachine.jl",
    devbranch="main",
)
