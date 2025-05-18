module IsingLogicMachine

include("logicinput.jl")
include("qubo_model.jl")
include("energy.jl")
include("optimization.jl")

export TruthTable, QUBOModel
export energy, calculate_energy_landscape, verify_low_energy_states
export construct_qubo, optimize_qubo_parameters, find_ground_states

end
