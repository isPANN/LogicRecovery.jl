"""
    calculate_energy_landscape(model::QUBOModel, states::Vector{Vector{Bool}})

Calculate the energy landscape for a set of logical states.
"""
function calculate_energy_landscape(model::QUBOModel, states::Vector{Vector{Bool}})
    return [energy(model, x) for x in states]
end

"""
    verify_low_energy_states(model::QUBOModel, target_states::Vector{Vector{Bool}}, threshold::Float64)

Verify if the target states are indeed low energy states in the QUBO model.
"""
function verify_low_energy_states(model::QUBOModel, target_states::Vector{Vector{Bool}}, threshold::Float64)
    target_energies = calculate_energy_landscape(model, target_states)
    return all(e <= threshold for e in target_energies)
end 