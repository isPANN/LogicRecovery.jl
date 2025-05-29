"""
    LogicCircuit

A simple structure representing a logic circuit with inputs and outputs.
"""
struct LogicCircuit
    n_inputs::Int
    n_outputs::Int
    input_labels::Vector{String}  # Labels for input bits
    output_labels::Vector{String} # Labels for output bits
    circuit_function::Function    # Function that takes input array and returns output array
end

"""
    create_logic_circuit(n_inputs::Int, n_outputs::Int, circuit_function::Function;
                        input_labels::Vector{String}=String[],
                        output_labels::Vector{String}=String[])::LogicCircuit

Create a logic circuit with specified number of inputs and outputs.
- n_inputs: number of input bits
- n_outputs: number of output bits
- circuit_function: function that takes input array and returns output array
- input_labels: labels for input bits (optional)
- output_labels: labels for output bits (optional)
"""
function create_logic_circuit(n_inputs::Int, n_outputs::Int, circuit_function::Function;
                              input_labels::Vector{String}=String[], 
                              output_labels::Vector{String}=String[])
    input_labels = isempty(input_labels) ? ["in$i" for i in 1:n_inputs] : input_labels
    output_labels = isempty(output_labels) ? ["out$i" for i in 1:n_outputs] : output_labels
    if length(input_labels) != n_inputs
        error("Expected $n_inputs input labels, got $(length(input_labels))")
    end
    if length(output_labels) != n_outputs
        error("Expected $n_outputs output labels, got $(length(output_labels))")
    end
    return LogicCircuit(n_inputs, n_outputs, input_labels, output_labels, circuit_function)
end

"""
    evaluate_circuit(circuit::LogicCircuit, inputs::Vector{Bool})::Vector{Bool}

Evaluate the circuit with given inputs and return the outputs.
"""
function evaluate_circuit(circuit::LogicCircuit, inputs::Vector{Bool})
    if length(inputs) != circuit.n_inputs
        error("Expected $(circuit.n_inputs) inputs, got $(length(inputs))")
    end
    return circuit.circuit_function(inputs)
end

"""
    get_all_circuit_states(circuit::LogicCircuit)::Vector{Vector{Bool}}

Return all possible input-output states of the circuit as a vector of vectors.
Each inner vector contains [inputs..., outputs...].
"""
function get_all_circuit_states(circuit::LogicCircuit; verbose::Bool=false)::Vector{BitVector}
    states = BitVector[]
    n_inputs = circuit.n_inputs

    for i in 0:(2^n_inputs - 1)
        inputs = digits(Bool, i, base=2, pad=n_inputs)
        outputs = evaluate_circuit(circuit, inputs)
        bits = BitVector(vcat(inputs, outputs))
        push!(states, bits)
        if verbose
            println("Input: ", inputs, " => Output: ", outputs)
        end
    end

    return states
end

"""
    print_circuit_truth_table(circuit::LogicCircuit)

Print the truth table of the circuit with labeled inputs and outputs.
"""
function print_circuit_truth_table(circuit::LogicCircuit)
    states = get_all_circuit_states(circuit)
    input_labels = circuit.input_labels
    output_labels = circuit.output_labels

    # Print header
    println("Idx", "\t|\t", join(input_labels, "\t"), "\t|\t", join(output_labels, "\t"))
    println("-"^70)

    for (idx, state) in enumerate(states)
        inputs = state[1:circuit.n_inputs]
        outputs = state[circuit.n_inputs+1:end]
        println(idx, "\t|\t", join(Bool.(inputs), "\t"), "\t|\t", join(Bool.(outputs), "\t"))
    end
end

# Example1: Full Adder implementation
function full_adder(inputs::Vector{Bool})
    a, b, cin = inputs
    sum = xor(xor(a, b), cin)
    cout = (a & b) | (cin & (a | b))
    return [sum, cout]
end

"""
    create_full_adder()::LogicCircuit

Create a full adder circuit instance.
"""
function create_full_adder()::LogicCircuit
    return create_logic_circuit(
        3, 2, 
        full_adder,
        input_labels=["a", "b", "cin"],
        output_labels=["sum", "cout"]
    )
end

#Example2: And + Xor
function and_xor(inputs::Vector{Bool})
    a, b, c = inputs
    and_result = a & b
    return [xor(and_result, c)]
end

function create_toy_model()::LogicCircuit
    return create_logic_circuit(
        3, 1,
        and_xor
    )
end