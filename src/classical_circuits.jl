module ClassicalCircuits

using ..LogicRecovery


struct FullAdder
    a::Bool
    b::Bool
    cin::Bool
    sum::Bool
    cout::Bool
end


function create_full_adder(a::Bool, b::Bool, cin::Bool)
    sum = xor(xor(a, b), cin)
    cout = (a & b) | (cin & (a | b))
    return FullAdder(a, b, cin, sum, cout)
end


function get_all_full_adder_states()
    states = FullAdder[]
    for a in [false, true]
        for b in [false, true]
            for cin in [false, true]
                push!(states, create_full_adder(a, b, cin))
            end
        end
    end
    return states
end


function convert_to_logic_input(circuit::FullAdder)
    inputs = Dict(
        "a" => circuit.a,
        "b" => circuit.b,
        "cin" => circuit.cin
    )
    outputs = Dict(
        "sum" => circuit.sum,
        "cout" => circuit.cout
    )
    return LogicInput(inputs, outputs)
end


struct LogicCircuit
    inputs::Dict{String, Bool}
    outputs::Dict{String, Bool}
    circuit_function::Function  # 用于计算输出的函数
end


function create_logic_circuit(input_names::Vector{String}, output_names::Vector{String}, circuit_function::Function)
    # 初始化所有输入为false
    inputs = Dict(name => false for name in input_names)
    # 计算初始输出
    outputs = circuit_function(inputs)
    return LogicCircuit(inputs, outputs, circuit_function)
end


function update_circuit(circuit::LogicCircuit, new_inputs::Dict{String, Bool})
    # 验证输入
    for (name, _) in new_inputs
        if !haskey(circuit.inputs, name)
            error("Unknown input: $name")
        end
    end
    # 更新输入并计算新输出
    new_outputs = circuit.circuit_function(new_inputs)
    return LogicCircuit(new_inputs, new_outputs, circuit.circuit_function)
end


function get_all_circuit_states(circuit::LogicCircuit)
    states = LogicCircuit[]
    input_names = collect(keys(circuit.inputs))
    n_inputs = length(input_names)
    
    # 生成所有可能的输入组合
    for i in 0:(2^n_inputs - 1)
        # 将数字转换为二进制输入
        binary = digits(Bool, i, base=2, pad=n_inputs)
        new_inputs = Dict(input_names[j] => binary[j] for j in 1:n_inputs)
        push!(states, update_circuit(circuit, new_inputs))
    end
    return states
end

function convert_to_logic_input(circuit::LogicCircuit)
    return LogicInput(circuit.inputs, circuit.outputs)
end

export LogicCircuit, create_logic_circuit, update_circuit, get_all_circuit_states, convert_to_logic_input,
       FullAdder, create_full_adder, get_all_full_adder_states

end # module 