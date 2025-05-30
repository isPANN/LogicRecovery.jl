using Flux, OMEinsum, Zygote
using LogicRecovery

# one-hot encoding
onehot(x::Int) = [1 - x, x]

# 抽象类型定义
abstract type LogicCircuit end

# 基础组件
struct Gate
    params::Matrix{Float64}
    input_size::Int
    output_size::Int
    input_ids::Vector{Symbol}
    output_ids::Vector{Symbol}
end

# create gate parameters
function create_gate_params(input_size::Int, output_size::Int)
    return randn(2^input_size, 2^output_size)
end

# create gate tensor from logits
function logic_tensor(gate::Gate)
    probs = Flux.softmax.(eachrow(gate.params))
    flat_probs = reduce(vcat, probs)    
    n_inputs = gate.input_size; n_outputs = gate.output_size
    T = zeros(fill(2, n_inputs)..., fill(2, n_outputs)...)
    T = permutedims(reshape(flat_probs, size(T)...), [n_inputs+1:n_inputs+n_outputs; 1:n_inputs])
    return T
end

# 前向传播
function evaluate(circuit::LogicCircuit, inputs::BitVector)
    # Initialize environment with input variables mapped to one-hot vectors
    env = Dict{Symbol, Vector{Float64}}()
    input_vars = circuit.gates[1].input_ids
    for (var, val) in zip(input_vars, inputs)
        env[var] = onehot(Int(val))
    end

    for gate in circuit.gates
        @show gate
        tensor = logic_tensor(gate)
        # Extract inputs from env
        gate_inputs = [env[id] for id in gate.input_ids]
        # Generate contraction string
        n_in = length(gate.input_ids)
        n_dims = ndims(tensor)
        input_indices = collect('a':'z')[1:n_in]
        tensor_indices = collect('a':'z')[n_in+1:n_in+n_dims-1]
        einsum_str = string(join(tensor_indices), ",", join(input_indices, ","), "->", join(tensor_indices))
        output = ein(einsum_str)(tensor, gate_inputs...)
        # Store outputs in env
        for (i, out_id) in enumerate(gate.output_ids)
            env[out_id] = output[i]
        end
    end

    # Return final output vector from the last gate's outputs
    final_outputs = [env[id] for id in circuit.gates[end].output_ids]
    return reduce(vcat, final_outputs)
end

# 数据生成器
function generate_data(circuit::LogicCircuit)
    data = Vector{Int}[]
    input_size = length(circuit.gates[1].input_ids)
    for inputs in Iterators.product(fill(0:1, input_size)...)
        output_vec = evaluate(circuit, Bool.(inputs))
        output_int = argmax(output_vec) - 1
        push!(data, [inputs..., output_int])
    end
    return data
end

# 损失函数
function loss_fn(data::Vector{BitVector}, gates::Vector{Gate})
    total = 0.0
    for row in data
        inputs = Bool.(row[1:end-1])
        target = row[end]
        circuit = AndXorCircuit(gates, length(inputs))
        output = evaluate(circuit, inputs)
        total += log(output[target + 1] + 1e-10)
    end
    return -total / length(data)
end

# 验证函数
function validate(gates::Vector{Gate}, data::Vector{Vector{Int}})
    correct = 0
    total = length(data)
    for row in data
        inputs = Bool.(row[1:end-1])
        target = row[end]
        circuit = AndXorCircuit(gates, length(inputs))
        output = evaluate(circuit, inputs)
        pred = argmax(output) - 1
        if pred == target
            correct += 1
        end
    end
    return correct / total
end

# 训练函数
function train(data::Vector{BitVector}, circuit::LogicCircuit; epochs=2000, lr=0.1, patience=50)
    gates = circuit.gates
    ps = Flux.params([gate.params for gate in gates]...)
    
    best_loss = Inf
    best_gates = deepcopy(gates)
    no_improve = 0
    
    for epoch in 1:epochs
        loss, grads = Zygote.withgradient(ps) do
            loss_fn(data, gates)
        end
        
        for p in ps
            p .-= lr .* grads[p]
        end
        
        if epoch % 100 == 0
            accuracy = validate(gates, data)
            println("Epoch $epoch, Loss = $(round(loss, digits=5)), Accuracy = $(round(accuracy, digits=3))")
        end
        
        if loss < best_loss
            best_loss = loss
            best_gates = deepcopy(gates)
            no_improve = 0
        else
            no_improve += 1
            if no_improve >= patience
                println("Early stopping at epoch $epoch")
                break
            end
        end
    end
    
    return best_gates
end

# 示例：AND-XOR 电路实现
struct AndXorCircuit <: LogicCircuit
    gates::Vector{Gate}
    input_size::Int
end

function AndXorCircuit()
    gates = [
        Gate(create_gate_params(2, 1), 2, 1, [:a, :b], [:c]),  # AND gate
        Gate(create_gate_params(2, 1), 2, 1, [:c, :d], [:e])   # XOR gate
    ]
    return AndXorCircuit(gates, 3)
end

data = get_all_circuit_states(create_toy_model())
# 使用示例
circuit = AndXorCircuit()
trained_gates = train(data, circuit)
@show logic_tensor(trained_gates[1])[1,1,:]  # AND gate
@show logic_tensor(trained_gates[2])[1,2,:]  # XOR gate