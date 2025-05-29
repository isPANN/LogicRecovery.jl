using OMEinsum
using Zygote
using LinearAlgebra
using Random
using Flux

# Helper: one-hot binary vector
onehot(x::Int) = [1 - x, x]

# Generate full truth table
function generate_data()
    data = []
    for a in 0:1, b in 0:1, d in 0:1
        c = a & b           # AND gate
        e = xor(c, d)       # XOR gate
        push!(data, (a, b, d, c, e))
    end
    @show data
    return data
end

# Instead of random probabilities, we now learn logits and apply softmax to each row
function create_gate_params()
    return randn(4, 2)  # 4 input combinations, 2 output logits
end

function logic_tensor(params::Matrix{Float64})
    probs = [Flux.softmax(params[i, :]) for i in 1:4]
    T = reshape(vcat(probs...), 2, 2, 2)
    return T
end

# Define loss function based on log-likelihood
function loss_fn(data, T1, T2)
    total_logp = 0.0
    for (a, b, d, c_true, e_true) in data
        va = onehot(a)
        vb = onehot(b)
        vd = onehot(d)

        c_vec = ein"abc, a, b -> c"(T1, va, vb)
        e_vec = ein"cde, c, d -> e"(T2, c_vec, vd)

        total_logp += log(e_vec[e_true + 1] + 1e-10)
    end
    return -total_logp / length(data)
end

# Helper function to print tensor
function print_tensor(T, name)
    println("\n$name gate probabilities:")
    for i in 1:2, j in 1:2
        println("Input ($(i-1), $(j-1)) -> Output probabilities: $(T[i,j,:])")
    end
end

# Training
function train(; epochs=500, lr=0.01)
    data = generate_data()
    gate1_params = create_gate_params()
    gate2_params = create_gate_params()
    ps = Flux.params(gate1_params, gate2_params)

    for epoch in 1:epochs
        loss, grads = Zygote.withgradient(ps) do
            T1 = logic_tensor(gate1_params)
            T2 = logic_tensor(gate2_params)
            loss_fn(data, T1, T2)
        end
        
        # Manual gradient descent update
        for p in ps
            p .-= lr .* grads[p]
        end

        if epoch % 10 == 0
            println("Epoch $epoch, Loss = $(round(loss, digits=5))")
        end
    end

    return logic_tensor(gate1_params), logic_tensor(gate2_params)
end

# Run
gate1, gate2 = train(epochs=5000, lr=0.1)
print_tensor(gate1, "AND-like")
print_tensor(gate2, "XOR-like")
