using Flux, OMEinsum, Zygote, Plots

# one-hot encoding
onehot(x::Int) = [1 - x, x]

# create gate parameters (2 input, 1 output)
function create_gate_params()
    return randn(4, 2)  # 4 input combinations, 2 output logits
end

# create gate tensor from logits
function logic_tensor(params::Matrix{Float64})
    probs = [Flux.softmax(params[i, :]) for i in 1:4]
    T = zeros(2, 2, 2)
    T = cat(
        cat(
            reshape(probs[1], 1, 1, 2),
            reshape(probs[2], 1, 1, 2),
            dims=2
        ),
        cat(
            reshape(probs[3], 1, 1, 2),
            reshape(probs[4], 1, 1, 2),
            dims=2
        ),
        dims=1
    )
    return T
end

function forward(a::Int, b::Int, cin::Int, T1, T2, T3, T4, T5)
    va = onehot(a)
    vb = onehot(b)
    vcin = onehot(cin)

    @assert size(T1) == (2, 2, 2) "T1 must be 2x2x2 tensor"
    @assert size(T2) == (2, 2, 2) "T2 must be 2x2x2 tensor"
    @assert size(T3) == (2, 2, 2) "T3 must be 2x2x2 tensor"
    @assert size(T4) == (2, 2, 2) "T4 must be 2x2x2 tensor"
    @assert size(T5) == (2, 2, 2) "T5 must be 2x2x2 tensor"

    # First XOR gate (a ⊕ b)
    xor1_vec = ein"abd, a, b -> d"(T1, va, vb)
    
    # Second XOR gate ((a ⊕ b) ⊕ cin)
    sum_vec = ein"dcy, d, c -> y"(T2, xor1_vec, vcin)
    
    # First AND gate (a & b)
    and1_vec = ein"abd, a, b -> d"(T3, va, vb)
    
    # Second AND gate (cin & (a ⊕ b))
    and2_vec = ein"dcy, d, c -> y"(T4, xor1_vec, vcin)
    
    # OR gate for final carry out
    cout_vec = ein"abd, a, b -> d"(T5, and1_vec, and2_vec)

    return sum_vec, cout_vec
end

function generate_data()
    data = []
    for a in 0:1, b in 0:1, cin in 0:1
        sum_val = xor(xor(a, b), cin)
        cout_val = (a & b) | (cin & xor(a, b))
        push!(data, (a, b, cin, sum_val, cout_val))
    end
    return data
end

function loss_fn(data, T1, T2, T3, T4, T5)
    total = 0.0
    for (a, b, cin, sum_val, cout_val) in data
        sum_vec, cout_vec = forward(a, b, cin, T1, T2, T3, T4, T5)
        # Negative log likelihood for both outputs
        total += log1p(sum_vec[sum_val + 1] - 1) + log1p(cout_vec[cout_val + 1] - 1)
    end
    return -total / length(data)
end

function validate(T1, T2, T3, T4, T5, data)
    correct = 0
    total = length(data)
    for (a, b, cin, sum_val, cout_val) in data
        sum_vec, cout_vec = forward(a, b, cin, T1, T2, T3, T4, T5)
        pred_sum = argmax(sum_vec) - 1
        pred_cout = argmax(cout_vec) - 1
        if pred_sum == sum_val && pred_cout == cout_val
            correct += 1
        end
    end
    return correct / total
end

function train(; epochs=10000, lr=0.1, patience=50)
    data = generate_data()
    
    gate1_params = create_gate_params()  # First XOR
    gate2_params = create_gate_params()  # Second XOR
    gate3_params = create_gate_params()  # First AND
    gate4_params = create_gate_params()  # Second AND
    gate5_params = create_gate_params()  # OR gate
    ps = Flux.params(gate1_params, gate2_params, gate3_params, gate4_params, gate5_params)
    
    best_loss = Inf
    best_params = (copy(gate1_params), copy(gate2_params), copy(gate3_params), 
                  copy(gate4_params), copy(gate5_params))
    no_improve = 0

    losses = Float64[]
    accuracies = Float64[]
    
    for epoch in 1:epochs
        loss, grads = Zygote.withgradient(ps) do
            T1 = logic_tensor(gate1_params)
            T2 = logic_tensor(gate2_params)
            T3 = logic_tensor(gate3_params)
            T4 = logic_tensor(gate4_params)
            T5 = logic_tensor(gate5_params)
            loss_fn(data, T1, T2, T3, T4, T5)
        end
        
        for p in ps
            p .-= lr .* grads[p]
        end
        
        if epoch % 100 == 0
            T1 = logic_tensor(gate1_params)
            T2 = logic_tensor(gate2_params)
            T3 = logic_tensor(gate3_params)
            T4 = logic_tensor(gate4_params)
            T5 = logic_tensor(gate5_params)
            accuracy = validate(T1, T2, T3, T4, T5, data)
            println("Epoch $epoch, Loss = $(round(loss, digits=5)), Accuracy = $(round(accuracy, digits=3))")

            push!(losses, loss)
            push!(accuracies, accuracy)
        end
        
        if loss < best_loss
            best_loss = loss
            best_params = (copy(gate1_params), copy(gate2_params), copy(gate3_params),
                         copy(gate4_params), copy(gate5_params))
            no_improve = 0
        else
            no_improve += 1
            if no_improve >= patience
                println("Early stopping at epoch $epoch")
                break
            end
        end
    end

    epochs_plot = 100:100:length(losses)*100
    p1 = plot(epochs_plot, losses, label="Loss", xlabel="Epoch", ylabel="Loss", 
             title="Training Loss", linewidth=2)
    p2 = plot(epochs_plot, accuracies, label="Accuracy", xlabel="Epoch", ylabel="Accuracy", 
             title="Training Accuracy", linewidth=2)
    plot(p1, p2, layout=(2,1), size=(800,600))
    savefig("training_curves.png")
    
    
    return logic_tensor(best_params[1]), logic_tensor(best_params[2]), 
           logic_tensor(best_params[3]), logic_tensor(best_params[4]),
           logic_tensor(best_params[5])
end

# Run training
T_xor1, T_xor2, T_and1, T_and2, T_or = train()
@show T_xor1[1,1,:]  # First XOR gate
@show T_xor2[1,1,:]  # Second XOR gate
@show T_and1[1,1,:]  # First AND gate
@show T_and2[1,1,:]  # Second AND gate
@show T_or[1,1,:]    # OR gate 