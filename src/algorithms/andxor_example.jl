using Flux, OMEinsum, Zygote

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
    # use cat function to build tensor
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


function forward(a::Int, b::Int, c::Int, T1, T2)
    va = onehot(a)
    vb = onehot(b)
    vc = onehot(c)

    @assert size(T1) == (2, 2, 2) "T1 must be 2x2x2 tensor"
    @assert size(T2) == (2, 2, 2) "T2 must be 2x2x2 tensor"

    d_vec = ein"abd, a, b -> d"(T1, va, vb)
    y_vec = ein"dcy, d, c -> y"(T2, d_vec, vc)

    return y_vec
end


function generate_data()
    data = []
    for a in 0:1, b in 0:1, c in 0:1
        d = a & b
        y = xor(d, c)
        push!(data, (a, b, c, y))
    end
    return data
end


function loss_fn(data, T1, T2)
    total = 0.0
    for (a, b, c, y) in data
        y_vec = forward(a, b, c, T1, T2)
        # The negative log likelihood of the correct output
        total += log1p(y_vec[y + 1] - 1)  # log1p(x) = log(1 + x)
    end
    return -total / length(data)
end


function validate(T1, T2, data)
    correct = 0
    total = length(data)
    for (a, b, c, y) in data
        y_vec = forward(a, b, c, T1, T2)
        pred = argmax(y_vec) - 1
        if pred == y
            correct += 1
        end
    end
    return correct / total
end


function train(; epochs=7000, lr=0.1, patience=50)
    data = generate_data()
    
    gate1_params = create_gate_params()
    gate2_params = create_gate_params()
    ps = Flux.params(gate1_params, gate2_params)
    
    best_loss = Inf
    best_params = (copy(gate1_params), copy(gate2_params))
    no_improve = 0
    
    for epoch in 1:epochs
        loss, grads = Zygote.withgradient(ps) do
            T1 = logic_tensor(gate1_params)
            T2 = logic_tensor(gate2_params)
            loss_fn(data, T1, T2)
        end
        
        for p in ps
            p .-= lr .* grads[p]
        end
        
        if epoch % 100 == 0
            T1 = logic_tensor(gate1_params)
            T2 = logic_tensor(gate2_params)
            accuracy = validate(T1, T2, data)
            println("Epoch $epoch, Loss = $(round(loss, digits=5)), Accuracy = $(round(accuracy, digits=3))")
        end
        
        # 早停检查
        if loss < best_loss
            best_loss = loss
            best_params = (copy(gate1_params), copy(gate2_params))
            no_improve = 0
        else
            no_improve += 1
            if no_improve >= patience
                println("Early stopping at epoch $epoch")
                break
            end
        end
    end
    
    return logic_tensor(best_params[1]), logic_tensor(best_params[2])
end

# Run training
T_and, T_xor = train()
@show T_and[1,1,:]
@show T_xor[1,2,:]