using Flux, OMEinsum, Zygote, Plots

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
    
    # 记录训练过程中的 loss 和 accuracy
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
            
            # 记录 loss 和 accuracy
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
    
    # 绘制训练曲线
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