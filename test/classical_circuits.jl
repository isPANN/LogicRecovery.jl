@testset "ClassicalCircuits" begin
    circuit = create_toy_model()
    print_circuit_truth_table(circuit)
    bitvectors = get_all_circuit_states(circuit; verbose=true)
    @test length(bitvectors) == 8
    
    circuit = create_full_adder()
    bitvectors = get_all_circuit_states(circuit; verbose=true)
    @test length(bitvectors) == 8
end