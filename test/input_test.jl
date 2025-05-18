@testset "TruthTable" begin
    @test TruthTable(Matrix{Bool}(undef, 2, 2)) isa TruthTable
end