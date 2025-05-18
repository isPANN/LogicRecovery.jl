"""
    TruthTable

A structure representing a truth table for logical operations.
"""
struct TruthTable
    matrix::Matrix{Bool}

    function TruthTable(matrix::Matrix{Bool})
        return new(matrix)
    end
end


