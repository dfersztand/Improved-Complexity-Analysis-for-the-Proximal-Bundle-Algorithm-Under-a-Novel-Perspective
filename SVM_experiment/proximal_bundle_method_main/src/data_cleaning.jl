"""
The following functions are used to process LIBSVM files.
"""
mutable struct LearningData
    feature_matrix
    labels::Vector{Float64}
end

function load_libsvm_file(file_name::String)
    open(file_name, "r") do io
        target = Array{Float64,1}()
        row_indicies = Array{Int64,1}()
        col_indicies = Array{Int64,1}()
        matrix_values = Array{Float64,1}()
        row_index = 0
        for line in eachline(io)
            row_index += 1
            split_line = split(line)
            label = parse(Float64, split_line[1])
            # This ensures that labels are 1 or -1. Different dataset use {-1, 1}, {0, 1}, and {1, 2}.
            if abs(label - 1.0) < 1e-05
                label = 1.0
            else
                label = -1.0
            end
            push!(target, label)
            for i = 2:length(split_line)
                push!(row_indicies, row_index)
                matrix_coef = split(split_line[i], ":")
                push!(col_indicies, parse(Int64, matrix_coef[1]))
                push!(matrix_values, parse(Float64, matrix_coef[2]))
            end
        end
        feature_matrix = sparse(row_indicies, col_indicies, matrix_values)
        return LearningData(feature_matrix, target)
    end
end

function normalize_columns(feature_matrix::SparseMatrixCSC{Float64,Int64})
    m = size(feature_matrix, 2)
    normalize_columns_by = ones(m)
    for j = 1:m
        col_vals = feature_matrix[:, j].nzval
        if length(col_vals) > 0
            normalize_columns_by[j] = 1.0 / norm(col_vals, 2)
        end
    end
    return feature_matrix * sparse(1:m, 1:m, normalize_columns_by)
end

function remove_empty_columns(feature_matrix::SparseMatrixCSC{Float64,Int64})
    keep_cols = Array{Int64,1}()
    for j = 1:size(feature_matrix, 2)
        if length(feature_matrix[:, j].nzind) > 0
            push!(keep_cols, j)
        end
    end
    return feature_matrix[:, keep_cols]
end

function add_intercept(feature_matrix::SparseMatrixCSC{Float64,Int64})
    return [sparse(ones(size(feature_matrix, 1))) feature_matrix]
end


function preprocess_learning_data(result::LearningData)
    result.feature_matrix = remove_empty_columns(result.feature_matrix)
    result.feature_matrix = add_intercept(result.feature_matrix)
    result.feature_matrix = normalize_columns(result.feature_matrix)
    result.feature_matrix = Matrix(result.feature_matrix)
    return result
end

function compute_minimum_with_cvx(Xp, y, reg_coeff)
    n, m = size(Xp)
    w = Convex.Variable(m)
    problem = Convex.minimize(
        (reg_coeff / 2) * Convex.sumsquares(w) +
        Convex.sum(Convex.pos((1 - y .* (Xp * w)) / n)),
    )
    
    Convex.solve!(
        problem,
        () -> Gurobi.Optimizer(),
        # () -> Gurobi.Optimizer(BarConvTol = 1e-10, BarQCPConvTol = 1e-10),
        # () -> SCS.Optimizer(verbose=false), 
        # verbose=false
    ) #
    return problem
end

function compute_minimum_with_jump(Xp::Matrix{Float64}, y::Vector{Float64}, reg_coeff::Float64, accuracy = 1e-10)
    n, m = size(Xp)
    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "BarConvTol", accuracy)
    set_optimizer_attribute(model, "BarQCPConvTol", accuracy)
    # Define the optimization variable
    @variable(model, w[1:m])
    # Define a variable to represent the hinge loss
    @variable(model, ξ[1:n] >= 0)
    # Define the objective function
    @objective(model, Min,
        (reg_coeff / 2) *sum(w[i]^2 for i in 1:m) +
         (1/n)*sum(ξ[i] for i in 1:n)
    )
    # Define constraints to represent the hinge loss
    for i in 1:n
        @constraint(model, ξ[i] >= 1 - y[i] * dot(Xp[i, :],w))
    end
    # Solve the problem
    optimize!(model)
    # Extract and return the solution
    w_opt = value.(w)
    return objective_value(model)
end