
@enum TerminationReason begin
    TERMINATION_REASON_UNSPECIFIED
    TERMINATION_REASON_OPTIMAL
    TERMINATION_REASON_ITERATION_LIMIT
    TERMINATION_REASON_TIMEOUT
end

struct BundleMethodParams
    eps_optimal::Float64
    iteration_limit::Int64
    contraction_factor::Float64
    verbose::Bool
    printing_frequency::Int64
    full_memory::Bool
end

struct SubgradientMethodParams
    eps_optimal::Float64
    iteration_limit::Int64
    verbose::Bool
    printing_frequency::Int64
    timeout::Float64
end

struct Cut
    constant_term::Float64
    gradient::Vector{Float64}
end

struct IterationInformation
    iteration::Int64
    is_null::Bool
    objective::Float64
    approximation_gap::Float64
    residual::Float64
    step_size::Float64
    time::Float64
end

struct BundleMethodOutput
    solution::Vector{Float64}
    iteration_stats::Vector{IterationInformation}
    termination_reason::TerminationReason
end

mutable struct SolverState
    current_objective::Float64
    current_gap::Float64
    current_residual::Float64
    latest_is_null::Bool
    current_solution::Vector{Float64}
    cuts::Vector{Cut}
end

function create_cut(point::Vector{Float64}, objective_function, subgradient_section)
    subgradient = subgradient_section(point)
    return Cut(objective_function(point) - dot(point, subgradient), subgradient)
end

function evaluate_cut(cut::Cut, point::Vector{Float64})
    return cut.constant_term + dot(cut.gradient, point)
end

function evaluate_model(cuts::Vector{Cut}, point::Vector{Float64})
    return maximum(map(cut -> evaluate_cut(cut, point), cuts))
end

function aggregate_cuts(cuts::Vector{Cut}, interpolation_coefficients::Vector{Float64})
    constant_term = cuts[1].constant_term * interpolation_coefficients[1]
    gradient = cuts[1].gradient * interpolation_coefficients[1]
    for i = 2:length(cuts)
        constant_term += cuts[i].constant_term * interpolation_coefficients[i]
        gradient += cuts[i].gradient * interpolation_coefficients[i]
    end
    return [Cut(constant_term, gradient)]
end

function check_termination(params::BundleMethodParams, iteration_info::IterationInformation)
    if params.verbose && mod(iteration_info.iteration, params.printing_frequency)
        iteration_log(params, iteration_log)
    end
    if iteration_info.iteration >= params.iteration_limit
        return TERMINATION_REASON_ITERATION_LIMIT
    end
    if iteration_info.approximation_gap <= eps_optimal
        return TERMINATION_REASON_OPTIMAL
    end
end

function iteration_log(
    params::SubgradientMethodParams,
    iteration_info::IterationInformation,
)
    if params.verbose && mod(iteration_info.iteration, params.printing_frequency) == 0
        Printf.@printf(
            " %5d obj=%12g residual=%12g step_size=%12g\n",
            iteration_info.iteration,
            iteration_info.objective,
            iteration_info.residual,
            iteration_info.step_size,
        )
    end
end


function iteration_log(params::BundleMethodParams, iteration_info::IterationInformation)
    if params.verbose && mod(iteration_info.iteration, params.printing_frequency) == 0
        Printf.@printf(
            " %s %5d obj=%12g gap=%12g residual=%12g step_size=%12g\n",
            iteration_info.is_null ? "n" : "d",
            iteration_info.iteration,
            iteration_info.objective,
            iteration_info.approximation_gap,
            iteration_info.residual,
            iteration_info.step_size,
        )
    end
end

function solve_model(cuts::Vector{Cut}, point::Vector{Float64}, stepsize::Float64)
    if length(cuts) == 1
        return point - cuts[1].gradient * stepsize, [1.0]
    elseif length(cuts) == 2

        candidate = point - cuts[1].gradient * stepsize
        if evaluate_cut(cuts[1], candidate) >= evaluate_cut(cuts[2], candidate)
            return candidate, [1.0, 0.0]
        end

        candidate = point - cuts[2].gradient * stepsize
        if evaluate_cut(cuts[2], candidate) >= evaluate_cut(cuts[1], candidate)
            return candidate, [0.0, 1.0]
        end

        grad_diff = cuts[1].gradient - cuts[2].gradient
        interpolation =
            1 / (stepsize * dot(grad_diff, grad_diff)) *
            (cuts[1].constant_term - cuts[2].constant_term + dot(grad_diff, candidate))
        return candidate - grad_diff * interpolation * stepsize,
        [interpolation, 1.0 - interpolation]
    else
        # This block of code is only called with the full-memory method.
        x = Convex.Variable(length(cuts[1].gradient))
        t = Convex.Variable(1)

        problem = minimize(t + (1 / (2 * stepsize)) * Convex.sumsquares(x - point))
        for cut in cuts
            problem.constraints += t >= cut.gradient' * x + cut.constant_term
        end
        Convex.solve!(problem, () -> SCS.Optimizer(verbose = false), verbose = false)
        # Convex.solve!(problem, Gurobi.Optimizer)
        # if problem.status != 1
        # error("The inner problem couldn't be solved to optimality.") # This should never happen.
        # end
        return Convex.evaluate(x), []
    end
end

function take_step(
    objective_function,
    subgradient_map,
    params,
    step_size,
    solver_state,
    iteration_stats,
)

    (new_point, interpolation_coefficients) =
        solve_model(solver_state.cuts, solver_state.current_solution, step_size)

    new_objective = objective_function(new_point)
    solver_state.current_residual = norm(new_point - solver_state.current_solution)
    solver_state.latest_is_null = true

    model_value = evaluate_model(solver_state.cuts, new_point)
    solver_state.current_gap = new_objective - model_value

    if params.contraction_factor * (solver_state.current_objective - model_value) <=
       solver_state.current_objective - new_objective
        solver_state.latest_is_null = false
        solver_state.current_solution = new_point
        solver_state.current_objective = new_objective
    end

    if !params.full_memory
        # This combines all the previous cuts into one, while mainteining the constrains
        # for the theory to hold.
        solver_state.cuts = aggregate_cuts(solver_state.cuts, interpolation_coefficients)
    end
    push!(solver_state.cuts, create_cut(new_point, objective_function, subgradient_map))
end

"""
Solves the problem with the subgradient method.
"""
function solve(
    objective_function::Function,
    subgradient_map::Function,
    params::SubgradientMethodParams,
    stepsize_policy::Function,
    initial_point::Vector{Float64},
)
    iteration_stats =
        [IterationInformation(0, false, objective_function(initial_point), Inf, Inf, 0.0, time())]
    solver_state = SolverState(
        objective_function(initial_point),
        Inf,
        Inf,
        false,
        initial_point,
        [create_cut(initial_point, objective_function, subgradient_map)],
    )
    best_solution = initial_point
    best_obj = objective_function(initial_point)
    for i = 1:params.iteration_limit
        step_size = stepsize_policy(
            solver_state.current_objective,
            solver_state.current_solution,
            i,
        )
        new_point = (
            solver_state.current_solution -
            stepsize_policy(
                solver_state.current_objective,
                solver_state.current_solution,
                i,
            ) * subgradient_map(solver_state.current_solution)
        )
        solver_state.current_residual = norm(new_point - solver_state.current_solution)

        solver_state.current_solution = new_point
        solver_state.current_objective = objective_function(new_point)

        if solver_state.current_objective < best_obj
            best_solution = solver_state.current_solution
        end

        push!(
            iteration_stats,
            IterationInformation(
                i,
                solver_state.latest_is_null,
                solver_state.current_objective,
                solver_state.current_gap,
                solver_state.current_residual,
                step_size,
                time()
            ),
        )
        iteration_log(params, last(iteration_stats))
        if solver_state.current_objective<eps_optimal
            return BundleMethodOutput(
                solver_state.current_solution,
                iteration_stats,
                TERMINATION_REASON_OPTIMAL,
            )
        end
        if time() - iteration_stats[1].time > params.timeout
            return BundleMethodOutput(
                solver_state.current_solution,
                iteration_stats,
                TERMINATION_REASON_TIMEOUT,
            )
        end
    end

    return BundleMethodOutput(
        best_solution,
        iteration_stats,
        TERMINATION_REASON_ITERATION_LIMIT,
    )
end


"""
Solves the problem with the bundle method.
"""
function solve(
    objective_function::Function,
    subgradient_map::Function,
    params::BundleMethodParams,
    stepsize_policy::Function,
    initial_point::Vector{Float64},
)

    iteration_stats =
        [IterationInformation(0, false, objective_function(initial_point), Inf, Inf, 0.0)]
    solver_state = SolverState(
        objective_function(initial_point),
        Inf,
        Inf,
        false,
        initial_point,
        [create_cut(initial_point, objective_function, subgradient_map)],
    )
    for i = 1:params.iteration_limit
        step_size = stepsize_policy(
            solver_state.current_objective,
            solver_state.current_solution,
            i,
        )
        take_step(
            objective_function,
            subgradient_map,
            params,
            step_size,
            solver_state,
            iteration_stats,
        )
        push!(
            iteration_stats,
            IterationInformation(
                i,
                solver_state.latest_is_null,
                solver_state.current_objective,
                solver_state.current_gap,
                solver_state.current_residual,
                1 / step_size,
            ),
        )
        iteration_log(params, last(iteration_stats))
    end

    return BundleMethodOutput(
        solver_state.current_solution,
        iteration_stats,
        TERMINATION_REASON_ITERATION_LIMIT,
    )
end

function combine_parallel_models(objective_function, subgradient_map, solver_states)
    min_fun, idx = findmin([state.current_objective for state in solver_states])
    new_solution = solver_states[idx].current_solution
    for state in solver_states
        if !state.latest_is_null && min_fun < state.current_objective
            state.current_residual = norm(state.current_solution - new_solution)
            state.current_solution = new_solution
            state.current_objective = objective_function(new_solution)
            state.cuts = [create_cut(new_solution, objective_function, subgradient_map)]
        end
    end
    return idx
end

function solve_adaptive(
    objective_function,
    subgradient_map,
    params,
    step_sizes,
    initial_point,
)
    iteration_stats =
        [IterationInformation(0, false, objective_function(initial_point), Inf, Inf, 0.0)]
    iteration_info_agents = [
        [IterationInformation(0, false, objective_function(initial_point), Inf, Inf, 0.0)] for _ in step_sizes
    ]
    solver_states = [
        SolverState(
            objective_function(initial_point),
            Inf,
            Inf,
            false,
            initial_point,
            [create_cut(initial_point, objective_function, subgradient_map)],
        ) for _ in step_sizes
    ]
    idx = 0
    for i = 1:params.iteration_limit
        for j = 1:length(step_sizes)
            take_step(
                objective_function,
                subgradient_map,
                params,
                step_sizes[j],
                solver_states[j],
                iteration_stats,
            )
        end
        idx = combine_parallel_models(objective_function, subgradient_map, solver_states)
        for idx = 1:length(step_sizes)
            push!(
                iteration_info_agents[idx],
                IterationInformation(
                    i,
                    solver_states[idx].latest_is_null,
                    solver_states[idx].current_objective,
                    solver_states[idx].current_gap,
                    solver_states[idx].current_residual,
                    step_sizes[idx],
                ),
            )
        end

        push!(
            iteration_stats,
            IterationInformation(
                i,
                solver_states[idx].latest_is_null,
                solver_states[idx].current_objective,
                solver_states[idx].current_gap,
                solver_states[idx].current_residual,
                1 / step_sizes[idx],
            ),
        )
        iteration_log(params, last(iteration_stats))
    end

    return BundleMethodOutput(
        solver_states[idx].current_solution,
        iteration_stats,
        TERMINATION_REASON_ITERATION_LIMIT,
    ),
    iteration_info_agents
end

function pegasos(Xp, Y, opt_val, x_init, k::Integer = 5, lambda::Real = 0.1, maxpasses::Integer = 100, printing_frequency = 50, eps_optimal = 1e-7,verbose = true, timeout = 3600)
    # p features, n observations
    
    X = transpose(Xp)
    coeff = lambda
    list_obj_value_gap = [svm_objective(x_init, Xp, Y, coeff, true) - opt_val]
    list_times = [0.]
    list_iter_index = [0.]
    p, n = size(X)
    new_printing_freq =  Int(floor(printing_frequency*n/k))
    has_timeout_pegasos = false
    ti = time()
    
    # Initialize weights so norm(w) <= 1 / sqrt(lambda)
    w = x_init
    sqrtlambda = sqrt(lambda)
    normalizer = sqrtlambda * norm(w)
    for j in 1:p
        w[j] /= normalizer
    end

    # Allocate storage for repeated used arrays
    deltaw = zeros(p)
    w_tmp = zeros(p)

    # Loop
    for t in 1:maxpasses
        # Calculate stepsize parameters
        alpha = 1.0 / t
        eta_t = 1.0 / (lambda * t)
        # Calculate scaled sum over misclassified examples
        # Subgradient over minibatch of size k
        fill!(deltaw, 0.0)
        for i in 1:k
            # Select a random item from X
            # This is one element of At of S
            index = rand(1:n)
            # Test if prediction isn't sufficiently good
            # If so, current item is element of At+
            pred = Y[index] * dot(w, X[:, index])
            if pred < 1.0
                # Update subgradient
                for j in 1:p
                    deltaw[j] += Y[index] * X[j, index]
                end
            end
        end
        
        for j in 1:p
            deltaw[j] *= (eta_t / k)
        end

        # Calculate tentative weight-update
        for j in 1:p
            w_tmp[j] = (1.0 - alpha) * w[j] + deltaw[j]
        end
        # # Rescale subgradient
        # for j in 1:p
        #     deltaw.*= (eta_t / k)
        # end
        # # Calculate tentative weight-update
        # for j in 1:p
        #     w_tmp = (1.0 - alpha) .* w .+ deltaw
        # end
        # Find projection of weights into L2 ball
        proj = min(1.0, 1.0 / (sqrtlambda * norm(w_tmp)))
        for j in 1:p
            w[j] = proj * w_tmp[j]
        end
        w = w_tmp
        
        if t%new_printing_freq==0
            this_val = svm_objective(w, Xp, Y, coeff, true) - opt_val
            push!(list_obj_value_gap, this_val)
            if verbose
                Printf.@printf("Gap=%12g\n", this_val)
                flush(stdout)
            end
            push!(list_times, time()-ti)
            push!(list_iter_index, t * k / size(Xp)[1])
            if this_val<eps_optimal
                break
            end
            if time()-ti>timeout
                has_timeout_pegasos = true
                break
            end
        end
    end
    return (svm_objective(x_init, Xp, Y, coeff, true), maxpasses, list_obj_value_gap, list_times, list_iter_index, w, has_timeout_pegasos)
end