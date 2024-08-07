# import Pkg
using JuMP
using Printf
using Convex
using Gurobi
using LinearAlgebra
using Random
using SparseArrays
using Plots
using Ipopt
using JLD

include("proximal_bundle_method_main/src/solver.jl")
include("proximal_bundle_method_main/src/data_cleaning.jl")
include("proximal_bundle_method_main/src/PBA_utils.jl")
include("proximal_bundle_method_main/src/output_utils.jl")
include("proximal_bundle_method_main/src/computations_utils.jl")

function build_data_matrix(name, reg_coeff, max_nb_samples)
    instance_names = [name]
    # coefficients = [0.001, 0.01, 0.1, 0.5, 1.0, 1.5, 2.0, 10.0]

    instances = Dict()
    for name in instance_names
        instances[name] = preprocess_learning_data(load_libsvm_file("proximal_bundle_method_main/data/" * name))
        samples, features  = size(instances[name].feature_matrix)
        println(name* " --- number of features = "*string(features)*" || number of samples = "*string(samples))
    end
    instance = instances[name]
    Xp = (instance.feature_matrix)
    y = instance.labels
    try
        # Determine the number of samples to draw
        nb_samples = min(max_nb_samples, size(Xp, 1))
        
        # Generate a random permutation of indices
        indices = randperm(size(Xp, 1))[1:nb_samples]
        
        # Select the samples using the random indices
        Xp = Xp[indices, :]
        y = y[indices]
    catch
        # Handle any potential errors
    end

    n, m = size(Xp)

    x_init = randn(m) / m
    coefficients = [reg_coeff/n]
    coeff = coefficients[1]
    print("\n------------------------------------\n")
    Printf.@printf("Instance %s, coeff %12g\n", name, coeff)
    println("Solving with Convex.jl first")
    opt_val = compute_minimum_with_jump(Xp, y, coeff)
    Printf.@printf("Obj=%12g\n", opt_val,)
    return Xp, y, x_init, coeff, opt_val
end
function main_experiment(seed_value, name, reg_coeff, max_nb_samples, rho_coef, eps_optimal,timeout ,output_path, verbose = false, iteration_limit = 10000000, printing_frequency = 50)
    Random.seed!(seed_value)
    # name = "covtype"
    # reg_coeff = 0.01
    # max_nb_samples = 1000#Inf
    # instance_names = ["colon-cancer", "duke", "leu", "ijcnn1", "skin_nonskin", "gisette_scale", "covtype"]
    Xp, y, x_init, coeff, opt_val = build_data_matrix(name, reg_coeff, max_nb_samples)

    full_function = true
    objective = (w -> svm_objective(w, Xp, y, coeff, full_function) - opt_val)
    gradient = (w -> svm_subgradient(w, Xp, y, coeff,full_function))

    #Pegasos
    batch_size = Int(floor(size(Xp)[1]/10))
    # batch_size = 5

    # eps_optimal = 1e-4
    epsilon = eps_optimal
    delta = eps_optimal/2
    rho = rho_coef/size(Xp)[1]

    manage_bundle_values = [Keep_active, Keep_all, Keep_active_null_step]
    #PEGASOS
    (obj, maxpasses, list_obj_value_gap_pegasos, list_times_pegasos,list_iter_index_pegasos, w, has_timeout_pegasos) = pegasos(
    Xp, y,opt_val, x_init, batch_size, coeff, Int(floor(iteration_limit * size(Xp)[1]/batch_size)), printing_frequency, eps_optimal, verbose, timeout);
    #SUBGRADIENT
    params_subgradient = SubgradientMethodParams(
        eps_optimal,
        iteration_limit,
        verbose,
        printing_frequency,
        timeout
    )
    poly_step_size = ((_, _, t) -> 1 / (coeff * t))
    sol_subgradient = solve(
                    objective,
                    gradient,
                    params_subgradient,
                    poly_step_size,
                    x_init,);
    #PBA    
    full_function = false
    objective = (w -> svm_objective(w, Xp, y, coeff, full_function) - opt_val)
    gradient = (w -> svm_subgradient(w, Xp, y, coeff,full_function))

        # manage_bundle_values = [Keep_active_null_step, Keep_active, Keep_all]
    # for rho in [1e-6, 1e-4, 1e-2, 1, 1e2, 1e4, 1e6]./size(Xp)[1]
    #     println("---------------")
    #     print("rho = ", rho)
    goal_obj = 0.
    dic_gaps = Dict()
    dic_times = Dict()
    dic_bundle_size = Dict()
    dic_has_timeout_bundle = Dict()
    dic_runtimes_bundle = Dict()
    # Plot the bundle sizes for each value of manage_bundle
    for manage_bundle in manage_bundle_values
        println(manage_bundle)
        flush(stdout)
        dict_cut,master_model, list_number_null_steps, number_serious_steps, bundle_size,runtime, list_obj_value_gap, list_times, has_timeout_bundle = solve_PBA(x_init,rho,epsilon,delta, manage_bundle, goal_obj,coeff,Xp, y,opt_val,  verbose, timeout)
        println(manage_bundle, " -- Runtime : ", runtime) 
        dic_bundle_size[manage_bundle] = bundle_size
        dic_gaps[manage_bundle] = list_obj_value_gap .- goal_obj
        dic_times[manage_bundle] = list_times
        dic_has_timeout_bundle[manage_bundle] = has_timeout_bundle
        dic_runtimes_bundle[manage_bundle] = runtime
    end
    # OUTPUT
    saved_output = [dic_bundle_size, dic_gaps, dic_times,dic_runtimes_bundle,  dic_has_timeout_bundle, list_iter_index_pegasos, list_obj_value_gap_pegasos, list_times_pegasos,has_timeout_pegasos, sol_subgradient]
    save(output_path[1:end-4], "saved_output", saved_output)
    # save("proximal_bundle_method_main/outputs/experiment_output_"*name*"_"*string(coeff)*"_"*string(max_nb_samples)*"_rho_"*string(rho)*".jld", "saved_output", saved_output)
    # plot_obj_value_gap_iterations(dic_gaps, dic_times, manage_bundle_values, dic_names_method, color_dic, name, coeff, max_nb_samples, rho, sol_subgradient, list_iter_index_pegasos, list_obj_value_gap_pegasos, eps_optimal)
    # plot_bundle_size(dic_bundle_size, manage_bundle_values, dic_names_method, color_dic, name, coeff, max_nb_samples, rho, size(Xp)[2])
    # plot_obj_value_gap_times(dic_gaps, dic_times, manage_bundle_values, dic_names_method, color_dic, name, coeff, max_nb_samples, rho, sol_subgradient, list_times_pegasos, list_obj_value_gap_pegasos, eps_optimal)
end
