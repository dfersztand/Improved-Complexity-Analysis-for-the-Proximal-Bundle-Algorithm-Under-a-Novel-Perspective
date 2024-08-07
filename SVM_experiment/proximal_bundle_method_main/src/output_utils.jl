function make_dic_runtime(dic_runtimes_bundle, dic_has_timeout_bundle, list_times_pegasos,has_timeout_pegasos, sol_subgradient)
    to_return_runtimes = Dict()
    for manage_bundle in keys(dic_runtimes_bundle)
        if dic_has_timeout_bundle[manage_bundle]
            to_return_runtimes[manage_bundle] = Inf
        else
            to_return_runtimes[manage_bundle] = dic_runtimes_bundle[manage_bundle]
        end
    end
    if has_timeout_pegasos
        to_return_runtimes["Pegasos"] = Inf
    else
        to_return_runtimes["Pegasos"] = list_times_pegasos[end]
    end
    if (sol_subgradient.termination_reason == TERMINATION_REASON_OPTIMAL)
        to_return_runtimes["Subgradient"] = sol_subgradient.iteration_stats[end].time - sol_subgradient.iteration_stats[1].time
    else
        to_return_runtimes["Subgradient"] = Inf
    end
    return to_return_runtimes
end

function make_dic_nb_iter(dic_gaps, dic_has_timeout_bundle, list_iter_index_pegasos,has_timeout_pegasos, sol_subgradient)
    to_return_iterations = Dict()
    for manage_bundle in keys(dic_gaps)
        if dic_has_timeout_bundle[manage_bundle]
            to_return_iterations[manage_bundle] = Inf
        else
            to_return_iterations[manage_bundle] = length(dic_gaps[manage_bundle])
        end
    end
    if has_timeout_pegasos
        to_return_iterations["Pegasos"] = Inf
    else
        to_return_iterations["Pegasos"] = Int(floor(list_iter_index_pegasos[end]))
    end
    if (sol_subgradient.termination_reason == TERMINATION_REASON_OPTIMAL)
        to_return_iterations["Subgradient"] = sol_subgradient.iteration_stats[end].iteration
    else
        to_return_iterations["Subgradient"] = Inf
    end
    return to_return_iterations
end

function plot_bundle_size(dic_bundle_size, manage_bundle_values, dic_names_method, color_dic, name, coeff, max_nb_samples, rho, nb_features)
        p = plot(size = (600,500))
    for manage_bundle in manage_bundle_values
        plot!(dic_bundle_size[manage_bundle], label=dic_names_method[manage_bundle], color = color_dic[manage_bundle],linewidth=4)
    end
    hline!([nb_features], linestyle=:dash, label="n+1", color = "grey")
    xlabel!("Iterations")
    ylabel!("Bundle Size")
    plot!(legendfontsize=14)
    # title!("Bundle Size for Different Bundle management strategies")
    savefig(p, "proximal_bundle_method_main/outputs/Bundle_size_"*name*"_"*string(coeff)*"_"*string(max_nb_samples)*"_rho_"*string(rho)*".pdf")  
    plot!()
    return p
end 

function plot_obj_value_gap_times(dic_gaps, dic_times, manage_bundle_values, dic_names_method, color_dic, name, coeff, max_nb_samples, rho, sol_subgradient, list_times_pegasos, list_obj_value_gap_pegasos, eps_optimal)
    p = plot(size = (600,500))
    for manage_bundle in manage_bundle_values
        # plot!(dic_gaps[manage_bundle], label = "", color = color_dic[manage_bundle],linewidth=4)
        plot!(dic_times[manage_bundle],dic_gaps[manage_bundle], label = dic_names_method[manage_bundle], color = color_dic[manage_bundle],linewidth=4)
    end
    plot!([s.time - sol_subgradient.iteration_stats[1].time for s in sol_subgradient.iteration_stats], ([s.objective for s in sol_subgradient.iteration_stats]), label = "subgradient", linewidth = 4, color = "blue")
    plot!(list_times_pegasos, list_obj_value_gap_pegasos, label = "Pegasos", linewidth = 4)

    hline!([eps_optimal], linestyle=:dash, label="Tolerance", color = "grey")
    xlabel!("Time (s)")
    ylabel!("Optimality Gap")
    plot!(legendfontsize=14)
    # y_ticks = range(1e-4, stop=1e1, length=20)
    # title!("Progress for different bundle management strategies")
    plot!(yaxis=:log, yticks = [1e-5, 1e-4, 1e-3, 1e-2,1e-1, 1, 10, 100])
    ylims!(1e-5, 100)
    # xlims!(-0.1, 500)
    savefig(p, "proximal_bundle_method_main/outputs/Optimality_gap_by_time_"*name*"_"*string(coeff)*"_"*string(max_nb_samples)*"_rho_"*string(rho)*".pdf") 
    return p
end

function plot_obj_value_gap_iterations(dic_gaps, dic_times, manage_bundle_values, dic_names_method, color_dic, name, coeff, max_nb_samples, rho, sol_subgradient, list_iter_index_pegasos, list_obj_value_gap_pegasos, eps_optimal)
    p = plot(size = (600,500))
    for manage_bundle in manage_bundle_values
        # plot!(dic_gaps[manage_bundle], label = "", color = color_dic[manage_bundle],linewidth=4)
        plot!(dic_gaps[manage_bundle], label = dic_names_method[manage_bundle], color = color_dic[manage_bundle],linewidth=4)
    end
    plot!([s.objective for s in sol_subgradient.iteration_stats], label = "subgradient", linewidth = 4, color = "blue")
    plot!(list_iter_index_pegasos, list_obj_value_gap_pegasos, label = "Pegasos", linewidth = 4)
    
    hline!([eps_optimal], linestyle=:dash, label="Tolerance", color = "grey")
    xlabel!("Iterations")
    ylabel!("Optimality Gap")
    plot!(legendfontsize=14)
    # y_ticks = range(1e-4, stop=1e1, length=20)
    # title!("Progress for different bundle management strategies")
    plot!(yaxis=:log, yticks = [1e-5, 1e-4, 1e-3, 1e-2,1e-1, 1, 10, 100])
    # ylims!(1e-5, 100)
    # xlims!(-10, 1000)
    savefig(p, "Optimality_gap.pdf") 
    savefig(p, "proximal_bundle_method_main/outputs/Optimality_gap_by_iteration_"*name*"_"*string(coeff)*"_"*string(max_nb_samples)*"_rho_"*string(rho)*".pdf") 
    plot!()
    return p
end 