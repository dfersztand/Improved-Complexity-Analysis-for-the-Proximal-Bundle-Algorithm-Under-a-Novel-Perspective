@enum Manage_bundle Keep_all Keep_active Keep_active_null_step
color_dic = Dict(Keep_all=>"green", Keep_active=>"red", Keep_active_null_step=>"blue", "Pegasos" => "gold",  "Subgradient" => "violet");
dic_names_method = Dict(Keep_all=>"All-cut", Keep_active=>"Active-cut", Keep_active_null_step=>"Single-cut", "Pegasos" => "Pegasos",  "Subgradient" => "Subgradient" );


function make_master_model(n, rho, x_center, coeff)
    # master_model =  Model(Ipopt.Optimizer)
    # set_optimizer_attribute(master_model, "print_level", 0)
    # set_optimizer_attribute(master_model, "tol", 1e-8)
    master_model =  Model(Gurobi.Optimizer)
    set_optimizer_attribute(master_model, "OutputFlag", 0)
    set_optimizer_attribute(master_model, "FeasibilityTol", 1e-8)
    # set_optimizer_attribute(master_model, "print_level", 0)
    # set_optimizer_attribute(master_model, "tol", 1e-8)
    
    @variable(master_model, x[1:n])
    @variable(master_model, z)
#     @NLobjective(master_model, Min,log(exp(sum(x[i]^2 for i in 1:n))+1) + z + rho*sum((x[i]-x_center[i])^2 for i in 1:n));
    @objective(master_model, Min,coeff/2*sum(x[i]^2 for i in 1:n) + z + rho/2*sum((x[i]-x_center[i])^2 for i in 1:n));
    
    return master_model
end

function make_cut(x_value, i_model)
    @objective(i_model, Max, sum(i_model.obj_dict[:y][1:end-1] .* x_value) + i_model.obj_dict[:y][end]);
    optimize!(i_model)
    obj_value = objective_value(i_model)
    y_opt = value.(i_model.obj_dict[:y])
    return obj_value, y_opt
end

function add_cut(x_value, m_model, dict_cut, iter, objective, gradient)
    obj_value = objective(x_value)
    y_opt = gradient(x_value)
    dict_cut[iter] = @constraint(m_model, m_model.obj_dict[:z]>= sum(y_opt.*(m_model.obj_dict[:x] .- x_value)) + obj_value);
    return obj_value
end

function update_proximal_center(m_model, x_center, rho, coeff)
    n = length(m_model.obj_dict[:x])
    
    @objective(m_model, Min, coeff/2*sum(m_model.obj_dict[:x][i]^2 for i in 1:n) + m_model.obj_dict[:z] + rho/2*sum((m_model.obj_dict[:x][i]-x_center[i])^2 for i in 1:n));

#     @NLobjective(m_model, Min, log(exp(sum(m_model.obj_dict[:x][i] for i in 1:n))+1) + m_model.obj_dict[:z] + rho*sum((m_model.obj_dict[:x][i]-x_center[i])^2 for i in 1:n));
end

function keep_active_cuts(dict_cut, m_model, verbose)
    dual_dict_cut = Dict(key => dual(value) for (key, value) in dict_cut)
    # d = length(m_model.obj_dict[:x])
    # n_cuts = length(dual_dict_cut)
    
    min_dual = 1e-8# * sum(values(dual_dict_cut))
    # if n_cuts>Inf
    #     l = reverse(sort([dual(value) for (key, value) in dict_cut]))[2*d+2]
    #     min_dual = max(min_dual, l)
    # end
    for k in (keys(dual_dict_cut))
        if (dual_dict_cut[k])<min_dual
            if verbose
                println("Cut ",k," deleted")
            end
            delete(m_model, dict_cut[k])
            pop!(dict_cut, k)
        end
    end
end
function keep_all_cuts(dict_cut, master_model, verbose)
end

function clear_all_cut_but(dict_cut, master_model, lst_to_keep, verbose)
    for k in keys(dict_cut)
        if !(k in lst_to_keep)
            if verbose
                println("Cut ",k," deleted")
            end
            delete(master_model, dict_cut[k])
            pop!(dict_cut, k)
        end
    end
end


function solve_PBA(x_init,rho,epsilon,delta, manage_bundle, goal_obj,coeff,Xp, y,opt_val, verbose = true, timeout = 3600)
    full_function = false
    objective = (w -> svm_objective(w, Xp, y, coeff, full_function) - opt_val)
    gradient = (w -> svm_subgradient(w, Xp, y, coeff,full_function))
    x_value = x_init
    prox_center = x_init
    n = length(x_init)
    master_model = make_master_model(n, rho, x_init, coeff)
    dict_cut = Dict()
    master_obj_value = Inf
    f_k_value = Inf
    g_value = Inf
    f_value = 0. 
    list_number_null_steps = [1]
    number_serious_steps = 0
    bundle_size = [0]
    list_obj_value_gap = [objective(x_init) + coeff/2*sum(x_init.^2)]
    list_times = [0.]
    has_timeout_bundle = false
    t = time()
    for iter in 1:150000
        f_value = add_cut(x_value, master_model, dict_cut, iter, objective, gradient)
        if iter >= 2
            master_obj_value = g_value + f_value
            push!(list_obj_value_gap, master_obj_value)
            push!(list_times, time()-t)
            if master_obj_value - goal_obj < epsilon
                println("Number of iterations : ", iter)
                break
            end
            if verbose
                println("Objective value master problem ", master_obj_value)
            end
            primal_gap = f_value - f_k_value
            if verbose
                println("----------- ITERATION ", iter, " ---------")
                println("The primal_gap is: ", primal_gap)
            end
            if primal_gap<delta + rho/2*sum((x_value-prox_center).^2)
                if goal_obj<-1e9
                   break 
                end
                number_serious_steps+=1

                push!(list_number_null_steps, 0)
                update_proximal_center(master_model, x_value, rho, coeff)
                prox_center = copy(x_value)
                if manage_bundle == Keep_active_null_step
                    clear_all_cut_but(dict_cut, master_model, [iter], verbose)
                end
                if verbose
                    println("*****NEW PROX CENTER*****")
                end
            else
                list_number_null_steps[number_serious_steps+1]+=1
            end
        end
        optimize!(master_model)
#         g_value = log(exp(value(sum(master_model.obj_dict[:x][i]^2 for i in 1:n)))+1)
        g_value =  value(coeff/2*sum(master_model.obj_dict[:x][i]^2 for i in 1:n))
        x_value = value.(master_model.obj_dict[:x])
        f_k_value = value(master_model.obj_dict[:z])
        if manage_bundle == Keep_all
            keep_all_cuts(dict_cut, master_model,verbose)
        else
            keep_active_cuts(dict_cut, master_model,verbose)
        end
        push!(bundle_size, length(dict_cut))
#         println(g_value, "  ", f_value)
        if time()-t>timeout
            runtime = time()-t
            has_timeout_bundle = true
            runtime = Inf
            break
        end
    end
    runtime = time()-t
    
    return dict_cut,master_model, list_number_null_steps, number_serious_steps, bundle_size,runtime, list_obj_value_gap, list_times,has_timeout_bundle
end