using JLD
include("run_SVM_experiment.jl")

path_to_params = ARGS[1]
output_path = ARGS[2]

if split(path_to_params,".")[end] == "ipynb_checkpoints"
    exit()
end

params = load(path_to_params)
(seed_value, name, reg_coeff, max_nb_samples, rho_coef, eps_optimal, timeout) = params["saved_params"]
# println("verbose")
# println(verbose)
main_experiment(seed_value, name, reg_coeff, max_nb_samples, rho_coef, eps_optimal,timeout, output_path)
print(path_to_params)
println("Done")