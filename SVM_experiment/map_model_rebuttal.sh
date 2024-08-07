#!/bin/bash

# Load Julia Module
module load julia/1.9.2
module load gurobi/gurobi-1000
echo $1
echo $2

# Call your script as you would from the command line, passing in $1 and $2 as arugments
# Note that $1 and $2 are the arguments passed into this script
julia map_launcher_rebuttal.jl $1 $2
# ./map_model_rebuttal.sh proximal_bundle_method_main/params_files/run_SVM_experiment_0.01_0.01.jld2 proximal_bundle_method_main/map_output_files/run_SVM_experiment_0.01_0.01.out