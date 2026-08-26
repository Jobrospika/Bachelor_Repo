import os
import pickle

import fire
import pandas as pd

from src.experiments.generic_grid_experiment import run_experiment

# Define valid algorithms, sampling methods and models
algorithms = ["losbo", "real_beta", "safeopt"]
sampling_methods = ["pre_rkhs_matern32", "onb_rkhs_se", "pre_rkhs_se"]
models = ["se", "matern32"]

# Experiment definitions
experiments = dict()

# ONB sampling SE-kernel
experiments[2] = {"slurm_seed": 0, "experiment_no": 2, "algorithm": "safeopt", "sampling_method": "onb_rkhs_se", "model": "se", "B": 10, "initial_points": 2, "in_reachable_set": False, "lengthscale_factor": 1}
experiments[30] = {"slurm_seed": 0, "experiment_no": 3_0, "algorithm": "real_beta", "sampling_method": "onb_rkhs_se", "model": "se", "B": 10, "initial_points": 2, "in_reachable_set": False, "lengthscale_factor": 1}

# misspecified B
experiments[31] = {"slurm_seed": 0, "experiment_no": 3_1, "algorithm": "real_beta", "sampling_method": "onb_rkhs_se", "model": "se", "B": 2.5, "initial_points": 2, "in_reachable_set": False, "lengthscale_factor": 1}
experiments[32] = {"slurm_seed": 0, "experiment_no": 3_2, "algorithm": "real_beta", "sampling_method": "onb_rkhs_se", "model": "se", "B": 20, "initial_points": 2, "in_reachable_set": False, "lengthscale_factor": 1}
experiments[33] = {"slurm_seed": 0, "experiment_no": 3_3, "algorithm": "losbo", "sampling_method": "onb_rkhs_se", "model": "se", "B": 10, "initial_points": 2, "in_reachable_set": False, "lengthscale_factor": 1}

experiments[40] = {"slurm_seed": 0, "experiment_no": 4_0, "algorithm": "real_beta", "sampling_method": "onb_rkhs_se", "model": "se", "B": 10, "initial_points": 1, "in_reachable_set": True, "lengthscale_factor": 1}
experiments[41] = {"slurm_seed": 0, "experiment_no": 4_1, "algorithm": "losbo", "sampling_method": "onb_rkhs_se", "model": "se", "B": 10, "initial_points": 1, "in_reachable_set": True, "lengthscale_factor": 1}

# Pre-RKHS sampling Matern32-kernel
experiments[42] = {"slurm_seed": 0, "experiment_no": 4_2, "algorithm": "real_beta", "sampling_method": "pre_rkhs_matern32", "model": "matern32", "B": 10, "initial_points": 1, "in_reachable_set": True, "lengthscale_factor": 1}
experiments[43] = {"slurm_seed": 0, "experiment_no": 4_3, "algorithm": "losbo", "sampling_method": "pre_rkhs_matern32", "model": "matern32", "B": 10, "initial_points": 1, "in_reachable_set": True, "lengthscale_factor": 1}

# misspecified lengthscale
experiments[50] = {"slurm_seed": 0, "experiment_no": 5_0, "algorithm": "real_beta", "sampling_method": "pre_rkhs_matern32", "model": "matern32", "B": 10, "initial_points": 1, "in_reachable_set": True, "lengthscale_factor": 4}
experiments[51] = {"slurm_seed": 0, "experiment_no": 5_1, "algorithm": "losbo", "sampling_method": "pre_rkhs_matern32", "model": "matern32", "B": 10, "initial_points": 1, "in_reachable_set": True, "lengthscale_factor": 4}
experiments[52] = {"slurm_seed": 0, "experiment_no": 5_2, "algorithm": "real_beta", "sampling_method": "pre_rkhs_matern32", "model": "matern32", "B": 10, "initial_points": 1, "in_reachable_set": True, "lengthscale_factor": 0.2}
experiments[53] = {"slurm_seed": 0, "experiment_no": 5_3, "algorithm": "losbo", "sampling_method": "pre_rkhs_matern32", "model": "matern32", "B": 10, "initial_points": 1, "in_reachable_set": True, "lengthscale_factor": 0.2}

# Pre-RKHS sampling SE-kernel
experiments[60] = {"slurm_seed": 0, "experiment_no": 6_0, "algorithm": "real_beta", "sampling_method": "pre_rkhs_matern32", "model": "se", "B": 10, "initial_points": 1, "in_reachable_set": True, "lengthscale_factor": 1}
experiments[61] = {"slurm_seed": 0, "experiment_no": 6_1, "algorithm": "losbo", "sampling_method": "pre_rkhs_matern32", "model": "se", "B": 10, "initial_points": 1, "in_reachable_set": True, "lengthscale_factor": 1}

experiments[70] = {"slurm_seed": 0, "experiment_no": 7_0, "algorithm": "real_beta", "sampling_method": "pre_rkhs_se", "model": "se", "B": 10, "initial_points": 1, "in_reachable_set": True, "lengthscale_factor": 1}
experiments[71] = {"slurm_seed": 0, "experiment_no": 7_1, "algorithm": "losbo", "sampling_method": "pre_rkhs_se", "model": "se", "B": 10, "initial_points": 1, "in_reachable_set": True, "lengthscale_factor": 1}

# Select experiment by number, which function to use and how many samples to run
def select_experiment(experiment_no, function_no, num_samples=10000, plot=False):
    """ Function to run a specific experiment with a given function and number of samples.
    
    Args:
        experiment_no (int): The number of the experiment to run.
        function_no (int): The number of the function to use.
        num_samples (int): The number of samples to run.
        plot (bool): Whether to plot the function.
    """

    # Check experiment number is valid
    if experiment_no not in experiments.keys():
        raise ValueError(f"Experiment number {experiment_no} not defined")
    
    # Check if algorithm, sampling method and model are valid
    if experiments[experiment_no]["algorithm"] not in algorithms:
        raise ValueError(f"Algorithm {experiments[experiment_no]['algorithm']} not defined")
    if experiments[experiment_no]["sampling_method"] not in sampling_methods:
        raise ValueError(f"Sampling method {experiments[experiment_no]['sampling_method']} not defined")
    if experiments[experiment_no]["model"] not in models:
        raise ValueError(f"Model {experiments[experiment_no]['model']} not defined")
    
    # Run experiment
    run_multiple_seeds(slurm_seed=experiments[experiment_no]["slurm_seed"], 
                       num_samples=num_samples, 
                       function_no=function_no, 
                       experiment_no=experiments[experiment_no]["experiment_no"], 
                       algorithm=experiments[experiment_no]["algorithm"], 
                       sampling_method=experiments[experiment_no]["sampling_method"],
                       model=experiments[experiment_no]["model"],
                       B=experiments[experiment_no]["B"], 
                       initial_points=experiments[experiment_no]["initial_points"], 
                       in_reachable_set=experiments[experiment_no]["in_reachable_set"], 
                       lengthscale_factor=experiments[experiment_no]["lengthscale_factor"], 
                       plot=plot)

# Run multiple seeds for a given experiment
def run_multiple_seeds(slurm_seed=1, num_samples=1000, function_no=20, experiment_no=5, algorithm="losbo", sampling_method = "pre_rkhs_matern32", model = "matern32", B=10, initial_points=1, in_reachable_set=True, lengthscale_factor=1, plot=False):
    """ Function to run multiple seeds for a given experiment.
    
    Args:
        slurm_seed (int): The slurm seed to use.
        num_samples (int): The number of samples to run.
        function_no (int): The identification number of the function to use.
        experiment_no (int): The number of the experiment to run.
        algorithm (str): The algorithm to use.
        sampling_method (str): The sampling method to use.
        model (str): The model to use.
        B (float): The beta value to use.
        initial_points (int): The number of initial points to use.
        in_reachable_set (bool): Whether to use the reachable set.
        lengthscale_factor (float): The lengthscale factor to use.
        plot (bool): Whether to plot the function.
    """
    # Set Path for results (e.g. results/experiment_1_losbo)
    results_path = os.path.join("results", f"experiment_{experiment_no}_{algorithm}")
    if os.path.isdir(results_path) == False:                                            # Create results folder if it does not exist
        os.mkdir(results_path)
    
    # Copy function config file to results folder
    file=f"config/rkhs_function_configs/{sampling_method}/function_{function_no}.yaml"
    
    # Get function name from file name
    file_name = os.path.split(file)[-1]
    dir_name = os.path.splitext(file_name)[0]
    
    # Create folder for results of specific function
    results_dir=os.path.join(results_path, dir_name)
    if os.path.isdir(results_dir) == False:                                             # Create results folder if it does not exist          
        os.mkdir(results_dir)
    
    # Initialise dataframe to store results
    df = pd.DataFrame(columns=["safety_violation", "no_opt_possible", "beta"])

    # Run experiment for each seed
    start_seed = slurm_seed * num_samples
    end_seed = (slurm_seed + 1) * num_samples
    dict_detailed_results = {}
    
    for i in range(start_seed, end_seed):
        # Run experiment
        dict_results, opt_results_df = run_experiment(seed=i, function_path=file, algorithm=algorithm, model_type = model, B=B, plot=plot, sample_points=initial_points, in_reachable_set=in_reachable_set, lengthscale_factor=lengthscale_factor)
        
        # Store detailed results
        dict_detailed_results[f"seed_{i}"] = opt_results_df

        # Cast columns to boolean if they contain only boolean values, to prevent pandas from throwing a warning
        for col in df.select_dtypes(include=['object']).columns:
            if all(item in [True, False, None] for item in df[col]):
                df[col] = df[col].astype('bool')

        # Append results to dataframe
        df = pd.concat([df, pd.DataFrame([dict_results])], ignore_index=True)
    
    # Save results to csv (overview) and pickle (detailed)
    df.to_csv(results_dir+f"/experiment_{experiment_no}_{start_seed}_{end_seed}.csv")
    with open(results_dir+f"/experiment_results_{experiment_no}_{start_seed}_{end_seed}.pkl", 'wb') as f:
        pickle.dump(dict_detailed_results, f)

# Use fire to run experiments from command line
if __name__ == '__main__':
  fire.Fire(select_experiment)
