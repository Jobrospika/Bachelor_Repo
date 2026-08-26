import math
import time

import fire
import gpytorch
import torch
import pandas as pd
import os

from src.algorithms.ucb_losbo import LosGPUCB
from src.experiments.testfunctions import observe_data
from src.gps.models import ExactGPSEModel
from src.utils.experiment_utils import (get_initial_safe_points, read_config,
                                  save_results, seed_everything,
                                  setup_experiment)


def experiment(config_path, seed=15):
    """
    Sets up and runs an experiment with predefined and random settings.

    Parameters:
        config_path (str): Path to the experiment configuration file.
        seed (int): Seed for random number generators to ensure reproducibility.
    """
    seed_everything(seed)
    losbo_config, function_config, kernel_config, config = setup_experiment(config_path, seed=seed, set_lengthscale=True)
    #run_random_experiment(seed=seed,  config=config, function_config=function_config, losbo_config=losbo_config, kernel_config=kernel_config)
    
    run_experiment(losbo_config, function_config, kernel_config, config)
    
def run_experiment(losbo_config, function_config, kernel_config, config):
    """
    Executes the main experiment loop for a LOS-GP-UCB algorithm configuration.

    Parameters:
        losbo_config (dict): LOS-GP-UCB algorithm configuration.
        function_config (dict): Configuration of the function being optimized.
        kernel_config (dict): Kernel configuration for the GP model.
        config (dict): General experiment configuration.
    """
    
    if "rkhs" in function_config["type"]:
        function_config["domain_bounds"]=[0,1]
        kernel_config["learn_hyperparameters"] = False
        kernel_config["lengthscale"]["prior_mean"]=function_config["gamma"]/math.sqrt(2)
        kernel_config["outputscale"]["prior_mean"]=1
        kernel_config["prior_mean"]=0
    
    train_X = get_initial_safe_points(observe_data, config, function_config, losbo_config, kernel_config)
    # explicit output dimension -- Y is 10 x 1
    train_Y = observe_data(train_X, function_config)

    mll, gp = initialize_gp(train_X, train_Y, kernel_config)
    
    # LosBO Loop
    iterations = losbo_config["max_iterations"]
    losbo_config["bounds"] = function_config["domain_bounds"]
    losbo_config["safety_threshhold"] = function_config["safety_threshold"]
    losbo_config["lipschitz_constant"] = function_config["lipschitz_constant"]
    
    losbo = LosGPUCB(losbo_config, gp)
    regret = torch.Tensor([[losbo.calculate_regret(function_config, lambda x: observe_data(x, function_config, False))]])
   
    for i in range(iterations):
        print(f"Iteration:{i}")
        x_next = losbo.optimize()
        Y = observe_data(x_next, function_config)
        losbo.add_new_point(x_next, Y)
        regret = torch.cat((regret, losbo.calculate_regret(function_config, lambda x: observe_data(x, function_config, False))),0)
        gp.set_train_data(train_X, train_Y, strict=False)

    save_results(config, function_config, losbo_config, losbo.X, losbo.Y, 0, regret, seed = config['seed'])

    # --- DIAGNOSTIC: dump ball-escape logs alongside the normal results ---
    #escape_path = os.path.join(config['results_path'], "escape_log.csv")
    #selected_path = os.path.join(config['results_path'], "selected_log.csv")
    #pd.DataFrame(losbo._escape_log).to_csv(escape_path, index=False)
    #pd.DataFrame(losbo._selected_log).to_csv(selected_path, index=False)
    #print(f"Escape log: {len(losbo._escape_log)} entries -> {escape_path}")
    #print(f"Selected log: {len(losbo._selected_log)} entries -> {selected_path}")
    # --- end diagnostic ---
    
def run_random_experiment(seed=1, config=None, function_config=None, losbo_config=None, kernel_config=None):
    """
    Executes a random baseline experiment and logs the results.

    Parameters:
        seed (int): Seed for reproducibility.
        config (dict): General experiment configuration.
        function_config (dict): Configuration of the function being optimized.
        losbo_config (dict): LOSBO related configuration.
        kernel_config (dict): Kernel configuration.
    """
    seed_everything(seed)
    bounds = function_config["domain_bounds"]
    start = time.time()
    X = torch.zeros((losbo_config["max_iterations"]+losbo_config["initial_safe_points"], len(bounds)))
    # sample initial safe set as first points
    x = get_initial_safe_points(observe_data, config, function_config, losbo_config, kernel_config=kernel_config)
    X[:losbo_config["initial_safe_points"], :] = x
    for j, bound in enumerate(bounds):
        X[losbo_config["initial_safe_points"]:, j] = torch.rand(losbo_config["max_iterations"], 1).squeeze(1) * (bound[1] - bound[0]) + bound[0]
    Y = observe_data(X, function_config, False)
    regret = function_config["optimum"]-Y
    for i in range(1, len(regret)):
        regret[i] = min(regret[i], regret[i-1])
    end = time.time()
    time_elapsed = end-start
    config["algorithm"] = "random"
 
    save_results(config, function_config, losbo_config, X, Y.squeeze(1), time_elapsed, regret, True)


def initialize_gp(train_X, train_Y, kernel_config):
    """ Initialize GP model and MLL object
    Args:
        train_X: torch.Tensor
        train_Y: torch.Tensor
        kernel_config: dict
    Returns:
        mll: gpytorch.mlls.ExactMarginalLogLikelihood
        model: ExactGPSEModel
    """
    if kernel_config["ard"]:
        ard_num_dims = train_X.shape[1]
    else:
        ard_num_dims = None
     # initialize GP
    model = ExactGPSEModel(
        train_X,
        train_Y,
        lengthscale_constraint=None,
        lengthscale_hyperprior=gpytorch.priors.NormalPrior(
            kernel_config["lengthscale"]["prior_mean"],
            kernel_config["lengthscale"]["prior_std"],
        ),
        outputscale_constraint=None,
        outputscale_hyperprior=gpytorch.priors.NormalPrior(
            kernel_config["outputscale"]["prior_mean"],
            kernel_config["outputscale"]["prior_std"],
        ),
        noise_constraint=None,
        noise_hyperprior=gpytorch.priors.NormalPrior(
            kernel_config["noise"]["prior_mean"],
            kernel_config["noise"]["prior_std"],
        ),
        ard_num_dims=ard_num_dims,
        prior_mean=kernel_config["prior_mean"],
    )

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    return mll, model

def experiment_rkhs(config_path, seed=15, function_no=0):
    """
    Sets up and runs an experiment specifically for RKHS functions.

    Parameters:
        config_path (str): Path to the experiment configuration file.
        seed (int): Seed for random number generators to ensure reproducibility.
        function_no (int): Function number to select specific function configuration.
    """
    seed_everything(seed)
    losbo_config, function_config, kernel_config, config = setup_experiment(config_path, seed=seed)
    function_config = read_config(os.path.join(function_config["path"], f"function_{function_no}.yaml"))
    run_experiment(losbo_config, function_config, kernel_config, config)


def select_experiment(config_path, seed=15, function_no=0):
    """
    Selects and runs the appropriate experiment based on the configuration path.

    Parameters:
        config_path (str): Path to the experiment configuration.
        seed (int): Seed for reproducibility.
        function_no (int): Function number to use in the experiment.
    """
    if "pre_rkhs" in config_path:
        experiment_rkhs(config_path, seed)
    else:
        experiment(config_path, seed)
    

if __name__ == "__main__":
    fire.Fire(select_experiment)

