import torch
import gpytorch
import math
import matplotlib.pyplot as plt
import sys
import pandas as pd
import yaml

from algorithms.losboAdaptive import LosboAdaptive
from src.utils.experiment_utils import seed_everything
from src.experiments.testfunctions import observe_data 
from src.gps.models import ExactGPSEModel, ExactGPMatern32Model, ExactGPMatern52Model
from src.algorithms.grid_implementation import SafeOptGrid, Losbo, RealBetaSafeOpt


def generate_initial_safe_set(function_info, seed, sample_points=1, in_reachable_set=True):
    '''
    function to generate initial safe set
    
    Args:
    function_info: dictionary containing the function information
    seed: seed for random number generator
    sample_points: number of initial points
    in_reachable_set: sample initial points in reachable set
    '''
    if in_reachable_set:
        init_x = torch.tensor(generate_random_safe_seed_set(lambda x: observe_data(x, function_info), function_info["safety_threshold"], seed = seed, intervall=function_info["reachable_safe_set"], sample_points=sample_points, function_info=function_info)).unsqueeze(-1)
    else:
        init_x = torch.tensor(generate_random_safe_seed_set(lambda x: observe_data(x, function_info), function_info["safety_threshold"], seed = seed, sample_points=sample_points, function_info=function_info)).unsqueeze(-1)  
    init_y = observe_data(init_x, function_info)
    return init_x, init_y

def generate_random_safe_seed_set(observe_data, safety_threshold, seed = 0, intervall=[0,1], sample_points=2, function_info=None):
    """
    generate seed set that is safe
    
    Args:
    observe_data: function to observe the data
    safety_threshold: safety threshold
    seed: seed for random number generator
    intervall: intervall to sample from
    sample_points: number of points to sample
    function_info: dictionary containing the function information
    """
    x = torch.linspace(0, 1, 10000).unsqueeze(1)
    y = observe_data(x)
    df = pd.DataFrame({'x': x.squeeze(-1).numpy(), 'y': y.squeeze(-1).numpy()})
    df = df[df['y'] > (safety_threshold+2*function_info["noise_lvl"])]
    
    # filter if df[x] is in intervall
    df = df[(df['x'] > intervall[0]) & (df['x'] < intervall[1])]
    
    # choose random points
    df = df.sample(n=sample_points, random_state=seed)
    
    return df['x'].values

def set_up_model(init_x, init_y, model_type, lengthscale):
    '''
    set up GP model
    
    Args:
    init_x: initial x values
    init_y: initial y values
    model_type: type of GP model
    lengthscale: lengthscale of the kernel
    '''

    # Diffentiate between Matern52 and SE kernel
    if model_type == "matern32":
        model = ExactGPMatern32Model(                                     # set up model
            init_x,
            init_y,
            lengthscale_constraint=None,
            lengthscale_hyperprior=gpytorch.priors.NormalPrior(lengthscale, 1),
            outputscale_constraint=None,
            outputscale_hyperprior=gpytorch.priors.NormalPrior(1, 1),
            noise_constraint=None,
            noise_hyperprior=None,
            ard_num_dims=None,
            prior_mean=0,
        )
    elif model_type == "matern52":
        model = ExactGPMatern52Model(
            init_x,
            init_y,
            lengthscale_constraint=None,
            lengthscale_hyperprior=gpytorch.priors.NormalPrior(lengthscale, 1),
            outputscale_constraint=None,
            outputscale_hyperprior=gpytorch.priors.NormalPrior(1, 1),
            noise_constraint=None,
            noise_hyperprior=None,
            ard_num_dims=None,
            prior_mean=0,
        )
    else:
        model = ExactGPSEModel(                                     # set up model it is not possible to set lengthscales and output scale directly, instead the mean of the hyper prior is used
            init_x,
            init_y,
            lengthscale_constraint=None,
            lengthscale_hyperprior=gpytorch.priors.NormalPrior(lengthscale, 1),
            outputscale_constraint=None,
            outputscale_hyperprior=gpytorch.priors.NormalPrior(1, 1),
            noise_constraint=None,
            noise_hyperprior=None,
            ard_num_dims=None,
            prior_mean=0,
        )
    
    # fix noise variance
    model.likelihood.noise_covar.noise = 0.01
    return model


def set_up_algorithm(algorithm, config, model):
    '''
    choose an algorithm
    
    Args:
    algorithm: algorithm to use
    config: algorithm configuration
    model: GP model
    '''
    if algorithm == "safeopt":
        opt = SafeOptGrid(config=config, gp=model)
    elif algorithm == "losbo":
        opt = Losbo(config=config, gp=model)
    elif algorithm == "losboadaptive":
        opt = LosboAdaptive(config=config, gp=model)
    elif algorithm == "real_beta":
        beta_dict = config["beta_dict"]
        opt = RealBetaSafeOpt(config=config, gp=model, beta_config=beta_dict)
    else:
        print("Algorithm not implemented")
        sys.exit()
    # Add initial safe set to GP
    opt.update_gp()
    return opt

def get_algorithm_config(function_info, init_x, B=10):
    '''
    config for the grid algorithm
    
    Args:
    function_info: dictionary containing the function information
    init_x: initial x values
    B: rkhs norm bound in the algoritm
    '''
    config = dict()
    config["bounds"] = [(0, 1)]
    config["safety_threshold"] = function_info["safety_threshold"]
    config["lipschitz_constant"] = function_info["lipschitz_constant"]*1.1
    beta_dict = {"B": B, "R": 0.01, "delta": 0.01, "lamb": 0.1}
    config["E"] = 2*0.01
    config["beta"] = 2
    config["points_per_axis"] = 501
    config["seed_set"] = init_x
    config["beta_dict"] = beta_dict
    config["iterations"] = 20
    return config

def run_loop(iterations, opt, function_info, df, plot=False):
    '''
    run the optimization loop
    
    Args:
    iterations: number of iterations
    opt: algorithm object
    function_info: dictionary containing the function information
    df: dataframe to save the results
    plot: boolean if the results should be plotted
    '''
    safety_violation = 0
    no_opt_possible = False
    for t in range(iterations):
        try:
            x_next = opt.optimize()
        except:
            no_opt_possible = True
            break
        y_next = observe_data(x_next, function_info)
        if y_next < function_info["safety_threshold"]:
            safety_violation += 1
        # if x_next in opt.X:
        #     no_opt_possible = True
        #     break
        opt.add_data_to_gp(x_next, y_next)
        opt.update_gp()
        dict_optimization_steps = {"iteration": t+1, "x": x_next.item(), "y": y_next.item(), "pred_opt":observe_data(opt.get_current_best_mean_x().unsqueeze(0), function_info, noise_on=False).item()}
        # add dict to dataframe
        df = pd.concat([df, pd.DataFrame([dict_optimization_steps])], ignore_index=True)
        if plot:
            fig, ax = opt.generate_plot(lambda x: observe_data(x, function_info))
            plt.show()
    return safety_violation, no_opt_possible, df



def run_experiment(seed=0, function_path="results/100_working_ONB_samples/onb_function_info_20.yaml", algorithm="safeopt", model_type = "matern32", B=10, plot=False, sample_points=1, in_reachable_set=True, lengthscale_factor=1):
    '''
    Run a single experiment
    
    Args:
    seed: seed for random number generator
    function_path: path to function info file
    algorithm: algorithm to use
    model_type: type of GP model to use
    B: number of samples to use for beta
    plot: plot results
    sample_points: number of initial points
    in_reachable_set: sample initial points in reachable set
    lengthscale_factor: factor to multiply lengthscale with
    '''
    # Set random seed
    seed_everything(seed)

    # Load function info from config file
    function_info = yaml.load(open(function_path), Loader=yaml.FullLoader)

    # Generate initial safe set
    init_x, init_y = generate_initial_safe_set(function_info, seed, sample_points, in_reachable_set)

    # Set up gaussian process model
    model = set_up_model(init_x, init_y, model_type = model_type, lengthscale = function_info["gamma"]/math.sqrt(2)*lengthscale_factor)
    
    # Set up algorithm
    config = get_algorithm_config(function_info, init_x, B=B)
    opt = set_up_algorithm(algorithm, config, model)
    
    # Different saving for single and multiple initial points
    if init_x.shape[0] == 1:
        dict_optimization = {"iteration": 0, "x": init_x.item(), "y": init_y.item(), "pred_opt":observe_data(opt.get_current_best_mean_x().unsqueeze(0), function_info, noise_on=False).item()}
        # create dataframe for results
        df = pd.DataFrame(dict_optimization, index=[0])
    else:
        for i, x in enumerate(init_x):
            dict_optimization = {"iteration": 0, "x": init_x.tolist()[i], "y": init_y.tolist()[i], "pred_opt":observe_data(opt.get_current_best_mean_x().unsqueeze(0), function_info, noise_on=False).item()}
            # create dataframe for results
            df = pd.DataFrame(dict_optimization, index=[i])

    # Optimization Loop
    safety_violation, no_opt_possible, df = run_loop(config["iterations"], opt, function_info, df, plot=plot)

    # Return results
    dict_results = {"seed": seed, "safety_violation": safety_violation, "no_opt_possible": no_opt_possible, "beta": opt.beta}
    return dict_results, df