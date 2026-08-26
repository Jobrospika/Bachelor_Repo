import gpytorch
import pandas as pd
import torch

from src.experiments.generic_grid_experiment import observe_data
from src.gps.models import ExactGPSEModel
from src.utils.experiment_utils import seed_everything, generate_random_safe_initial_points, read_config
from src.algorithms.losbo_higher_d import LosboAdaptive



def set_up_model(init_x, init_y, lengthscale):
    model = ExactGPSEModel(
        init_x, init_y,
        lengthscale_constraint=None,
        lengthscale_hyperprior=gpytorch.priors.NormalPrior(lengthscale, 1),
        outputscale_constraint=None,
        outputscale_hyperprior=gpytorch.priors.NormalPrior(1, 1),
        noise_constraint=None,
        noise_hyperprior=None,
        ard_num_dims=None,
        prior_mean=0,
    )
    model.likelihood.noise_covar.noise = 0.01
    return model


def run_adaptive(function_config_path, local_std_scale, seed=0, iterations=50,
                  n_safe_samples=1000, n_frontier_samples=1000, lipschitz_factor=1.1):
    seed_everything(seed)
    function_info = read_config(function_config_path)

    init_x = generate_random_safe_initial_points(
        lambda x: observe_data(x, function_info, noise_on=False),
        function_info["domain_bounds"],
        safety_threshold=function_info["safety_threshold"] + 0.2,
        num_points=1,
    )
    init_y = observe_data(init_x, function_info, noise_on=True).squeeze(-1)

    lengthscale = 1.0 / (function_info["lipschitz_constant"] * lipschitz_factor)
    model = set_up_model(init_x, init_y, lengthscale)

    config = dict(
        bounds=function_info["domain_bounds"],
        safety_threshold=function_info["safety_threshold"],
        lipschitz_constant=function_info["lipschitz_constant"] * lipschitz_factor,
        E=0.02,
        beta=2,
        seed_set=init_x,
        n_safe_samples=n_safe_samples,
        n_frontier_samples=n_frontier_samples,
        local_std=local_std_scale,
        candidate_seed=seed,
    )

    opt = LosboAdaptive(config, model)
    opt.update_gp()

    safety_violations = 0
    no_opt_possible = False

    for t in range(iterations):
        try:
            x_next = opt.optimize()
        except Exception as ex:
            no_opt_possible = True
            print(f"    stopped at iteration {t}: {ex}")
            break

        y_next = observe_data(x_next, function_info).squeeze(-1)
        if y_next < function_info["safety_threshold"]:
            safety_violations += 1

        opt.add_data_to_gp(x_next, y_next)
        opt.update_gp()


    if opt.safe_set.shape[0] > 0:
        pred_opt = observe_data(
            opt.get_current_best_mean_x().unsqueeze(0), function_info, noise_on=False
        ).item()
    else:
        best_idx = torch.argmax(opt.Y)
        pred_opt = observe_data(
            opt.X[best_idx].unsqueeze(0), function_info, noise_on=False
        ).item()

    return {
        "safety_violations": safety_violations,
        "no_opt_possible": no_opt_possible,
        "final_pred_opt": pred_opt,
        "true_optimum": function_info["optimum"],
        "n_observations": opt.X.shape[0],
    }


# One representative function per dimension.
TEST_FUNCTIONS = [
    ("rosenbrock_4D", "config/function_config/rosenbrock_4D.yaml"),
    ("hartmann_6D",   "config/function_config/hartmann_6D.yaml"),
    ("camel_10D",     "config/function_config/camel_10D.yaml"),
]

# Fraction of each axis range used as jitter std -- tight, medium, wide.
LOCAL_STD_SCALES = {
    "tight":  0.02,
    "medium": 0.05,
    "wide":   0.15,
}

N_SEEDS = 10

if __name__ == "__main__":
    rows = []
    for fn_label, fn_path in TEST_FUNCTIONS:
        for scale_label, scale in LOCAL_STD_SCALES.items():
            print(f"[{fn_label}] local_std={scale_label} ({scale})")
            for seed in range(N_SEEDS):
                try:
                    result = run_adaptive(fn_path, scale, seed=seed)
                    result.update(function=fn_label, local_std_label=scale_label,
                                   local_std_value=scale, seed=seed)
                    rows.append(result)
                    print(f"    seed={seed}: violations={result['safety_violations']} "
                          f"final={result['final_pred_opt']:.4f} "
                          f"(true opt={result['true_optimum']}) "
                          f"n_obs={result['n_observations']}")
                except Exception as ex:
                    print(f"    !! seed={seed} failed: {ex}")

    df = pd.DataFrame(rows)
    print("\n=== Summary (mean over seeds) ===")
    print(df.groupby(["function", "local_std_label"])[
        ["safety_violations", "final_pred_opt", "no_opt_possible"]
    ].mean().round(4))

    df.to_csv("results/losbo_adaptive_scale_test.csv", index=False)
    print("\nSaved raw results to results/losbo_adaptive_scale_test.csv")