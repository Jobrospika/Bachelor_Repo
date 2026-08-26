import os
import itertools
from datetime import datetime

import gpytorch
import pandas as pd
import torch
import math

from src.experiments.generic_grid_experiment import observe_data
from src.gps.models import ExactGPSEModel, ExactGPMatern52Model
from src.utils.experiment_utils import seed_everything, generate_random_safe_initial_points, read_config

from src.algorithms.losboAdaptive import LosboAdaptive

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))


def set_up_model_extended(init_x, init_y, model_type, lengthscale):
    shared_kwargs = dict(
        lengthscale_constraint=None,
        lengthscale_hyperprior=gpytorch.priors.NormalPrior(lengthscale, 1),
        outputscale_constraint=None,
        outputscale_hyperprior=gpytorch.priors.NormalPrior(1, 1),
        noise_constraint=None,
        noise_hyperprior=None,
        ard_num_dims=None,
        prior_mean=0,
    )
    if model_type == "matern52":
        model = ExactGPMatern52Model(init_x, init_y, **shared_kwargs)
    else:
        model = ExactGPSEModel(init_x, init_y, **shared_kwargs)
    model.likelihood.noise_covar.noise = 0.01
    return model


# ============================================================================
# Single-run adaptive LoSBO experiment
# ============================================================================

def run_adaptive_experiment(
    function_config_path,
    function_label,
    E_multiplier=1.0,
    lipschitz_factor=1.1,
    kernel="se",
    local_std=0.05,
    local_std_label="medium",
    seed=0,
    couple_lengthscale=True,
    iterations=20,
    n_safe_samples=1000,
    n_frontier_samples=1000,
    save_trajectory_dir=None,
):
    seed_everything(seed)
    function_info = read_config(function_config_path)

    E = E_multiplier * function_info["noise_lvl"]

    init_x = generate_random_safe_initial_points(
        lambda x: observe_data(x, function_info, noise_on=False),
        function_info["domain_bounds"],
        safety_threshold=function_info["safety_threshold"] + 0.2,
        num_points=1,
    )
    init_y = observe_data(init_x, function_info, noise_on=True).squeeze(-1)

    if couple_lengthscale:
        # main sweep setting: the padded Lipschitz belief also sets GP smoothness
        lengthscale = 1.0 / (function_info["lipschitz_constant"] * lipschitz_factor)
    else:
        # control setting: GP smoothness held fixed, lipschitz_factor drives only the safety margin
        lengthscale = 1.0 / function_info["lipschitz_constant"]

    model = set_up_model_extended(init_x, init_y, model_type=kernel, lengthscale=lengthscale)

    config = dict(
        bounds=function_info["domain_bounds"],
        safety_threshold=function_info["safety_threshold"],
        lipschitz_constant=function_info["lipschitz_constant"] * lipschitz_factor,
        E=E,
        beta=2,
        seed_set=init_x,
        n_safe_samples=n_safe_samples,
        n_frontier_samples=n_frontier_samples,
        local_std=local_std,
        candidate_seed=seed,
        lipschitz_scaling=False,
    )

    opt = LosboAdaptive(config, model)
    opt.update_gp()

    traj_rows = [{
        "iteration": 0,
        "x": init_x[0].tolist(),
        "y": init_y[0].item(),
        "pred_opt": observe_data(
            opt.get_current_best_mean_x().unsqueeze(0), function_info, noise_on=False
        ).item(),
    }]

    safety_violations = 0
    true_safety_violations = 0
    no_opt_possible = False

    for t in range(iterations):
        try:
            x_next = opt.optimize()
        except Exception as ex:
            no_opt_possible = True
            print(f"    [{function_label} seed={seed}] stopped at iteration {t}: {ex}")
            break

        y_next = observe_data(x_next, function_info).squeeze(-1)
        y_next_true = observe_data(x_next, function_info, noise_on=False).squeeze(-1)
        if y_next < function_info["safety_threshold"]:
            safety_violations += 1
        if y_next_true < function_info["safety_threshold"]:
            true_safety_violations += 1

        opt.add_data_to_gp(x_next, y_next)
        opt.update_gp()

        pred_opt = observe_data(
            opt.get_current_best_mean_x().unsqueeze(0), function_info, noise_on=False
        ).squeeze()

        traj_rows.append({
            "iteration": t + 1,
            "x": x_next.squeeze().tolist(),
            "y": y_next.squeeze().item(),
            "pred_opt": pred_opt.item(),
        })

    traj_df = pd.DataFrame(traj_rows)

    if opt.safe_set.shape[0] > 0:
        final_pred_opt = observe_data(
            opt.get_current_best_mean_x().unsqueeze(0), function_info, noise_on=False
        ).item()
    else:
        best_idx = torch.argmax(opt.Y)
        final_pred_opt = observe_data(
            opt.X[best_idx].unsqueeze(0), function_info, noise_on=False
        ).item()

    if save_trajectory_dir is not None:
        os.makedirs(save_trajectory_dir, exist_ok=True)
        fname = f"{function_label}_E{E}_L{lipschitz_factor}_{kernel}_std{local_std_label}_seed{seed}.csv"
        traj_df.to_csv(os.path.join(save_trajectory_dir, fname), index=False)

    true_optimum = function_info["optimum"]
    result = {
        "function": function_label,
        "E": E,
        "lipschitz_factor": lipschitz_factor,
        "kernel": kernel,
        "local_std_label": local_std_label,
        "local_std_value": local_std,
        "seed": seed,
        "couple_lengthscale": couple_lengthscale,
        "init_x": init_x[0].tolist(),
        "safety_violations": safety_violations,
        "true_safety_violations": true_safety_violations,
        "no_opt_possible": no_opt_possible,
        "final_pred_opt": final_pred_opt,
        "avg_pred_opt": traj_df["pred_opt"].mean(),
        "true_optimum": true_optimum,
        "ratio": final_pred_opt / true_optimum,
        "n_observations": opt.X.shape[0],
    }
    return result


# ============================================================================
# Sweep configuration
# ============================================================================

E_multiplier = {
    #"small": 0.5,
    "exact": 1.0,
    #"large": 2.0,
}

L_FACTORS = {
    "underestimated": 0.5,
    #"exact": 1.0,
    "low_safety_net": 1.5,
    "high_safety_net": 2.0,
}

LOCAL_STD_VALUES = {
    "tight": 0.02,
    "medium": 0.05,
    "wide": 0.15,
}

KERNELS = [
    #"se",
    "matern52"
]


FUNCTIONS_ND = [
    # 2D
    #("braninmod_2D", os.path.join(REPO_ROOT, "config/function_config/braninmod_2D.yaml")),
    #("camel_2D", os.path.join(REPO_ROOT, "config/function_config/camel_2D.yaml")),
    ("rosenbrock_2D", os.path.join(REPO_ROOT, "config/function_config/rosenbrock_2D.yaml")),
    #("sphere_2D", os.path.join(REPO_ROOT, "config/function_config/sphere_2D.yaml")),

    # 3D
    #("hartmann_3D", os.path.join(REPO_ROOT, "config/function_config/hartmann_3D.yaml")),

    # 4D
    #("rosenbrock_4D", os.path.join(REPO_ROOT, "config/function_config/rosenbrock_4D.yaml")),
    #("sphere_4D", os.path.join(REPO_ROOT, "config/function_config/sphere_4D.yaml")),
    #("styblinski_tang_4D", os.path.join(REPO_ROOT, "config/function_config/styblinski_tang_4D.yaml")),

    # 5D
    #("sum_squares_5D", os.path.join(REPO_ROOT, "config/function_config/sum_squares_5D.yaml")),

    # 6D
    ("hartmann_6D", os.path.join(REPO_ROOT, "config/function_config/hartmann_6D.yaml")),
    #("sphere_6D", os.path.join(REPO_ROOT, "config/function_config/sphere_6D.yaml")),
    ("griewank_6D", os.path.join(REPO_ROOT, "config/function_config/griewank_6D.yaml")),

    # 7D
    #("dixon_price_7D", os.path.join(REPO_ROOT, "config/function_config/dixon_price_7D.yaml")),


    # 8D
    #("cosine8", os.path.join(REPO_ROOT, "config/function_config/cosine8.yaml")),

    # 9D
    #("rastrigin_9D", os.path.join(REPO_ROOT, "config/function_config/rastrigin_9D.yaml")),

    # 10D
    #("camel_10D", os.path.join(REPO_ROOT, "config/function_config/camel_10D.yaml")),
    ("gaussian_10D", os.path.join(REPO_ROOT, "config/function_config/gaussian_10D.yaml")),
    #("schwefel_10D", os.path.join(REPO_ROOT, "config/function_config/schwefel_10D.yaml")),
    #("trid_10D", os.path.join(REPO_ROOT, "config/function_config/trid_10D.yaml")),

]

N_SEEDS = 10

COUPLE_LENGTHSCALE = False

def classify_violation(v):
    if v == 0:
        return "none"
    elif v <= 2:
        return "minor"
    else:
        return "major"


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ls_tag = "coupled" if COUPLE_LENGTHSCALE else "decoupled"
    run_dir = os.path.join(REPO_ROOT, f"results/bachelor_tests_nd_adaptive/run_{timestamp}_{ls_tag}")
    trajectory_dir = os.path.join(run_dir, "trajectories")
    os.makedirs(run_dir, exist_ok=True)

    grid = list(itertools.product(
        E_multiplier.items(),
        L_FACTORS.items(),
        KERNELS,
        LOCAL_STD_VALUES.items(),
        FUNCTIONS_ND,
    ))

    print(f"Total configurations: {len(grid)}")
    print(f"Runs per config (seeds): {N_SEEDS}")
    print(f"Total runs: {len(grid) * N_SEEDS}\n")
    print(f"Lengthscale: {'coupled to lipschitz_factor' if COUPLE_LENGTHSCALE else 'decoupled'}")

    all_rows = []

    for i, ((e_label, e_mult), (l_label, L), kernel, (std_label, std_val), (fn_label, fn_path)) in enumerate(grid):
        config_label = f"E={e_label}_L={l_label}_kernel={kernel}_std={std_label}_fn={fn_label}"
        print(f"[{i+1}/{len(grid)}] {config_label}")

        function_config = read_config(fn_path)
        d = function_config["domain_size"]
        iterations = round(20+40*math.sqrt(d))

        for seed in range(N_SEEDS):
            try:
                result = run_adaptive_experiment(
                    function_config_path=fn_path,
                    function_label=fn_label,
                    E_multiplier=e_mult,
                    lipschitz_factor=L,
                    kernel=kernel,
                    local_std=std_val,
                    local_std_label=std_label,
                    seed=seed,
                    couple_lengthscale=COUPLE_LENGTHSCALE,
                    iterations=iterations,
                    save_trajectory_dir=trajectory_dir,
                )
                result["E_label"] = e_label
                result["L_label"] = l_label
                result["E_value"] = e_mult * function_config["noise_lvl"]
                result["violation_class"] = classify_violation(result["safety_violations"])
                all_rows.append(result)
            except Exception as ex:
                print(f"  !! seed={seed} failed: {ex}")

    df = pd.DataFrame(all_rows)
    raw_path = f"{run_dir}/RAW_RUNS_{timestamp}.csv"
    df.to_csv(raw_path, index=False)
    print(f"\nSaved {len(df)} raw run rows to {raw_path}")

    agg = df.groupby(["function", "E_label", "L_label", "kernel", "local_std_label"]).agg(
        n_runs=("ratio", "count"),
        mean_ratio=("ratio", "mean"),
        std_ratio=("ratio", "std"),
        pct_no_violation=("violation_class", lambda s: round(100 * (s == "none").sum() / len(s), 1)),
        pct_no_opt=("no_opt_possible", lambda s: round(100 * s.sum() / len(s), 1)),
        mean_final=("final_pred_opt", "mean"),
        mean_avg=("avg_pred_opt", "mean"),
    ).reset_index()

    summary_path = f"{run_dir}/SUMMARY_ND_ADAPTIVE_{timestamp}.csv"
    agg.to_csv(summary_path, index=False)
    print(f"Saved aggregated summary to {summary_path}")