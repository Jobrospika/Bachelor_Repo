import os
import math
import yaml
import itertools
import gpytorch
import pandas as pd
import datetime


from src.experiments.generic_grid_experiment import (
    generate_initial_safe_set,
    set_up_algorithm,
    run_loop,
    observe_data
)
from src.gps.models import (
    ExactGPSEModel,
    ExactGPMatern32Model,
    ExactGPMatern52Model
)
from src.utils.experiment_utils import seed_everything

def set_up_model_extended(init_x, init_y, model_type, lengthscale):
    shared_kwargs = dict(lengthscale_constraint=None,
                         lengthscale_hyperprior=gpytorch.priors.NormalPrior(lengthscale,1),
                         outputscale_constraint=None,
                         outputscale_hyperprior=gpytorch.priors.NormalPrior(1, 1),
                         noise_constraint=None,
                         noise_hyperprior=None,
                         ard_num_dims=None,
                         prior_mean=0,
                         )
    if model_type == "matern32":
        model = ExactGPMatern32Model(init_x, init_y, **shared_kwargs)

    elif model_type == "matern52":
        model = ExactGPMatern52Model(init_x, init_y, **shared_kwargs)

    elif model_type == "rq":

        class ExactGPRQModel(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y):
                likelihood = gpytorch.likelihoods.GaussianLikelihood()
                super().__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RQKernel()
                )
                self.covar_module.base_kernel.lengthscale = lengthscale

            def forward(self, x):
                mean = self.mean_module(x)
                covar = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean, covar)

        model = ExactGPRQModel(init_x, init_y)

    else:
        model = ExactGPSEModel(init_x, init_y, **shared_kwargs)

    model.likelihood.noise_covar.noise = 0.01
    return model

def run_custom_experiment(
    seed=0,
    function_path="results/100_working_ONB_samples/onb_function_info_20.yaml",
    algorithm="losbo",
    model_type="se",
    E=0.02,
    lipschitz_factor=1.1,
    sample_points=1,
    in_reachable_set=True,
    lengthscale_factor=1.0,
    iterations=20,
    plot=False,
):
    seed_everything(seed)

    function_info = yaml.load(open(function_path), Loader=yaml.FullLoader)

    init_x, init_y = generate_initial_safe_set(function_info, seed, sample_points, in_reachable_set)

    model = set_up_model_extended(
        init_x,
        init_y,
        model_type=model_type,
        lengthscale=function_info["gamma"] / math.sqrt(2) * lengthscale_factor
    )

    config = dict()
    config["bounds"] = [(0, 1)]
    config["safety_threshold"] = function_info["safety_threshold"]
    config["lipschitz_constant"] = function_info["lipschitz_constant"] * lipschitz_factor
    config["E"] = E
    config["beta"] = 2
    config["points_per_axis"] = 501
    config["seed_set"] = init_x
    config["beta_dict"] = {"B": 10, "R": 0.01, "delta": 0.01, "lamb": 0.1}
    config["iterations"] = iterations

    opt = set_up_algorithm(algorithm, config, model)

    dict_init = {
        "iteration": 0,
        "x": init_x.item(),
        "y": init_y.item(),
        "pred_opt": observe_data(
            opt.get_current_best_mean_x().unsqueeze(0), function_info, noise_on=False
        ).item()
    }
    df = pd.DataFrame(dict_init, index=[0])

    safety_violation, no_opt_possible, df = run_loop(config["iterations"], opt, function_info, df, plot=plot)

    final_performance = df["pred_opt"].iloc[-1]
    avg_performance = df["pred_opt"].mean()

    result = {
        "safety_violation": safety_violation,
        "no_opt_possible": no_opt_possible,
        "final_performance": final_performance,
        "avg_performance": avg_performance,
    }

    return result, df


E_VALUES = {
    "super_small": 0.001,
    "baseline": 0.02,
    "super_large": 1
}

L_FACTORS = {
    "super_small": 0.1,
    "baseline": 1.1,
    "super_large": 4.0
}

KERNELS = ["se", "matern52"]

FUNCTION_FAMILIES = [
    ("onb_rkhs_se", "config/rkhs_function_configs/onb_rkhs_se"),
    ("pre_rkhs_se", "config/rkhs_function_configs/pre_rkhs_se"),
    ("pre_rkhs_matern32", "config/rkhs_function_configs/pre_rkhs_matern32"),
]

FUNCTION_IDS = list(range(2))
N_SEEDS = 2

def classify_violation(v):
    if v == 0:
        return "none"
    elif v <= 2:
        return "minor"
    else:
        return "major"


if __name__ == "__main__":
    os.makedirs("results/bachelor_tests", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    grid = list(itertools.product(
        E_VALUES.items(),
        L_FACTORS.items(),
        KERNELS,
        FUNCTION_FAMILIES,
    ))

    print(f"Total configurations: {len(grid)}")
    print(f"Runs per config: {len(FUNCTION_IDS) * N_SEEDS}")
    print(f"Total runs: {len(grid) * len(FUNCTION_IDS) * N_SEEDS}\n")

    all_summary_rows = []

    for i, ((e_label, E), (l_label, L), kernel, (family_label, family_dir)) in enumerate(grid):

        config_label = f"E={e_label}_L={l_label}_kernel={kernel}_fn={family_label}"
        print(f"[{i + 1}/{len(grid)}] {config_label}")

        run_results = []

        for fn_id in FUNCTION_IDS:
            function_path = f"{family_dir}/function_{fn_id}.yaml"

            for seed in range(N_SEEDS):
                try:
                    result, _ = run_custom_experiment(
                        seed=seed,
                        function_path=function_path,
                        model_type=kernel,
                        E=E,
                        lipschitz_factor=L,
                    )
                    result["function_id"] = fn_id
                    result["seed"] = seed
                    result["violation_class"] = classify_violation(result["safety_violation"])
                    run_results.append(result)
                except Exception as ex:
                    print(f"  !! Error fn={fn_id} seed={seed}: {ex}")

        if not run_results:
            continue

        df_runs = pd.DataFrame(run_results)

        # Aggregate into one summary row
        total = len(df_runs)
        summary = {
            "E_label": e_label,
            "E_value": E,
            "L_label": l_label,
            "L_factor": L,
            "kernel": kernel,
            "function_family": family_label,
            "n_runs": total,
            "pct_no_violation": round(100 * (df_runs["violation_class"] == "none").sum() / total, 1),
            "pct_minor": round(100 * (df_runs["violation_class"] == "minor").sum() / total, 1),
            "pct_major": round(100 * (df_runs["violation_class"] == "major").sum() / total, 1),
            "pct_no_opt": round(100 * df_runs["no_opt_possible"].sum() / total, 1),
            "mean_final_perf": round(df_runs["final_performance"].mean(), 4),
            "mean_avg_perf": round(df_runs["avg_performance"].mean(), 4),
        }
        all_summary_rows.append(summary)
        print(f"  safe={summary['pct_no_violation']}%  minor={summary['pct_minor']}%  "
              f"major={summary['pct_major']}%  final_perf={summary['mean_final_perf']}")

    # Save master summary table
    df_summary = pd.DataFrame(all_summary_rows)
    df_summary.to_csv(f"results/bachelor_tests/SUMMARY_{timestamp}.csv", index=False)
    print(f"\nDone. Summary saved to results/bachelor_tests/SUMMARY_{timestamp}.csv")