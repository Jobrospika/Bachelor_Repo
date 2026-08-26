import math
import os
import itertools
import gpytorch
import pandas as pd
import torch
from matplotlib import pyplot as plt
from datetime import datetime
from PIL import Image
import glob
import shutil

from src.experiments.generic_grid_experiment import (
    set_up_algorithm,
    observe_data
)
from src.gps.models import (
    ExactGPSEModel,
    ExactGPMatern52Model
)
from src.utils.experiment_utils import seed_everything, generate_random_safe_initial_points
from src.utils.experiment_utils import read_config

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


def run_2d_experiment(
    seed=0,
    function_config_path="config/function_config/rosenbrock_2D.yaml",
    model_type="se",
    E_factor=1.0,
    lipschitz_factor=1.1,
    iterations= round(20+40*math.sqrt(2)),
    points_per_axis=50,
    plot=False,
    run_dir=None,
):
    seed_everything(seed)

    function_info = read_config(function_config_path)

    E = E_factor * function_info["noise_lvl"]

    init_x = generate_random_safe_initial_points(
        lambda x: observe_data(x, function_info, noise_on=False),
        function_info["domain_bounds"],
        safety_threshold=function_info["safety_threshold"] + 0.2,
        num_points=1,
    )
    init_y = observe_data(init_x, function_info, noise_on=True).squeeze(-1)

    print(f"init_x shape: {init_x.shape}, dtype: {init_x.dtype}")
    print(f"init_y shape: {init_y.shape}, dtype: {init_y.dtype}")
    print(f"init_x value: {init_x}")
    print(f"init_y value: {init_y}")


    lengthscale = 1.0 / (function_info["lipschitz_constant"] * lipschitz_factor)

    model = set_up_model_extended(init_x, init_y, model_type=model_type, lengthscale=lengthscale)

    print(f"Model train_x shape: {model.train_inputs[0].shape}")
    print(f"Model train_y shape: {model.train_targets.shape}")

    config = dict()
    config["bounds"] = function_info["domain_bounds"]
    config["safety_threshold"] = function_info["safety_threshold"]
    config["lipschitz_constant"] = function_info["lipschitz_constant"] * lipschitz_factor
    config["E"] = E
    config["beta"] = 2
    config["points_per_axis"] = points_per_axis
    config["seed_set"] = init_x
    config["beta_dict"] = {"B": 10, "R": 0.01, "delta": 0.01, "lamb": 0.1}
    config["iterations"] = iterations

    opt = set_up_algorithm("losbo", config, model)

    best_x = opt.get_current_best_mean_x()
    print(f"best_x shape: {best_x.shape}, value: {best_x}")
    print(f"best_x.unsqueeze(0) shape: {best_x.unsqueeze(0).shape}")

    obs = observe_data(opt.get_current_best_mean_x().unsqueeze(0), function_info, noise_on=False)
    print(f"observe_data output shape: {obs.shape}, value: {obs}")

    dict_init = {
        "iteration": 0,
        "x": init_x[0, 0].item(),
        "y": init_y[0].item(),
        "pred_opt": observe_data(
            opt.get_current_best_mean_x().unsqueeze(0), function_info, noise_on=False
        ).item()
    }
    df = pd.DataFrame(dict_init, index=[0])

    plot_dir = f"{run_dir}/_tmp_frames" if plot else None



    import traceback
    try:
        safety_violation, no_opt_possible, df = run_loop_2d(
            config["iterations"], opt, function_info, df, plot=plot, plot_dir=plot_dir
        )
    except Exception:
        traceback.print_exc()
        raise

    if plot:
        fn_name = os.path.splitext(os.path.basename(function_config_path))[0]
        config_tag = f"E{E}_L{lipschitz_factor}_{model_type}"
        gif_path = f"{run_dir}/{fn_name}_{config_tag}_seed{seed}.gif"
        make_gif(plot_dir, gif_path)

    final_performance = df["pred_opt"].iloc[-1]
    avg_performance = df["pred_opt"].mean()

    result = {
        "safety_violation": safety_violation,
        "no_opt_possible": no_opt_possible,
        "final_performance": final_performance,
        "avg_performance": avg_performance,
        "E_resolved": E,
    }

    return result, df



E_FACTORS = {
    #"super_small": 0.5,
    "baseline": 1.0,
    #"super_large": 1.5,
}

L_FACTORS = {
    #"super_small": 0.1,
    "baseline": 1.1,
    #"super_large": 4.0,
}

KERNELS = [
    "se",
    "matern52"
]

FUNCTIONS_2D = [
    ("branin_2D",    "config/function_config/braninmod_2D.yaml"),
    ("camel_2D",     "config/function_config/camel_2D.yaml"),
    #("rosenbrock_2D","config/function_config/rosenbrock_2D.yaml"),
    #("sphere_2D",    "config/function_config/sphere_2D.yaml"),
]

N_SEEDS = 1

def classify_violation(v):
    if v == 0:
        return "none"
    elif v <= 2:
        return "minor"
    else:
        return "major"

def run_loop_2d(iterations, opt, function_info, df, plot=False, plot_dir=None):
    """Patched version of run_loop for 2D inputs."""
    safety_violation = 0
    no_opt_possible = False

    if plot:
        os.makedirs(plot_dir, exist_ok=True)

    for t in range(iterations):
        try:
            x_next = opt.optimize()
        except:
            no_opt_possible = True
            break

        y_next = observe_data(x_next, function_info).squeeze(-1)

        if y_next < function_info["safety_threshold"]:
            safety_violation += 1

        opt.add_data_to_gp(x_next, y_next)
        opt.update_gp()

        pred_opt = observe_data(
            opt.get_current_best_mean_x().unsqueeze(0), function_info, noise_on=False
        ).squeeze()

        dict_step = {
            "iteration": t + 1,
            "x": x_next.squeeze().tolist(),  # list of [x1, x2] instead of .item()
            "y": y_next.squeeze().item(),
            "pred_opt": pred_opt.item(),
        }
        df = pd.concat([df, pd.DataFrame([dict_step])], ignore_index=True)

        if plot:
            bounds = function_info["domain_bounds"]
            x1 = torch.linspace(bounds[0][0], bounds[0][1], 100)
            x2 = torch.linspace(bounds[1][0], bounds[1][1], 100)
            grid = torch.stack(torch.meshgrid(x1, x2, indexing='ij'), dim=-1).reshape(-1, 2)
            y_grid = observe_data(grid, function_info, noise_on=False).reshape(100, 100)

            fig, ax = plt.subplots(figsize=(6, 5))
            contour = ax.contourf(x1.numpy(), x2.numpy(), y_grid.numpy().T, levels=50, cmap='RdYlGn')
            ax.contour(x1.numpy(), x2.numpy(), y_grid.numpy().T,
                      levels=[function_info["safety_threshold"]], colors='red', linewidths=2)
            plt.colorbar(contour, ax=ax)

            X = opt.X.detach().numpy()
            ax.scatter(X[:, 0], X[:, 1], c='blue', s=20, zorder=5, label='Observed')
            ax.scatter(x_next[0, 0].item(), x_next[0, 1].item(),
                      c='yellow', s=60, zorder=6, marker='*', label='Latest')
            ax.legend()
            ax.set_title(f"Iteration {t+1}")
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            plt.tight_layout()
            plt.savefig(f"{plot_dir}/frame_{t+1:03d}.png", dpi=100)
            plt.close()

    return safety_violation, no_opt_possible, df

def make_gif(plot_dir, gif_path, duration=500):
    frame_files = sorted(glob.glob(f"{plot_dir}/frame_*.png"))
    if not frame_files:
        return
    frames = [Image.open(f) for f in frame_files]
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
    )
    shutil.rmtree(plot_dir)

def analyze_safe_region(fn_label, fn_path):
    function_info = read_config(fn_path)
    bounds = function_info["domain_bounds"]
    threshold = function_info["safety_threshold"]

    axes = [torch.linspace(b[0], b[1], 100) for b in bounds]
    grid = torch.stack(torch.meshgrid(*axes, indexing='ij'), dim=-1).reshape(-1, len(bounds))
    y = observe_data(grid, function_info, noise_on=False)

    safe_mask = y.squeeze() > threshold
    print(f"\n[{fn_label}]")
    print(f"  Safe fraction:   {safe_mask.float().mean():.2%}")
    print(f"  Max value:       {y.max():.4f}")
    print(f"  Min value:       {y.min():.4f}")
    print(f"  Safety threshold:{threshold}")
    print(f"  Lipschitz const: {function_info['lipschitz_constant']}")


def plot_safe_region_2d(fn_label, fn_path, opt=None):
    function_info = read_config(fn_path)
    bounds = function_info["domain_bounds"]

    x1 = torch.linspace(bounds[0][0], bounds[0][1], 100)
    x2 = torch.linspace(bounds[1][0], bounds[1][1], 100)
    grid = torch.stack(torch.meshgrid(x1, x2, indexing='ij'), dim=-1).reshape(-1, 2)
    y = observe_data(grid, function_info, noise_on=False).reshape(100, 100)

    fig, ax = plt.subplots(figsize=(6, 5))
    contour = ax.contourf(x1.numpy(), x2.numpy(), y.numpy().T, levels=50, cmap='RdYlGn')
    ax.contour(x1.numpy(), x2.numpy(), y.numpy().T,
               levels=[function_info["safety_threshold"]], colors='red', linewidths=2)
    plt.colorbar(contour, ax=ax)

    # plot observed points if optimizer is passed
    if opt is not None:
        X = opt.gp.train_inputs[0].detach().numpy()
        ax.scatter(X[:, 0], X[:, 1], c='blue', s=20, zorder=5, label='Observed')
        ax.legend()

    ax.set_title(fn_label)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    plt.tight_layout()
    plt.savefig(f"results/bachelor_tests_2d/plot_{fn_label}.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    print("=== Safe Region Analysis ===")
    for fn_label, fn_path in FUNCTIONS_2D:
        analyze_safe_region(fn_label, fn_path)
    print("============================\n")


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"results/bachelor_tests_2d/run_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)


    grid = list(itertools.product(
        E_FACTORS.items(),
        L_FACTORS.items(),
        KERNELS,
        FUNCTIONS_2D,
    ))

    """###only baselines###
    grid = [
        (e, l, kernel, fn)
        for e, l, kernel, fn in grid
        if e[0] == "baseline" and l[0] == "baseline"
    ]
    ####################"""

    print(f"Total configurations: {len(grid)}")
    print(f"Runs per config (seeds): {N_SEEDS}")
    print(f"Total runs: {len(grid) * N_SEEDS}\n")

    all_summary_rows = []
    all_raw_rows = []

    for i, ((e_label, E_factor), (l_label, L), kernel, (fn_label, fn_path)) in enumerate(grid):

        config_label = f"E={e_label}_L={l_label}_kernel={kernel}_fn={fn_label}"
        print(f"[{i+1}/{len(grid)}] {config_label}")

        run_results = []

        for seed in range(N_SEEDS):
            try:
                result, _ = run_2d_experiment(
                    seed=seed,
                    function_config_path=fn_path,
                    model_type=kernel,
                    E_factor=E_factor,
                    lipschitz_factor=L,
                    run_dir=run_dir
                )
                result["seed"] = seed
                result["violation_class"] = classify_violation(result["safety_violation"])
                result["E_label"] = e_label
                result["E_factor"] = E_factor
                result["L_label"] = l_label
                result["L_factor"] = L
                result["kernel"] = kernel
                result["function"] = fn_label
                run_results.append(result)
                all_raw_rows.append(result)
            except Exception as ex:
                print(f"  !! Error seed={seed}: {ex}")

        if not run_results:
            continue

        df_runs = pd.DataFrame(run_results)

        total = len(df_runs)
        summary = {
            "E_label":        e_label,
            "E_factor":        E_factor,
            "L_label":        l_label,
            "L_factor":       L,
            "kernel":         kernel,
            "function":       fn_label,
            "n_runs":         total,
            "pct_no_violation": round(100 * (df_runs["violation_class"] == "none").sum() / total, 1),
            "pct_minor":      round(100 * (df_runs["violation_class"] == "minor").sum() / total, 1),
            "pct_major":      round(100 * (df_runs["violation_class"] == "major").sum() / total, 1),
            "pct_no_opt":     round(100 * df_runs["no_opt_possible"].sum() / total, 1),
            "mean_final_perf":round(df_runs["final_performance"].mean(), 4),
            "mean_avg_perf":  round(df_runs["avg_performance"].mean(), 4),
        }
        all_summary_rows.append(summary)
        print(f"  safe={summary['pct_no_violation']}%  minor={summary['pct_minor']}%  "
              f"major={summary['pct_major']}%  no_opt={summary['pct_no_opt']}%  "
              f"final_perf={summary['mean_final_perf']}")

    df_summary = pd.DataFrame(all_summary_rows)
    df_summary.to_csv(f"{run_dir}/SUMMARY_2D_{timestamp}.csv", index=False)

    df_raw = pd.DataFrame(all_raw_rows)
    df_raw.to_csv(f"{run_dir}/RAW_2D_{timestamp}.csv", index=False)
    print(f"\nDone. Summary saved to {run_dir}")