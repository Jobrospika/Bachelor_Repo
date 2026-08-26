"""
resume_sweep.py

Diffs an already-compiled results CSV against the full intended sweep grid
(E x L x kernel x local_std x seed) and reruns ONLY the missing / partial
configs. Safe to run multiple times -- if nothing is missing, it does nothing.

Adjust the imports below to match your actual module paths.
"""

import math
import os
from itertools import product

import pandas as pd

# --- adjust these imports to your actual codebase -------------------------
from src.utils.experiment_utils import read_config
from experiment_scripts.custom_experiments_ND import run_adaptive_experiment
from experiment_scripts.run_L_comparison import classify_violation
# ---------------------------------------------------------------------------


# ---- grid definition: keep this in sync with your main sweep script -------
E_multiplier = {
    "small": 0.5,
    "exact": 1.0,
    "large": 2.0,
}

L_FACTORS = {
    "underestimated": 0.5,
    "exact": 1.0,
    "low_safety_net": 1.5,
    "high_safety_net": 2.0,
}

LOCAL_STD_VALUES = {
    "tight": 0.02,
    "medium": 0.05,
    "wide": 0.15,
}

KERNELS = [
    "se",
    "matern52"
]

N_SEEDS = 30
FLOAT_TOL = 1e-6  # tolerance for matching epsilon/lipschitz floats from the compiled csv
# ---------------------------------------------------------------------------


def iterations_for_dim(d: int) -> int:
    return int(round(20 + 40 * math.sqrt(d)))


def build_full_grid(noise_lvl: float):
    """All (E_label, E_value, L_label, L_value, kernel, std_label, std_value) combos."""
    grid = []
    for e_label, e_mult in E_multiplier.items():
        e_val = e_mult * noise_lvl
        for l_label, l_val in L_FACTORS.items():
            for kernel in KERNELS:
                for std_label, std_val in LOCAL_STD_VALUES.items():
                    grid.append((e_label, e_val, l_label, l_val, kernel, std_label, std_val))
    return grid


def load_completed_seeds(compiled_csv_path: str):
    """
    Returns: dict[(round(epsilon), round(lipschitz), kernel, noise_std)] -> set of seeds present.
    Empty dict if the file doesn't exist yet (nothing done so far).
    """
    if not os.path.exists(compiled_csv_path):
        return {}

    df = pd.read_csv(compiled_csv_path)
    present = df[["epsilon", "lipschitz", "kernel", "noise_std", "seed"]].drop_duplicates()

    completed = {}
    for (eps, lip, kernel, std), sub in present.groupby(["epsilon", "lipschitz", "kernel", "noise_std"]):
        key = (round(eps, 6), round(lip, 6), kernel, std)
        completed[key] = set(sub["seed"].tolist())
    return completed


def find_matching_key(completed: dict, e_val: float, l_val: float, kernel: str, std_label: str):
    """Match a grid cell to a key in `completed`, allowing for float rounding drift."""
    for (eps, lip, k, s) in completed.keys():
        if k == kernel and s == std_label and abs(eps - e_val) < FLOAT_TOL and abs(lip - l_val) < FLOAT_TOL:
            return (eps, lip, k, s)
    return None


def resume_function(function_config_path: str, function_label: str, compiled_csv_path: str,
                     trajectory_dir: str, dry_run: bool = False):
    function_info = read_config(function_config_path)
    noise_lvl = function_info["noise_lvl"]
    dim = function_info["domain_size"]
    iterations = iterations_for_dim(dim)

    full_grid = build_full_grid(noise_lvl)
    completed = load_completed_seeds(compiled_csv_path)

    todo = []
    for (e_label, e_val, l_label, l_val, kernel, std_label, std_val) in full_grid:
        key = find_matching_key(completed, e_val, l_val, kernel, std_label)
        done_seeds = completed.get(key, set())
        missing_seeds = [s for s in range(N_SEEDS) if s not in done_seeds]
        if missing_seeds:
            todo.append((e_label, e_val, l_label, l_val, kernel, std_label, std_val, missing_seeds))

    total_missing_runs = sum(len(t[-1]) for t in todo)
    print(f"[{function_label}] {len(todo)} configs with missing seeds, "
          f"{total_missing_runs} total runs to do (of {len(full_grid) * N_SEEDS} full grid)")

    if dry_run:
        for (e_label, e_val, l_label, l_val, kernel, std_label, std_val, missing_seeds) in todo:
            print(f"  E={e_label} L={l_label} kernel={kernel} std={std_label}: "
                  f"missing seeds {missing_seeds}")
        return

    all_rows = []
    for i, (e_label, e_val, l_label, l_val, kernel, std_label, std_val, missing_seeds) in enumerate(todo):
        config_label = f"E={e_label}_L={l_label}_kernel={kernel}_std={std_label}_fn={function_label}"
        print(f"[{i+1}/{len(todo)}] {config_label} -- {len(missing_seeds)} seeds")

        for seed in missing_seeds:
            try:
                result = run_adaptive_experiment(
                    function_config_path=function_config_path,
                    function_label=function_label,
                    E_multiplier=E_multiplier[e_label],
                    lipschitz_factor=l_val,
                    kernel=kernel,
                    local_std=std_val,
                    local_std_label=std_label,
                    seed=seed,
                    iterations=iterations,
                    save_trajectory_dir=trajectory_dir,
                )
                result["E_label"] = e_label
                result["L_label"] = l_label
                result["violation_class"] = classify_violation(result["safety_violations"])
                all_rows.append(result)
            except Exception as ex:
                print(f"  !! seed={seed} failed: {ex}")

    return all_rows


if __name__ == "__main__":
    FUNCTIONS = [
        ("config/function_config/rosenbrock_2D.yaml", "rosenbrock_2D",
         "results/bachelor_tests_nd_adaptive/run_20260811_232714/compiled/rosenbrock_compiled.csv", "results/bachelor_tests_nd_adaptive/run_20260811_232714/trajectories"),
        ("config/function_config/hartmann_6D.yaml", "hartmann_6D",
         "results/bachelor_tests_nd_adaptive/run_20260811_232714/compiled/hartmann_compiled.csv", "results/bachelor_tests_nd_adaptive/run_20260811_232714/trajectories"),
        #("config/function_config/griewank_6d.yaml", "griewank_6D",
         #"results/bachelor_tests_nd_adaptive/run_20260811_232714/compiled/griewank_compiled.csv", "results/trajectories"),
        #("config/function_config/gaussian_10D.yaml", "gaussian_10D",
         #"results/bachelor_tests_nd_adaptive/run_20260811_232714/compiled/gaussian_compiled.csv", "results/trajectories"),
    ]

    DRY_RUN = False

    for fn_config_path, fn_label, compiled_csv, traj_dir in FUNCTIONS:
        resume_function(fn_config_path, fn_label, compiled_csv, traj_dir, dry_run=DRY_RUN)