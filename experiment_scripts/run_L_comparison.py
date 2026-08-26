"""
run_L_comparison.py

Runs LosboAdaptive across every benchmark function under 4 different
Lipschitz-constant settings, to see how much the tightened L estimates from
lipschitz_estimator.py actually change safety/performance outcomes compared
to the L values already stored in the yaml configs.

The 4 settings, per function:
  1. old_L_factored : yaml's existing lipschitz_constant * 1.1,  E=0.05
  2. new_L_factored : estimator's computed L         * 1.1,  E=0.05
  3. old_L_raw      : yaml's existing lipschitz_constant (no factor), E=0.0
  4. new_L_raw      : estimator's computed L          (no factor), E=0.0

NOT executed/tested end-to-end here - this environment doesn't have gpytorch,
your GP model classes, or LosboAdaptive installed, so this has only been
checked by careful reading against the scripts you provided, not by running
it. Strongly recommend a smoke test first: set N_SEEDS=1 and ITERATIONS=3
below and confirm one function runs cleanly before doing the full sweep.

This script does NOT modify your original yaml configs. For the "new_L"
settings, it writes a temporary copy of the function's yaml (with
lipschitz_constant overridden) into <run_dir>/temp_configs/ and points
run_adaptive_experiment at that copy instead. These temp copies are left on
disk (not deleted) for reproducibility/debugging.
"""

import os
from datetime import datetime

import pandas as pd
import yaml

from experiment_scripts.custom_experiments_ND import run_adaptive_experiment
from src.utils.experiment_utils import read_config
from src.experiments.lipschitz_estimator import dispatch_type, is_rkhs_type

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# --------------------------------------------------------------------------- #
# HARD-CODED CONFIG - edit directly, no CLI args (matching lipschitz_estimator.py)
# --------------------------------------------------------------------------- #

ALL_FUNCTIONS = [
    ("forrester_1D", "config/function_config/forrester_1D.yaml"),
    ("braninmod_2D",  "config/function_config/braninmod_2D.yaml"),
    ("camel_2D",       "config/function_config/camel_2D.yaml"),
    ("rosenbrock_2D",  "config/function_config/rosenbrock_2D.yaml"),
    ("sphere_2D",      "config/function_config/sphere_2D.yaml"),
    ("hartmann_3D", "config/function_config/hartmann_3D.yaml"),
    ("rosenbrock_4D",  "config/function_config/rosenbrock_4D.yaml"),
    ("styblinski_tang_4D", "config/function_config/styblinski_tang_4D.yaml"),
    ("sphere_4D",      "config/function_config/sphere_4D.yaml"),
    ("sum_squares_5D", "config/function_config/sum_squares_5D.yaml"),
    ("sphere_6D",      "config/function_config/sphere_6D.yaml"),
    ("hartmann_6D",    "config/function_config/hartmann_6D.yaml"),
    ("dixon_price_7D", "config/function_config/dixon_price_7D.yaml"),
    ("cosine_8D", "config/function_config/cosine8.yaml"),
    ("rastrigin_9D", "config/function_config/rastrigin_9D.yaml"),
    ("trid_10D", "config/function_config/trid_10D.yaml"),
    ("schwefel_10D", "config/function_config/schwefel_10D.yaml"),
    ("camel_10D",      "config/function_config/camel_10D.yaml"),
    ("gaussian_10D",   "config/function_config/gaussian_10D.yaml"),
]

KERNEL = "matern52"
LOCAL_STD_LABEL = "medium"
LOCAL_STD_VALUE = 0.05

N_SEEDS = 5
ITERATIONS = 10

#(setting_label, use_new_L: bool, lipschitz_factor, E)
SETTINGS = [
    ("old_L_factored", False, 1.1, 0.05),
    ("new_L_factored", True,  1.1, 0.05),
    ("old_L_raw",      False, 1.0, 0.0),
    ("new_L_raw",      True,  1.0, 0.0),
]


def classify_violation(v):
    if v == 0:
        return "none"
    elif v <= 2:
        return "minor"
    else:
        return "major"


def compute_new_L(function_info):
    """
    Uses the same closed-form/interval-arithmetic calculators as
    lipschitz_estimator.py - imported directly, not reimplemented, so a fix
    there automatically applies here too.
    """
    type_str = function_info["type"]
    bounds = [tuple(b) for b in function_info["domain_bounds"]]
    fn = dispatch_type(type_str)
    if fn is None:
        raise ValueError(f"No calculator registered for type={type_str!r}")
    L, method = fn(bounds)
    return L, method


def write_temp_config(original_path, new_L, temp_dir, fn_label):
    """
    Copies the function's yaml with lipschitz_constant overridden to new_L.
    Leaves the file on disk for reproducibility - not cleaned up.
    """
    with open(original_path) as f:
        raw = yaml.safe_load(f)
    raw["lipschitz_constant"] = float(new_L)
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"{fn_label}_newL.yaml")
    with open(temp_path, "w") as f:
        yaml.safe_dump(raw, f)
    return temp_path


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(REPO_ROOT, f"results/L_comparison/run_{timestamp}")
    trajectory_dir = os.path.join(run_dir, "trajectories")
    temp_config_dir = os.path.join(run_dir, "temp_configs")
    os.makedirs(run_dir, exist_ok=True)

    all_rows = []

    for fn_label, rel_path in ALL_FUNCTIONS:
        fn_path = os.path.join(REPO_ROOT, rel_path)
        if not os.path.exists(fn_path):
            print(f"[skip: file not found] {fn_label} ({fn_path})")
            continue

        function_info = read_config(fn_path)
        type_str = function_info.get("type", "")

        if is_rkhs_type(type_str):
            print(f"[skip: RKHS type] {fn_label} - no closed-form L available, "
                  f"not comparable here")
            continue

        old_L = function_info["lipschitz_constant"]
        try:
            new_L, method = compute_new_L(function_info)
        except Exception as e:
            print(f"[skip: L computation failed] {fn_label}: {e}")
            continue

        print(f"\n=== {fn_label} ===  old_L={old_L:.6f}  new_L={new_L:.6f} ({method})")

        new_L_config_path = write_temp_config(fn_path, new_L, temp_config_dir, fn_label)

        for setting_label, use_new_L, lipschitz_factor, E in SETTINGS:
            config_path = new_L_config_path if use_new_L else fn_path
            L_used = new_L if use_new_L else old_L
            print(f"  [{setting_label}]  L_base={L_used:.6f}  factor={lipschitz_factor}  E={E}")

            for seed in range(N_SEEDS):
                try:
                    result = run_adaptive_experiment(
                        function_config_path=config_path,
                        function_label=fn_label,
                        E=E,
                        lipschitz_factor=lipschitz_factor,
                        kernel=KERNEL,
                        local_std=LOCAL_STD_VALUE,
                        local_std_label=LOCAL_STD_LABEL,
                        seed=seed,
                        iterations=ITERATIONS,
                        save_trajectory_dir=trajectory_dir,
                    )
                    result["setting"] = setting_label
                    result["L_base"] = L_used
                    result["L_source"] = "new" if use_new_L else "old"
                    result["violation_class"] = classify_violation(result["safety_violations"])
                    all_rows.append(result)
                except Exception as ex:
                    print(f"    !! seed={seed} failed: {ex}")

    if not all_rows:
        print("\nNo runs completed - nothing to summarize.")
        return

    df = pd.DataFrame(all_rows)
    raw_path = os.path.join(run_dir, f"RAW_RUNS_{timestamp}.csv")
    df.to_csv(raw_path, index=False)
    print(f"\nSaved {len(df)} raw run rows to {raw_path}")

    agg = df.groupby(["function", "setting"]).agg(
        L_base=("L_base", "first"),
        n_runs=("ratio", "count"),
        mean_ratio=("ratio", "mean"),
        std_ratio=("ratio", "std"),
        pct_no_violation=("violation_class", lambda s: round(100 * (s == "none").sum() / len(s), 1)),
        mean_safety_violations=("safety_violations", "mean"),
        pct_no_opt=("no_opt_possible", lambda s: round(100 * s.sum() / len(s), 1)),
        mean_final_pred_opt=("final_pred_opt", "mean"),
        mean_n_observations=("n_observations", "mean"),
    ).reset_index()

    summary_path = os.path.join(run_dir, f"SUMMARY_L_COMPARISON_{timestamp}.csv")
    agg.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()