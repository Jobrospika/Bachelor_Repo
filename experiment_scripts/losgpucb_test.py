import subprocess

# ===== Configuration =====
PYTHON_EXE = r"C:\Users\josch\miniconda3\envs\LoSBO\python.exe"
EXPERIMENT_SCRIPT = "experiment_scripts/losgpucb_experiment.py"

CONFIGS = [
    #"config/experiment_se_rosenbrock_2D.yaml",
    #"config/experiment_se_hartmann_6D.yaml",
    #"config/experiment_se_griewank_6D.yaml",
    #"config/experiment_se_gaussian_10D.yaml",
    "config/experiment_matern_rosenbrock_2D.yaml",
    "config/experiment_matern_hartmann_6D.yaml",
    "config/experiment_matern_griewank_6D.yaml",
    "config/experiment_matern_gaussian_10D.yaml",
]

SEEDS = range(30)

for config in CONFIGS:
    for seed in SEEDS:
        print(f"\n=== Running {config} (seed={seed}) ===")
        result = subprocess.run(
            [PYTHON_EXE, EXPERIMENT_SCRIPT, "--config_path", config, "--seed", str(seed)],
            check=False,
        )
        if result.returncode != 0:
            print(f"!! {config} seed={seed} exited with code {result.returncode}")