import torch
from src.utils.experiment_utils import read_config
from src.experiments.testfunctions import observe_data

def compute_scale_stats(function_config_path, n_samples=500_000, seed=0):
    torch.manual_seed(seed)
    function_info = read_config(function_config_path)
    bounds = function_info["domain_bounds"]
    threshold = function_info["safety_threshold"]

    lows = torch.tensor([b[0] for b in bounds])
    highs = torch.tensor([b[1] for b in bounds])
    X = lows + torch.rand(n_samples, len(bounds)) * (highs - lows)

    Y = observe_data(X, function_info, noise_on=False).squeeze(-1)

    full_std = Y.std().item()
    safe_mask = Y >= threshold
    safe_std = Y[safe_mask].std().item() if safe_mask.any() else float("nan")
    safe_frac = safe_mask.float().mean().item()

    return {
        "function": function_info.get("type", function_config_path),
        "full_std": full_std,
        "safe_region_std": safe_std,
        "safe_fraction": safe_frac,
        "n_safe_samples": int(safe_mask.sum()),
    }

for path in [
    "config/function_config/rosenbrock_2D.yaml",
    "config/function_config/hartmann_6D.yaml",
    "config/function_config/gaussian_10D.yaml",
    "config/function_config/griewank_6d.yaml",
]:
    print(compute_scale_stats(path))