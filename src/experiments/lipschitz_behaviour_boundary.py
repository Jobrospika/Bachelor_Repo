import torch
from src.utils.experiment_utils import read_config
from src.experiments.testfunctions import observe_data

def boundary_vs_global_L(function_config_path, band=0.02, n_samples=2_000_000, seed=0):
    """
    Compares the global Lipschitz constant against the true local gradient
    magnitude specifically near the safety boundary (Y close to threshold),
    rather than across the whole domain.
    """
    torch.manual_seed(seed)
    function_info = read_config(function_config_path)
    bounds = function_info["domain_bounds"]
    threshold = function_info["safety_threshold"]
    L_global = function_info["lipschitz_constant"]

    lows = torch.tensor([b[0] for b in bounds])
    highs = torch.tensor([b[1] for b in bounds])
    X = lows + torch.rand(n_samples, len(bounds)) * (highs - lows)
    X.requires_grad_(True)

    Y = observe_data(X, function_info, noise_on=False).squeeze(-1)
    grad = torch.autograd.grad(Y.sum(), X)[0]
    grad_norm = grad.norm(dim=-1)

    near_boundary = (Y - threshold).abs() < band
    if near_boundary.sum() == 0:
        print("No samples landed near the boundary — widen `band`.")
        return

    local_max = grad_norm[near_boundary].max().item()
    local_mean = grad_norm[near_boundary].mean().item()

    print(f"{function_info.get('type', function_config_path)}: "
          f"L_global={L_global:.4f}, "
          f"local grad near boundary (mean={local_mean:.4f}, max={local_max:.4f}), "
          f"conservatism ratio (global/local_max)={L_global/local_max:.1f}x")

for path in [
    "config/function_config/rosenbrock_2D.yaml",
    "config/function_config/hartmann_6D.yaml",
    "config/function_config/griewank_6d.yaml",
]:
    boundary_vs_global_L(path)