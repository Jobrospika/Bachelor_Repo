import glob
import os

import pandas as pd
import torch

from src.experiments.generic_grid_experiment import observe_data
from src.utils.experiment_utils import read_config


class SafeRegionAnalyzer:

    def __init__(self, config_dir, n_samples=500000, seed=0):
        self.config_dir = config_dir
        self.n_samples = n_samples
        self.seed = seed
        self.configs = {}  # label -> path

    def discover(self, pattern="*.yaml"):
        """
        Finds all config files matching `pattern` in config_dir. Label is the
        filename without extension (e.g. "rosenbrock_4D.yaml" -> "rosenbrock_4D").
        Re-running discover() refreshes self.configs from disk, so newly added
        yaml files are picked up automatically without editing this class.
        """
        self.configs = {}
        for path in sorted(glob.glob(os.path.join(self.config_dir, pattern))):
            label = os.path.splitext(os.path.basename(path))[0]
            self.configs[label] = path
        return self.configs

    def exclude(self, labels):
        for label in labels:
            self.configs.pop(label, None)
        return self

    def analyze_one(self, label, path=None):
        """Monte Carlo safe-region stats for a single config."""
        if path is None:
            path = self.configs[label]

        torch.manual_seed(self.seed)
        function_info = read_config(path)
        bounds = function_info["domain_bounds"]
        threshold = function_info["safety_threshold"]
        dim = function_info["domain_size"]

        lows = torch.tensor([b[0] for b in bounds])
        highs = torch.tensor([b[1] for b in bounds])
        X = lows + torch.rand(self.n_samples, dim) * (highs - lows)
        y = observe_data(X, function_info, noise_on=False)

        safe_mask = y.squeeze() > threshold
        return {
            "function": label,
            "type": function_info.get("type", label),
            "dim": dim,
            "safe_fraction": safe_mask.float().mean().item(),
            "max_value": y.max().item(),
            "min_value": y.min().item(),
            "safety_threshold": threshold,
            "lipschitz_constant": function_info["lipschitz_constant"],
            "optimum": function_info["optimum"],
            "n_samples": self.n_samples,
        }

    def analyze_all(self, labels=None):
        """
        Runs analyze_one over either all discovered configs (default) or a
        specified subset of labels. Calls discover() first if it hasn't been
        called yet.
        """
        if not self.configs:
            self.discover()

        targets = labels if labels is not None else list(self.configs.keys())
        rows = [self.analyze_one(label) for label in targets]
        return pd.DataFrame(rows)

    def save(self, df, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        return path

    def report(self, df=None, labels=None):
        """Analyze (if df not already computed) and print a readable summary."""
        if df is None:
            df = self.analyze_all(labels=labels)
        print("\n=== Safe Region Analysis (Monte Carlo) ===")
        print(df.to_string(index=False))
        print("===========================================\n")
        return df


if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))  # -> LoSBO/
    CONFIG_DIR = os.path.join(REPO_ROOT, "config/function_config")

    analyzer = SafeRegionAnalyzer(CONFIG_DIR)
    analyzer.discover()
    print(f"Discovered {len(analyzer.configs)} config files: {list(analyzer.configs.keys())}")
    analyzer.exclude(["base_onb_rkhs_se","base_pre_rkhs_matern32", "base_pre_rkhs_se", "pre_rkhs_function", "example_config"])
    print(f"Curated config files: {list(analyzer.configs.keys())}")

    df = analyzer.report()
    out_path = os.path.join(REPO_ROOT, "results/safety_region_report.csv")
    analyzer.save(df, out_path)
    print(f"Saved to {out_path}")