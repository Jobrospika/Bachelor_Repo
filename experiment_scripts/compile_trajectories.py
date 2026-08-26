#!/usr/bin/env python3
"""
Compile per-run LoSBOAdaptive trajectory CSVs into

  1. <function>_compiled.csv          long format, one row per (run, iteration)
  2. <function>_iteration_summary.csv per (config, iteration) aggregate, for convergence plots
  3. <function>_run_metrics.csv       per (run) convergence metrics, for tables

Expects per-run files named like
  griewank_6D_E0.0024075_L0.5_matern52_stdtight_seed7.csv
with at least the columns  iteration, pred_opt  (x and y are carried through if present).

Usage
  python compile_trajectories.py --traj-dir results/trajectories --out-dir results/compiled
  python compile_trajectories.py --traj-dir results/trajectories --thresholds 0.8 0.9 0.95
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# griewank_6D_E0.0024075_L0.5_matern52_stdtight_seed7.csv
FNAME_RE = re.compile(
    r"^(?P<function>.+?)_(?P<dim>\d+)D"
    r"_E(?P<epsilon>[0-9.eE+-]+)"
    r"_L(?P<lipschitz>[0-9.eE+-]+)"
    r"_(?P<kernel>[A-Za-z0-9]+)"
    r"_std(?P<noise_std>[A-Za-z]+)"
    r"_seed(?P<seed>\d+)\.csv$"
)

# lipschitz_factor -> label used in the summary CSVs
L_LABELS = {0.5: "underestimated", 1.0: "exact", 1.5: "low_safety_net", 2.0: "high_safety_net"}
# rank of the epsilon value within each function -> label
E_LABELS = {0: "small", 1: "exact", 2: "large"}

CONFIG_COLS = ["function", "dim", "epsilon", "lipschitz", "kernel", "noise_std"]


def parse_name(path):
    m = FNAME_RE.match(path.name)
    if m is None:
        return None
    d = m.groupdict()
    return {
        "function": d["function"],
        "dim": int(d["dim"]),
        "epsilon": float(d["epsilon"]),
        "lipschitz": float(d["lipschitz"]),
        "kernel": d["kernel"],
        "noise_std": d["noise_std"],
        "seed": int(d["seed"]),
        "source_file": path.name,
    }


def load_runs(traj_dir, dim_suffix):
    files = sorted(Path(traj_dir).rglob("*.csv"))
    if not files:
        sys.exit(f"no CSV files found under {traj_dir}")

    frames, skipped = [], []
    for p in files:
        meta = parse_name(p)
        if meta is None:
            skipped.append(p.name)
            continue
        df = pd.read_csv(p)
        if "pred_opt" not in df.columns:
            skipped.append(p.name + " (no pred_opt column)")
            continue
        if "iteration" not in df.columns:
            df.insert(0, "iteration", np.arange(len(df)))
        df = df.sort_values("iteration").reset_index(drop=True)
        if dim_suffix:
            meta["function"] = f'{meta["function"]}_{meta["dim"]}D'
        for k, v in meta.items():
            df[k] = v
        frames.append(df)

    if skipped:
        print(f"skipped {len(skipped)} file(s), first few {skipped[:5]}", file=sys.stderr)
    if not frames:
        sys.exit("no parseable trajectory files")

    out = pd.concat(frames, ignore_index=True)
    keep = [c for c in ["iteration", "x", "y", "pred_opt"] if c in out.columns]
    ordered = keep + CONFIG_COLS + ["seed", "source_file"]
    return out[ordered + [c for c in out.columns if c not in ordered]]


def add_labels(df):
    df = df.copy()
    df["L_label"] = df["lipschitz"].map(L_LABELS)
    if df["L_label"].isna().any():
        unknown = sorted(df.loc[df.L_label.isna(), "lipschitz"].unique())
        print(f"warning, unlabelled lipschitz factors {unknown}", file=sys.stderr)
    # epsilon labels are relative to each function's own three values
    mapping = {}
    for fn, grp in df.groupby("function"):
        vals = sorted(grp["epsilon"].unique())
        for i, v in enumerate(vals):
            mapping[(fn, v)] = E_LABELS.get(i, f"E{i}")
    df["E_label"] = [mapping[(f, e)] for f, e in zip(df["function"], df["epsilon"])]
    return df


def run_metrics(df, thresholds, optimum, stall_tol):
    """One row per run."""
    rows = []
    key = CONFIG_COLS + ["seed", "source_file", "L_label", "E_label"]
    for meta, g in df.groupby(key, sort=False):
        v = g["pred_opt"].to_numpy(dtype=float)
        best = np.maximum.accumulate(v)
        ratio = best / optimum
        rec = dict(zip(key, meta))
        rec.update(
            n_iterations=len(v),
            start=v[0],
            final=v[-1],
            best=best[-1],
            gain=best[-1] - v[0],
            final_ratio=v[-1] / optimum,
            best_ratio=best[-1] / optimum,
            anytime_mean_ratio=float(ratio.mean()),  # normalised area under the best-so-far curve
        )
        # first iteration where the best-so-far curve crosses each threshold
        for t in thresholds:
            hit = np.flatnonzero(ratio >= t)
            rec[f"iter_to_{t:g}"] = int(hit[0]) if hit.size else np.nan
            rec[f"reached_{t:g}"] = bool(hit.size)
        # last iteration that improved the best-so-far curve by more than stall_tol
        improved = np.flatnonzero(np.diff(best, prepend=-np.inf) > stall_tol)
        last_imp = int(improved[-1]) if improved.size else 0
        rec["last_improve_iter"] = last_imp
        rec["stall_length"] = len(v) - 1 - last_imp
        rec["stall_frac"] = rec["stall_length"] / max(len(v) - 1, 1)
        rows.append(rec)
    return pd.DataFrame(rows)


def iteration_summary(df, optimum):
    """One row per (config, iteration), averaged over seeds."""
    df = df.copy()
    df["best_so_far"] = df.groupby("source_file", sort=False)["pred_opt"].cummax()
    df["ratio"] = df["pred_opt"] / optimum
    df["best_ratio"] = df["best_so_far"] / optimum

    key = CONFIG_COLS + ["L_label", "E_label", "iteration"]
    agg = df.groupby(key, sort=True).agg(
        n_runs=("pred_opt", "size"),
        mean_pred_opt=("pred_opt", "mean"),
        mean_best_ratio=("best_ratio", "mean"),
        median_best_ratio=("best_ratio", "median"),
        std_best_ratio=("best_ratio", "std"),
        q25_best_ratio=("best_ratio", lambda s: s.quantile(0.25)),
        q75_best_ratio=("best_ratio", lambda s: s.quantile(0.75)),
        min_best_ratio=("best_ratio", "min"),
        max_best_ratio=("best_ratio", "max"),
    )
    agg["sem_best_ratio"] = agg["std_best_ratio"] / np.sqrt(agg["n_runs"])
    return agg.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-dir", required=True)
    ap.add_argument("--out-dir", default="compiled")
    ap.add_argument("--optimum", type=float, default=1.0,
                    help="true optimum used to normalise pred_opt into a ratio")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.8, 0.9, 0.95])
    ap.add_argument("--stall-tol", type=float, default=1e-6,
                    help="improvement below this does not count as progress")
    ap.add_argument("--dim-suffix", action="store_true",
                    help="store 'griewank_6D' rather than 'griewank' in the function column, "
                         "matching the SUMMARY_ND_ADAPTIVE files. Default is the bare name, "
                         "matching the existing hartmann/rosenbrock compiled files.")
    ap.add_argument("--function", nargs="+", default=None,
                    help="only process these functions, e.g. --function hartmann_6D. "
                         "Use this to run twice over a folder holding several functions "
                         "when they need different --thresholds.")
    ap.add_argument("--min-iterations", type=int, default=0,
                    help="drop runs shorter than this many iterations (truncated/no-opt runs) "
                         "so they do not create survivorship jumps in the iteration summary")
    ap.add_argument("--no-compiled", action="store_true",
                    help="skip writing the large long-format file")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_runs(args.traj_dir, args.dim_suffix)

    if args.function:
        norm = lambda n: re.sub(r"_\d+D$", "", n)
        wanted = {norm(n) for n in args.function}
        found = {norm(n) for n in raw["function"].unique()}
        missing = wanted - found
        if missing:
            print(f"warning, requested function(s) not present {sorted(missing)}, "
                  f"folder holds {sorted(found)}", file=sys.stderr)
        raw = raw[raw["function"].map(norm).isin(wanted)]
        if raw.empty:
            sys.exit("no runs left after --function filter")

    if args.min_iterations > 0:
        lens = raw.groupby("source_file")["iteration"].transform("size")
        dropped = raw.loc[lens < args.min_iterations, "source_file"].nunique()
        if dropped:
            print(f"dropped {dropped} run(s) shorter than {args.min_iterations} iterations")
        raw = raw[lens >= args.min_iterations]

    raw = add_labels(raw)

    for fn, df in raw.groupby("function"):
        n_runs = df["source_file"].nunique()
        n_cfg = df.groupby(CONFIG_COLS, sort=False).ngroups
        lens = df.groupby("source_file")["iteration"].size()
        print(f"{fn}  {n_runs} runs, {n_cfg} configs, "
              f"iterations {lens.min()}-{lens.max()}, {len(df)} rows")
        if n_runs != n_cfg * df["seed"].nunique():
            print(f"  warning, grid looks incomplete for {fn}", file=sys.stderr)

        if not args.no_compiled:
            cols = [c for c in ["iteration", "x", "y", "pred_opt"] if c in df.columns]
            cols += CONFIG_COLS + ["seed", "source_file"]
            df[cols].to_csv(out_dir / f"{fn}_compiled.csv", index=False)

        iteration_summary(df, args.optimum).to_csv(
            out_dir / f"{fn}_iteration_summary.csv", index=False)
        run_metrics(df, args.thresholds, args.optimum, args.stall_tol).to_csv(
            out_dir / f"{fn}_run_metrics.csv", index=False)

    print(f"wrote outputs to {out_dir.resolve()}")


if __name__ == "__main__":
    main()