#!/usr/bin/env python3
"""
compile_trajectories.py

Recombines individual per-run trajectory CSVs (one file per run) into
one aggregated CSV per benchmark function type (e.g. rosenbrock, hartmann).

Expected input filename pattern (as produced by the experiment pipeline):

    <timestamp>_<function>_<dim>D_E<epsilon>_L<lipschitz>_<kernel>_std<noise>_seed<seed>.csv

    e.g. 1786517641507_hartmann_6D_E0_013887_L0_5_matern52_stdmedium_seed0.csv
         1786517655886_rosenbrock_2D_E0_0139335_L2_0_se_stdtight_seed29.csv

Note: in these filenames, decimal points inside the E and L values are
encoded as underscores (e.g. "E0_013887" -> epsilon=0.013887,
"L2_0" -> lipschitz=2.0). This script decodes that.

Each individual CSV is expected to have columns: iteration, x, y, pred_opt

Output: for every distinct function name found, writes
    <outdir>/<function>_compiled.csv
containing all rows from every matching run, with extra columns:
    source_file, function, dim, epsilon, lipschitz, kernel, noise_std, seed

Usage:
    python compile_trajectories.py --input-dir path/to/runs --output-dir path/to/compiled
    python compile_trajectories.py --input-dir path/to/runs --output-dir path/to/compiled --summary
"""

import argparse
import re
import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd

FNAME_RE = re.compile(
    r"^(?:(?P<timestamp>\d+)_)?"
    r"(?P<function>.+?)_"
    r"(?P<dim>\d+)D_"
    r"E(?P<epsilon>[\d_.]+)_"
    r"L(?P<lipschitz>[\d_.]+)_"
    r"(?P<kernel>[A-Za-z0-9]+)_"
    r"std(?P<noise>[A-Za-z0-9]+)_"
    r"seed(?P<seed>\d+)"
    r"\.csv$"
)


def decode_underscore_float(s: str) -> float:
    """Handles both encodings seen in the wild:
    '0_013887' -> 0.013887, '2_0' -> 2.0 (underscore-as-decimal-point)
    '0.0069435' -> 0.0069435, '0.5' -> 0.5 (plain decimal point)
    """
    if "_" in s:
        s = s.replace("_", ".", 1)
    return float(s)


def parse_filename(path: Path):
    m = FNAME_RE.match(path.name)
    if not m:
        return None
    d = m.groupdict()
    return {
        "function": d["function"],
        "dim": int(d["dim"]),
        "epsilon": decode_underscore_float(d["epsilon"]),
        "lipschitz": decode_underscore_float(d["lipschitz"]),
        "kernel": d["kernel"],
        "noise_std": d["noise"],
        "seed": int(d["seed"]),
    }


def compile_trajectories(input_dir: Path, output_dir: Path, make_summary: bool):
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()

    if not input_dir.exists():
        print(f"ERROR: input directory does not exist: {input_dir}", file=sys.stderr)
        print("Tip: --input-dir is resolved relative to your current working directory, "
              "not relative to where this script lives. Use an absolute path if unsure.",
              file=sys.stderr)
        sys.exit(1)
    if not input_dir.is_dir():
        print(f"ERROR: input path is not a directory: {input_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading from:  {input_dir}")
    print(f"Writing to:    {output_dir}")

    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {input_dir}", file=sys.stderr)
        return

    by_function = defaultdict(list)
    skipped = []

    for f in csv_files:
        meta = parse_filename(f)
        if meta is None:
            skipped.append(f.name)
            continue
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"  ! failed to read {f.name}: {e}", file=sys.stderr)
            continue

        for key, val in meta.items():
            df[key] = val
        df["source_file"] = f.name

        by_function[meta["function"]].append(df)

    if skipped:
        print(f"Skipped {len(skipped)} file(s) that didn't match the expected naming pattern:")
        for name in skipped:
            print(f"  - {name}")

    for function, frames in by_function.items():
        combined = pd.concat(frames, ignore_index=True)
        out_path = output_dir / f"{function}_compiled.csv"
        combined.to_csv(out_path, index=False)
        n_runs = combined[["dim", "epsilon", "lipschitz", "kernel", "noise_std", "seed"]] \
            .drop_duplicates().shape[0]
        print(f"{function}: {n_runs} run(s), {len(combined)} rows -> {out_path}")

        if make_summary:
            summary = (
                combined.groupby("iteration")[["y", "pred_opt"]]
                .agg(["mean", "std", "count"])
            )
            summary_path = output_dir / f"{function}_summary_by_iteration.csv"
            summary.to_csv(summary_path)
            print(f"  summary (mean/std/count per iteration) -> {summary_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", required=True, type=Path,
                         help="Directory containing the individual per-run trajectory CSVs")
    parser.add_argument("--output-dir", required=True, type=Path,
                         help="Directory to write the compiled per-function CSVs into")
    parser.add_argument("--summary", action="store_true",
                         help="Also write a mean/std/count-per-iteration summary CSV per function")
    args = parser.parse_args()

    compile_trajectories(args.input_dir, args.output_dir, args.summary)


if __name__ == "__main__":
    main()