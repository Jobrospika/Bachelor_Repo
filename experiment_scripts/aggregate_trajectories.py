

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


RUN_FILENAME_RE = re.compile(
    r"(?P<function>[a-zA-Z]+)_(?P<dim>\d+)D_E(?P<epsilon>\d+_\d+)_"
    r"L(?P<lipschitz>\d+_\d+)_(?P<kernel>[a-zA-Z0-9]+)_std(?P<local_std>[a-zA-Z]+)_"
    r"seed(?P<seed>\d+)"
)


def parse_run_filename(name: str) -> Optional[dict]:
    """Extract function/dim/epsilon/lipschitz/kernel/local_std/seed from a
    per-run filename. Returns None if the filename doesn't match."""
    m = RUN_FILENAME_RE.search(name)
    if not m:
        return None
    d = m.groupdict()
    return {
        "function": d["function"],
        "dim": int(d["dim"]),
        "epsilon": float(d["epsilon"].replace("_", ".")),
        "lipschitz": float(d["lipschitz"].replace("_", ".")),
        "kernel": d["kernel"],
        "noise_std": d["local_std"],
        "seed": int(d["seed"]),
    }

# ---------------------------------------------------------------------
# CONFIG — edit these before running
# ---------------------------------------------------------------------

SAFETY_THRESHOLDS = {
    "rosenbrock": 0.0,
    "hartmann": 0.1,
}

TRUE_OPTIMUM = {}
DEFAULT_TRUE_OPTIMUM = 1.0

GROUP_COLS = ["function", "E_label", "L_label", "kernel", "local_std_label"]


KNOWN_EPSILON_MAPS = {
    "hartmann": {0.0069435: "small", 0.013887: "exact", 0.027774: "large"},
    "rosenbrock": {0.0139335: "small", 0.027867: "exact", 0.055734: "large"},
}

KNOWN_LIPSCHITZ_MAP = {
    0.5: "underestimated", 1.0: "exact", 1.5: "low_safety_net", 2.0: "high_safety_net",
}

# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------


METADATA_COLS = {"function", "epsilon", "lipschitz", "kernel", "seed", "noise_std"}


def _read_one_run_file(f: Path) -> pd.DataFrame | None:
    """Read a single per-run CSV. If it's missing the metadata columns
    (function/epsilon/lipschitz/kernel/seed/noise_std) — i.e. it's a bare
    iteration,x,y,pred_opt file — parse them from the filename instead."""
    try:
        df = pd.read_csv(f)
    except Exception as e:
        print(f"[load] SKIPPED unreadable file {f.name}: {e}")
        return None

    if "source_file" not in df.columns:
        df["source_file"] = f.name

    missing = METADATA_COLS - set(df.columns)
    if missing:
        parsed = parse_run_filename(f.name)
        if parsed is None:
            print(f"[load] SKIPPED {f.name}: missing columns {missing} "
                  f"and filename doesn't match the expected "
                  f"'<function>_<dim>D_E<eps>_L<lip>_<kernel>_std<label>_seed<n>' pattern")
            return None
        for col, val in parsed.items():
            df[col] = val
    return df


def load_raw(inputs: list[str]) -> pd.DataFrame:
    """Load one or more compiled CSVs, individual per-run CSVs (bare or
    with metadata columns), and/or directories of per-run CSVs into a
    single per-iteration dataframe."""
    frames = []
    for inp in inputs:
        p = Path(inp)
        if p.is_dir():
            csvs = sorted(p.glob("*.csv"))
            print(f"[load] {p}: found {len(csvs)} per-run CSV files")
            for f in csvs:
                df = _read_one_run_file(f)
                if df is not None:
                    frames.append(df)
        elif p.is_file():
            df = _read_one_run_file(p)
            if df is not None:
                frames.append(df)
                if METADATA_COLS - set(pd.read_csv(p, nrows=0).columns):
                    print(f"[load] {p.name}: parsed metadata from filename")
                else:
                    print(f"[load] {p.name}: reading (has metadata columns)")
        else:
            print(f"[load] WARNING: {p} not found, skipping")

    if not frames:
        raise SystemExit("No input data loaded — check your paths.")

    raw = pd.concat(frames, ignore_index=True)
    required = {"iteration", "y", "pred_opt", "function", "epsilon",
                "lipschitz", "kernel", "seed"}
    missing = required - set(raw.columns)
    if missing:
        raise SystemExit(f"Input is missing required columns: {missing}")

    # your compiled files call the local_std column "noise_std" —
    # normalize to local_std_label here so the rest of the script (and
    # the final summary) uses one consistent name
    if "noise_std" in raw.columns and "local_std_label" not in raw.columns:
        raw = raw.rename(columns={"noise_std": "local_std_label"})

    return raw


# ---------------------------------------------------------------------
# Label reconstruction (epsilon / lipschitz -> E_label / L_label)
# ---------------------------------------------------------------------


def add_grid_labels(raw: pd.DataFrame) -> pd.DataFrame:
    """Map literal epsilon values to small/exact/large and literal
    lipschitz-factor values to underestimated/exact/low_safety_net/
    high_safety_net, per function (grids can differ per function)."""
    raw = raw.copy()
    e_names = ["small", "exact", "large"]
    l_names = ["underestimated", "exact", "low_safety_net", "high_safety_net"]

    raw["E_label"] = None
    raw["L_label"] = None

    for fn, sub in raw.groupby("function"):
        eps_vals = sorted(sub["epsilon"].unique())
        lip_vals = sorted(sub["lipschitz"].unique())

        if fn in KNOWN_EPSILON_MAPS:
            e_map = KNOWN_EPSILON_MAPS[fn]
            unknown = [v for v in eps_vals if v not in e_map]
            if unknown:
                print(f"[labels] WARNING: {fn} has epsilon values not in "
                      f"KNOWN_EPSILON_MAPS: {unknown} — these will get no E_label.")
        else:
            # fallback: infer from sorted order within *this batch*. Only
            # safe if this batch actually contains the full grid of E
            # values for this function — a partial upload will mislabel.
            print(f"[labels] WARNING: no fixed epsilon map for '{fn}' — "
                  f"inferring E_label from sorted order of the {len(eps_vals)} "
                  f"epsilon value(s) present in THIS batch: {eps_vals}. "
                  f"This is only correct if the batch contains the full "
                  f"grid for {fn} — add a KNOWN_EPSILON_MAPS entry to be safe.")
            e_map = {v: e_names[i] for i, v in enumerate(eps_vals[:3])}

        if all(v in KNOWN_LIPSCHITZ_MAP for v in lip_vals):
            l_map = KNOWN_LIPSCHITZ_MAP
        else:
            unknown = [v for v in lip_vals if v not in KNOWN_LIPSCHITZ_MAP]
            print(f"[labels] WARNING: lipschitz values not in "
                  f"KNOWN_LIPSCHITZ_MAP: {unknown} — inferring from sorted "
                  f"order within this batch instead, which is only correct "
                  f"if the batch contains the full grid for {fn}.")
            l_map = {v: l_names[i] for i, v in enumerate(lip_vals[:4])}

        mask = raw["function"] == fn
        raw.loc[mask, "E_label"] = raw.loc[mask, "epsilon"].map(e_map)
        raw.loc[mask, "L_label"] = raw.loc[mask, "lipschitz"].map(l_map)

    return raw


# ---------------------------------------------------------------------
# Completeness check — this is the "broke in the middle" handling
# ---------------------------------------------------------------------


def flag_incomplete_runs(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Determine the 'complete' iteration count per function as the mode
    of max-iteration across all runs of that function, then split runs
    into complete / incomplete. Returns (clean_raw, audit_df)."""
    run_keys = ["function", "E_label", "L_label", "kernel",
                "local_std_label", "seed"]

    per_run_max = raw.groupby(run_keys)["iteration"].max().rename("max_iter")
    per_run_count = raw.groupby(run_keys)["iteration"].count().rename("n_rows")
    run_info = pd.concat([per_run_max, per_run_count], axis=1).reset_index()

    expected_per_fn = run_info.groupby("function")["max_iter"].agg(
        lambda s: s.mode().iloc[0]
    )
    run_info["expected_max_iter"] = run_info["function"].map(expected_per_fn)
    run_info["complete"] = run_info["max_iter"] >= run_info["expected_max_iter"]

    n_incomplete = (~run_info["complete"]).sum()
    if n_incomplete:
        print(f"[completeness] {n_incomplete} / {len(run_info)} runs are "
              f"incomplete and will be EXCLUDED from the summary.")
        print(run_info.loc[~run_info["complete"], run_keys + ["max_iter", "expected_max_iter"]]
              .to_string(index=False))
    else:
        print(f"[completeness] all {len(run_info)} runs are complete.")

    complete_keys = run_info.loc[run_info["complete"], run_keys]
    clean_raw = raw.merge(complete_keys, on=run_keys, how="inner")
    return clean_raw, run_info


# ---------------------------------------------------------------------
# Per-run derived metrics
# ---------------------------------------------------------------------


def classify_violation(y: pd.Series, threshold: float) -> str:
    if threshold is None:
        return "unknown"  # no threshold configured for this function
    return "none" if (y >= threshold).all() else "violation"


def determine_no_opt_possible(run_df: pd.DataFrame) -> bool:
    """STUB. Replace with your actual definition of 'this run could not
    have found the optimum.' Currently always False."""
    return False


def compute_ratio(final_pred_opt: float, function: str) -> float:
    true_opt = TRUE_OPTIMUM.get(function, DEFAULT_TRUE_OPTIMUM)
    return final_pred_opt / true_opt


def build_trajectory_table(clean_raw: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-iteration rows into one row per run with the columns
    the standard aggregation expects: ratio, violation_class,
    no_opt_possible, final_pred_opt, avg_pred_opt."""
    run_keys = ["function", "E_label", "L_label", "kernel",
                "local_std_label", "seed"]
    warned = set()
    rows = []
    for keys, run_df in clean_raw.groupby(run_keys):
        run_df = run_df.sort_values("iteration")
        fn = keys[0]
        threshold = SAFETY_THRESHOLDS.get(fn)
        if threshold is None and fn not in warned:
            print(f"[warn] no safety threshold configured for '{fn}' — "
                  f"violation_class will be 'unknown' for this function.")
            warned.add(fn)

        final_pred_opt = run_df["pred_opt"].iloc[-1]
        avg_pred_opt = run_df["pred_opt"].mean()

        row = dict(zip(run_keys, keys))
        row["final_pred_opt"] = final_pred_opt
        row["avg_pred_opt"] = avg_pred_opt
        row["ratio"] = compute_ratio(final_pred_opt, fn)
        row["violation_class"] = classify_violation(run_df["y"], threshold)
        row["no_opt_possible"] = determine_no_opt_possible(run_df)
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Standard aggregation (exactly your snippet)
# ---------------------------------------------------------------------


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby(GROUP_COLS).agg(
        n_runs=("ratio", "count"),
        mean_ratio=("ratio", "mean"),
        std_ratio=("ratio", "std"),
        pct_no_violation=("violation_class", lambda s: round(100 * (s == "none").sum() / len(s), 1)),
        pct_no_opt=("no_opt_possible", lambda s: round(100 * s.sum() / len(s), 1)),
        mean_final=("final_pred_opt", "mean"),
        mean_avg=("avg_pred_opt", "mean"),
    ).reset_index()
    return agg


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+",
                     help="compiled CSV(s) and/or directories of per-run CSVs")
    ap.add_argument("-o", "--output", default="summary.csv",
                     help="output summary CSV path (default: summary.csv)")
    ap.add_argument("--audit-output", default="run_audit.csv",
                     help="output path for the per-run completeness audit "
                          "(default: run_audit.csv)")
    ap.add_argument("--trajectory-output", default=None,
                     help="optionally also save the per-run trajectory "
                          "table (ratio/violation_class/etc per run)")
    args = ap.parse_args()

    raw = load_raw(args.inputs)
    raw = add_grid_labels(raw)
    clean_raw, audit = flag_incomplete_runs(raw)
    audit.to_csv(args.audit_output, index=False)
    print(f"[output] run completeness audit -> {args.audit_output}")

    traj = build_trajectory_table(clean_raw)
    if args.trajectory_output:
        traj.to_csv(args.trajectory_output, index=False)
        print(f"[output] per-run trajectory table -> {args.trajectory_output}")

    summary = aggregate(traj)
    summary.to_csv(args.output, index=False)
    print(f"[output] summary -> {args.output}")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

#python3 aggregate_trajectories.py results/bachelor_tests_nd_adaptive/run_20260811_232714/compiled/hartmann_compiled.csv results/bachelor_tests_nd_adaptive/run_20260811_232714/compiled/rosenbrock_compiled.csv -o summary.csv --audit-output run_audit.csv