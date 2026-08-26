import glob
import os
import re
from datetime import datetime
import pandas as pd


CUTOFF_TIME = datetime(2026, 8, 10, 11, 19, 0)
RESULTS_ROOT = r"C:\Users\josch\OneDrive\Desktop\Coding\Bachelor\LoSBO\results"
OUTPUT_CSV = fr"C:\Users\josch\OneDrive\Desktop\Coding\Bachelor\LoSBO\results\ucb\compiled_results_{CUTOFF_TIME.strftime('%Y-%m-%d_%H-%M-%S')}.csv"

run_folder_pattern = re.compile(r"^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_(.+)_seed_(\d+)$")

rows = []
for filepath in glob.glob(os.path.join(RESULTS_ROOT, "**", "seed_*_results.csv"), recursive=True):
    run_folder = os.path.basename(os.path.dirname(filepath))
    config_name = os.path.basename(os.path.dirname(os.path.dirname(filepath)))

    match = run_folder_pattern.match(run_folder)
    if not match:
        print(f"Skipping unrecognized folder: {run_folder}")
        continue

    timestamp_str, run_label, seed = match.groups()
    run_time = datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")

    if run_time < CUTOFF_TIME:
        continue

    seed = int(seed)

    df = pd.read_csv(filepath)
    df["seed"] = seed
    df["config"] = config_name
    df["timestamp"] = timestamp_str
    df["run_label"] = run_label
    rows.append(df)

if not rows:
    print(f"No result files found under {RESULTS_ROOT} after {CUTOFF_TIME}")
else:
    combined = pd.concat(rows, ignore_index=True)

    # if the same (config, seed) was run more than once after the cutoff, keep only the latest
    latest_per_seed = (
        combined[["config", "seed", "timestamp"]]
        .drop_duplicates()
        .sort_values("timestamp")
        .groupby(["config", "seed"], as_index=False)
        .last()
    )
    combined = combined.merge(latest_per_seed, on=["config", "seed", "timestamp"], how="inner")

    combined.to_csv(OUTPUT_CSV, index=False)
    print(f"Compiled {len(rows)} files into {OUTPUT_CSV} ({len(combined)} rows total)")