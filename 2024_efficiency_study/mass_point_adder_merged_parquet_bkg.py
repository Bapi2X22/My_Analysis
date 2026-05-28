import os
import glob
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# =========================================================
# Base merged background directory
# =========================================================

base_dir = "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/NTuples_BKG_2024_HDNA_presel_official_full/merged"

# Allowed signal mass points
mass_points = np.array([12, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])

# =========================================================
# Find all background datasets
# =========================================================

dataset_dirs = glob.glob(os.path.join(base_dir, "*/nominal"))

print(f"Found {len(dataset_dirs)} background datasets")

# =========================================================
# Loop over datasets
# =========================================================

for dataset_dir in dataset_dirs:

    print("\n=================================================")
    print(f"Processing:")
    print(dataset_dir)

    parquet_files = glob.glob(os.path.join(dataset_dir, "*.parquet"))

    if len(parquet_files) == 0:
        print("No parquet files found")
        continue

    # -----------------------------------------------------
    # Process parquet files
    # -----------------------------------------------------

    for parquet_file in parquet_files:

        print(f"\n  -> {os.path.basename(parquet_file)}")

        # =================================================
        # Read parquet file
        # =================================================

        table = pq.read_table(parquet_file)

        df = table.to_pandas()

        n_events = len(df)

        # =================================================
        # Assign random mass point per event
        # =================================================

        df["mass_point"] = np.random.choice(
            mass_points,
            size=n_events,
            replace=True
        )

        # =================================================
        # Save parquet file
        # =================================================

        new_table = pa.Table.from_pandas(df)

        pq.write_table(new_table, parquet_file)

        print(f"     Events: {n_events}")
        print("     Saved with mass_point")

print("\nDone!")
