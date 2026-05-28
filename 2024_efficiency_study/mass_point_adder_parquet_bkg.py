import os
import glob
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# =========================================================
# Base background directory
# =========================================================

base_dir = "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/NTuples_BKG_2024_HDNA_presel_official_full"

# Allowed mass points
mass_points = np.array([12, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])

# =========================================================
# Find all background dataset directories
# =========================================================

dataset_dirs = glob.glob(os.path.join(base_dir, "*/nominal"))

print(f"Found {len(dataset_dirs)} background datasets")

# =========================================================
# Loop over all background datasets
# =========================================================

for dataset_dir in dataset_dirs:

    print(f"\nProcessing: {dataset_dir}")

    parquet_files = glob.glob(os.path.join(dataset_dir, "*.parquet"))

    if len(parquet_files) == 0:
        print("  [WARNING] No parquet files found")
        continue

    print(f"  Found {len(parquet_files)} parquet files")

    # -----------------------------------------------------
    # Process each parquet file
    # -----------------------------------------------------

    for parquet_file in parquet_files:

        print(f"    -> {os.path.basename(parquet_file)}")

        # Read parquet file
        table = pq.read_table(parquet_file)

        # Convert to pandas
        df = table.to_pandas()

        # -------------------------------------------------
        # Assign random mass point per event
        # -------------------------------------------------

        random_mass_points = np.random.choice(
            mass_points,
            size=len(df),
            replace=True
        )

        df["mass_point"] = random_mass_points

        # Convert back to parquet
        new_table = pa.Table.from_pandas(df)

        # Overwrite original file
        pq.write_table(new_table, parquet_file)

print("\nDone!")
