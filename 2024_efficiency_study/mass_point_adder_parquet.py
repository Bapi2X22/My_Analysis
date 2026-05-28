import os
import glob
import re
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# =========================================================
# Base directory
# =========================================================

base_dir = "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/NTuples_WH_2024_HDNA_presel_official_full"

# Mass points to process
mass_points = [12, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]

# =========================================================
# Loop over all mass point directories
# =========================================================

for mass in mass_points:

    dir_path = f"{base_dir}/WH-2024M{mass}/nominal"

    if not os.path.isdir(dir_path):
        print(f"[WARNING] Directory not found: {dir_path}")
        continue

    parquet_files = glob.glob(os.path.join(dir_path, "*.parquet"))

    if len(parquet_files) == 0:
        print(f"[WARNING] No parquet files found in: {dir_path}")
        continue

    print(f"\nProcessing mass point M{mass}")
    print(f"Found {len(parquet_files)} parquet files")

    # -----------------------------------------------------
    # Process each parquet file
    # -----------------------------------------------------

    for parquet_file in parquet_files:

        print(f"  -> {os.path.basename(parquet_file)}")

        # Read parquet
        table = pq.read_table(parquet_file)

        # Convert to pandas
        df = table.to_pandas()

        # Add mass_point column
        df["mass_point"] = mass

        # Convert back to parquet table
        new_table = pa.Table.from_pandas(df)

        # Overwrite original file
        pq.write_table(new_table, parquet_file)

print("\nDone!")
