import os
import glob
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# =========================================================
# Base merged background directory
# =========================================================

base_dir = "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/NTuples_BKG_2024_HDNA_presel_official_full/merged"

# BDT cut
bdt_cut = 0.7

# =========================================================
# Find all background datasets
# =========================================================

dataset_dirs = glob.glob(os.path.join(base_dir, "*/nominal"))

print(f"Found {len(dataset_dirs)} background datasets")

# =========================================================
# Loop over all background datasets
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

        n_before = len(df)

        # =================================================
        # Apply BDT cut
        # =================================================

        df_cut = df[df["BDT_score"] > bdt_cut].copy()

        n_after = len(df_cut)

        efficiency = (
            100.0 * n_after / n_before
            if n_before > 0 else 0
        )

        print(f"     Before cut : {n_before}")
        print(f"     After cut  : {n_after}")
        print(f"     Efficiency : {efficiency:.2f}%")

        # =================================================
        # Save filtered parquet file
        # Overwrite original file
        # =================================================

        new_table = pa.Table.from_pandas(df_cut)

        pq.write_table(new_table, parquet_file)

        print("     Saved filtered events")

print("\nDone!")
