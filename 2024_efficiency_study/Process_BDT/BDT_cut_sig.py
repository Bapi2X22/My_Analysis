import os
import glob
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# =========================================================
# Base merged signal directory
# =========================================================

base_dir = "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/NTuples_WH_2024_HDNA_presel_official_full/merged"

# Signal mass points
mass_points = [12, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]

# BDT cut
bdt_cut = 0.7

# =========================================================
# Loop over signal datasets
# =========================================================

for mass in mass_points:

    signal_dir = f"{base_dir}/WH-2024M{mass}/nominal"

    print("\n=================================================")
    print(f"Processing signal M{mass}")
    print(signal_dir)

    if not os.path.isdir(signal_dir):
        print("Directory not found")
        continue

    parquet_files = glob.glob(os.path.join(signal_dir, "*.parquet"))

    if len(parquet_files) == 0:
        print("No parquet files found")
        continue

    # -----------------------------------------------------
    # Process each parquet file
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
