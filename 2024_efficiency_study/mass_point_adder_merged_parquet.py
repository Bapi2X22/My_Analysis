import os
import glob
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# =========================================================
# Base merged signal directory
# =========================================================

base_dir = "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/NTuples_WH_2024_HDNA_presel_official_full/merged_old"

# Signal mass points
mass_points = [12, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]

# =========================================================
# Loop over all signal mass points
# =========================================================

for mass in mass_points:

    signal_dir = f"{base_dir}/WH-2024M{mass}/nominal"

    print("\n=================================================")
    print(f"Processing M{mass}")
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

        print(f"  -> {os.path.basename(parquet_file)}")

        # Read parquet
        table = pq.read_table(parquet_file)

        # Convert to pandas
        df = table.to_pandas()

        # Add mass point branch
        df["mass_point"] = mass

        # Convert back to parquet
        new_table = pa.Table.from_pandas(df)

        # Overwrite parquet file
        pq.write_table(new_table, parquet_file)

        print("     Saved")

print("\nDone!")
