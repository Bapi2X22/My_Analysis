import os
import glob
import pandas as pd
import xgboost as xgb
import pickle
import pyarrow as pa
import pyarrow.parquet as pq

# =========================================================
# Load trained model
# =========================================================

with open("trained_model.pkl", "rb") as f:
    modelDict = pickle.load(f)

model = modelDict["xgbModel"]

# =========================================================
# Input variables
# MUST match training order exactly
# =========================================================

inputVars = [
    "leppt","lepeta",
    "first_jet_eta","second_jet_eta",
    "pT1_by_mbb","pT2_by_mbb",
    "n_bJets",
    "pholead_eta","phosublead_eta",
    "first_jet_B","second_jet_B",
    "pholead_mvaID","phosublead_mvaID",
    "pT1_by_mgg","pT2_by_mgg",
    "delphi_gg","delphi_bb","delphi_bbgg",
    "mass_point"
]

# =========================================================
# Base merged background directory
# =========================================================

base_dir = "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/NTuples_BKG_2024_HDNA_presel_official_full/merged_old"

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

        # =================================================
        # Build feature matrix
        # =================================================

        X = df[inputVars].values

        dmatrix = xgb.DMatrix(
            X,
            feature_names=inputVars
        )

        # =================================================
        # Predict BDT score
        # =================================================

        bdt_score = model.predict(dmatrix)

        print(f"     Events: {len(bdt_score)}")

        # =================================================
        # Add BDT score branch
        # =================================================

        df["BDT_score"] = bdt_score

        # =================================================
        # Save parquet file
        # =================================================

        new_table = pa.Table.from_pandas(df)

        pq.write_table(new_table, parquet_file)

        print("     Saved with BDT_score")

print("\nDone!")
