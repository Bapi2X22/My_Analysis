#!/usr/bin/env python3

import os
import glob
import pickle

import awkward as ak
import numpy as np
import xgboost as xgb


# -------------------------------------------------------
# Configuration
# -------------------------------------------------------

INPUT_DIR = "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/NTuples_WH_2024_HDNA_presel_latest_with_BDT_score"

OUTPUT_DIR = "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/NTuples_WH_2024_HDNA_presel_withAllBDTs"

MODEL_FILE = "/eos/user/b/bbapi/HiggsDNA_220526/HiggsDNA/higgs_dna/workflows/models/model_htoaaTo2b2g.pkl"

INPUT_VARS = [
    "leppt",
    "lepeta",
    "first_jet_eta",
    "second_jet_eta",
    "pT1_by_mbb",
    "pT2_by_mbb",
    "first_jet_B",
    "second_jet_B",
    "Njets",
    "pholead_eta",
    "phosublead_eta",
    "pT1_by_mgg",
    "pT2_by_mgg",
    "pholead_mvaID",
    "phosublead_mvaID",
    "delphi_gg",
    "delphi_bb",
    "delphi_bbgg",
    "mass_point",
]



# -------------------------------------------------------
# Load model
# -------------------------------------------------------

with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

bdt_model = model["xgbModel"]


# -------------------------------------------------------
# Find parquet files
# -------------------------------------------------------

files = sorted(
    glob.glob(os.path.join(INPUT_DIR, "WH-2024M*", "**", "*.parquet"), recursive=True)
)

print(f"Found {len(files)} parquet files")


# -------------------------------------------------------
# Loop
# -------------------------------------------------------

# -------------------------------------------------------
# Helper
# -------------------------------------------------------

import pyarrow.parquet as pq


def write_preserve_metadata(events, infile, outfile):
    """
    Write parquet while preserving the original schema metadata.
    """

    pf = pq.ParquetFile(infile)
    metadata = pf.schema_arrow.metadata

    # table = ak.to_arrow_table(events)
    table = ak.to_arrow_table(events, extensionarray=False)

    if metadata is not None:
        table = table.replace_schema_metadata(metadata)

    pq.write_table(
        table,
        outfile,
        compression="snappy",
    )


# -------------------------------------------------------
# Loop
# -------------------------------------------------------

for infile in files:

    relpath = os.path.relpath(infile, INPUT_DIR)
    outfile = os.path.join(OUTPUT_DIR, relpath)

    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    print(f"\nProcessing:\n  {relpath}")

    try:
        events = ak.from_parquet(infile)
    except Exception as e:
        print("Cannot read:", e)
        continue

    # ----------------------------
    # Empty file
    # ----------------------------

    if len(events) == 0:
        print("  Empty file")


        events["bdt_score"] = ak.Array(
            np.empty(0, dtype=np.float32)
        )

        write_preserve_metadata(
            events,
            infile,
            outfile,
        )

        continue

    # ----------------------------
    # Feature matrix
    # ----------------------------

    X = np.column_stack(
        [
            ak.to_numpy(events[var])
            for var in INPUT_VARS
        ]
    )

    n_bjets = ak.to_numpy(events["n_bJets"])
    has2b = n_bjets >= 2

    # ----------------------------
    # Evaluate BDTs
    # ----------------------------

    score = np.full(
        len(events),
        -999,
        dtype=np.float32,
    )

    if np.any(has2b):
        dmat = xgb.DMatrix(
            X[has2b],
            feature_names=INPUT_VARS,
        )

        score[has2b] = bdt_model.predict(dmat).astype(np.float32)

    events["bdt_score"] = score

    # ----------------------------
    # Save output
    # ----------------------------

    write_preserve_metadata(
        events,
        infile,
        outfile,
    )

print("\nDone.")