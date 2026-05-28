import uproot
import awkward as ak
import pandas as pd
import xgboost as xgb
import pickle
import numpy as np

# =========================================================
# Load trained model
# =========================================================

with open("trained_model.pkl", "rb") as f:
    modelDict = pickle.load(f)

model = modelDict["xgbModel"]

# =========================================================
# Input variables (MUST match training order)
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
# Open ROOT file
# =========================================================

input_file = "merged_bkg_withMass.root"
tree_name  = "DiphotonTree"

tree = uproot.open(input_file)[tree_name]

# Read ALL branches
arrays_all = tree.arrays()

# Convert only training variables to dataframe
df = ak.to_dataframe(arrays_all[inputVars])

# =========================================================
# Convert to XGBoost DMatrix
# =========================================================

X = df[inputVars].values

dmatrix = xgb.DMatrix(X, feature_names=inputVars)

# =========================================================
# Predict BDT score
# =========================================================

bdt_score = model.predict(dmatrix)

print("Number of events:", len(bdt_score))
print("First 10 scores:", bdt_score[:10])

# =========================================================
# Save score to ROOT file
# =========================================================

# =========================================================
# Save score to ROOT file
# =========================================================

output_dict = {}

# Copy ALL original branches
for branch in arrays_all.fields:
    output_dict[branch] = ak.to_numpy(arrays_all[branch])

# Add BDT score
output_dict["BDT_score"] = bdt_score

# Write output ROOT file
outfile = uproot.recreate("scored_background.root")

outfile["DiphotonTree"] = output_dict

outfile.close()

print("Saved: scored_background.root")
