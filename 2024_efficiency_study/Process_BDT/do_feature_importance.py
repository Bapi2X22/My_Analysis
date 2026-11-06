import pickle
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

with open("Models/BDT_reference_model_resol_061126.pkl", "rb") as f:
    model = pickle.load(f)

print(type(model))
print(model.keys())

for key, value in model.items():
    print(key, type(value))

# Feature names in exactly the training order
feature_names = [
    "leppt", "lepeta",
    "first_jet_eta", "second_jet_eta",
    "pT1_by_mbb", "pT2_by_mbb",
    "Njets",
    "pholead_eta", "phosublead_eta",
    "first_jet_B", "second_jet_B",
    "pholead_mvaID", "phosublead_mvaID",
    "pT1_by_mgg", "pT2_by_mgg",
    "delphi_gg", "delphi_bb", "delphi_bbgg", "delphi_ggMET_Puppi", "energy_resol_lead", "energy_resol_sublead",
    "mass_point"
]

# Get importance scores
xgb_model = model["xgbModel"]

print(xgb_model.feature_names)

importance = xgb_model.get_score(
    importance_type="gain"
)

# Include features that were never used in a split
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": [
        importance.get(f, 0.0) for f in feature_names
    ]
})

importance_df = importance_df.sort_values(
    "Importance", ascending=False
)

# Normalize to percentage
importance_df["Importance (%)"] = (
    importance_df["Importance"] /
    importance_df["Importance"].sum() * 100
)

print(importance_df)

# Plot
plt.figure(figsize=(10, 8))
plt.barh(
    importance_df["Feature"],
    importance_df["Importance (%)"]
)
plt.xlabel("Feature Importance (%)")
plt.ylabel("Features")
plt.title("XGBoost BDT Feature Importance (Gain)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("BDT_feature_importance_reference_resol.png", dpi=300)
plt.show()
