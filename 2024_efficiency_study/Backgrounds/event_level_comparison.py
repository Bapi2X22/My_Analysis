import awkward as ak
import numpy as np

skim_file = (
    "NTuples_2024_BKG_skim/merged/"
    "TTto2L2Nu_24SummerRun3/CAT1_merged.parquet"
)

raw_file = (
    "NTuples_BKG_2024_with_MET/merged/"
    "TTto2L2Nu-24SummerRun3/CAT1_merged.parquet"
)

cols = [
    "run",
    "lumi",
    "event",
    "electron_pt",
    "weight",
]

skim = ak.to_dataframe(
    ak.from_parquet(skim_file, columns=cols)
).reset_index(drop=True)

raw = ak.to_dataframe(
    ak.from_parquet(raw_file, columns=cols)
).reset_index(drop=True)

# ------------------------------------------------------------
# Event identifier
# ------------------------------------------------------------

keys = ["run", "lumi", "event"]

skim = skim.set_index(keys)
raw  = raw.set_index(keys)

common = skim.index.intersection(raw.index)

print("Raw events :", len(raw))
print("Skim events:", len(skim))
print("Common     :", len(common))

# ------------------------------------------------------------
# Compare common events
# ------------------------------------------------------------

s = skim.loc[common]
r = raw.loc[common]

# electron pT difference
dpt = s["electron_pt"] - r["electron_pt"]

mask = (
    np.isfinite(dpt)
    & (s["electron_pt"] > -900)
    & (r["electron_pt"] > -900)
)

print()
print("Common valid electrons:", mask.sum())

print(
    "Mean ΔpT:",
    dpt[mask].mean()
)

print(
    "RMS ΔpT:",
    dpt[mask].std()
)

print(
    "Maximum |ΔpT|:",
    np.max(np.abs(dpt[mask]))
)

# ------------------------------------------------------------
# Weight comparison
# ------------------------------------------------------------

dw = s["weight"] - r["weight"]

print()
print("Weight comparison")

print(
    "Mean Δweight:",
    dw.mean()
)

print(
    "Maximum |Δweight|:",
    np.max(np.abs(dw))
)
