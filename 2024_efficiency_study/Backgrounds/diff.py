import awkward as ak
import numpy as np
import pandas as pd

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
    "electron_eta",
    "electron_phi",

    "muon_pt",
    "muon_eta",
    "muon_phi",

    "pholead_pt",
    "pholead_eta",
    "phosublead_pt",
    "phosublead_eta",

    "first_jet_pt",
    "second_jet_pt",

    "n_bJets",
    "Njets",

    "PFMET_pt",
    "PuppiMET_pt",

    "mass",

    "weight",
]


# ============================================================
# Load
# ============================================================

raw = ak.to_dataframe(
    ak.from_parquet(
        raw_file,
        columns=cols
    )
).reset_index(drop=True)

skim = ak.to_dataframe(
    ak.from_parquet(
        skim_file,
        columns=cols
    )
).reset_index(drop=True)


# ============================================================
# Event keys
# ============================================================

keys = [
    "run",
    "lumi",
    "event"
]

raw_keys = set(
    zip(
        raw["run"],
        raw["lumi"],
        raw["event"]
    )
)

skim_keys = set(
    zip(
        skim["run"],
        skim["lumi"],
        skim["event"]
    )
)


# ============================================================
# Raw-only events
# ============================================================

missing_keys = raw_keys - skim_keys

print()
print("=" * 80)
print(f"Raw events       : {len(raw_keys)}")
print(f"Skim events      : {len(skim_keys)}")
print(f"Raw-only events  : {len(missing_keys)}")
print("=" * 80)


# ============================================================
# Extract missing events
# ============================================================

missing = raw[
    raw.apply(
        lambda row:
            (
                row["run"],
                row["lumi"],
                row["event"]
            ) in missing_keys,
        axis=1
    )
].copy()


# ============================================================
# Print them
# ============================================================

print()
print("RAW-ONLY EVENTS")
print("=" * 80)

print(
    missing[
        [
            "run",
            "lumi",
            "event",

            "electron_pt",
            "electron_eta",

            "muon_pt",
            "muon_eta",

            "pholead_pt",
            "phosublead_pt",

            "first_jet_pt",
            "second_jet_pt",

            "n_bJets",
            "Njets",

            "PFMET_pt",
            "PuppiMET_pt",

            "mass",

            "weight",
        ]
    ].to_string(index=False)
)


# ============================================================
# Count valid electrons
# ============================================================

valid_electron = (
    np.isfinite(missing["electron_pt"])
    & (missing["electron_pt"] > -900)
)

print()
print(
    "Raw-only events with valid electron:",
    valid_electron.sum()
)

print(
    "Raw-only events with electron_pt = -999:",
    (~valid_electron).sum()
)


# ============================================================
# Contribution to electron histogram
# ============================================================

electron_events = missing[
    valid_electron
]

electron_yield = (
    electron_events["weight"].sum()
    * 405.87
    * 109000
)

print()
print(
    "Yield contribution of missing electrons:",
    electron_yield
)


# ============================================================
# Save
# ============================================================

missing.to_csv(
    "raw_only_events.csv",
    index=False
)

print()
print("Saved: raw_only_events.csv")
