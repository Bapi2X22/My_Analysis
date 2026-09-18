#!/usr/bin/env python3

import awkward as ak
import pandas as pd
import pyarrow.parquet as pq


# ============================================================
# Files
# ============================================================

RAW_FILE = (
    "NTuples_BKG_2024_with_MET/merged/"
    "TTto2L2Nu-24SummerRun3/CAT1_merged.parquet"
)

SKIM_FILE = (
    "NTuples_2024_BKG_skim/merged/"
    "TTto2L2Nu_24SummerRun3/CAT1_merged.parquet"
)


# ============================================================
# Object-selection requirements
# ============================================================

PHOTON_PT_MIN = 12.0
PHOTON_ETA_MAX = 2.5

MUON_PT_MIN = 20.0
MUON_ETA_MAX = 2.4

ELECTRON_PT_MIN = 25.0
ELECTRON_ETA_MAX = 2.5

JET_PT_MIN = 15.0
JET_ETA_MAX = 2.4


# ============================================================
# Event identifiers
# ============================================================

KEYS = [
    "run",
    "lumi",
    "event",
]


# ============================================================
# Variables to load from Raw
# ============================================================

VARIABLES = [

    # --------------------------------------------------------
    # Event
    # --------------------------------------------------------

    "run",
    "lumi",
    "event",
    "weight",
    "genWeight",

    # --------------------------------------------------------
    # Electron
    # --------------------------------------------------------

    "electron_pt",
    "electron_eta",
    "electron_phi",
    "electron_mass",
    "electron_charge",

    # --------------------------------------------------------
    # Muon
    # --------------------------------------------------------

    "muon_pt",
    "muon_eta",
    "muon_phi",
    "muon_mass",
    "muon_charge",

    # --------------------------------------------------------
    # Generic lepton
    # --------------------------------------------------------

    "leppt",
    "lepeta",
    "charge",
    "dZ",

    # --------------------------------------------------------
    # Leading photon
    # --------------------------------------------------------

    "pholead_pt_raw",
    "pholead_pt",
    "pholead_eta",
    "pholead_phi",
    "pholead_mass",

    # Photon ID
    "pholead_cutBased",
    "pholead_mvaID",
    "pholead_mvaID_WP80",
    "pholead_mvaID_WP90",
    "pholead_electronVeto",
    "pholead_pixelSeed",

    # Photon shower / isolation
    "pholead_hoe",
    "pholead_sieie",
    "pholead_sieip",
    "pholead_sipip",
    "pholead_r9",

    "pholead_pfChargedIso",
    "pholead_pfPhoIso03",
    "pholead_pfRelIso03_all_quadratic",
    "pholead_pfRelIso03_chg_quadratic",

    # Photon geometry
    "pholead_ScEta",
    "pholead_superclusterEta",
    "pholead_isScEtaEB",
    "pholead_isScEtaEE",

    # --------------------------------------------------------
    # Subleading photon
    # --------------------------------------------------------

    "phosublead_pt_raw",
    "phosublead_pt",
    "phosublead_eta",
    "phosublead_phi",
    "phosublead_mass",

    # Photon ID
    "phosublead_cutBased",
    "phosublead_mvaID",
    "phosublead_mvaID_WP80",
    "phosublead_mvaID_WP90",
    "phosublead_electronVeto",
    "phosublead_pixelSeed",

    # Photon shower / isolation
    "phosublead_hoe",
    "phosublead_sieie",
    "phosublead_sieip",
    "phosublead_sipip",
    "phosublead_r9",

    "phosublead_pfChargedIso",
    "phosublead_pfPhoIso03",
    "phosublead_pfRelIso03_all_quadratic",
    "phosublead_pfRelIso03_chg_quadratic",

    # Photon geometry
    "phosublead_ScEta",
    "phosublead_superclusterEta",
    "phosublead_isScEtaEB",
    "phosublead_isScEtaEE",

    # --------------------------------------------------------
    # Jets
    # --------------------------------------------------------

    "Njets",
    "n_bJets",

    "first_jet_pt_raw",
    "first_jet_pt",
    "first_jet_eta",
    "first_jet_phi",
    "first_jet_mass",

    "second_jet_pt_raw",
    "second_jet_pt",
    "second_jet_eta",
    "second_jet_phi",
    "second_jet_mass",

    # b-tagging
    "first_jet_B",
    "first_jet_probb",
    "first_jet_probbb",

    "second_jet_B",
    "second_jet_probb",
    "second_jet_probbb",

    "diff_first_jet_probb_probbb",

    # --------------------------------------------------------
    # Diphoton
    # --------------------------------------------------------

    "mass",
    "dipho_pt",

    "delphi_gg",
    "delphi_bb",
    "delphi_bbgg",

    # --------------------------------------------------------
    # MET
    # --------------------------------------------------------

    "PFMET_pt",
    "PFMET_phi",
    "PFMET_sumEt",

    "PuppiMET_pt",
    "PuppiMET_phi",
    "PuppiMET_sumEt",

    # --------------------------------------------------------
    # Other
    # --------------------------------------------------------

    "nPV",
    "fixedGridRhoAll",
]


# ============================================================
# Read event IDs
# ============================================================

print("Reading event IDs...")

raw_id = ak.to_dataframe(
    ak.from_parquet(
        RAW_FILE,
        columns=KEYS
    )
).reset_index(drop=True)

skim_id = ak.to_dataframe(
    ak.from_parquet(
        SKIM_FILE,
        columns=KEYS
    )
).reset_index(drop=True)


# ============================================================
# Build event-key sets
# ============================================================

raw_keys = set(
    zip(
        raw_id["run"],
        raw_id["lumi"],
        raw_id["event"],
    )
)

skim_keys = set(
    zip(
        skim_id["run"],
        skim_id["lumi"],
        skim_id["event"],
    )
)


# ============================================================
# Find Raw-only events
# ============================================================

missing_keys = raw_keys - skim_keys


print()
print("=" * 100)
print(
    f"Raw events      : {len(raw_keys)}"
)
print(
    f"Skim events     : {len(skim_keys)}"
)
print(
    f"Raw-only events : {len(missing_keys)}"
)
print("=" * 100)


# ============================================================
# Check Raw parquet schema
# ============================================================

raw_fields = pq.read_schema(
    RAW_FILE
).names


missing_variables = [
    x for x in VARIABLES
    if x not in raw_fields
]


if missing_variables:

    print()
    print("ERROR: The following variables are missing:")
    print()

    for x in missing_variables:
        print("   ", x)

    raise RuntimeError(
        "Requested variables are not available in Raw parquet."
    )


# ============================================================
# Load Raw variables
# ============================================================

print()
print("Reading Raw diagnostic variables...")

raw = ak.to_dataframe(
    ak.from_parquet(
        RAW_FILE,
        columns=VARIABLES
    )
).reset_index(drop=True)


# ============================================================
# Build event key
# ============================================================

raw["event_key"] = list(
    zip(
        raw["run"],
        raw["lumi"],
        raw["event"],
    )
)


# ============================================================
# Select Raw-only events
# ============================================================

missing = raw[
    raw["event_key"].isin(
        missing_keys
    )
].copy()


missing.drop(
    columns=["event_key"],
    inplace=True
)


# ============================================================
# Sort by event
# ============================================================

missing.sort_values(
    ["run", "lumi", "event"],
    inplace=True
)

missing.reset_index(
    drop=True,
    inplace=True
)


# ============================================================
# Object selections
#
# IMPORTANT:
#
# The skim selection was applied BEFORE correction.
#
# Therefore:
#
# Photon selection -> *_pt_raw
# Jet selection     -> *_pt_raw
#
# Corrected pT is evaluated separately only to see whether
# correction causes any selection migration.
# ============================================================


# ============================================================
# Leading photon
# ============================================================

missing["leadPho_pt_pass"] = (
    missing["pholead_pt_raw"]
    > PHOTON_PT_MIN
)

missing["leadPho_eta_pass"] = (
    missing["pholead_eta"].abs()
    < PHOTON_ETA_MAX
)

missing["leadPho_raw_pass"] = (
    missing["leadPho_pt_pass"]
    &
    missing["leadPho_eta_pass"]
)


# Corrected version

missing["leadPho_corr_pt_pass"] = (
    missing["pholead_pt"]
    > PHOTON_PT_MIN
)

missing["leadPho_corr_eta_pass"] = (
    missing["pholead_eta"].abs()
    < PHOTON_ETA_MAX
)

missing["leadPho_corr_pass"] = (
    missing["leadPho_corr_pt_pass"]
    &
    missing["leadPho_corr_eta_pass"]
)


# ============================================================
# Subleading photon
# ============================================================

missing["subleadPho_pt_pass"] = (
    missing["phosublead_pt_raw"]
    > PHOTON_PT_MIN
)

missing["subleadPho_eta_pass"] = (
    missing["phosublead_eta"].abs()
    < PHOTON_ETA_MAX
)

missing["subleadPho_raw_pass"] = (
    missing["subleadPho_pt_pass"]
    &
    missing["subleadPho_eta_pass"]
)


# Corrected version

missing["subleadPho_corr_pt_pass"] = (
    missing["phosublead_pt"]
    > PHOTON_PT_MIN
)

missing["subleadPho_corr_eta_pass"] = (
    missing["phosublead_eta"].abs()
    < PHOTON_ETA_MAX
)

missing["subleadPho_corr_pass"] = (
    missing["subleadPho_corr_pt_pass"]
    &
    missing["subleadPho_corr_eta_pass"]
)


# ============================================================
# Combined photon selection
# ============================================================

missing["photons_raw_pass"] = (
    missing["leadPho_raw_pass"]
    &
    missing["subleadPho_raw_pass"]
)

missing["photons_corr_pass"] = (
    missing["leadPho_corr_pass"]
    &
    missing["subleadPho_corr_pass"]
)


# ============================================================
# Electron
# ============================================================

missing["electron_pt_pass"] = (
    missing["electron_pt"]
    > ELECTRON_PT_MIN
)

missing["electron_eta_pass"] = (
    missing["electron_eta"].abs()
    < ELECTRON_ETA_MAX
)

missing["electron_pass"] = (
    missing["electron_pt_pass"]
    &
    missing["electron_eta_pass"]
)


# ============================================================
# Muon
# ============================================================

missing["muon_pt_pass"] = (
    missing["muon_pt"]
    > MUON_PT_MIN
)

missing["muon_eta_pass"] = (
    missing["muon_eta"].abs()
    < MUON_ETA_MAX
)

missing["muon_pass"] = (
    missing["muon_pt_pass"]
    &
    missing["muon_eta_pass"]
)


# ============================================================
# Leading jet
# ============================================================

missing["firstJet_pt_pass"] = (
    missing["first_jet_pt_raw"]
    > JET_PT_MIN
)

missing["firstJet_eta_pass"] = (
    missing["first_jet_eta"].abs()
    < JET_ETA_MAX
)

missing["firstJet_raw_pass"] = (
    missing["firstJet_pt_pass"]
    &
    missing["firstJet_eta_pass"]
)


# Corrected

missing["firstJet_corr_pt_pass"] = (
    missing["first_jet_pt"]
    > JET_PT_MIN
)

missing["firstJet_corr_eta_pass"] = (
    missing["first_jet_eta"].abs()
    < JET_ETA_MAX
)

missing["firstJet_corr_pass"] = (
    missing["firstJet_corr_pt_pass"]
    &
    missing["firstJet_corr_eta_pass"]
)


# ============================================================
# Subleading jet
# ============================================================

missing["secondJet_pt_pass"] = (
    missing["second_jet_pt_raw"]
    > JET_PT_MIN
)

missing["secondJet_eta_pass"] = (
    missing["second_jet_eta"].abs()
    < JET_ETA_MAX
)

missing["secondJet_raw_pass"] = (
    missing["secondJet_pt_pass"]
    &
    missing["secondJet_eta_pass"]
)


# Corrected

missing["secondJet_corr_pt_pass"] = (
    missing["second_jet_pt"]
    > JET_PT_MIN
)

missing["secondJet_corr_eta_pass"] = (
    missing["second_jet_eta"].abs()
    < JET_ETA_MAX
)

missing["secondJet_corr_pass"] = (
    missing["secondJet_corr_pt_pass"]
    &
    missing["secondJet_corr_eta_pass"]
)


# ============================================================
# Print RAW vs CORRECTED pT and eta
# ============================================================

print()
print("=" * 160)
print("RAW vs CORRECTED OBJECT pT AND eta FOR RAW-ONLY EVENTS")
print("=" * 160)

pt_eta_columns = [

    "run",
    "lumi",
    "event",

    # --------------------------------------------------------
    # Leading photon
    # --------------------------------------------------------

    "pholead_pt_raw",
    "pholead_pt",
    "pholead_eta",

    # --------------------------------------------------------
    # Subleading photon
    # --------------------------------------------------------

    "phosublead_pt_raw",
    "phosublead_pt",
    "phosublead_eta",

    # --------------------------------------------------------
    # Leading jet
    # --------------------------------------------------------

    "first_jet_pt_raw",
    "first_jet_pt",
    "first_jet_eta",

    # --------------------------------------------------------
    # Subleading jet
    # --------------------------------------------------------

    "second_jet_pt_raw",
    "second_jet_pt",
    "second_jet_eta",
]


print(
    missing[
        pt_eta_columns
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)


# ============================================================
# Print selection details
# ============================================================

print()
print("=" * 180)
print("OBJECT SELECTION DETAILS")
print("=" * 180)

selection_columns = [

    "run",
    "lumi",
    "event",

    # --------------------------------------------------------
    # Leading photon
    # --------------------------------------------------------

    "pholead_pt_raw",
    "pholead_eta",
    "leadPho_pt_pass",
    "leadPho_eta_pass",
    "leadPho_raw_pass",

    # --------------------------------------------------------
    # Subleading photon
    # --------------------------------------------------------

    "phosublead_pt_raw",
    "phosublead_eta",
    "subleadPho_pt_pass",
    "subleadPho_eta_pass",
    "subleadPho_raw_pass",

    # --------------------------------------------------------
    # Electron
    # --------------------------------------------------------

    "electron_pt",
    "electron_eta",
    "electron_pt_pass",
    "electron_eta_pass",
    "electron_pass",

    # --------------------------------------------------------
    # Muon
    # --------------------------------------------------------

    "muon_pt",
    "muon_eta",
    "muon_pt_pass",
    "muon_eta_pass",
    "muon_pass",

    # --------------------------------------------------------
    # Leading jet
    # --------------------------------------------------------

    "first_jet_pt_raw",
    "first_jet_eta",
    "firstJet_pt_pass",
    "firstJet_eta_pass",
    "firstJet_raw_pass",

    # --------------------------------------------------------
    # Subleading jet
    # --------------------------------------------------------

    "second_jet_pt_raw",
    "second_jet_eta",
    "secondJet_pt_pass",
    "secondJet_eta_pass",
    "secondJet_raw_pass",

    # --------------------------------------------------------
    # Event counts
    # --------------------------------------------------------

    "n_bJets",
    "Njets",
]


print(
    missing[
        selection_columns
    ].to_string(
        index=False
    )
)


# ============================================================
# Summary
# ============================================================

print()
print("=" * 100)
print("SELECTION SUMMARY")
print("=" * 100)


print(
    f"Leading photon fails pT : "
    f"{(~missing['leadPho_pt_pass']).sum()}"
)

print(
    f"Leading photon fails eta: "
    f"{(~missing['leadPho_eta_pass']).sum()}"
)

print(
    f"Subleading photon fails pT : "
    f"{(~missing['subleadPho_pt_pass']).sum()}"
)

print(
    f"Subleading photon fails eta: "
    f"{(~missing['subleadPho_eta_pass']).sum()}"
)

print(
    f"Events failing combined photon selection: "
    f"{(~missing['photons_raw_pass']).sum()}"
)

print()

print(
    f"Electron fails pT : "
    f"{(~missing['electron_pt_pass']).sum()}"
)

print(
    f"Electron fails eta: "
    f"{(~missing['electron_eta_pass']).sum()}"
)

print(
    f"Muon fails pT : "
    f"{(~missing['muon_pt_pass']).sum()}"
)

print(
    f"Muon fails eta: "
    f"{(~missing['muon_eta_pass']).sum()}"
)

print()

print(
    f"Leading jet fails pT : "
    f"{(~missing['firstJet_pt_pass']).sum()}"
)

print(
    f"Leading jet fails eta: "
    f"{(~missing['firstJet_eta_pass']).sum()}"
)

print(
    f"Subleading jet fails pT : "
    f"{(~missing['secondJet_pt_pass']).sum()}"
)

print(
    f"Subleading jet fails eta: "
    f"{(~missing['secondJet_eta_pass']).sum()}"
)


# ============================================================
# Raw -> corrected migrations
# ============================================================

print()
print("=" * 100)
print("RAW -> CORRECTED SELECTION MIGRATIONS")
print("=" * 100)


lead_photon_migration = (
    (~missing["leadPho_raw_pass"])
    &
    missing["leadPho_corr_pass"]
)

sublead_photon_migration = (
    (~missing["subleadPho_raw_pass"])
    &
    missing["subleadPho_corr_pass"]
)

first_jet_migration = (
    (~missing["firstJet_raw_pass"])
    &
    missing["firstJet_corr_pass"]
)

second_jet_migration = (
    (~missing["secondJet_raw_pass"])
    &
    missing["secondJet_corr_pass"]
)


print(
    "Leading photon:"
)

print(
    "  raw FAIL -> corrected PASS = "
    f"{lead_photon_migration.sum()}"
)

print(
    "Subleading photon:"
)

print(
    "  raw FAIL -> corrected PASS = "
    f"{sublead_photon_migration.sum()}"
)

print(
    "Leading jet:"
)

print(
    "  raw FAIL -> corrected PASS = "
    f"{first_jet_migration.sum()}"
)

print(
    "Subleading jet:"
)

print(
    "  raw FAIL -> corrected PASS = "
    f"{second_jet_migration.sum()}"
)


# ============================================================
# Print events failing only because of pT
# ============================================================

print()
print("=" * 100)
print("PHOTON EVENTS FAILING pT BUT PASSING eta")
print("=" * 100)

lead_pt_only = (
    (~missing["leadPho_pt_pass"])
    &
    missing["leadPho_eta_pass"]
)

sublead_pt_only = (
    (~missing["subleadPho_pt_pass"])
    &
    missing["subleadPho_eta_pass"]
)

print(
    "Leading photon:"
)

print(
    missing.loc[
        lead_pt_only,
        [
            "run",
            "lumi",
            "event",
            "pholead_pt_raw",
            "pholead_pt",
            "pholead_eta",
        ]
    ].to_string(
        index=False
    )
)

print()

print(
    "Subleading photon:"
)

print(
    missing.loc[
        sublead_pt_only,
        [
            "run",
            "lumi",
            "event",
            "phosublead_pt_raw",
            "phosublead_pt",
            "phosublead_eta",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# Print events failing only because of eta
# ============================================================

print()
print("=" * 100)
print("PHOTON EVENTS FAILING eta BUT PASSING pT")
print("=" * 100)

lead_eta_only = (
    missing["leadPho_pt_pass"]
    &
    (~missing["leadPho_eta_pass"])
)

sublead_eta_only = (
    missing["subleadPho_pt_pass"]
    &
    (~missing["subleadPho_eta_pass"])
)

print(
    "Leading photon:"
)

print(
    missing.loc[
        lead_eta_only,
        [
            "run",
            "lumi",
            "event",
            "pholead_pt_raw",
            "pholead_pt",
            "pholead_eta",
        ]
    ].to_string(
        index=False
    )
)

print()

print(
    "Subleading photon:"
)

print(
    missing.loc[
        sublead_eta_only,
        [
            "run",
            "lumi",
            "event",
            "phosublead_pt_raw",
            "phosublead_pt",
            "phosublead_eta",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# Save complete diagnostic table
# ============================================================

output_columns = list(dict.fromkeys(
    pt_eta_columns
    + selection_columns
))

missing[
    output_columns
].to_csv(
    "raw_only_events_pt_eta_selection.csv",
    index=False
)


print()
print("=" * 100)
print(
    "Saved: raw_only_events_pt_eta_selection.csv"
)
print("=" * 100)
