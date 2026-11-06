import uproot
import awkward as ak
import numpy as np
import glob
import argparse



parser = argparse.ArgumentParser(
    description=(
        "Merge signal ROOT files of various mass points for training BDT"
    )
)

parser.add_argument(
    "-o", "--output",
    required=True,
    help="Output ROOT file name",
)

args = parser.parse_args()

def add_features(arr):
    """
    Add derived features to an Awkward Array.
    All input branches must already exist in arr.
    """

    # Example: ratio of two existing features
    arr["pt_ratio"] = ak.where(arr["pholead_pt"] != 0,arr["phosublead_pt"] / arr["pholead_pt"],-999.0)
    arr["energy_resol_lead"] = ak.where(arr["pholead_energyRaw"] != 0,arr["pholead_energyErr"] / arr["pholead_energyRaw"],-999.0)
    arr["energy_resol_sublead"] = ak.where(arr["phosublead_energyRaw"] != 0,arr["phosublead_energyErr"] / arr["phosublead_energyRaw"],-999.0)
    # Add more derived features here

    return arr

# cross sections in pb
# ============================================================
# Cross sections in pb
# ============================================================
xsecs = {
    "WGtoLNuG": 671.5,
    "WtoENu0J": 55850,
    "WtoENu2J": 3474,
    "WtoMuNu2J": 3490,
    "DYto2Mu50": 2124.08,
    "DYto2E50": 2124.08,
    "DYGto2LG4": 88.13,
    "DYGto2LG50": 126.7,
    "TTG1Jets": 4.634,
    "TTto2L2Nu": 98.04,
    "TTtoLNu2Q": 405.87

}

process_map = {
    "WGtoLNuG": 0,
    "WtoENu0J": 1,
    "WtoENu2J": 2,
    "WtoMuNu2J": 3,
    "DYto2Mu50": 4,
    "DYto2E50":  5,
    "DYGto2LG4": 6,
    "DYGto2LG50": 7,
    "TTG1Jets": 8,
    "TTto2L2Nu": 9,
    "TTtoLNu2Q": 10
}

# all_arrays = []
# total_sumw = 0.0

# files = sorted(glob.glob("Background/output_*root"))

# print(files)

# # ---------------- first pass ----------------
# for fname in files:

#     process = None

#     for key in xsecs:
#         if key in fname:
#             process = key
#             break

#     if process is None:
#         print(f"Skipping {fname}")
#         continue

#     xsec = xsecs[process]

#     f = uproot.open(fname)

#     print(f.keys())

#     # Get nominal CAT1 tree automatically
#     tree_keys = [
#         k for k in f.keys(recursive=True)
#         if ("Resolved" in k) and ("sigma" not in k)
#     ]

#     if not tree_keys:
#         print(f"  No Resolved tree found, skipping {fname}")
#         continue

#     tree_key = tree_keys[0]
#     tree = f[tree_key]

#     arr = tree.arrays(library="ak")

#     arr = add_features(arr)

#     phys_wgt = arr["weight"] * xsec

#     total_sumw += ak.sum(phys_wgt)

#     arr["phys_wgt"] = phys_wgt
#     arr["process_id"] = process_map[process]

#     all_arrays.append(arr)

# # ---------------- second pass ----------------
# normed_arrays = []

# for arr in all_arrays:

#     arr["evt_wgt"] = arr["phys_wgt"] / total_sumw

#     normed_arrays.append(arr)

# merged = ak.concatenate(normed_arrays)

# print("Final sum =", ak.sum(merged["evt_wgt"]))

# with uproot.recreate(args.output) as fout:
#     fout["DiphotonTree"] = merged


# Replace with your actual tree names
tree_names = [
    "Resolved",
    "Inclusive",
    "1-bJet"
]

all_arrays = {name: [] for name in tree_names}
total_sumws = {name: 0.0 for name in tree_names}

files = sorted(glob.glob("Background/output_*root"))

print(files)

# ---------------- first pass ----------------

for fname in files:

    process = None

    for key in xsecs:
        if key in fname:
            process = key
            break

    if process is None:
        print(f"Skipping {fname}")
        continue

    xsec = xsecs[process]

    f = uproot.open(fname)

    print(f.keys())

    # Find all three nominal trees
    for tree_name in tree_names:

        tree_keys = [
            k for k in f.keys(recursive=True)
            if tree_name in k and "sigma" not in k
        ]

        if not tree_keys:
            print(f"  No {tree_name} tree found in {fname}")
            continue

        tree_key = tree_keys[0]

        tree = f[tree_key]

        arr = tree.arrays(library="ak")

        arr = add_features(arr)

        phys_wgt = arr["weight"] * xsec

        total_sumws[tree_name] += ak.sum(phys_wgt)

        arr["phys_wgt"] = phys_wgt
        arr["process_id"] = process_map[process]

        all_arrays[tree_name].append(arr)


# ---------------- second pass ----------------

merged_arrays = {}

for tree_name in tree_names:

    normed_arrays = []

    for arr in all_arrays[tree_name]:

        arr["evt_wgt"] = arr["phys_wgt"] / total_sumws[tree_name]

        normed_arrays.append(arr)

    if not normed_arrays:
        print(f"No arrays available for {tree_name}")
        continue

    merged_arrays[tree_name] = ak.concatenate(normed_arrays)

    print(
        f"{tree_name}: Final sum =",
        ak.sum(merged_arrays[tree_name]["evt_wgt"])
    )


# ---------------- write output ----------------

with uproot.recreate(args.output) as fout:

    for tree_name, merged in merged_arrays.items():

        fout[f"DiphotonTree/{tree_name}"] = merged


# # ============================================================
# # Mass points
# # ============================================================
# mass_points = np.array(
#     [12, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
#     dtype=np.int32
# )

# rng = np.random.default_rng(22345)

# # ============================================================
# # Input files
# # ============================================================
# files = sorted(glob.glob("Background/output_*root"))

# print("Input files:")
# for f in files:
#     print("  ", f)

# all_arrays = []
# total_sumw = 0.0

# # ============================================================
# # First pass:
# #   - identify process
# #   - calculate physical weight
# #   - calculate total normalization
# # ============================================================
# for fname in files:

#     process = None

#     for key in xsecs:
#         if key in fname:
#             process = key
#             break

#     if process is None:
#         print(f"Skipping {fname}")
#         continue

#     xsec = xsecs[process]

#     print(f"\nProcessing: {fname}")
#     print(f"  Process: {process}")
#     print(f"  Xsec:    {xsec} pb")

#     f = uproot.open(fname)

#     print("  Keys:", f.keys())

#     # Get nominal CAT1 tree automatically
#     tree_keys = [
#         k for k in f.keys(recursive=True)
#         if ("Resolved" in k) and ("sigma" not in k)
#     ]

#     if not tree_keys:
#         print(f"  No Resolved tree found, skipping {fname}")
#         continue

#     tree_key = tree_keys[0]
#     tree = f[tree_key]

#     arr = tree.arrays(library="ak")
#     arr = add_features(arr)

#     # Physical weight
#     phys_wgt = arr["weight"] * xsec

#     total_sumw += ak.sum(phys_wgt)

#     arr["phys_wgt"] = phys_wgt
#     arr["process_id"] = process_map[process]

#     all_arrays.append(arr)

# # ============================================================
# # Second pass:
# #   - normalize weights
# #   - assign random mass point
# # ============================================================
# normed_arrays = []

# for arr in all_arrays:

#     arr["evt_wgt"] = arr["phys_wgt"] / total_sumw

#     # Randomly assign one of the A mass points to every event
#     # arr["mass_point"] = rng.choice(
#     #     mass_points,
#     #     size=len(arr),
#     #     replace=True
#     # ).astype(np.int32)

#     normed_arrays.append(arr)

# # ============================================================
# # Merge
# # ============================================================
# merged = ak.concatenate(normed_arrays)

# print("\n========================================")
# print("Final checks")
# print("========================================")
# print("Number of events =", len(merged))
# print("Total sumw       =", total_sumw)
# print("Final evt_wgt    =", ak.sum(merged["evt_wgt"]))

# # Mass-point distribution
# print("\nMass point distribution:")
# unique_mass, counts = np.unique(
#     ak.to_numpy(merged["mass_point"]),
#     return_counts=True
# )

# for mass, count in zip(unique_mass, counts):
#     print(f"  {mass:.2f} GeV : {count}")

# # ============================================================
# # Write final file
# # ============================================================
# output_file = "merged_bkg_with_allCorr.root"

# with uproot.recreate(output_file) as fout:
#     fout["DiphotonTree"] = merged

# print(f"\nDone: {output_file}")