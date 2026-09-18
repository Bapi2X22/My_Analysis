# import uproot
# import awkward as ak
# import glob

# # cross sections in pb
# xsecs = {
#     "TTG1Jets": 4.634,
#     "TTto2L2Nu": 98.04,
#     "TTtoLNu2Q": 405.87,
#     "WGtoLNuG": 671.5,
#     "DYto2Mu50": 2230.0,
#     "DYto2E50": 2244.0,
# }

# process_map = {
#     "TTG1Jets": 0,
#     "TTto2L2Nu": 1,
#     "TTtoLNu2Q": 2,
#     "WGtoLNuG": 3,
#     "DYto2Mu50": 4,
#     "DYto2E50": 5,
# }

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

#     # get nominal CAT1 tree automatically
#     tree_key = [k for k in f.keys(recursive=True)
#                 if ("CAT1" in k)
#                 and ("sigma" not in k)][0]


#     tree = f[tree_key]

#     arr = tree.arrays(library="ak")

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

# with uproot.recreate("merged_bkg.root") as fout:
#     fout["DiphotonTree"] = merged



import uproot
import awkward as ak
import numpy as np
import glob

# ============================================================
# Cross sections in pb
# ============================================================
xsecs = {
    "TTG1Jets":   4.634,
    "TTto2L2Nu":  98.04,
    "TTtoLNu2Q":  405.87,
    "WGtoLNuG":   671.5,
    "DYto2Mu50":  2124.08,
    "DYto2E50":   2124.08,
    "DYto2Mu10":  21190.0,
    "DYto2E10":   21140.0

}

process_map = {
    "TTG1Jets":   0,
    "TTto2L2Nu": 1,
    "TTtoLNu2Q": 2,
    "WGtoLNuG":  3,
    "DYto2Mu50": 4,
    "DYto2E50":  5,
    "DYto2Mu10": 6,
    "DYto2E10":  7
}

# ============================================================
# Mass points
# ============================================================
mass_points = np.array(
    [12, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
    dtype=np.int32
)

rng = np.random.default_rng(12345)

# ============================================================
# Input files
# ============================================================
files = sorted(glob.glob("Background/output_*root"))

print("Input files:")
for f in files:
    print("  ", f)

all_arrays = []
total_sumw = 0.0

# ============================================================
# First pass:
#   - identify process
#   - calculate physical weight
#   - calculate total normalization
# ============================================================
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

    print(f"\nProcessing: {fname}")
    print(f"  Process: {process}")
    print(f"  Xsec:    {xsec} pb")

    f = uproot.open(fname)

    print("  Keys:", f.keys())

    # Get nominal CAT1 tree automatically
    tree_keys = [
        k for k in f.keys(recursive=True)
        if ("CAT1" in k) and ("sigma" not in k)
    ]

    if not tree_keys:
        print(f"  No CAT1 tree found, skipping {fname}")
        continue

    tree_key = tree_keys[0]
    tree = f[tree_key]

    arr = tree.arrays(library="ak")

    # Physical weight
    phys_wgt = arr["weight"] * xsec

    total_sumw += ak.sum(phys_wgt)

    arr["phys_wgt"] = phys_wgt
    arr["process_id"] = process_map[process]

    all_arrays.append(arr)

# ============================================================
# Second pass:
#   - normalize weights
#   - assign random mass point
# ============================================================
normed_arrays = []

for arr in all_arrays:

    arr["evt_wgt"] = arr["phys_wgt"] / total_sumw

    # Randomly assign one of the A mass points to every event
    arr["mass_point"] = rng.choice(
        mass_points,
        size=len(arr),
        replace=True
    ).astype(np.int32)

    normed_arrays.append(arr)

# ============================================================
# Merge
# ============================================================
merged = ak.concatenate(normed_arrays)

print("\n========================================")
print("Final checks")
print("========================================")
print("Number of events =", len(merged))
print("Total sumw       =", total_sumw)
print("Final evt_wgt    =", ak.sum(merged["evt_wgt"]))

# Mass-point distribution
print("\nMass point distribution:")
unique_mass, counts = np.unique(
    ak.to_numpy(merged["mass_point"]),
    return_counts=True
)

for mass, count in zip(unique_mass, counts):
    print(f"  {mass:2d} GeV : {count}")

# ============================================================
# Write final file
# ============================================================
output_file = "merged_bkg_withMET.root"

with uproot.recreate(output_file) as fout:
    fout["DiphotonTree"] = merged

print(f"\nDone: {output_file}")