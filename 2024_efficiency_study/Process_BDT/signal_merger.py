import uproot
import awkward as ak
import glob

import awkward as ak

def add_features(arr):
    """
    Add derived features to an Awkward Array.
    All input branches must already exist in arr.
    """

    # Example: ratio of two existing features
    arr["pt_ratio"] = ak.where(arr["pholead_pt"] != 0,arr["phosublead_pt"] / arr["pholead_pt"],-999.0)
    # Add more derived features here

    return arr

all_arrays = []

total_sumw = 0.0

# first pass
for fname in sorted(glob.glob("Signal/output_WHM*root")):

    mass = int(fname.split("_M")[1].split("_")[0])

    f = uproot.open(fname)

    tree = f[f"DiphotonTree/WHM{mass}Y2024_{mass}_13TeV_Resolved"]

    arr = tree.arrays(library="ak")

    arr = add_features(arr)

    total_sumw += ak.sum(arr["weight"])

    # arr["mass_point"] = mass

    all_arrays.append(arr)

# second pass
normed = []

for arr in all_arrays:

    arr["evt_wgt"] = arr["weight"] / total_sumw

    normed.append(arr)

merged = ak.concatenate(normed)

print("Final sum =", ak.sum(merged["evt_wgt"]))

with uproot.recreate("merged_signal_with_allCorr_uniform_mass_weight.root") as fout:
    fout["DiphotonTree"] = merged


# all_arrays = []

# total_sumws = []

# # first pass
# for fname in sorted(glob.glob("Signal/output_WHM*root")):

#     mass = int(fname.split("_M")[1].split("_")[0])

#     f = uproot.open(fname)

#     tree = f[f"DiphotonTree/WHM{mass}Y2024_{mass}_13TeV_Resolved"]

#     arr = tree.arrays(library="ak")

#     arr = add_features(arr)

#     total_sumw = ak.sum(arr["weight"])

#     total_sumws.append(total_sumw)

#     # arr["mass_point"] = mass

#     all_arrays.append(arr)

# # second pass
# normed = []

# for arr, sumw in zip(all_arrays, total_sumws):

#     arr["evt_wgt"] = arr["weight"] / (len(total_sumws)*sumw)

#     normed.append(arr)

# merged = ak.concatenate(normed)

# print("Final sum =", ak.sum(merged["evt_wgt"]))

# with uproot.recreate("merged_signal_with_allCorr.root") as fout:
#     fout["DiphotonTree"] = merged