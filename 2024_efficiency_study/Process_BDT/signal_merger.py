import uproot
import awkward as ak
import glob
import argparse
import awkward as ak


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

parser.add_argument(
    "--equalize_weight",
    action="store_true",
    help="Equalize the total event weight contribution from each mass point",
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

# all_arrays = []

# total_sumw = 0.0

# # first pass
# for fname in sorted(glob.glob("Signal/output_WHM*root")):

#     mass = int(fname.split("_M")[1].split("_")[0])

#     f = uproot.open(fname)

#     tree = f[f"DiphotonTree/WHM{mass}Y2024_{mass}_13TeV_Resolved"]

#     arr = tree.arrays(library="ak")

#     arr = add_features(arr)

#     total_sumw += ak.sum(arr["weight"])

#     # arr["mass_point"] = mass

#     all_arrays.append(arr)

# # second pass
# normed = []

# for arr in all_arrays:

#     arr["evt_wgt"] = arr["weight"] / total_sumw

#     normed.append(arr)

# merged = ak.concatenate(normed)

# print("Final sum =", ak.sum(merged["evt_wgt"]))

# with uproot.recreate("merged_signal_with_allCorr_uniform_mass_weight.root") as fout:
#     fout["DiphotonTree"] = merged


# all_arrays = []

# total_sumws = []

# # first pass
# for fname in sorted(glob.glob("Signal/output_WHM*root")):
# # for fname in sorted(glob.glob("Signal/output_WHM55Y2024_M55_13p6TeV_amcatnlo_pythia8.root")):

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

#     if args.equalize_weight:
#         arr["evt_wgt"] = arr["weight"] / (len(total_sumws)*sumw)
#     else:
#         arr["evt_wgt"] = arr["weight"] / (ak.sum(total_sumws))
#     normed.append(arr)

# merged = ak.concatenate(normed)

# print("Final sum =", ak.sum(merged["evt_wgt"]))

# with uproot.recreate(args.output) as fout:
#     fout["DiphotonTree"] = merged


# Define the three trees
tree_names = [
    "Resolved",
    "Inclusive",
    "1-bJet"
]

# Store arrays and sum of weights separately for each tree
all_arrays = {name: [] for name in tree_names}
total_sumws = {name: [] for name in tree_names}

# --------------------------------------------------
# First pass: Read all mass points
# --------------------------------------------------

for fname in sorted(glob.glob("Signal/output_WHM*root")):

    mass = int(fname.split("_M")[1].split("_")[0])

    f = uproot.open(fname)

    for tree_name in tree_names:

        tree_path = f"DiphotonTree/WHM{mass}Y2024_{mass}_13TeV_{tree_name}"

        tree = f[tree_path]

        arr = tree.arrays(library="ak")

        arr = add_features(arr)

        total_sumw = ak.sum(arr["weight"])

        total_sumws[tree_name].append(total_sumw)

        all_arrays[tree_name].append(arr)


# --------------------------------------------------
# Second pass: Normalize and merge separately
# --------------------------------------------------

merged_arrays = {}

for tree_name in tree_names:

    normed = []

    for arr, sumw in zip(
        all_arrays[tree_name],
        total_sumws[tree_name]
    ):

        if args.equalize_weight:
            arr["evt_wgt"] = arr["weight"] / (
                len(total_sumws[tree_name]) * sumw
            )

        else:
            arr["evt_wgt"] = arr["weight"] / ak.sum(
                total_sumws[tree_name]
            )

        normed.append(arr)

    merged_arrays[tree_name] = ak.concatenate(normed)

    print(
        f"{tree_name}: Final sum =",
        ak.sum(merged_arrays[tree_name]["evt_wgt"])
    )


# --------------------------------------------------
# Write all trees into one ROOT file
# --------------------------------------------------

with uproot.recreate(args.output) as fout:

    for tree_name in tree_names:

        fout[f"DiphotonTree/{tree_name}"] = merged_arrays[tree_name]