import uproot
import awkward as ak
import glob

all_arrays = []

total_sumw = 0.0

# first pass
for fname in sorted(glob.glob("Signal/output_WHM*root")):

    mass = int(fname.split("_M")[1].split("_")[0])

    f = uproot.open(fname)

    tree = f[f"DiphotonTree/WHM{mass}Y2024_{mass}_13TeV_CAT1"]

    arr = tree.arrays(library="ak")

    total_sumw += ak.sum(arr["weight"])

    arr["mass_point"] = mass

    all_arrays.append(arr)

# second pass
normed = []

for arr in all_arrays:

    arr["evt_wgt"] = arr["weight"] / total_sumw

    normed.append(arr)

merged = ak.concatenate(normed)

print("Final sum =", ak.sum(merged["evt_wgt"]))

with uproot.recreate("merged_signal.root") as fout:
    fout["DiphotonTree"] = merged
