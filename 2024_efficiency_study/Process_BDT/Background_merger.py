import uproot
import awkward as ak
import glob

# cross sections in pb
xsecs = {
    "TTG1Jets": 4.634,
    "TTto2L2Nu": 98.04,
    "TTtoLNu2Q": 405.87,
    "WGtoLNuG": 671.5,
}

process_map = {
    "TTG1Jets": 0,
    "TTto2L2Nu": 1,
    "TTtoLNu2Q": 2,
    "WGtoLNuG": 3
}

all_arrays = []
total_sumw = 0.0

files = sorted(glob.glob("Background/output_*root"))

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

    # get nominal CAT1 tree automatically
    tree_key = [k for k in f.keys(recursive=True)
                if ("CAT1" in k)
                and ("sigma" not in k)][0]

    tree = f[tree_key]

    arr = tree.arrays(library="ak")

    phys_wgt = arr["weight"] * xsec

    total_sumw += ak.sum(phys_wgt)

    arr["phys_wgt"] = phys_wgt
    arr["process_id"] = process_map[process]

    all_arrays.append(arr)

# ---------------- second pass ----------------
normed_arrays = []

for arr in all_arrays:

    arr["evt_wgt"] = arr["phys_wgt"] / total_sumw

    normed_arrays.append(arr)

merged = ak.concatenate(normed_arrays)

print("Final sum =", ak.sum(merged["evt_wgt"]))

with uproot.recreate("merged_bkg.root") as fout:
    fout["DiphotonTree"] = merged
