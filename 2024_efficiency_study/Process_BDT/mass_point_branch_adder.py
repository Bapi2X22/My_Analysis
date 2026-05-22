import uproot
import awkward as ak
import numpy as np

# Read existing file
f = uproot.open("merged_bkg.root")
arr = f["DiphotonTree"].arrays(library="ak")

# Random mass points
mass_points = np.array([12, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60], dtype=np.int32)

rng = np.random.default_rng(12345)

arr["mass_point"] = rng.choice(
    mass_points,
    size=len(arr),
    replace=True
).astype(np.int32)

# Write new file
with uproot.recreate("merged_bkg_withMass.root") as fout:
    fout["DiphotonTree"] = arr

print("Done")
