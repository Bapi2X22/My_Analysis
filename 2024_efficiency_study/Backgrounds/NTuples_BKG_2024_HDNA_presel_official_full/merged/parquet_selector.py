import awkward as ak
import glob
import os

files = glob.glob(
    "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/NTuples_BKG_2024_HDNA_presel_official_full/merged/*/nominal/CAT1_merged.parquet"
)

for f in files:
    size = os.path.getsize(f)
    if size < 100:
        print("CORRUPTED:", f, size)

for f in files:
    events = ak.from_parquet(f)
    n_before = len(events)

    events = events[(events.mass >= 20) & (events.mass <= 70)]

    n_after = len(events)

    print(f"{f}: {n_before} -> {n_after}")

    if n_after > 0:
        ak.to_parquet(events, f, compression=None)
    else:
        print("Skipping empty file")
    check = ak.from_parquet(f)
    print(ak.min(check.mass), ak.max(check.mass))
