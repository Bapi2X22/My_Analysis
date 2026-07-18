import awkward as ak
import glob
import os

files = sorted(
    glob.glob(
        "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/NTuples_BKG_2024_HDNA_presel_pBDT_score/merged/*/M*_merged.parquet"
    )
)

print(f"Found {len(files)} parquet files")

for f in files:
    size = os.path.getsize(f)
    if size < 100:
        print(f"CORRUPTED: {f} ({size} bytes)")
        continue

    try:
        events = ak.from_parquet(f)
    except Exception as e:
        print(f"Cannot read {f}: {e}")
        continue

    n_before = len(events)

    events = events[(events.mass >= 10) & (events.mass <= 70)]

    n_after = len(events)

    print(f"{os.path.basename(os.path.dirname(f))}/{os.path.basename(f)}: "
          f"{n_before} -> {n_after}")

    if n_after == 0:
        print("  Skipping empty file")
        continue

    ak.to_parquet(events, f, compression=None)

    # Verify
    check = ak.from_parquet(f)
    print(f"  Mass range: {ak.min(check.mass):.3f} - {ak.max(check.mass):.3f}")
