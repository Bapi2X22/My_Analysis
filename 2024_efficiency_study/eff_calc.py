import os
import glob
import csv

import pyarrow.parquet as pq
import coffea.nanoevents as nanoevents

BASE_TOP = (
    "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/"
    "NTuples_bbgg_2024_modified_presel"
)

OUTPUT_CSV = "efficiency_summary.csv"


def read_sum_genw_presel(parquet_file):
    pf = pq.ParquetFile(parquet_file)
    return float(pf.metadata.metadata[b"sum_genw_presel"].decode())


rows = []

# discover all mass points automatically
mass_dirs = sorted(
    d for d in glob.glob(os.path.join(BASE_TOP, "WH-2024M*"))
    if os.path.isdir(d)
)

print("Found mass points:")
for d in mass_dirs:
    print("  ", os.path.basename(d))

for mass_dir in mass_dirs:
    mass_point = os.path.basename(mass_dir)

    diphoton_dir = os.path.join(mass_dir, "nominal", "diphoton")
    if not os.path.isdir(diphoton_dir):
        print(f"[skip] no diphoton dir for {mass_point}")
        continue

    parquet_files = glob.glob(os.path.join(diphoton_dir, "*.parquet"))
    if not parquet_files:
        print(f"[skip] no parquet files for {mass_point}")
        continue

    print(f"Processing {mass_point} ({len(parquet_files)} files)")

    total_sumw = 0.0
    total_survived = 0

    for pf in parquet_files:
        total_sumw += read_sum_genw_presel(pf)

        events = nanoevents.NanoEventsFactory.from_parquet(pf).events()
        total_survived += len(events)

    eff = round((total_survived / total_sumw) * 100, 2) if total_sumw > 0 else 0.00

    rows.append(
        {
            "mass_point": mass_point,
            "total_events": total_sumw,
            "survived_events": total_survived,
            "efficiency": eff,
        }
    )

# write CSV
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "mass_point",
            "total_events",
            "survived_events",
            "efficiency",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"\nWrote {OUTPUT_CSV}")



