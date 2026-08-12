import glob
import awkward as ak
import json
from collections import defaultdict

f = "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_Data_validation_debug/merged/Data_EGamma-Data-2024E/allData_CAT1_merged.parquet"

run_lumis = defaultdict(set)

events = ak.from_parquet(f)

for run, lumi in zip(
    ak.to_numpy(events.run),
    ak.to_numpy(events.lumi),
):
    run_lumis[int(run)].add(int(lumi))

# Convert to CMS JSON format
output = {}

for run, lumis in run_lumis.items():
    lumis = sorted(lumis)

    ranges = []
    start = prev = lumis[0]

    for lumi in lumis[1:]:
        if lumi == prev + 1:
            prev = lumi
        else:
            ranges.append([start, prev])
            start = prev = lumi

    ranges.append([start, prev])
    output[str(run)] = ranges

# Save JSON
with open("processed_EGamma.json", "w") as f:
    json.dump(output, f, indent=2, sort_keys=True)

print(f"Saved processed_EGamma.json with {len(output)} runs.")
