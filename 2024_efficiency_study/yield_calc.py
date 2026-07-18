import os
import glob
import awkward as ak

base_dir = "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/NTuples_WH_2024_HDNA_presel_latest_with_BDT_score/merged/"

lumi = 109.0  # fb^-1

xsecs = {
    "WH-2024M12": 0.32054,  # pb
    "WH-2024M15": 0.32054,  # pb
    "WH-2024M20": 0.32054,  # pb
    "WH-2024M25": 0.32054,  # pb
    "WH-2024M30": 0.32054,  # pb
    "WH-2024M35": 0.32054,  # pb
    "WH-2024M40": 0.32054,  # pb
    "WH-2024M45": 0.32054,  # pb
    "WH-2024M50": 0.32054,  # pb
    "WH-2024M55": 0.32054,  # pb
    "WH-2024M60": 0.32054,  # pb
}

print(f"{'Process':30s} {'Events':>10s} {'Σweight':>15s} {'Yield':>15s}")
print("-"*75)

for proc, xsec in xsecs.items():

    files = glob.glob(os.path.join(base_dir, proc, "nominal", "*.parquet"))

    if not files:
        print(f"{proc:30s} No parquet files found.")
        continue

    arr = ak.concatenate([ak.from_parquet(f) for f in files])

    sum_weight = ak.sum(arr.weight)
    yield_events = sum_weight * xsec * lumi * 1000.0

    print(f"{proc:30s} {len(arr):10d} {sum_weight:15.6f} {yield_events:15.2f}")
