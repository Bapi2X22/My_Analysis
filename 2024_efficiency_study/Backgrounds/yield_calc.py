import os
import awkward as ak

base_dir = "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/NTuples_BKG_2024_HDNA_presel_pBDT_score_pho10/merged"

lumi = 109.0  # fb^-1

xsecs = {
    "TTG1Jets-24SummerRun3":   4.634,
    "DYto2Mu50-24SummerRun3": 2230.0,
    "DYto2E50-24SummerRun3":  2244.0,
    "TTto2L2Nu-24SummerRun3":   98.04,
    "TTtoLNu2Q-24SummerRun3":  405.87,
    "WGtoLNuG-24SummerRun3":   671.5,
}

# Get the list of merged files from the first process directory
first_proc = next(iter(xsecs))
merged_files = sorted(
    f for f in os.listdir(os.path.join(base_dir, first_proc))
    if f.endswith("_merged.parquet")
)

for merged_file in merged_files:

    print("\n" + "=" * 100)
    print(merged_file)
    print("=" * 100)
    print(f"{'Process':30s} {'Events':>10s} {'Yield':>15s}")

    total_yield = 0.0

    for proc, xsec in xsecs.items():

        f = os.path.join(base_dir, proc, merged_file)

        if not os.path.exists(f):
            continue

        arr = ak.from_parquet(f)

        sum_weight = ak.sum(arr.weight)
        yield_events = sum_weight * xsec * lumi * 1000.0
        total_yield += yield_events

        print(f"{proc:30s} {len(arr):10d} {yield_events:15.2f}")

    print("-" * 60)
    print(f"{'Total background':30s} {'':10s} {total_yield:15.2f}")
