import numpy as np
import pandas as pd
import awkward as ak
import uproot

filename = "merged_signal_allCat.root"

weight_branch = "evt_wgt"
mass_branch = "mass_point"

all_stats = {}

with uproot.open(filename) as f:

    tree_names = [
        key.split(";")[0]
        for key in f["DiphotonTree"].keys()
    ]

    print("Available trees:", tree_names)

    for tree_name in tree_names:

        print(f"\nProcessing {tree_name}")

        tree = f[f"DiphotonTree/{tree_name}"]

        dataset = tree.arrays(library="ak")
        df_sig = ak.to_dataframe(dataset)

        # # Group by mass point
        # stats = df_sig.groupby(mass_branch).agg(
        #     Events=(mass_branch, "size"),
        #     SumWeights=(weight_branch, "sum"),
        # )

        # # Event fraction within each category
        # stats["EventFraction (%)"] = (
        #     100 * stats["Events"] / stats["Events"].sum()
        # )

        # # Weight fraction within each category
        # total_weight = stats["SumWeights"].sum()

        # stats["WeightFraction (%)"] = (
        #     100 * stats["SumWeights"] / total_weight
        #     if total_weight != 0 else np.nan
        # )
        # Assign events to nominal mass points
        df_sig["Mass Point"] = df_sig["mass_point"].round().astype(int)

        # Group by nominal mass point
        stats = df_sig.groupby("Mass Point").agg(
            Events=("Mass Point", "size"),
            SumWeights=(weight_branch, "sum"),
        )

        # Event fraction
        stats["EventPercentage (%)"] = (
            100 * stats["Events"] / stats["Events"].sum()
        )

        # Weight fraction
        total_weight = stats["SumWeights"].sum()

        stats["WeightPercentage (%)"] = (
            100 * stats["SumWeights"] / total_weight
            if total_weight != 0 else np.nan
        )

        stats.index.name = "Mass Point"

        # Round all floating-point columns to 2 decimal places
        stats = stats.round(2)

        all_stats[tree_name] = stats

        print(f"\nSignal statistics: {tree_name}")
        print(stats.to_string())

        print("\nTotal events:", stats["Events"].sum())
        print(f"Total weight: {stats['SumWeights'].sum():.2f}")

        stats.to_csv(f"signal_statistics_{tree_name}.csv", float_format="%.2f")


# Combined CSV
combined_stats = pd.concat(
    all_stats,
    names=["Category", "Mass Point"]
)

combined_stats.to_csv("signal_statistics_all_categories.csv")

print("\nSaved signal statistics.")