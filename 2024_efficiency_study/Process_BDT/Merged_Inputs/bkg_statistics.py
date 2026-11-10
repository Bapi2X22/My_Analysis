import numpy as np
import pandas as pd
import awkward as ak
import uproot

filename = "merged_bkg_allCat.root"

process_map = {
    "WGtoLNuG": 0,
    "WtoENu0J": 1,
    "WtoENu2J": 2,
    "WtoMuNu2J": 3,
    "DYto2Mu50": 4,
    "DYto2E50": 5,
    "DYGto2LG4": 6,
    "DYGto2LG50": 7,
    "TTG1Jets": 8,
    "TTto2L2Nu": 9,
    "TTtoLNu2Q": 10,
}

process_id_to_name = {v: k for k, v in process_map.items()}

weight_branch = "evt_wgt"

# --------------------------------------------------
# Get all category trees
# --------------------------------------------------

with uproot.open(filename) as f:

    tree_names = [
        key.split(";")[0]
        for key in f["DiphotonTree"].keys()
    ]

    print("Available trees:", tree_names)

    all_stats = {}

    # --------------------------------------------------
    # Loop over categories
    # --------------------------------------------------

    for tree_name in tree_names:

        print(f"\nProcessing {tree_name}")

        tree = f[f"DiphotonTree/{tree_name}"]

        dataset = tree.arrays(library="ak")

        df_bkg = ak.to_dataframe(dataset)

        # Process identification
        df_bkg["Process"] = (
            df_bkg["process_id"].map(process_id_to_name)
        )

        # Check unknown process IDs
        if df_bkg["Process"].isna().any():
            print("WARNING: Unknown process IDs:")
            print(
                df_bkg.loc[
                    df_bkg["Process"].isna(), "process_id"
                ].unique()
            )

        # Statistics
        stats = df_bkg.groupby("Process").agg(
            Events=("Process", "size"),
        )

        stats["EventPercentage (%)"] = (
            100 * stats["Events"] / stats["Events"].sum()
        )

        if weight_branch in df_bkg.columns:

            stats["SumWeights"] = (
                df_bkg.groupby("Process")[weight_branch].sum()
            )

            total_weight = stats["SumWeights"].sum()

            stats["WeightPercentage (%)"] = (
                100 * stats["SumWeights"] / total_weight
                if total_weight != 0 else np.nan
            )

        # Preserve process ordering
        stats = stats.reindex(process_map.keys(), fill_value=0)

        # Round floating-point columns to 2 decimal places
        stats = stats.round(2)

        all_stats[tree_name] = stats

        print(f"\nBackground composition: {tree_name}")
        print(stats.to_string())

        print("\nTotal events:", len(df_bkg))

        if weight_branch in df_bkg.columns:
            print(f"Total weight: {df_bkg[weight_branch].sum():.2f}")

# --------------------------------------------------
# Save separate CSV files for each category
# --------------------------------------------------

for tree_name, stats in all_stats.items():

    stats.to_csv(f"background_statistics_{tree_name}.csv")


# --------------------------------------------------
# Save combined CSV
# --------------------------------------------------

combined_stats = pd.concat(
    all_stats,
    names=["Category", "Process"]
)

combined_stats.to_csv("background_statistics_all_categories_DS11.csv")

print("\nSaved all background statistics.")