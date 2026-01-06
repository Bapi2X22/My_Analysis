import os
import awkward as ak
import pandas as pd
from tqdm import tqdm
import argparse

# ------------------------------------------------------
# Argument parser
# ------------------------------------------------------
parser = argparse.ArgumentParser(description="Generate cutflow table with optional efficiencies.")
parser.add_argument("--eff", action="store_true", help="Add efficiency columns (as percentages).")
parser.add_argument("--base", default="/eos/user/b/bbapi/My_Analysis/NTuples_ggH_v15/",
                    help="Base path containing the cut directories.")
parser.add_argument("--output", default="cutflow_table_csvv2_ggH.csv",
                    help="Output CSV file name.")
args = parser.parse_args()

# ------------------------------------------------------
# Configuration
# ------------------------------------------------------
full_files = args.base
add_eff = args.eff
out_csv = args.output

# cut_dirs = [
#     "without_presel_trigger",
#     "without_presel",
#     "lead_pt_cut",
#     "sublead_pt_cut",
#     "eta_cut",
#     "photon_mva",
#     "photon_pixel_seed",
#     "Jet_pt_eta_cut",
#     "btagged"
# ]

cut_dirs = [
    "btagged"
]

# Manual total events for each mass
manual_totals = {
    "M15": 95231,
    "M20": 94709,
    "M30": 95224,
    "M45": 94346,
    "M60": 95411
}

# ------------------------------------------------------
# Discover and process
# ------------------------------------------------------
# ref_dir = os.path.join(full_files, "without_presel")
ref_dir = os.path.join(full_files, "btagged")
mass_years = sorted(
    [d for d in os.listdir(ref_dir) if d.startswith("ggH_M") and "Run3Summer20" in d]
)

cutflow_data = []

for my in tqdm(mass_years, desc="Processing mass–year samples", unit="sample"):
    try:
        # Extract clean mass label like "M60"
        mass = my.split("-")[0].replace("ggH_", "").strip().replace("_", "")
        year = "2022" if "2022" in my else "Unknown"
        row = {"Year": year, "Mass": mass}

        for cut in cut_dirs:
            path = os.path.join(full_files, cut, my, "nominal/")

            if not os.path.exists(path):
                row[cut] = None
                continue

            # try:
            #     arr = ak.from_parquet(path+"/*.parquet")
            #     row[cut] = len(arr)
            #     del arr
            # except Exception as e:
            #     print(f"Error loading {path}: {e}")
            #     row[cut] = None
            try:
                # find ANY parquet file
                import glob
                files = glob.glob(os.path.join(path, "*parquet"))

                if not files:
                    raise FileNotFoundError(f"No .parquet files in {path}")

                # load the (single) file
                arr = ak.from_parquet(files[0])
                row[cut] = len(arr)
                del arr

            except Exception as e:
                print(f"Error loading {path}: {e}")
                row[cut] = None


        cutflow_data.append(row)

    except Exception as e:
        print(f"Error parsing folder name '{my}': {e}")
        continue

# ------------------------------------------------------
# Build DataFrame
# ------------------------------------------------------
df = pd.DataFrame(cutflow_data)
df["Mass_num"] = df["Mass"].str.extract(r"M(\d+)").astype(float)
df["Year"] = pd.Categorical(df["Year"], categories=["2022"], ordered=True)
df = df.sort_values(["Year", "Mass_num"]).drop(columns=["Mass_num"]).reset_index(drop=True)

# ------------------------------------------------------
# Compute efficiencies inline (as percentages)
# ------------------------------------------------------
# if add_eff:
#     new_cols = ["Year", "Mass"]
#     eff_cols_data = {}

#     for cut in cut_dirs:
#         new_cols.append(cut)
#         eff_col = f"{cut}_eff"
#         new_cols.append(eff_col)
#         eff_cols_data[eff_col] = []

if add_eff:
    new_cols = ["Year", "Mass", "Total_events"]
    eff_cols_data = {}

    for cut in cut_dirs:
        new_cols.append(cut)
        eff_col = f"{cut}_eff"
        new_cols.append(eff_col)
        eff_cols_data[eff_col] = []

    # Fill manual_total column
    df["Total_events"] = df["Mass"].str.strip().map(manual_totals)


    for _, row in df.iterrows():
        mass = row["Mass"].strip()
        total_events = manual_totals.get(mass, None)
        if total_events is None:
            print(f"Warning: No total events provided for {mass}. Efficiency = NaN.")

        for cut in cut_dirs:
            if total_events and row[cut] is not None:
                eff_val = (row[cut] / total_events) * 100  # percentage
                eff_val = round(eff_val, 2)  # two decimal places
            else:
                eff_val = None
            eff_cols_data[f"{cut}_eff"].append(eff_val)

    for eff_col, eff_vals in eff_cols_data.items():
        df[eff_col] = eff_vals

    df = df[new_cols]

# ------------------------------------------------------
# Save and print
# ------------------------------------------------------
df.to_csv(out_csv, index=False)
print(f"Cutflow table created successfully: {out_csv}\n")

# Show table with percentages formatted nicely
pd.set_option('display.float_format', '{:.2f}'.format)
print(df)

