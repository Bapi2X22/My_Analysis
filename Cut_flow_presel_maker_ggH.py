import os
import awkward as ak
import pandas as pd
from tqdm import tqdm

# Base path (adjust if needed)
full_files = "/eos/user/b/bbapi/My_Analysis/NTuples_ggH/"

# Ordered list of cut directories
cut_dirs = [
    "without_presel_trigger",
    "without_presel",
    "lead_pt_cut",
    "sublead_pt_cut",
    "eta_cut",
    "photon_mva",
    "photon_pixel_seed",
    "Jet_pt_eta_cut",
    "btagged"
]

# Reference directory to discover all WH mass–year folders
ref_dir = os.path.join(full_files, "without_presel")

# Find all directories matching WH_Mxx-RunIISummer20*
mass_years = sorted(
    [d for d in os.listdir(ref_dir) if d.startswith("ggH_M") and "Run3Summer20" in d]
)

cutflow_data = []

for my in tqdm(mass_years, desc="Processing mass–year samples", unit="sample"):
    # Example: WH_M60-RunIISummer20UL16NanoAODAPVv2
    try:
        # Extract mass
        mass = my.split("-")[0].replace("ggH_", "")  # e.g., M60

        # Extract year/version (UL16, UL16APV, UL17, UL18)
        if "2022" in my:
            year = "2022"
        else:
            year = "Unknown"

        row = {"Year": year, "Mass": mass}

        # Loop through cut stages
        for cut in cut_dirs:
            path = os.path.join(full_files, cut, my, "nominal")

            if not os.path.exists(path):
                row[cut] = None
                continue

            try:
                arr = ak.from_parquet(path)
                row[cut] = len(arr)
                del arr  # free memory
            except Exception as e:
                print(f"Error loading {path}: {e}")
                row[cut] = None

        cutflow_data.append(row)

    except Exception as e:
        print(f"Error parsing folder name '{my}': {e}")
        continue

# Build dataframe
df = pd.DataFrame(cutflow_data)

# Sort by year (UL16, UL16APV, UL17, UL18) and numeric mass
df["Mass_num"] = df["Mass"].str.extract(r"M(\d+)").astype(float)
df["Year"] = pd.Categorical(df["Year"], categories=["2022"], ordered=True)
df = df.sort_values(["Year", "Mass_num"]).drop(columns=["Mass_num"]).reset_index(drop=True)

# Reorder columns: Year, Mass, then cuts
cols = ["Year", "Mass"] + [c for c in cut_dirs if c in df.columns]
df = df[cols]

# Save to CSV
out_csv = "cutflow_table_csvv2_ggH.csv"
df.to_csv(out_csv, index=False)

print(f"Cutflow table created successfully: {out_csv}\n")
print(df)

