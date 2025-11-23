import os
import awkward as ak
import pandas as pd
from tqdm import tqdm

# Base path (adjust if needed)
full_files = "/eos/user/b/bbapi/My_Analysis/NTuples_pretest_full_csvv2/"

# Ordered list of cut directories
cut_dirs = [
    "Without_presel_trigger",
    "Without_presel",
    "pT_eta_cut",
    "electron_mva",
    "electron_pf_iso",
    "Muon_pf_iso",
    "one_electron",
    "b_tagged",
    "photon_mva",
    "photon_pixel_seed",
    "dr_ele_pho",
]

# Reference directory to discover all WH mass–year folders
ref_dir = os.path.join(full_files, "Without_presel")

# Find all directories matching WH_Mxx-RunIISummer20*
mass_years = sorted(
    [d for d in os.listdir(ref_dir) if d.startswith("WH_M") and "RunIISummer20" in d]
)

cutflow_data = []

for my in tqdm(mass_years, desc="Processing mass–year samples", unit="sample"):
    # Example: WH_M60-RunIISummer20UL16NanoAODAPVv2
    try:
        # Extract mass
        mass = my.split("-")[0].replace("WH_", "")  # e.g., M60

        # Extract year/version (UL16, UL16APV, UL17, UL18)
        if "APV" in my:
            year = "UL16APV"
        elif "UL16" in my:
            year = "UL16"
        elif "UL17" in my:
            year = "UL17"
        elif "UL18" in my:
            year = "UL18"
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
df["Year"] = pd.Categorical(df["Year"], categories=["UL16", "UL16APV", "UL17", "UL18"], ordered=True)
df = df.sort_values(["Year", "Mass_num"]).drop(columns=["Mass_num"]).reset_index(drop=True)

# Reorder columns: Year, Mass, then cuts
cols = ["Year", "Mass"] + [c for c in cut_dirs if c in df.columns]
df = df[cols]

# Save to CSV
out_csv = "cutflow_table_csvv2_WH.csv"
df.to_csv(out_csv, index=False)

print(f"Cutflow table created successfully: {out_csv}\n")
print(df)

