# import glob
# import pyarrow.parquet as pq
# import awkward as ak

# base_dir = "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/NTuples_WH_2024_HDNA_presel_final_good_selections/merged/"

# Mass_points = ["12", "15", "20", "25", "30", "35", "40", "45", "50", "55", "60"]
# #Mass_points = ["30"]

# categories = ["CAT1", "CAT2", "CAT3"]


# for mass in Mass_points:
#     for cat in categories:
#         inside_dir = f"{base_dir}WH-2024M{mass}/nominal/diphoton/"
#         file = ak.from_parquet(inside_dir+f"{cat}_merged.parquet")
#         weight = file.weight

#         eff = ak.sum(weight)*100


import awkward as ak
import pandas as pd

base_dir = "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/NTuples_WH_2024_HDNA_presel_final_good_selections/merged/"

Mass_points = ["12", "15", "20", "25", "30", "35", "40", "45", "50", "55", "60"]
categories = ["CAT1", "CAT2", "CAT3"]

rows = []

for mass in Mass_points:
    row = {"Mass": mass}

    for cat in categories:
        inside_dir = f"{base_dir}WH-2024M{mass}/nominal/diphoton/"
        file = ak.from_parquet(inside_dir + f"{cat}_merged.parquet")

        eff = ak.sum(file.weight) * 100
        row[cat] = float(eff)

    rows.append(row)

# Create DataFrame
df = pd.DataFrame(rows)

# Save to CSV
output_file = "efficiency_WH_2024_table_all_cat.csv"
df.to_csv(output_file, index=False, float_format="%.2f")

print(f"Saved CSV to: {output_file}")
