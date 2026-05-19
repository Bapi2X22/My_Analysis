import awkward as ak
import pandas as pd

base_dir = "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/NTuples_BKG_2024_HDNA_presel_final_good_selections/merged/"

Mass_points = ["TTG1Jets_24SummerRun3", "TTtoLNu2Q_24SummerRun3", "TTto2L2Nu_24SummerRun3"]
categories = ["CAT1", "CAT2", "CAT3"]

rows = []

for mass in Mass_points:
    row = {"Background": mass}

    for cat in categories:
        inside_dir = f"{base_dir}{mass}/nominal/diphoton/"
        file = ak.from_parquet(inside_dir + f"{cat}_merged.parquet")

        eff = ak.sum(file.weight) * 100
        row[cat] = float(eff)

    rows.append(row)

# Create DataFrame
df = pd.DataFrame(rows)

# Save to CSV
output_file = "efficiency_BKG_2024_table_all_cat.csv"
df.to_csv(output_file, index=False, float_format="%.4f")

print(f"Saved CSV to: {output_file}")
