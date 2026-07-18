import glob
import pyarrow.parquet as pq
import pandas as pd

# ==========================================
# User input
# ==========================================
parquet_dir = "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/tau_test/WH-2024M30/nominal/"

# ==========================================
# Read metadata from all parquet files
# ==========================================
files = sorted(glob.glob(f"{parquet_dir}/*.parquet"))

metadata_sum = {}

for f in files:
    schema = pq.read_schema(f)
    meta = schema.metadata

    if meta is None:
        continue

    for key, value in meta.items():
        key = key.decode()

        try:
            value = int(value.decode())
        except ValueError:
            try:
                value = float(value.decode())
            except ValueError:
                continue

        metadata_sum[key] = metadata_sum.get(key, 0) + value

# ==========================================
# Convert to table
# ==========================================
df = (
    pd.DataFrame(
        {
            "Metadata": metadata_sum.keys(),
            "Total": metadata_sum.values(),
        }
    )
    .sort_values("Metadata")
    .reset_index(drop=True)
)

print(df.to_string(index=False))

# Optional: save
df.to_csv("merged_metadata_summary.csv", index=False)
df.to_excel("merged_metadata_summary.xlsx", index=False)

print("\nSaved:")
print("  merged_metadata_summary.csv")
print("  merged_metadata_summary.xlsx")
