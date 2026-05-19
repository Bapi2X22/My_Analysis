import glob
import pyarrow.parquet as pq

files = glob.glob("*.parquet")

sum_genw_beforesel = 0.0

for f in files:
    pf = pq.ParquetFile(f)
    meta = pf.schema_arrow.metadata

    if meta and b"sum_genw_presel" in meta:
        val = meta[b"sum_genw_presel"].decode()
        if val != "Data":
            sum_genw_beforesel += float(val)

print("Total sum_genw:", sum_genw_beforesel)
