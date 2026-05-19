import glob
import pyarrow.parquet as pq
import awkward as ak

base_dir = "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/NTuples_WH_2024_HDNA_presel_final_good_selections_pixel/"

Mass_points = ["12", "15", "20", "25", "30", "35", "40", "45", "50", "55", "60"]
#Mass_points = ["30"]

for mass in Mass_points:
    inside_dir = f"{base_dir}WH-2024M{mass}/nominal/diphoton/"
    files = glob.glob(f"{inside_dir}/*.parquet")

    sum_genw_beforesel = 0.0

    for f in files:
        pf = pq.ParquetFile(f)
        meta = pf.schema_arrow.metadata

        if meta and b"sum_genw_presel" in meta:
            val = meta[b"sum_genw_presel"].decode()
            if val != "Data":
                sum_genw_beforesel += float(val)

    array = ak.from_parquet(files)
    array_length = len(array.pholead_pt)

    eff = (array_length / sum_genw_beforesel)*100

    print(f"Mass: {mass} GeV, sum_genw_presel: {sum_genw_beforesel}, number of events after selection: {array_length}, efficiency: {eff:.4f}%")
