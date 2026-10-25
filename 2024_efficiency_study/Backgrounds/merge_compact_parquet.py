import os
import glob
import pyarrow as pa
import pyarrow.parquet as pq


# ============================================================
# Configuration
# ============================================================

BASE_DIR = "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds"
OUTPUT_DIR = os.path.join(BASE_DIR, "merged_compact_parquet")

INPUT_DIRS = {
    "TTG1Jets": os.path.join(BASE_DIR, "TTG1Jets_24SummerRun3_parquet"),
    "TTto2L2Nu": os.path.join(BASE_DIR, "TTto2L2Nu_24SummerRun3_parquet"),
    "TTtoLNu2Q": os.path.join(BASE_DIR, "TTtoLNu2Q_24SummerRun3_parquet"),

    "DYto2E10": os.path.join(BASE_DIR, "DYto2E10_24SummerRun3_parquet"),
    "DYto2E50": os.path.join(BASE_DIR, "DYto2E50_24SummerRun3_parquet"),

    "DYto2Mu10": os.path.join(BASE_DIR, "DYto2Mu10_24SummerRun3_parquet"),
    "DYto2Mu50": os.path.join(BASE_DIR, "DYto2Mu50_24SummerRun3_parquet"),

    "DYGto2LG4": os.path.join(BASE_DIR, "DYGto2LG4_24SummerRun3_parquet"),
    "DYGto2LG50": os.path.join(BASE_DIR, "DYGto2LG50_24SummerRun3_parquet"),

    "WGtoLNuG": os.path.join(BASE_DIR, "WGtoLNuG_24SummerRun3_parquet"),

    "WtoENu0J": os.path.join(BASE_DIR, "WtoENu0J_24SummerRun3_parquet"),
    "WtoENu1J": os.path.join(BASE_DIR, "WtoENu1J_24SummerRun3_parquet"),
    "WtoENu2J": os.path.join(BASE_DIR, "WtoENu2J_24SummerRun3_parquet"),

    "WtoMuNu0J": os.path.join(BASE_DIR, "WtoMuNu0J_24SummerRun3_parquet"),
    "WtoMuNu1J": os.path.join(BASE_DIR, "WtoMuNu1J_24SummerRun3_parquet"),
    "WtoMuNu2J": os.path.join(BASE_DIR, "WtoMuNu2J_24SummerRun3_parquet"),
}


os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Merge each process
# ============================================================

for process, input_dir in INPUT_DIRS.items():

    print("\n" + "=" * 70)
    print(f"Process: {process}")
    print(f"Input:   {input_dir}")
    print("=" * 70)

    files = sorted(
        glob.glob(os.path.join(input_dir, "*.parquet"))
    )

    if not files:
        print("No parquet files found. Skipping.")
        continue

    print(f"Found {len(files)} parquet files")

    tables = []
    total_sum_genw_presel = 0.0

    for i, filename in enumerate(files, 1):

        print(
            f"[{i}/{len(files)}] "
            f"{os.path.basename(filename)}"
        )

        table = pq.read_table(filename)

        # ----------------------------------------------------
        # Read metadata
        # ----------------------------------------------------

        metadata = table.schema.metadata

        if metadata is None or b"sum_genw_presel" not in metadata:
            raise RuntimeError(
                f"Missing sum_genw_presel metadata in:\n{filename}"
            )

        sum_genw_presel = float(
            metadata[b"sum_genw_presel"].decode("utf-8")
        )

        total_sum_genw_presel += sum_genw_presel

        # Remove individual-file metadata
        table = table.replace_schema_metadata(None)

        tables.append(table)

    # --------------------------------------------------------
    # Concatenate all files for this process
    # --------------------------------------------------------

    merged_table = pa.concat_tables(
        tables,
        promote_options="default"
    )

    # --------------------------------------------------------
    # Add combined metadata
    # --------------------------------------------------------

    metadata = {
        b"sum_genw_presel":
            str(total_sum_genw_presel).encode("utf-8")
    }

    merged_table = merged_table.replace_schema_metadata(metadata)

    # --------------------------------------------------------
    # Write merged file
    # --------------------------------------------------------

    output_file = os.path.join(
        OUTPUT_DIR,
        f"{process}.parquet"
    )

    pq.write_table(
        merged_table,
        output_file,
        compression=None
    )

    print(f"\nMerged file : {output_file}")
    print(f"Events      : {merged_table.num_rows}")
    print(
        f"sum_genw_presel : "
        f"{total_sum_genw_presel:.10e}"
    )

print("\nAll processes merged.")
