#!/usr/bin/env python3

import os
import glob
import argparse
import awkward as ak
import pyarrow.parquet as pq


PROCESSES = [
    "TTto2L2Nu",
    "TTG1Jets",
    "TTtoLNu2Q",
    "WGtoLNuG",
    "DYto2E10",
    "DYto2E50",
    "DYto2Mu10",
    "DYto2Mu50",
    "DYGto2LG4",
    "DYGto2LG50",
    "WtoENu0J",
    "WtoENu1J",
    "WtoENu2J",
    "WtoMuNu0J",
    "WtoMuNu1J",
    "WtoMuNu2J"
]


SUFFIX = "_24SummerRun3_selected_with_corrections"


def get_sum_genw_presel(filename):
    metadata = pq.ParquetFile(filename).metadata.metadata
    if metadata is None or b"sum_genw_presel" not in metadata:
        raise RuntimeError(f"sum_genw_presel not found in {filename}")
    return float(metadata[b"sum_genw_presel"])


def process_directory(process, input_base, output_dir):
    input_dir = os.path.join(input_base, process + SUFFIX)
    files = sorted(glob.glob(os.path.join(input_dir, "*.parquet")))

    if not files:
        print(f"[WARNING] No parquet files found: {input_dir}")
        return

    print("\n" + "=" * 80)
    print(f"Processing: {process}")
    print(f"Directory: {input_dir}")
    print(f"Number of files: {len(files)}")
    print("=" * 80)

    total_sum_genw_presel = 0.0

    for filename in files:
        value = get_sum_genw_presel(filename)
        total_sum_genw_presel += value
        print(f"{os.path.basename(filename):30s} {value:.10e}")

    print("-" * 80)
    print(f"Total sum_genw_presel = {total_sum_genw_presel:.10e}")

    if total_sum_genw_presel == 0:
        raise RuntimeError(f"Total sum_genw_presel is zero for {process}")

    arrays = []

    for filename in files:
        print(f"Reading: {os.path.basename(filename)}")
        arrays.append(ak.from_parquet(filename))

    merged = ak.concatenate(arrays, axis=0)

    print(f"Total events = {len(merged)}")

    if "weight" not in merged.fields:
        raise RuntimeError(f"'weight' branch not found for {process}")

    merged["weight"] = merged["weight"] / total_sum_genw_presel

    table = ak.to_arrow_table(merged)

    metadata = dict(table.schema.metadata or {})
    metadata[b"sum_genw_presel"] = str(total_sum_genw_presel).encode()
    table = table.replace_schema_metadata(metadata)

    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"{process}.parquet")

    pq.write_table(table, output_file, compression=None)

    print(f"Written: {output_file}")
    print(f"Stored sum_genw_presel = {total_sum_genw_presel:.10e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-base", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    for process in PROCESSES:
        try:
            process_directory(process, args.input_base, args.output_dir)
        except Exception as e:
            print(f"[ERROR] {process}: {e}")


if __name__ == "__main__":
    main()
