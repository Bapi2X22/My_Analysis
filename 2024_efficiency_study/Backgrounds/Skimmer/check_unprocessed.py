#!/usr/bin/env python3

import json
import sys
from pathlib import Path


def check_unprocessed(json_file, skim_dir, dataset):

    # --------------------------------------------------
    # Read input JSON
    # --------------------------------------------------
    with open(json_file) as f:
        data = json.load(f)

    if dataset not in data:
        print(f"ERROR: Dataset '{dataset}' not found in JSON.")
        print("Available datasets:")
        for key in data:
            print(f"  {key}")
        sys.exit(1)

    input_files = data[dataset]

    # --------------------------------------------------
    # Find processed skim files
    # --------------------------------------------------
    skim_dir = Path(skim_dir)

    if not skim_dir.exists():
        print(f"ERROR: Skim directory does not exist:")
        print(skim_dir)
        sys.exit(1)

    processed = set()

    for f in skim_dir.glob("*_skim.root"):
        uuid = f.name.removesuffix("_skim.root")
        processed.add(uuid)

    # --------------------------------------------------
    # Find unprocessed files
    # --------------------------------------------------
    unprocessed = []

    for input_file in input_files:

        filename = input_file.rstrip("/").split("/")[-1]

        if not filename.endswith(".root"):
            continue

        uuid = filename.removesuffix(".root")

        if uuid not in processed:
            unprocessed.append(input_file)

    # --------------------------------------------------
    # Output filenames
    # --------------------------------------------------
    txt_file = f"unprocessed_{dataset}.txt"
    json_file_out = f"unprocessed_{dataset}.json"

    # --------------------------------------------------
    # Write TXT
    # --------------------------------------------------
    with open(txt_file, "w") as f:
        for input_file in unprocessed:
            f.write(input_file + "\n")

    # --------------------------------------------------
    # Write JSON
    # --------------------------------------------------
    output_data = {
        dataset: unprocessed
    }

    with open(json_file_out, "w") as f:
        json.dump(output_data, f, indent=4)

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------
    print("=" * 70)
    print(f"Dataset        : {dataset}")
    print(f"JSON input     : {json_file}")
    print(f"Skim directory : {skim_dir}")
    print("=" * 70)

    print(f"Input files    : {len(input_files)}")
    print(f"Processed      : {len(input_files) - len(unprocessed)}")
    print(f"Unprocessed    : {len(unprocessed)}")
    print("=" * 70)

    print(f"\nTXT file  : {txt_file}")
    print(f"JSON file : {json_file_out}")

    if unprocessed:
        print("\nUnprocessed files:")
        for i, input_file in enumerate(unprocessed, 1):
            print(f"{i:4d}  {input_file}")
    else:
        print("\nAll files have been processed.")


if __name__ == "__main__":

    if len(sys.argv) != 4:
        print(
            "Usage:\n"
            f"  python {sys.argv[0]} <json_file> <skim_directory> <dataset>"
        )
        sys.exit(1)

    json_file = sys.argv[1]
    skim_dir = sys.argv[2]
    dataset = sys.argv[3]

    check_unprocessed(json_file, skim_dir, dataset)
