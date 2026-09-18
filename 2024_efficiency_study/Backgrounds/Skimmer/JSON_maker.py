#!/usr/bin/env python3

import os
import json
import sys


EOS_PREFIX = "root://eosuser.cern.ch/"


def make_json(folder_dataset_pairs, output_json):

    output = {}

    for skim_folder, dataset in folder_dataset_pairs:

        skim_folder = os.path.abspath(skim_folder)

        if not os.path.isdir(skim_folder):
            print(f"WARNING: Directory does not exist, skipping:")
            print(f"  {skim_folder}")
            continue

        if not skim_folder.startswith("/eos/"):
            print(f"WARNING: Not an EOS path, skipping:")
            print(f"  {skim_folder}")
            continue

        root_files = []

        for filename in sorted(os.listdir(skim_folder)):

            if not filename.endswith(".root"):
                continue

            local_path = os.path.join(skim_folder, filename)

            if not os.path.isfile(local_path):
                continue

            # /eos/... -> root://eosuser.cern.ch//eos/...
            xrootd_path = EOS_PREFIX + local_path

            root_files.append(xrootd_path)

        output[dataset] = root_files

        print(f"{dataset:40s} : {len(root_files)} files")

    # Write JSON
    with open(output_json, "w") as f:
        json.dump(output, f, indent=4)

    print("\n" + "=" * 70)
    print(f"Output JSON: {output_json}")
    print("=" * 70)


if __name__ == "__main__":

    if len(sys.argv) < 4 or (len(sys.argv) - 2) % 2 != 0:
        print(
            "\nUsage:\n"
            "  python make_skim_json.py "
            "<folder1> <dataset1> "
            "[<folder2> <dataset2> ...] "
            "<output.json>\n"
        )
        sys.exit(1)

    # Last argument is output JSON
    output_json = sys.argv[-1]

    # Everything before it is folder/dataset pairs
    args = sys.argv[1:-1]

    folder_dataset_pairs = []

    for i in range(0, len(args), 2):
        folder = args[i]
        dataset = args[i + 1]

        folder_dataset_pairs.append(
            (folder, dataset)
        )

    make_json(folder_dataset_pairs, output_json)
