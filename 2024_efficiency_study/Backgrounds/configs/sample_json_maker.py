#!/usr/bin/env python3

import json
import subprocess
import argparse


XROOTD_PREFIX = "root://xrootd-cms.infn.it//"


def get_files(dataset):
    """Get all files for a DAS dataset."""

    command = [
        "dasgoclient",
        "-query",
        f"file dataset={dataset}",
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
    )

    files = []

    for line in result.stdout.splitlines():
        line = line.strip()

        if not line:
            continue

        # Make sure there is only one leading /
        line = line.lstrip("/")

        files.append(XROOTD_PREFIX + "/" + line)

    return files


def read_input_file(filename):
    """
    Read input file.

    Format:

        sample_name
        dataset1
        dataset2

        another_sample
        dataset3
        dataset4
    """

    samples = {}
    current_sample = None

    with open(filename) as f:

        for line in f:

            line = line.strip()

            # Ignore empty lines
            if not line:
                current_sample = None
                continue

            # Ignore comments
            if line.startswith("#"):
                continue

            # First line of a block = sample name
            if current_sample is None:
                current_sample = line
                samples[current_sample] = []

            # Following lines = datasets
            else:
                samples[current_sample].append(line)

    return samples


def main():

    parser = argparse.ArgumentParser(
        description="Create JSON containing DAS ROOT files."
    )

    parser.add_argument(
        "input",
        help="Input text file containing sample names and DAS datasets",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="samples.json",
        help="Output JSON file",
    )

    args = parser.parse_args()

    samples = read_input_file(args.input)

    output = {}

    for sample, datasets in samples.items():

        print()
        print("=" * 70)
        print(f"Sample: {sample}")
        print("=" * 70)

        output[sample] = []

        for dataset in datasets:

            print(f"Querying DAS:")
            print(f"  {dataset}")

            try:
                files = get_files(dataset)

                print(f"  Found {len(files)} files")

                output[sample].extend(files)

            except subprocess.CalledProcessError as e:

                print("  ERROR: dasgoclient failed")
                print(e.stderr)

        # Remove duplicate files while preserving order
        output[sample] = list(dict.fromkeys(output[sample]))

        print(f"Total files for {sample}: {len(output[sample])}")

    with open(args.output, "w") as f:
        json.dump(output, f, indent=4)

    print()
    print(f"JSON written to: {args.output}")


if __name__ == "__main__":
    main()
