#!/usr/bin/env python3

import argparse
import subprocess
import csv

def bytes_to_gb(size_bytes):
    return round(float(size_bytes) / (1024**3), 2)

def das_query(dataset):
    """Query DAS for dataset-level information."""

    query = f"dataset dataset={dataset}"

    try:
        result = subprocess.run(
            ["dasgoclient", "-query", query],
            capture_output=True,
            text=True,
            check=True,
        )

        return result.stdout.strip()

    except subprocess.CalledProcessError as e:
        print(f"ERROR querying {dataset}")
        print(e.stderr)
        return ""


def get_dataset_info(dataset):
    """Get number of events and size from DAS."""

    # Query number of events
    nevents_query = f"dataset dataset={dataset} | grep dataset.nevents"
    size_query = f"dataset dataset={dataset} | grep dataset.size"

    try:
        nevents_result = subprocess.run(
            ["dasgoclient", "-query", nevents_query],
            capture_output=True,
            text=True,
            check=True,
        )

        size_result = subprocess.run(
            ["dasgoclient", "-query", size_query],
            capture_output=True,
            text=True,
            check=True,
        )

        nevents = nevents_result.stdout.strip()
        size = size_result.stdout.strip()

    except subprocess.CalledProcessError as e:
        print(f"ERROR querying {dataset}: {e}")
        return "", ""

    return nevents, size


def main():

    parser = argparse.ArgumentParser(
        description="Get DAS dataset event counts and sizes"
    )

    parser.add_argument(
        "input",
        help="Text file containing DAS dataset names"
    )

    parser.add_argument(
        "-o",
        "--output",
        default="dataset_info.csv",
        help="Output CSV file"
    )

    args = parser.parse_args()

    rows = []

    with open(args.input) as f:

        datasets = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]

    for i, dataset in enumerate(datasets, 1):

        print(f"[{i}/{len(datasets)}] {dataset}")

        nevents, size = get_dataset_info(dataset)

        rows.append({
            "dataset": dataset,
            "events": nevents,
            "size_GB": bytes_to_gb(size),
        })

        print(f"  events = {nevents}")
        print(f"  size   = {size}")

    with open(args.output, "w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "events", "size_GB"]
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
