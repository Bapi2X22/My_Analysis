#!/usr/bin/env python3

import os
import argparse
import pyarrow.parquet as pq

def check_parquet(root_dir):
    corrupted = []

    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if not fname.endswith(".parquet"):
                continue

            fpath = os.path.join(dirpath, fname)

            try:
                pq.ParquetFile(fpath)
            except Exception as e:
                corrupted.append((fpath, str(e)))

    return corrupted


def main():
    parser = argparse.ArgumentParser(
        description="Find corrupted Parquet files."
    )
    parser.add_argument("directory", help="Directory to scan")
    parser.add_argument(
        "-o",
        "--output",
        default="corrupted_parquet_files.txt",
        help="Output file containing corrupted file list",
    )

    args = parser.parse_args()

    corrupted = check_parquet(args.directory)

    print(f"Scanned directory: {args.directory}")
    print(f"Corrupted files found: {len(corrupted)}")

    with open(args.output, "w") as fout:
        for fpath, err in corrupted:
            print(f"[BAD] {fpath}")
            print(f"      {err}")
            fout.write(f"{fpath}\n")

    print(f"\nList written to: {args.output}")


if __name__ == "__main__":
    main()
