import json
import sys


json_file = sys.argv[1]

with open(json_file, "r") as f:
    data = json.load(f)

print("=" * 70)
print(f"{'Dataset':<50} {'N files':>10}")
print("=" * 70)

total_files = 0

for dataset, files in data.items():

    n_files = len(files)
    total_files += n_files

    print(f"{dataset:<50} {n_files:>10}")

print("=" * 70)
print(f"{'Total':<50} {total_files:>10}")
print("=" * 70)
