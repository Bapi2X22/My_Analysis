#!/usr/bin/env python3

import os
import glob
import json
import re

LOG_BASE = "condor_logs"
OUTPUT_JSON = "unprocessed_files.json"

unprocessed = {}

for log_file in glob.glob(os.path.join(LOG_BASE, "*", "*")):

    if not os.path.isfile(log_file):
        continue

    dataset = os.path.basename(os.path.dirname(log_file))

    with open(log_file, "r", errors="ignore") as f:
        content = f.read()

    # Only consider jobs that reached maximum attempts
    if "Maximum attempts reached." not in content:
        continue

    # Extract the Input ROOT file
    match = re.search(
        r"Input\s*:\s*(root://\S+)",
        content
    )

    if not match:
        print(f"WARNING: Could not find Input in {log_file}")
        continue

    input_file = match.group(1)

    unprocessed.setdefault(dataset, []).append(input_file)


# Remove duplicates
for dataset in unprocessed:
    unprocessed[dataset] = sorted(set(unprocessed[dataset]))


# Write JSON
with open(OUTPUT_JSON, "w") as f:
    json.dump(unprocessed, f, indent=4)


print(f"Written: {OUTPUT_JSON}")
print()

for dataset, files in unprocessed.items():
    print(f"{dataset}: {len(files)} unprocessed files")
