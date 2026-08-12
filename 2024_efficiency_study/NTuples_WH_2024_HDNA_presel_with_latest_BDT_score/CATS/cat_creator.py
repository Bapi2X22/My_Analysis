#!/usr/bin/env python3

import re

# File containing the AMS output
input_file = "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/AMS_scan_latest_7/summary.txt" 

# Template cat.json
template_file = "cat_25.json"

# Read template
with open(template_file, "r") as f:
    template = f.read()

# Read AMS results
with open(input_file, "r") as f:
    text = f.read()

# Extract (mass, best cut)
pattern = re.compile(
    r"Mass\s*=\s*(\d+).*?"
    r"Best cut\s*:\s*([0-9.]+)",
    re.DOTALL,
)

matches = pattern.findall(text)

print(f"Found {len(matches)} mass points")

for mass, cut in matches:
    # Replace the BDT cut in the template
    new_text = re.sub(
        r'("BDT_score"\s*,\s*">\s*",?\s*)([0-9.]+)',
        rf'\g<1>{cut}',
        template,
    )

    # If your template has exactly ["BDT_score", ">", 0.74]
    new_text = re.sub(
        r'(\["BDT_score"\s*,\s*">\s*,\s*)[0-9.]+',
        rf'\g<1>{cut}',
        new_text,
    )

    outfile = f"cat_M{mass}.json"

    with open(outfile, "w") as f:
        f.write(new_text)

    print(f"Created {outfile}")
