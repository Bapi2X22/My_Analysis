import json

# Input and output file paths
input_json = "samples_local_full.json"   # replace with your actual filename
output_json = "output_prefixed.json"

# Load the original JSON
with open(input_json, "r") as f:
    data = json.load(f)

# Add prefix "WH_" to all top-level keys
prefixed_data = {f"WH_{k}": v for k, v in data.items()}

# Save the new JSON
with open(output_json, "w") as f:
    json.dump(prefixed_data, f, indent=2)

print(f"Done! Saved with 'WH_' prefix to: {output_json}")

