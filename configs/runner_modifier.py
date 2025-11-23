import json

# Input and output file paths
input_json = "my_runner_full.json"
output_json = "my_runner_full_WH.json"

# Load the JSON file
with open(input_json, "r") as f:
    data = json.load(f)

# Modify keys inside the "year" dictionary
if "year" in data:
    new_year_dict = {}
    for key, value in data["year"].items():
        new_key = f"WH_{key}" if not key.startswith("WH_") else key
        new_year_dict[new_key] = value
    data["year"] = new_year_dict

# Save to new JSON
with open(output_json, "w") as f:
    json.dump(data, f, indent=4)

print(f"Prefixed dataset keys saved to: {output_json}")

