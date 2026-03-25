import json
import itertools
import pandas as pd

# -----------------------------
# Load correction JSON
# -----------------------------
json_path = "/eos/user/b/bbapi/HiggsDNA/higgs_dna/systematics/JSONs/scaleAndSmearing/EGMScalesSmearing_Pho_2024_mvaID2G.v1.json"

with open(json_path) as f:
    corr = json.load(f)

name_filter = "EGMSmearAndSyst_PhoPTsplit_20242G"   # None → do ALL (current behavior)
# name_filter = "FNUF"   # example filter


# -----------------------------
# Multibinning extractor
# -----------------------------
def extract_multibinning(node, labels):
    edges = node["edges"]
    contents = node["content"]
    inputs = node["inputs"]

    bin_ranges = [list(zip(ax[:-1], ax[1:])) for ax in edges]

    rows = []
    idx = 0

    for bins in itertools.product(*bin_ranges):
        row = dict(labels)
        for name, (lo, hi) in zip(inputs, bins):
            row[f"{name}_min"] = lo
            row[f"{name}_max"] = hi
        row["value"] = contents[idx]
        rows.append(row)
        idx += 1

    return rows

# -----------------------------
# Recursive node walker
# -----------------------------
def walk_node(node, labels):
    rows = []

    nodetype = node["nodetype"]

    # Case 1: multibinning → extract table
    if nodetype == "multibinning":
        rows.extend(extract_multibinning(node, labels))

    # Case 2: category → recurse
    elif nodetype == "category":
        cat_name = node["input"]
        for entry in node["content"]:
            new_labels = dict(labels)
            new_labels[cat_name] = entry["key"]
            rows.extend(walk_node(entry["value"], new_labels))

    # Case 3: scalar / formula → skip safely
    else:
        # examples: constant, formula, transform
        pass

    return rows

# -----------------------------
# Main loop
# -----------------------------
# all_rows = []

# for corr_def in corr["corrections"]:
#     corr_name = corr_def["name"]
#     data = corr_def["data"]

#     all_rows.extend(
#         walk_node(
#             data,
#             {"correction": corr_name},
#         )
#     )

all_rows = []

for corr_def in corr["corrections"]:
    corr_name = corr_def["name"]

    # Apply filter only if name_filter is not None
    if name_filter is not None and name_filter not in corr_name:
        continue

    data = corr_def["data"]

    all_rows.extend(
        walk_node(
            data,
            {"correction": corr_name},
        )
    )


# -----------------------------
# Save CSV
# -----------------------------
df = pd.DataFrame(all_rows)
df.to_csv("EGMSmearAndSyst_PhoPTsplit_20242G.csv", index=False)

print("Saved EGMSmearAndSyst_PhoPTsplit_20242G.csv")
print(df.head())
