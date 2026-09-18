import json
import re

LOG_FILES = [
    "condor_logs/TTtoLNu2Q_24SummerRun3/TTtoLNu2Q_24SummerRun3.15862039.13.out",
    "condor_logs/TTtoLNu2Q_24SummerRun3/TTtoLNu2Q_24SummerRun3.15862039.14.out",
    "condor_logs/TTtoLNu2Q_24SummerRun3/TTtoLNu2Q_24SummerRun3.15862039.17.out",
    "condor_logs/TTtoLNu2Q_24SummerRun3/TTtoLNu2Q_24SummerRun3.15862039.1.out",
    "condor_logs/TTtoLNu2Q_24SummerRun3/TTtoLNu2Q_24SummerRun3.15862039.21.out",
    "condor_logs/TTtoLNu2Q_24SummerRun3/TTtoLNu2Q_24SummerRun3.15862039.22.out",
    "condor_logs/TTtoLNu2Q_24SummerRun3/TTtoLNu2Q_24SummerRun3.15862039.24.out",
    "condor_logs/TTtoLNu2Q_24SummerRun3/TTtoLNu2Q_24SummerRun3.15862039.2.out",
    "condor_logs/TTtoLNu2Q_24SummerRun3/TTtoLNu2Q_24SummerRun3.15862039.9.out",
]

OUTPUT_JSON = "unprocessed_files_small.json"

EOS_PREFIX = (
    "root://eosuser.cern.ch//eos/user/b/bbapi/"
    "My_Analysis/2024_efficiency_study/Backgrounds/Skimmer/"
)

data = {}

for log_file in LOG_FILES:

    with open(log_file, "r", errors="ignore") as f:
        text = f.read()

    # Check that this job really reached maximum attempts
    if "Maximum attempts reached." not in text:
        print(f"WARNING: Maximum attempts not found: {log_file}")
        continue

    # Extract dataset
    dataset_match = re.search(
        r"Dataset\s+:\s*(\S+)",
        text
    )

    if not dataset_match:
        print(f"WARNING: Dataset not found: {log_file}")
        continue

    dataset = dataset_match.group(1)

    # Extract original input ROOT filename
    input_match = re.search(
        r"Input\s+:\s*.*?/([^/\s]+\.root)",
        text
    )

    if not input_match:
        print(f"WARNING: Input file not found: {log_file}")
        continue

    input_filename = input_match.group(1)

    # Convert:
    # xxx.root -> xxx_skim.root
    skim_filename = input_filename.replace(
        ".root",
        "_skim.root"
    )

    # Construct EOS output path
    output_file = (
        EOS_PREFIX
        + dataset
        + "/"
        + skim_filename
    )

    data.setdefault(dataset, []).append(output_file)

    print(f"Found: {output_file}")


# Remove duplicates while preserving sorting
for dataset in data:
    data[dataset] = sorted(set(data[dataset]))


with open(OUTPUT_JSON, "w") as f:
    json.dump(data, f, indent=2)


print()
print(f"Written: {OUTPUT_JSON}")

for dataset, files in data.items():
    print(f"{dataset}: {len(files)} files")
