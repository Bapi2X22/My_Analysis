import json
import subprocess

XROOTD_PREFIX = "root://xrootd-cms.infn.it//"

datasets = {
    "WtoMuNu0J_24SummerRun3":
        "/WtoMuNu-2Jets_Bin-0J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",

    "WtoMuNu1J_24SummerRun3":
        "/WtoMuNu-2Jets_Bin-1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",

    "WtoMuNu2J_24SummerRun3":
        "/WtoMuNu-2Jets_Bin-2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",

    "WtoENu0J_24SummerRun3":
        "/WtoENu-2Jets_Bin-0J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",

    "WtoENu1J_24SummerRun3":
        "/WtoENu-2Jets_Bin-1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",

    "WtoENu2J_24SummerRun3":
        "/WtoENu-2Jets_Bin-2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
}

output = {}

for key, dataset in datasets.items():

    print(f"Querying {key}...")

    result = subprocess.run(
        [
            "dasgoclient",
            "-query",
            f"file dataset={dataset}"
        ],
        capture_output=True,
        text=True,
        check=True
    )

    files = [
        XROOTD_PREFIX + line.strip().lstrip("/")
        for line in result.stdout.splitlines()
        if line.strip()
    ]

    output[key] = files

    print(f"  Found {len(files)} files")

with open("WtoLNu_24SummerRun3.json", "w") as f:
    json.dump(output, f, indent=4)

print("\nCreated WtoLNu_24SummerRun3.json")
