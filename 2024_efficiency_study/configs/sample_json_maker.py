import subprocess
import json

masses = [12,15,20,25,30,35,40,45,50,55,60]

base = "/WHToAA-AToBB-AToGG_Par-M-{m}_TuneCP5_13p6TeV_madgraph-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM"

output = {}

for m in masses:
    key = f"WH-2024M{m}"
    dataset = base.format(m=m)

    print(f"Querying M={m}...")

    cmd = f'dasgoclient --query="file dataset={dataset}"'
    files = subprocess.check_output(cmd, shell=True).decode().splitlines()

    # Optional: prepend xrootd
    files = [f"root://cms-xrd-global.cern.ch/{f}" for f in files]

    output[key] = files

# Save JSON
with open("sample_WH_v15.json", "w") as f:
    json.dump(output, f, indent=2)

print("Saved to sample_WH_v15.json")
