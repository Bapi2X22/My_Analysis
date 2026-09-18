import json
import glob
import argparse


def combine_efficiencies(input_dir, output_file):
    json_files = glob.glob(f"{input_dir}/*.json")

    process_content = []

    for json_file in json_files:
        with open(json_file, "r") as f:
            data = json.load(f)

        process_name = json_file.split("/")[-1].replace(".json", "")

        correction = data["corrections"][0]

        process_content.append({
            "key": process_name,
            "value": correction["data"]
        })

    mega_correction = {
        "name": "btagging_efficiencies",
        "description": "Correction that contains the b-tagging efficiencies as a function of the process, the jet hadron flavour, transverse momentum and absolute eta.",
        "version": 1,
        "inputs": [
            {
                "name": "process",
                "type": "string",
                "description": "Simulation process"
            },
            {
                "name": "wp",
                "type": "string",
                "description": "B-tagging working point"
            },
            {
                "name": "flav",
                "type": "int",
                "description": "Jet hadron flavour"
            },
            {
                "name": "abseta",
                "type": "real",
                "description": "Jet absolute eta"
            },
            {
                "name": "pt",
                "type": "real",
                "description": "Jet transverse momentum"
            }
        ],
        "output": {
            "name": "efficiency",
            "type": "real",
            "description": "B-tagging efficiency"
        },
        "data": {
            "nodetype": "category",
            "input": "process",
            "content": process_content
        }
    }

    mega_json = {
        "schema_version": 2,
        "description": "Correction set for b-tagging efficiencies",
        "corrections": [
            mega_correction
        ]
    }

    with open(output_file, "w") as f:
        json.dump(mega_json, f, indent=2)

    print(f"Combined {len(json_files)} JSON files")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine b-tagging efficiency JSON files.")

    parser.add_argument("--input-dir", required=True, help="Directory containing individual efficiency JSON files.")
    parser.add_argument("--output", required=True, help="Output mega JSON file.")

    args = parser.parse_args()

    combine_efficiencies(args.input_dir, args.output)
