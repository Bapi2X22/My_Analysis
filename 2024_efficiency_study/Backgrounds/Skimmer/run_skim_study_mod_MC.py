#!/usr/bin/env python3

import argparse
import csv
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path


# ============================================================
# DATASETS
# ============================================================

DATASETS = [
    "/TTG-1Jets_TuneCP5_13p6TeV_amcatnloFXFXold-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",

    # "/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3/NANOAODSIM"

    "/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM"

    # "/DYto2E-2Jets_Bin-MLL-10to50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",

    # "/DYto2Mu-2Jets_Bin-MLL-10to50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",

    # "/DYto2E-2Jets_Bin-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v4/NANOAODSIM",

    # "/DYto2Mu-2Jets_Bin-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v6/NANOAODSIM",

    # "/WGtoLNuG-1Jets_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",

    # "/DYGto2LG-1Jets_Bin-MLL-4to50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",

    # "/DYGto2LG-1Jets_Bin-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",

    # "/WZ_TuneCP5_13p6TeV_pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",

    # "/ZZ_TuneCP5_13p6TeV_pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",

    # "/WW_TuneCP5_13p6TeV_pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",

    # "/EGamma0/Run2024E-MINIv6NANOv15-v1/NANOAOD", 
    # "/Muon0/Run2024E-MINIv6NANOv15-v1/NANOAOD"
]


# ============================================================
# SKIM STAGES
# ============================================================

STAGES = [
    # (
        # "Branch",
        # [
        #     "--no-jet-selection",
        #     "--no-electron-selection",
        #     "--no-muon-selection",
        #     "--no-photon-selection",
        #     "--no-event-selection"
        # ],
    # ),

    # (
    #     "Branch+Pho",
    #     [
    #         "--no-jet-selection",
    #         "--no-electron-selection",
    #         "--no-muon-selection",
    #         "--no-event-selection",
    #     ],
    # ),

    # (
    #     "Branch+Pho+Mu",
    #     [
    #         "--no-jet-selection",
    #         "--no-electron-selection",
    #         "--no-event-selection",
    #     ],
    # ),

    # (
    #     "Branch+Pho+Mu+ele",
    #     [
    #         "--no-jet-selection",
    #         "--no-event-selection",
    #     ],
    # ),

    # (
    #     "Branch+Pho+Mu+ele+jets",
    #     [
    #         "--no-event-selection",
    #         "--apply_kinematic_cuts_jet"
    #     ],
    # ),

    (
        "Branch+Pho+Mu+ele+jets+pixel",
        [
            "--apply_pixelSeed",
            "--no-event-selection",
            "--apply_kinematic_cuts_jet"
        ],
    ),

    (
        "Branch+Pho+Mu+ele+jets+pixel+event",
        [
            "--apply_pixelSeed",
            "--apply_kinematic_cuts_jet"
        ],
    ),

    (
        "Branch+Pho+Mu+ele+jets+pixel+event+btag",
        [
            "--apply_pixelSeed",
            "--apply_bJet_tagger",
            "--apply_kinematic_cuts_jet"
        ],
    ),

    (
        "Branch+Pho+Mu+ele+pixel+event+btag+nojetkin",
        [
            "--apply_bJet_tagger",
            "--apply_pixelSeed"
        ],
    ),
]


# ============================================================
# RUN COMMAND
# ============================================================

def run_command(cmd):

    print(
        "\n$ " + " ".join(cmd),
        flush=True,
    )

    result = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
    )

    if result.returncode != 0:

        print("\n========== COMMAND FAILED ==========")

        print("\nSTDOUT:")
        print(result.stdout)

        print("\nSTDERR:")
        print(result.stderr)

        print("====================================\n")

        raise RuntimeError(
            f"Command failed with exit code "
            f"{result.returncode}"
        )

    return result.stdout


# ============================================================
# GET DAS FILE LIST
# ============================================================

def get_files(dataset):

    output = run_command(
        [
            "dasgoclient",
            "-query",
            f"file dataset={dataset}",
        ]
    )

    return [
        line.strip()
        for line in output.splitlines()
        if line.strip()
    ]


# ============================================================
# RECURSIVE JSON SEARCH
# ============================================================

def find_value_recursive(obj, key):

    if isinstance(obj, dict):

        if key in obj:
            return obj[key]

        for value in obj.values():

            result = find_value_recursive(
                value,
                key,
            )

            if result is not None:
                return result

    elif isinstance(obj, list):

        for item in obj:

            result = find_value_recursive(
                item,
                key,
            )

            if result is not None:
                return result

    return None


# ============================================================
# GET TOTAL EVENTS IN DATASET
# ============================================================

def get_dataset_events(dataset):

    output = run_command(
        [
            "dasgoclient",
            "-query",
            f"summary dataset={dataset}",
            "--format=json",
        ]
    )

    data = json.loads(output)

    nevents = find_value_recursive(
        data,
        "nevents",
    )

    if nevents is None:

        raise RuntimeError(
            f"Could not find nevents for:\n{dataset}"
        )

    return int(nevents)


# ============================================================
# GET EVENTS + SIZE OF FILE
# ============================================================

def get_file_metadata(filename):

    output = run_command(
        [
            "dasgoclient",
            "-query",
            f"file={filename}",
            "--format=json",
        ]
    )

    data = json.loads(output)

    file_info = None

    def search(obj):

        nonlocal file_info

        if file_info is not None:
            return

        if isinstance(obj, dict):

            if (
                obj.get("name") == filename
                or
                obj.get("name")
                == "/" + filename.lstrip("/")
            ):
                file_info = obj
                return

            for value in obj.values():
                search(value)

        elif isinstance(obj, list):

            for item in obj:
                search(item)

    search(data)

    if file_info is None:

        raise RuntimeError(
            f"Could not find metadata for:\n{filename}"
        )

    return (
        int(file_info["nevents"]),
        int(file_info["size"]),
    )


# ============================================================
# GET EVENTS FROM OUTPUT ROOT FILE
# ============================================================

def get_root_events(filename):

    python_code = r"""
import sys
import uproot

filename = sys.argv[1]

f = uproot.open(filename)

keys = [
    key.split(";")[0]
    for key in f.keys()
]

if "Events" in keys:

    tree_name = "Events"

else:

    tree_name = None

    for key in keys:

        try:

            obj = f[key]

            if hasattr(obj, "num_entries"):
                tree_name = key
                break

        except Exception:
            pass

if tree_name is None:

    raise RuntimeError(
        f"No TTree/RNTuple found in {filename}"
    )

print(f[tree_name].num_entries)
"""

    output = run_command(
        [
            sys.executable,
            "-c",
            python_code,
            str(filename),
        ]
    )

    return int(output.strip())


# ============================================================
# FORMAT SIZE
# ============================================================

def format_size(size_bytes):

    if size_bytes >= 1024**4:

        return (
            size_bytes / 1024**4,
            "TB",
        )

    if size_bytes >= 1024**3:

        return (
            size_bytes / 1024**3,
            "GB",
        )

    if size_bytes >= 1024**2:

        return (
            size_bytes / 1024**2,
            "MB",
        )

    return (
        size_bytes / 1024,
        "KB",
    )


# ============================================================
# PROCESS ONE DATASET
#
# Returns:
#
#     True  -> successful
#     False -> failed
#
# If one stage fails, we immediately leave this dataset
# and move to the next dataset.
# ============================================================

def copy_remote_file(input_file, local_file, max_retries=3):
    """Copy the selected DAS file locally with retries."""
    input_url = (
        "root://xrootd-cms.infn.it//"
        + input_file.lstrip("/")
    )

    for copy_attempt in range(1, max_retries + 1):
        print()
        print("-" * 100)
        print(
            f"Downloading input file "
            f"(attempt {copy_attempt}/{max_retries})"
        )
        print(f"Remote : {input_url}")
        print(f"Local  : {local_file}")
        print("-" * 100)

        try:
            local_file.unlink()
        except FileNotFoundError:
            pass

        try:
            run_command(
                [
                    "xrdcp",
                    "--force",
                    input_url,
                    str(local_file),
                ]
            )

            if not local_file.exists():
                raise RuntimeError(
                    f"xrdcp completed but local file does not exist:\n"
                    f"{local_file}"
                )

            local_size = local_file.stat().st_size
            if local_size == 0:
                raise RuntimeError(
                    f"xrdcp produced an empty file:\n{local_file}"
                )

            value, unit = format_size(local_size)
            print(f"Download successful: {value:.3f} {unit}")
            return

        except Exception as error:
            print(
                f"xrdcp attempt {copy_attempt} failed: {error}"
            )

            if copy_attempt < max_retries:
                sleep_seconds = 5 * copy_attempt
                print(
                    f"Retrying xrdcp in {sleep_seconds} seconds..."
                )
                time.sleep(sleep_seconds)
            else:
                raise


def run_stage_with_retries(command, stage_name, max_retries=3):
    """Retry an individual nano_reduce stage without re-downloading."""
    for stage_attempt in range(1, max_retries + 1):
        print()
        print(
            f"Running stage '{stage_name}' "
            f"(attempt {stage_attempt}/{max_retries})"
        )

        try:
            run_command(command)
            return

        except Exception as error:
            print(
                f"Stage '{stage_name}' failed on attempt "
                f"{stage_attempt}: {error}"
            )

            if stage_attempt < max_retries:
                sleep_seconds = 5 * stage_attempt
                print(
                    f"Retrying stage in {sleep_seconds} seconds..."
                )
                time.sleep(sleep_seconds)
            else:
                raise


# ============================================================
# PROCESS ONE DATASET
#
# DAS -> select third file -> xrdcp once -> run all stages
# locally -> collect results -> automatically delete local file.
# ============================================================

def process_dataset(
    dataset,
    dataset_number,
    total_datasets,
    output_dir,
    nano_reduce,
    rows,
    attempt,
    stage_retries=3,
    xrdcp_retries=3,
):

    dataset_name = (
        dataset.strip("/")
        .split("/")[0]
    )

    safe_dataset_name = (
        dataset_name
        .replace("/", "_")
        .replace(":", "_")
    )

    print()
    print("=" * 100)
    print(
        f"[{dataset_number}/{total_datasets}] "
        f"{dataset_name}"
    )
    print(f"ATTEMPT: {attempt}")
    print("=" * 100)

    try:

        # ====================================================
        # GET FILE LIST
        # ====================================================

        files = get_files(dataset)

        if len(files) < 3:
            raise RuntimeError(
                f"Only {len(files)} files found by DAS"
            )

        # Keep your original choice: third DAS file.
        input_file = files[2]

        print()
        print("Third DAS file:")
        print(input_file)

        # ====================================================
        # DATASET EVENTS
        # ====================================================

        total_dataset_events = get_dataset_events(dataset)

        print()
        print(
            f"Total dataset events: "
            f"{total_dataset_events:,}"
        )

        # ====================================================
        # FILE EVENTS + SIZE
        # ====================================================

        file_events, file_size = get_file_metadata(input_file)

        size_value, size_unit = format_size(file_size)

        print(f"Third-file events: {file_events:,}")
        print(
            f"Third-file size: "
            f"{size_value:.3f} {size_unit}"
        )

        # ====================================================
        # DOWNLOAD ONCE
        # ====================================================

        with tempfile.TemporaryDirectory(
            prefix=f"nano_reduce_{safe_dataset_name}_"
        ) as temp_dir:

            local_input = Path(temp_dir) / "input.root"

            copy_remote_file(
                input_file=input_file,
                local_file=local_input,
                max_retries=xrdcp_retries,
            )

            # ====================================================
            # RUN ALL STAGES ON LOCAL FILE
            # ====================================================

            for stage_number, (stage_name, flags) in enumerate(
                STAGES,
                start=1,
            ):

                print()
                print("-" * 100)
                print(f"Dataset : {dataset_name}")
                print(f"Attempt : {attempt}")
                print(
                    f"Stage   : {stage_number}/{len(STAGES)}"
                )
                print(f"Name    : {stage_name}")
                print("-" * 100)

                output_file = (
                    output_dir
                    / f"{safe_dataset_name}_stage{stage_number}.root"
                )

                # Remove stale output from an earlier attempt.
                if output_file.exists():
                    print(
                        f"Removing existing output: {output_file}"
                    )
                    output_file.unlink()

                command = [
                    sys.executable,
                    nano_reduce,
                    "--input",
                    str(local_input),
                    "--output",
                    str(output_file),
                ]
                command.extend(flags)

                run_stage_with_retries(
                    command=command,
                    stage_name=stage_name,
                    max_retries=stage_retries,
                )

                # ------------------------------------------------
                # Validate output
                # ------------------------------------------------

                if not output_file.exists():
                    raise RuntimeError(
                        f"Output file does not exist:\n"
                        f"{output_file}"
                    )

                output_size = output_file.stat().st_size

                if output_size == 0:
                    raise RuntimeError(
                        f"Output file is empty:\n"
                        f"{output_file}"
                    )

                # ------------------------------------------------
                # Output events
                # ------------------------------------------------

                output_events = get_root_events(output_file)

                inferred_dataset_events = (
                    output_events
                    * total_dataset_events
                    / file_events
                )

                # ------------------------------------------------
                # Infer total dataset size
                # ------------------------------------------------

                inferred_dataset_size = (
                    output_size
                    * total_dataset_events
                    / file_events
                )

                output_value, output_unit = format_size(
                    output_size
                )

                inferred_value, inferred_unit = format_size(
                    inferred_dataset_size
                )

                print()
                print(
                    f"Events after skim : "
                    f"{output_events:,}"
                )
                print(
                    f"Output size        : "
                    f"{output_value:.3f} {output_unit}"
                )
                print(
                    f"Inferred dataset events    : "
                    f"{inferred_dataset_events:,.0f}"
                )
                print(
                    f"Inferred dataset size : "
                    f"{inferred_value:.3f} {inferred_unit}"
                )

                # ------------------------------------------------
                # Save row
                # ------------------------------------------------

                rows.append(
                    {
                        "dataset": dataset_name,
                        "dataset_path": dataset,
                        "stage": stage_name,
                        "stage_number": stage_number,
                        "input_file": input_file,
                        "dataset_total_events":
                            total_dataset_events,
                        "file_events_before_skim":
                            file_events,
                        "file_size_GB":
                            file_size / 1024**3,
                        "events_after_skim":
                            output_events,
                        "output_file_size_GB":
                            output_size / 1024**3,
                        "inferred_dataset_events":
                            inferred_dataset_events,
                        "inferred_total_dataset_size_GB":
                            inferred_dataset_size / 1024**3,
                        "output_file":
                            str(output_file),
                        "attempt":
                            attempt,
                    }
                )

            print()
            print("=" * 100)
            print(f"SUCCESS: {dataset_name}")
            print(
                "All stages completed using the local input file."
            )
            print("=" * 100)

        # TemporaryDirectory deletes the downloaded ROOT file here.
        print(
            f"Temporary input deleted for {dataset_name}"
        )

        return True

    except Exception as error:

        print()
        print("!" * 100)
        print(f"FAILED: {dataset_name}")
        print(f"Attempt: {attempt}")
        print(f"Error: {error}")
        print(
            "Temporary input will be cleaned automatically."
        )
        print("Moving to the next dataset...")
        print("!" * 100)

        return False


# ============================================================
# WRITE CSV
# ============================================================

def write_csv(filename, rows):

    columns = [
        "dataset",
        "dataset_path",
        "stage",
        "stage_number",
        "input_file",
        "dataset_total_events",
        "file_events_before_skim",
        "file_size_GB",
        "events_after_skim",
        "output_file_size_GB",
        "inferred_dataset_events",
        "inferred_total_dataset_size_GB",
        "output_file",
        "attempt",
    ]

    with open(
        filename,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=columns,
        )

        writer.writeheader()

        writer.writerows(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--nano-reduce",
        default="nano_reduce.py",
    )

    parser.add_argument(
        "--output-dir",
        default="skimmed_stage",
    )

    parser.add_argument(
        "--csv",
        default="skim_study.csv",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help=(
            "Number of retry passes after the initial "
            "pass. Default: 3"
        ),
    )

    parser.add_argument(
        "--stage-retries",
        type=int,
        default=3,
        help=(
            "Number of attempts for each nano_reduce stage. "
            "Default: 3"
        ),
    )

    parser.add_argument(
        "--xrdcp-retries",
        type=int,
        default=3,
        help=(
            "Number of attempts for downloading each input file. "
            "Default: 3"
        ),
    )

    args = parser.parse_args()

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    # ========================================================
    # FIRST PASS
    # ========================================================

    failed_datasets = []

    print()
    print("#" * 100)
    print("INITIAL PASS")
    print("#" * 100)

    for dataset_number, dataset in enumerate(
        DATASETS,
        start=1,
    ):

        success = process_dataset(
            dataset=dataset,
            dataset_number=dataset_number,
            total_datasets=len(DATASETS),
            output_dir=output_dir,
            nano_reduce=args.nano_reduce,
            rows=rows,
            attempt=0,
            stage_retries=args.stage_retries,
            xrdcp_retries=args.xrdcp_retries,
        )

        if not success:

            failed_datasets.append(
                dataset
            )

    # ========================================================
    # RETRY PASSES
    #
    # Only failed datasets are retried.
    #
    # This means:
    #
    # Dataset 1 failed
    # Dataset 2
    # Dataset 3
    # ...
    # Dataset 13
    #
    # THEN:
    #
    # retry Dataset 1
    #
    # ========================================================

    for retry_number in range(
        1,
        args.max_retries + 1,
    ):

        if not failed_datasets:
            break

        print()
        print("#" * 100)
        print(
            f"RETRY PASS {retry_number}/{args.max_retries}"
        )
        print(
            f"Datasets to retry: "
            f"{len(failed_datasets)}"
        )
        print("#" * 100)

        new_failed_datasets = []

        for dataset in failed_datasets:

            dataset_number = (
                DATASETS.index(dataset) + 1
            )

            success = process_dataset(
                dataset=dataset,
                dataset_number=dataset_number,
                total_datasets=len(DATASETS),
                output_dir=output_dir,
                nano_reduce=args.nano_reduce,
                rows=rows,
                attempt=retry_number,
                stage_retries=args.stage_retries,
                xrdcp_retries=args.xrdcp_retries,
            )

            if not success:

                new_failed_datasets.append(
                    dataset
                )

        failed_datasets = (
            new_failed_datasets
        )

    # ========================================================
    # WRITE CSV
    # ========================================================

    write_csv(
        args.csv,
        rows,
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("#" * 100)
    print("FINAL SUMMARY")
    print("#" * 100)

    successful = (
        len(DATASETS)
        - len(failed_datasets)
    )

    print(
        f"Successful datasets : "
        f"{successful}/{len(DATASETS)}"
    )

    if failed_datasets:

        print()
        print(
            "Datasets still failing "
            f"after {args.max_retries} retries:"
        )

        for dataset in failed_datasets:

            print(
                f"  - {dataset}"
            )

    else:

        print()
        print(
            "All datasets completed successfully."
        )

    print()
    print(
        f"CSV written to: {args.csv}"
    )

    print("#" * 100)


if __name__ == "__main__":
    main()