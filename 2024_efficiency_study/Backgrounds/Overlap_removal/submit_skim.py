#!/usr/bin/env python3

import os
import subprocess
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

DATASETS = [
    "TTto2L2Nu_24SummerRun3",
    "TTG1Jets_24SummerRun3",
    "TTtoLNu2Q_24SummerRun3",
    "WGtoLNuG_24SummerRun3",
    "DYto2E10_24SummerRun3",
    "DYto2E50_24SummerRun3",
    "DYto2Mu10_24SummerRun3",
    "DYto2Mu50_24SummerRun3",
    "DYGto2LG4_24SummerRun3",
    "DYGto2LG50_24SummerRun3",
    "WtoENu0J_24SummerRun3",
    "WtoENu1J_24SummerRun3",
    "WtoENu2J_24SummerRun3",
    "WtoMuNu0J_24SummerRun3",
    "WtoMuNu1J_24SummerRun3",
    "WtoMuNu2J_24SummerRun3",
]


# ------------------------------------------------------------
# Base directories
# ------------------------------------------------------------

BASE_INPUT_DIR = (
    "/eos/user/b/bbapi/"
    "My_Analysis/2024_efficiency_study/"
    "Backgrounds/Skimmer"
)

BASE_OUTPUT_DIR = (
    "/eos/user/b/bbapi/"
    "My_Analysis/2024_efficiency_study/"
    "Backgrounds"
)

BASE_CONDOR_LOG_DIR = (
    "/eos/user/b/bbapi/"
    "My_Analysis/2024_efficiency_study/"
    "Backgrounds/CondorLogs/skim"
)


# ------------------------------------------------------------
# Executables
# ------------------------------------------------------------

EXECUTABLE = "run_skim.sh"
SKIM_SCRIPT = "skim_to_parquet.py"


# ============================================================
# Helpers
# ============================================================

def eos_mkdir(path):

    print(f"Creating EOS directory: {path}")

    subprocess.run(
        [
            "xrdfs",
            "eosuser.cern.ch",
            "mkdir",
            "-p",
            path,
        ],
        check=True,
    )


def get_dataset_config(dataset):

    input_dir = Path(BASE_INPUT_DIR) / dataset

    eos_output_dir = (
        Path(BASE_OUTPUT_DIR)
        / f"{dataset}_parquet"
    )

    condor_log_dir = (
        Path(BASE_CONDOR_LOG_DIR)
        / dataset
    )

    eos_output_remote = (
        "root://eosuser.cern.ch//"
        + str(eos_output_dir).lstrip("/")
    )

    condor_log_remote = (
        "root://eosuser.cern.ch//"
        + str(condor_log_dir).lstrip("/")
    )

    return (
        input_dir,
        eos_output_dir,
        condor_log_dir,
        eos_output_remote,
        condor_log_remote,
    )


# ============================================================
# Submit one dataset
# ============================================================

def submit_dataset(dataset):

    print()
    print("=" * 70)
    print(f"Preparing dataset: {dataset}")
    print("=" * 70)

    (
        input_dir,
        eos_output_dir,
        condor_log_dir,
        eos_output_remote,
        condor_log_remote,
    ) = get_dataset_config(dataset)

    # --------------------------------------------------------
    # Check executable files
    # --------------------------------------------------------

    if not Path(EXECUTABLE).exists():
        raise FileNotFoundError(
            f"Cannot find executable: {EXECUTABLE}"
        )

    if not Path(SKIM_SCRIPT).exists():
        raise FileNotFoundError(
            f"Cannot find skim script: {SKIM_SCRIPT}"
        )

    # --------------------------------------------------------
    # Get ROOT files
    # --------------------------------------------------------

    input_files = sorted(
        str(f)
        for f in input_dir.glob("*.root")
    )

    if not input_files:
        print(
            f"WARNING: No ROOT files found in: "
            f"{input_dir}"
        )
        return False

    # input_files = input_files[:1]

    # print(f"TEST MODE: processing only first ROOT file")
    # print(f"Selected file    : {input_files[0]}")


    # --------------------------------------------------------
    # Dataset information
    # --------------------------------------------------------

    print(f"Dataset           : {dataset}")
    print(f"Input directory   : {input_dir}")
    print(f"Number of files   : {len(input_files)}")
    print(f"EOS output        : {eos_output_remote}")
    print(f"Condor logs       : {condor_log_remote}")

    # --------------------------------------------------------
    # EOS directories
    # --------------------------------------------------------

    eos_mkdir(str(eos_output_dir))
    eos_mkdir(str(condor_log_dir))

    # --------------------------------------------------------
    # Local Condor directory
    # --------------------------------------------------------

    submit_dir = Path(
        f"condor_skim_{dataset}"
    )

    submit_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"Submit directory  : {submit_dir}"
    )

    # --------------------------------------------------------
    # Create arguments file
    # --------------------------------------------------------

    argument_file = submit_dir / "arguments.txt"

    print()
    print("-" * 70)
    print("Creating arguments file")
    print("-" * 70)

    n_jobs = 0

    with open(argument_file, "w") as f:

        for root_file in input_files:

            basename = Path(root_file).stem

            parquet_name = (
                f"{basename}.parquet"
            )

            eos_output = (
                f"{eos_output_remote}/"
                f"{parquet_name}"
            )

            local_output = parquet_name

            # ------------------------------------------------
            # Skip already existing output
            # ------------------------------------------------

            check_result = subprocess.run(
                [
                    "xrdfs",
                    "eosuser.cern.ch",
                    "stat",
                    f"{str(eos_output).replace('root://eosuser.cern.ch/', '')}",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            if check_result.returncode == 0:
                print(
                    f"Skipping existing output: "
                    f"{parquet_name}"
                )
                continue

            f.write(
                f"{root_file} "
                f"{local_output} "
                f"{eos_output} "
                f"{dataset}\n"
            )

            n_jobs += 1

    print()
    print(f"Arguments file    : {argument_file}")
    print(f"Jobs to submit    : {n_jobs}")

    if n_jobs == 0:
        print()
        print("All outputs already exist.")
        print("Nothing to submit.")
        return True

    # --------------------------------------------------------
    # Create Condor submit file
    # --------------------------------------------------------

    submit_file = submit_dir / "submit.sub"

    with open(submit_file, "w") as f:

        # ----------------------------------------------------
        # Executable
        # ----------------------------------------------------

        f.write(
            f"executable = "
            f"{os.path.abspath(EXECUTABLE)}\n"
        )

        f.write(
            "arguments = "
            "$(ROOT_FILE) "
            "$(LOCAL_OUTPUT) "
            "$(EOS_OUTPUT) "
            "$(PROCESS_NAME)\n\n"
        )

        # ----------------------------------------------------
        # stdout
        # ----------------------------------------------------

        f.write(
            f"output = "
            f"{dataset}.$(ClusterId).$(ProcId).out\n"
        )

        # ----------------------------------------------------
        # stderr
        # ----------------------------------------------------

        f.write(
            f"error = "
            f"{dataset}.$(ClusterId).$(ProcId).err\n"
        )

        # ----------------------------------------------------
        # Condor log
        # ----------------------------------------------------

        f.write(
            f"log = "
            f"{submit_dir}/{dataset}.$(ClusterId).log\n"
        )

        # ----------------------------------------------------
        # Send stdout/stderr to EOS
        # ----------------------------------------------------

        f.write(
            f"output_destination = "
            f"{condor_log_remote}\n\n"
        )

        # ----------------------------------------------------
        # Environment
        # ----------------------------------------------------

        f.write("getenv = True\n")
        f.write("use_x509userproxy = true\n\n")

        # ----------------------------------------------------
        # Resources
        # ----------------------------------------------------

        f.write("request_cpus = 1\n")
        f.write("request_memory = 4096 MB\n\n")

        # ----------------------------------------------------
        # Job flavour
        # ----------------------------------------------------

        f.write(
            '+JobFlavour = "workday"\n\n'
        )

        # ----------------------------------------------------
        # Dataset identification
        # ----------------------------------------------------

        f.write(
            f'+SkimmerDataset = "{dataset}"\n\n'
        )

        # ----------------------------------------------------
        # Retry / hold behaviour
        # ----------------------------------------------------

        f.write(
            "on_exit_remove = "
            "(ExitBySignal == False) && "
            "(ExitCode == 0)\n"
        )

        f.write(
            "on_exit_hold = "
            "(ExitBySignal == True) || "
            "(ExitCode != 0)\n"
        )

        f.write(
            "periodic_hold = "
            "(JobStatus == 7) && "
            "((CurrentTime - EnteredCurrentStatus) > 300)\n"
        )

        f.write(
            'periodic_hold_reason = '
            '"Job stuck suspended >5m"\n'
        )

        f.write(
            "periodic_release = "
            "(JobStatus == 5) && "
            "((CurrentTime - EnteredCurrentStatus) > 60)\n"
        )

        f.write("max_retries = 10\n")

        f.write(
            "requirements = "
            "Machine =!= LastRemoteHost\n\n"
        )

        # ----------------------------------------------------
        # File transfer
        # ----------------------------------------------------

        f.write(
            "should_transfer_files = YES\n"
        )

        f.write(
            "when_to_transfer_output = ON_EXIT\n"
        )

        # transfer_files = [
        #     os.path.abspath(EXECUTABLE),
        #     os.path.abspath(SKIM_SCRIPT),
        # ]

        transfer_files = [
            os.path.abspath(EXECUTABLE),
            os.path.abspath(SKIM_SCRIPT),
            os.path.abspath("analysis.py"),
            os.path.abspath("data_loader.py"),
            os.path.abspath("physics.py"),
            os.path.abspath("config.py"),
            os.path.abspath("plot_config.py"),
        ]

        f.write(
            "transfer_input_files = "
            + ",".join(transfer_files)
            + "\n\n"
        )

        # ----------------------------------------------------
        # Queue
        # ----------------------------------------------------

        f.write(
            "queue "
            "ROOT_FILE,LOCAL_OUTPUT,"
            "EOS_OUTPUT,PROCESS_NAME "
            f"from {argument_file}\n"
        )

    print()
    print(f"Submit file      : {submit_file}")

    # --------------------------------------------------------
    # Submit using spool
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        f"Submitting {n_jobs} Condor jobs "
        f"for {dataset}..."
    )
    print("=" * 70)

    subprocess.run(
        [
            "condor_submit",
            "-spool",
            str(submit_file),
        ],
        check=True,
    )

    print()
    print("=" * 70)
    print(f"Submission completed: {dataset}")
    print("=" * 70)

    print(f"Output directory: {eos_output_remote}")
    print(f"Condor logs     : {condor_log_remote}")

    return True


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 70)
    print("SKIM TO PARQUET CONDOR SUBMISSION")
    print("=" * 70)

    print()
    print("Datasets to submit:")

    for dataset in DATASETS:
        print(f"  - {dataset}")

    successful = []
    failed = []

    # --------------------------------------------------------
    # Submit each dataset
    # --------------------------------------------------------

    for dataset in DATASETS:

        try:

            success = submit_dataset(dataset)

            if success:
                successful.append(dataset)
            else:
                failed.append(dataset)

        except Exception as e:

            print()
            print("=" * 70)
            print(
                f"ERROR processing dataset: "
                f"{dataset}"
            )
            print("=" * 70)
            print(str(e))
            print()

            failed.append(dataset)

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SUBMISSION SUMMARY")
    print("=" * 70)

    print()
    print(
        f"Total datasets : {len(DATASETS)}"
    )

    print(
        f"Successful     : {len(successful)}"
    )

    print(
        f"Failed         : {len(failed)}"
    )

    if successful:

        print()
        print("Successfully submitted:")

        for dataset in successful:
            print(f"  ✓ {dataset}")

    if failed:

        print()
        print("Failed:")

        for dataset in failed:
            print(f"  ✗ {dataset}")

    print()
    print("=" * 70)

    if failed:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())