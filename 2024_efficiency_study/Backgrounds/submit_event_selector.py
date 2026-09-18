#!/usr/bin/env python3

import os
import subprocess
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

DATASET = "TTtoLNu2Q_24SummerRun3"

# ------------------------------------------------------------
# EOS input directory
# ------------------------------------------------------------

INPUT_DIR = (
    "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/"
    "Backgrounds/Skimmer/TTtoLNu2Q_24SummerRun3"
)

# ------------------------------------------------------------
# EOS output directory
# ------------------------------------------------------------

EOS_OUTPUT_DIR = (
    "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/"
    "Backgrounds/TTtoLNu2Q_24SummerRun3_selected2"
)

# Remote EOS URL
EOS_OUTPUT_REMOTE = (
    "root://eosuser.cern.ch//"
    + EOS_OUTPUT_DIR.lstrip("/")
)

# ------------------------------------------------------------
# Condor logs
# ------------------------------------------------------------

CONDOR_LOG_DIR = (
    "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/"
    "Backgrounds/CondorLogs/event_selector/"
    + DATASET
)

CONDOR_LOG_REMOTE = (
    "root://eosuser.cern.ch//"
    + CONDOR_LOG_DIR.lstrip("/")
)

# ------------------------------------------------------------
# Number of ROOT files per Condor job
# ------------------------------------------------------------

CHUNK_SIZE = 10

# ------------------------------------------------------------
# Files
# ------------------------------------------------------------

EXECUTABLE = "run_event_selector.sh"
EVENT_SELECTOR = "event_selector.py"


# ============================================================
# Helper
# ============================================================

def eos_mkdir(path):
    """
    Create an EOS directory using xrdfs.
    """

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


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 70)
    print(f"Preparing dataset: {DATASET}")
    print("=" * 70)

    # --------------------------------------------------------
    # Check executable files
    # --------------------------------------------------------

    if not Path(EXECUTABLE).exists():
        raise FileNotFoundError(
            f"Cannot find executable: {EXECUTABLE}"
        )

    if not Path(EVENT_SELECTOR).exists():
        raise FileNotFoundError(
            f"Cannot find event_selector.py: {EVENT_SELECTOR}"
        )

    # --------------------------------------------------------
    # Get ROOT files
    # --------------------------------------------------------

    input_dir = Path(INPUT_DIR)

    input_files = sorted(
        str(f)
        for f in input_dir.glob("*.root")
    )

    if not input_files:

        print(
            f"ERROR: No ROOT files found in:\n"
            f"{INPUT_DIR}"
        )

        return 1

    # --------------------------------------------------------
    # Dataset information
    # --------------------------------------------------------

    print(
        f"Dataset          : {DATASET}"
    )

    print(
        f"Input directory   : {INPUT_DIR}"
    )

    print(
        f"Number of files   : {len(input_files)}"
    )

    print(
        f"Chunk size        : {CHUNK_SIZE}"
    )

    # --------------------------------------------------------
    # Make chunks
    # --------------------------------------------------------

    chunks = [
        input_files[i:i + CHUNK_SIZE]
        for i in range(
            0,
            len(input_files),
            CHUNK_SIZE
        )
    ]

    print(
        f"Number of jobs    : {len(chunks)}"
    )

    # --------------------------------------------------------
    # EOS directories
    # --------------------------------------------------------

    eos_mkdir(
        EOS_OUTPUT_DIR
    )

    eos_mkdir(
        CONDOR_LOG_DIR
    )

    print(
        f"EOS output        : {EOS_OUTPUT_REMOTE}"
    )

    print(
        f"Condor logs       : {CONDOR_LOG_REMOTE}"
    )

    # --------------------------------------------------------
    # Local Condor directory
    # --------------------------------------------------------

    submit_dir = Path(
        f"condor_{DATASET}"
    )

    submit_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"Submit directory  : {submit_dir}"
    )

    # --------------------------------------------------------
    # Create batch file lists
    # --------------------------------------------------------

    print()
    print(
        "-" * 70
    )

    print(
        "Creating batch file lists"
    )

    print(
        "-" * 70
    )

    batch_files = []

    for batch_number, chunk in enumerate(chunks):

        batch_file = (
            submit_dir
            / f"batch_{batch_number:05d}.txt"
        )

        with open(batch_file, "w") as f:

            for root_file in chunk:
                f.write(
                    root_file + "\n"
                )

        batch_files.append(
            batch_file
        )

        print(
            f"Batch {batch_number:05d}: "
            f"{len(chunk)} ROOT files"
        )

    # --------------------------------------------------------
    # Create arguments file
    # --------------------------------------------------------

    argument_file = (
        submit_dir
        / "arguments.txt"
    )

    with open(argument_file, "w") as f:

        for batch_number in range(
            len(chunks)
        ):

            batch_name = (
                f"batch_{batch_number:05d}.txt"
            )

            local_output = (
                f"part_{batch_number:05d}.parquet"
            )

            eos_output = (
                f"{EOS_OUTPUT_REMOTE}/"
                f"{local_output}"
            )

            f.write(
                f"{batch_name} "
                f"{local_output} "
                f"{eos_output}\n"
            )

    print()
    print(
        f"Argument file    : {argument_file}"
    )

    # --------------------------------------------------------
    # Create Condor submit file
    # --------------------------------------------------------

    submit_file = (
        submit_dir
        / "submit.sub"
    )

    with open(
        submit_file,
        "w"
    ) as f:

        # ----------------------------------------------------
        # Executable
        # ----------------------------------------------------

        f.write(
            f"executable = "
            f"{os.path.abspath(EXECUTABLE)}\n"
        )

        f.write(
            "arguments = $(FILE_LIST) "
            "$(LOCAL_OUTPUT) "
            "$(EOS_OUTPUT)\n\n"
        )

        # ----------------------------------------------------
        # stdout
        # ----------------------------------------------------

        f.write(
            f"output = "
            f"{DATASET}.$(ClusterId).$(ProcId).out\n"
        )

        # ----------------------------------------------------
        # stderr
        # ----------------------------------------------------

        f.write(
            f"error = "
            f"{DATASET}.$(ClusterId).$(ProcId).err\n"
        )

        # ----------------------------------------------------
        # Condor log
        # ----------------------------------------------------

        f.write(
            f"log = "
            f"{submit_dir}/"
            f"{DATASET}.$(ClusterId).log\n"
        )

        # ----------------------------------------------------
        # Send stdout/stderr to EOS
        # ----------------------------------------------------

        f.write(
            f"output_destination = "
            f"{CONDOR_LOG_REMOTE}\n\n"
        )

        # ----------------------------------------------------
        # Environment
        # ----------------------------------------------------

        f.write(
            "getenv = True\n"
        )

        f.write(
            "use_x509userproxy = true\n\n"
        )

        # ----------------------------------------------------
        # Resources
        # ----------------------------------------------------

        f.write(
            "request_cpus = 1\n"
        )

        f.write(
            "request_memory = 4096 MB\n\n"
        )

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
            f'+SkimmerDataset = '
            f'"{DATASET}"\n\n'
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

        f.write(
            "max_retries = 10\n"
        )

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

        # Transfer executable, selector and all batch lists
        transfer_files = [
            os.path.abspath(EXECUTABLE),
            os.path.abspath(EVENT_SELECTOR),
        ]

        transfer_files += [
            os.path.abspath(batch_file)
            for batch_file in batch_files
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
            f"queue FILE_LIST,LOCAL_OUTPUT,"
            f"EOS_OUTPUT from {argument_file}\n"
        )

    print()
    print(
        f"Submit file     : {submit_file}"
    )

    # --------------------------------------------------------
    # Submit using spool
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        f"Submitting {len(chunks)} Condor jobs..."
    )
    print("=" * 70)

    subprocess.run(
        [
            "condor_submit",
            "-spool",
            str(submit_file)
        ],
        check=True
    )

    print()
    print(
        "=" * 70
    )

    print(
        "Submission completed."
    )

    print(
        f"Output directory: {EOS_OUTPUT_REMOTE}"
    )

    print(
        f"Condor logs     : {CONDOR_LOG_REMOTE}"
    )

    print(
        "=" * 70
    )

    return 0


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    raise SystemExit(
        main()
    )
