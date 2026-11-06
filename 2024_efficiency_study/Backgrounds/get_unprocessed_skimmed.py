import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

import uproot
import pyarrow.parquet as pq

from higgs_dna.utils.logger_utils import setup_logger


logger = setup_logger(level="INFO")


def get_fetcher_args():
    parser = argparse.ArgumentParser(
        description=(
            "Find unprocessed ROOT files by matching internal UUIDs "
            "and validating HiggsDNA Parquet outputs."
        )
    )

    parser.add_argument(
        "-s", "--source",
        required=True,
        help="Parent directory containing dataset directories.",
    )

    parser.add_argument(
        "-j", "--json",
        required=True,
        help="Input JSON mapping dataset names to ROOT file paths.",
    )

    parser.add_argument(
        "-o", "--output",
        default="unprocessed_samples.json",
        help="Output JSON path.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Check only the first N ROOT files per dataset.",
    )

    parser.add_argument(
        "--nominal-only",
        action="store_true",
        help="Check only the nominal systematic directory.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of worker threads (default: 8).",
    )

    return parser.parse_args()


def get_root_uuid(filepath):
    """Return the internal UUID of a ROOT file."""
    with uproot.open(filepath) as root_file:
        return str(root_file.file.uuid)


def get_parquet_files(directory):
    """
    Return a dictionary mapping Parquet UUIDs to file paths.

    Expected filename:
        <uuid>_Events_0-47767.parquet
    """
    parquet_files = {}

    if not os.path.isdir(directory):
        logger.warning(f"Systematic directory not found: {directory}")
        return parquet_files

    for filename in os.listdir(directory):
        if filename.endswith(".parquet") and "_Events_" in filename:
            parquet_uuid = filename.split("_Events_", 1)[0]
            parquet_files[parquet_uuid] = os.path.join(
                directory, filename
            )

    return parquet_files


def validate_parquet(filepath):
    """
    Validate a Parquet file by reading its metadata, schema,
    and all data pages in batches.

    Returns:
        filepath, valid, error
    """
    try:
        parquet_file = pq.ParquetFile(filepath)

        # Reading the schema and metadata is performed by ParquetFile.
        # Iterate over every batch to detect unreadable data pages.
        for _ in parquet_file.iter_batches(batch_size=65536):
            pass

        return filepath, True, None

    except Exception as exc:
        return filepath, False, str(exc)


def validate_parquet_files(parquet_by_syst, workers):
    """
    Validate each discovered Parquet file once.

    Returns:
        syst -> UUID -> (valid, error)
    """
    validation_results = {
        syst: {} for syst in parquet_by_syst
    }

    tasks = []

    for syst, uuid_to_path in parquet_by_syst.items():
        for parquet_uuid, filepath in uuid_to_path.items():
            tasks.append((syst, parquet_uuid, filepath))

    if not tasks:
        return validation_results

    logger.info(
        f"Validating {len(tasks)} Parquet files using "
        f"{workers} workers."
    )

    def run_validation(task):
        syst, parquet_uuid, filepath = task
        result = validate_parquet(filepath)
        return syst, parquet_uuid, result

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for syst, parquet_uuid, result in executor.map(
            run_validation, tasks
        ):
            filepath, valid, error = result

            validation_results[syst][parquet_uuid] = (
                valid, error
            )

            if not valid:
                logger.warning(
                    f"INVALID Parquet [{syst}]: {filepath}: {error}"
                )

    return validation_results


def check_root_file(filepath, validation_by_syst):
    """
    Check the ROOT UUID against every requested systematic.

    A ROOT file is considered processed only if its matching
    Parquet exists and passes validation in every requested
    systematic directory.
    """
    try:
        root_uuid = get_root_uuid(filepath)

        missing_systs = []
        invalid_systs = []

        for syst, uuid_results in validation_by_syst.items():
            result = uuid_results.get(root_uuid)

            if result is None:
                missing_systs.append(syst)
                continue

            valid, error = result

            if not valid:
                invalid_systs.append(syst)

        if missing_systs or invalid_systs:
            details = []

            if missing_systs:
                details.append(
                    "missing Parquet in: " + ", ".join(missing_systs)
                )

            if invalid_systs:
                details.append(
                    "invalid Parquet in: " + ", ".join(invalid_systs)
                )

            return filepath, False, "; ".join(details)

        return filepath, True, None

    except Exception as exc:
        return filepath, False, f"Could not inspect ROOT file: {exc}"


def main():
    args = get_fetcher_args()

    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")

    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be non-negative.")

    if not os.path.isfile(args.json):
        raise FileNotFoundError(
            f"Input JSON does not exist: {args.json}"
        )

    if not os.path.isdir(args.source):
        raise NotADirectoryError(
            f"Dataset parent directory does not exist: {args.source}"
        )

    with open(args.json) as f:
        samples = json.load(f)

    unprocessed = {}

    for dataset, all_root_files in samples.items():
        root_files = (
            all_root_files[:args.limit]
            if args.limit is not None
            else all_root_files
        )

        dataset_dir = os.path.join(args.source, dataset)

        if not os.path.isdir(dataset_dir):
            logger.warning(
                f"Dataset directory not found: {dataset_dir}"
            )
            unprocessed[dataset] = root_files
            continue

        if args.nominal_only:
            systematic_dirs = ["nominal"]
        else:
            systematic_dirs = sorted(
                entry
                for entry in os.listdir(dataset_dir)
                if os.path.isdir(
                    os.path.join(dataset_dir, entry)
                )
            )

        if not systematic_dirs:
            logger.warning(
                f"No systematic directories found in {dataset_dir}"
            )
            unprocessed[dataset] = root_files
            continue

        # Discover Parquet files for each systematic.
        parquet_by_syst = {
            syst: get_parquet_files(
                os.path.join(dataset_dir, syst)
            )
            for syst in systematic_dirs
        }

        logger.info(
            f"Dataset '{dataset}': {len(root_files)} ROOT files; "
            f"{len(systematic_dirs)} systematic directories."
        )

        # Validate each Parquet file once, in parallel.
        validation_by_syst = validate_parquet_files(
            parquet_by_syst, args.workers
        )

        missing_files = []
        matched_files = 0

        # Match ROOT UUIDs against validated Parquet outputs.
        with ThreadPoolExecutor(
            max_workers=args.workers
        ) as executor:
            results = executor.map(
                lambda filepath: check_root_file(
                    filepath, validation_by_syst
                ),
                root_files,
            )

            for filepath, matched, error in results:
                if matched:
                    matched_files += 1
                else:
                    missing_files.append(filepath)

                    if error:
                        logger.warning(
                            f"{filepath}: {error}"
                        )

        logger.info(
            f"{dataset}: {matched_files}/{len(root_files)} "
            f"ROOT files have valid matching Parquet outputs."
        )

        if missing_files:
            unprocessed[dataset] = missing_files

    if unprocessed:
        with open(args.output, "w") as f:
            json.dump(unprocessed, f, indent=4)

        logger.info(
            f"Unprocessed file list saved to: {args.output}"
        )
    else:
        logger.info(
            "All checked ROOT files have valid matching Parquet outputs."
        )
        logger.info("No output JSON was created.")


if __name__ == "__main__":
    main()

