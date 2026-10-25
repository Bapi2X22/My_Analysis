import argparse
import time

import numpy as np
import awkward as ak
import uproot
import pyarrow as pa
import pyarrow.parquet as pq

from data_loader import load_root_file
from analysis import analyze_events
from physics import is_allowed_parent, is_allowed_parent_TT


def get_sum_genweight_from_metadata(filename):

    with uproot.open(filename) as f:

        metadata = f["Metadata"]

        sum_genw_presel = metadata[
            "sum_genw_presel"
        ].array(
            library="np"
        )[0]

    return float(
        sum_genw_presel
    )


def get_parentage_function(process):

    if process in {
        "TTto2L2Nu",
        "TTtoLNu2Q",
        "TTG1Jets"
    }:

        return is_allowed_parent_TT

    return is_allowed_parent


def result_to_parquet(
    result,
    output_file,
    sum_genw_presel
):

    data = {}

    for field in result.__dataclass_fields__:

        value = getattr(
            result,
            field
        )

        if isinstance(value, ak.Array):

            data[field] = value

        else:

            data[field] = np.asarray(
                value
            )

    # ------------------------------------------------------------
    # Check event alignment
    # ------------------------------------------------------------

    print()
    print("Checking parquet column lengths...")

    lengths = {}

    for name, value in data.items():

        lengths[name] = len(value)

        print(
            f"{name:45s} : {len(value)}"
        )

    unique_lengths = set(
        lengths.values()
    )

    if len(unique_lengths) != 1:

        print()
        print(
            "ERROR: AnalysisResult contains "
            "columns with different lengths!"
        )

        for name, length in lengths.items():

            print(
                f"{name:45s} : {length}"
            )

        raise RuntimeError(
            "All parquet columns must have "
            "the same number of events."
        )

    n_events = next(
        iter(unique_lengths)
    )

    print()
    print(
        f"All columns contain {n_events} events."
    )

    # ------------------------------------------------------------
    # Arrow table
    # ------------------------------------------------------------

    table = pa.table(
        data
    )

    # ------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------

    metadata = {
        b"sum_genw_presel":
            str(
                sum_genw_presel
            ).encode("utf-8")
    }

    table = table.replace_schema_metadata(
        metadata
    )

    # ------------------------------------------------------------
    # Write
    # ------------------------------------------------------------

    pq.write_table(
        table,
        output_file,
        compression=None
    )

    print()
    print(
        f"Written parquet: {output_file}"
    )


def process_file(
    filename,
    output_file,
    sum_genw_presel,
    photon_pt_min,
    photon_eta_max,
    other_pt_min,
    parentage_func
):

    print()
    print("=" * 70)
    print(
        f"Input : {filename}"
    )
    print(
        f"Output: {output_file}"
    )
    print("=" * 70)

    t0 = time.perf_counter()

    # ------------------------------------------------------------
    # Load ROOT
    # ------------------------------------------------------------

    events = load_root_file(
        filename
    )

    t1 = time.perf_counter()

    print(
        f"Load time          : "
        f"{t1 - t0:.2f} s"
    )

    # ------------------------------------------------------------
    # Raw genWeight
    #
    # NO cross section
    # NO luminosity
    # NO normalization
    # ------------------------------------------------------------

    genweight = ak.to_numpy(
        events.genWeight
    )

    # ------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------

    result = analyze_events(
        events,
        genweight,
        photon_pt_min,
        photon_eta_max,
        other_pt_min,
        parentage_func
    )

    t2 = time.perf_counter()

    print(
        f"Analysis time      : "
        f"{t2 - t1:.2f} s"
    )

    # ------------------------------------------------------------
    # Write parquet
    # ------------------------------------------------------------

    result_to_parquet(
        result,
        output_file,
        sum_genw_presel
    )

    t3 = time.perf_counter()

    print(
        f"Parquet write time : "
        f"{t3 - t2:.2f} s"
    )

    print(
        f"Total time         : "
        f"{t3 - t0:.2f} s"
    )

    del result
    del genweight
    del events


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Process one ROOT file "
            "and save it to Parquet"
        )
    )

    parser.add_argument(
        "--input-file",
        required=True,
        help="Input ROOT file"
    )

    parser.add_argument(
        "--output-file",
        required=True,
        help="Output Parquet file"
    )

    parser.add_argument(
        "--process",
        required=True,
        help="Process name"
    )

    parser.add_argument(
        "--photon-pt-min",
        type=float,
        default=10.0
    )

    parser.add_argument(
        "--photon-eta-max",
        type=float,
        default=3.0
    )

    parser.add_argument(
        "--other-pt-min",
        type=float,
        default=5.0
    )

    args = parser.parse_args()

    # ------------------------------------------------------------
    # Parentage function
    # ------------------------------------------------------------

    parentage_func = get_parentage_function(
        args.process
    )

    print()
    print("=" * 70)
    print(
        f"Process         : {args.process}"
    )
    print(
        f"Input file      : {args.input_file}"
    )
    print(
        f"Output file     : {args.output_file}"
    )
    print(
        f"Parentage       : "
        f"{parentage_func.__name__}"
    )
    print("=" * 70)

    # ------------------------------------------------------------
    # Read sum_genw_presel from ROOT metadata
    # ------------------------------------------------------------

    sum_genw_presel = (
        get_sum_genweight_from_metadata(
            args.input_file
        )
    )

    print(
        f"sum_genw_presel : "
        f"{sum_genw_presel:.6e}"
    )

    # ------------------------------------------------------------
    # Process
    # ------------------------------------------------------------

    process_file(
        filename=args.input_file,
        output_file=args.output_file,
        sum_genw_presel=sum_genw_presel,
        photon_pt_min=args.photon_pt_min,
        photon_eta_max=args.photon_eta_max,
        other_pt_min=args.other_pt_min,
        parentage_func=parentage_func
    )

    print()
    print("=" * 70)
    print("Finished")
    print("=" * 70)


if __name__ == "__main__":

    main()