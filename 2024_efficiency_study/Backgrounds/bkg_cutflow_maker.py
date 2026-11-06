#!/usr/bin/env python3
"""
Process cumulative BBgg background cutflows from Parquet files.

Stage 1 (expensive):
  Parquet -> batched Awkward arrays -> weighted ROOT histograms + cutflow.
  Saves:
    cutflow_histograms.root
    cutflow_metadata.json
    cutflow_yields.csv

After this has run once, use plot_from_cache.py to change plot styling
without reading the Parquet files again.
"""

import os
import glob
import json
from datetime import datetime
from array import array

import numpy as np
import awkward as ak
import pyarrow.parquet as pq
import ROOT


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = (
    "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/"
    "Backgrounds/NTuples_2024_BKG_step_by_step_bool_cumulative_latest/merged/"
)

OUTPUT_DIR = (
    "/eos/user/b/bbapi/www/Analysis_plots/BKG_cutflow/"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

LUMI_FB = 109.0
BATCH_SIZE = 100_000
WEIGHT_BRANCH = "weight"

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)
ROOT.gStyle.SetLegendBorderSize(0)


# ============================================================
# CROSS SECTIONS (pb)
# ============================================================

XSECS = {
    "WGtoLNuG": 671.5,
    "WtoENu0J": 55850,
    "WtoENu1J": 9177,
    "WtoENu2J": 3474,
    "WtoMuNu0J": 55920,
    "WtoMuNu1J": 9202,
    "WtoMuNu2J": 3490,
    "DYto2Mu10": 21190.0,
    "DYto2Mu50": 2124.08,
    "DYto2E10": 21140.0,
    "DYto2E50": 2124.08,
    "DYGto2LG4": 88.13,
    "DYGto2LG50": 126.7,
    "TTG1Jets": 4.634,
    "TTto2L2Nu": 98.04,
    "TTtoLNu2Q": 405.87,
}


# ============================================================
# CUMULATIVE SELECTIONS
# Each mask is already cumulative. Do not AND it with earlier
# masks again.
# ============================================================

SELECTIONS = [
    ("Lepton pT cut", None),
    ("Muon ID", "pass_apply_muon_id"),
    ("Electron ID", "pass_apply_electron_id"),
    ("Muon Iso", "pass_apply_muon_iso"),
    ("Global Muon", "pass_apply_global_muon"),
    ("Photon pT", "pass_apply_photon_pt"),
    ("Photon MVA ID", "pass_apply_photon_mvaid"),
    ("Electron-Photon dR", "pass_apply_electron_photon_dr"),
    ("Muon-Photon dR", "pass_apply_muon_photon_dr"),
    ("Jet pT/Eta", "pass_apply_jet_pt_eta"),
    ("Jet ID", "pass_apply_jet_id"),
    ("Photon-Jet dR", "pass_apply_photon_jet_dr"),
    ("Electron-Jet dR", "pass_apply_electron_jet_dr"),
    ("Muon-Jet dR", "pass_apply_muon_jet_dr"),
    ("Medium b-tag", "medium_btag_mask"),
    ("At least 2 b-jets", "atleast_two_bjets_mask"),
]

STAGE_NAMES = [stage for stage, _ in SELECTIONS]
N_STAGES = len(SELECTIONS)


# ============================================================
# VARIABLES
# ============================================================

VARIABLES = [
    "leppt",
    "lepeta",
    "first_jet_eta",
    "second_jet_eta",
    "pT1_by_mbb",
    "pT2_by_mbb",
    "first_jet_B",
    "second_jet_B",
    "Njets",
    "pholead_eta",
    "phosublead_eta",
    "pT1_by_mgg",
    "pT2_by_mgg",
    "pholead_mvaID",
    "phosublead_mvaID",
    "delphi_gg",
    "delphi_bb",
    "delphi_bbgg",
    "delphi_ggMET_Puppi",
    "mass",
]


# ============================================================
# PLOT CONFIGURATION
# ============================================================

PLOT_CONFIG = {
    "leppt": {
        "range": (0.0, 200.0),
        "nbins": 40,
        "xlabel": "Lepton p_{T} [GeV]",
    },
    "lepeta": {
        "range": (-2.5, 2.5),
        "nbins": 25,
        "xlabel": "Lepton #eta",
    },
    "first_jet_eta": {
        "range": (-4.0, 4.0),
        "nbins": 20,
        "xlabel": "Leading b-Jet #eta",
    },
    "second_jet_eta": {
        "range": (-4.0, 4.0),
        "nbins": 20,
        "xlabel": "Subleading b-Jet #eta",
    },
    "pT1_by_mbb": {
        "range": (0.0, 5.0),
        "nbins": 20,
        "xlabel": "Leading jet p_{T} / m_{bb}",
    },
    "pT2_by_mbb": {
        "range": (0.0, 3.0),
        "nbins": 12,
        "xlabel": "Subleading jet p_{T} / m_{bb}",
    },
    "first_jet_B": {
        "range": (0.0, 1.0),
        "nbins": 40,
        "xlabel": "Leading b-Jet B-tag score",
    },
    "second_jet_B": {
        "range": (0.0, 1.0),
        "nbins": 40,
        "xlabel": "Subleading b-Jet B-tag score",
    },
    "Njets": {
        "range": (0, 10),
        "nbins": 10,
        "xlabel": "Number of Jets per event",
    },
    "pholead_eta": {
        "range": (-2.5, 2.5),
        "nbins": 25,
        "xlabel": "Leading photon #eta",
    },
    "phosublead_eta": {
        "range": (-2.5, 2.5),
        "nbins": 25,
        "xlabel": "Subleading photon #eta",
    },
    "pT1_by_mgg": {
        "range": (0.0, 10.0),
        "nbins": 40,
        "xlabel": "Leading photon p_{T} / m_{#gamma#gamma}",
    },
    "pT2_by_mgg": {
        "range": (0.0, 6.0),
        "nbins": 24,
        "xlabel": "Subleading photon p_{T} / m_{#gamma#gamma}",
    },
    "pholead_mvaID": {
        "range": (-1.0, 1.0),
        "nbins": 40,
        "xlabel": "Leading photon MVA ID",
    },
    "phosublead_mvaID": {
        "range": (-1.0, 1.0),
        "nbins": 40,
        "xlabel": "Subleading photon MVA ID",
    },
    "delphi_gg": {
        "range": (0.0, 3.2),
        "nbins": 32,
        "xlabel": "#Delta#phi_{#gamma#gamma}",
    },
    "delphi_bb": {
        "range": (0.0, 3.2),
        "nbins": 32,
        "xlabel": "#Delta#phi_{bb}",
    },
    "delphi_bbgg": {
        "range": (0.0, 3.2),
        "nbins": 32,
        "xlabel": "#Delta#phi_{bb,#gamma#gamma}",
    },
    "delphi_ggMET_Puppi": {
        "range": (0.0, 3.2),
        "nbins": 16,
        "xlabel": "#Delta#phi between #gamma#gamma and MET",
    },
    "mass": {
        "range": (10.0, 70.0),
        "nbins": 60,
        "xlabel": "Diphoton invariant mass (m_{#gamma#gamma}) [GeV]",
    },
}


# ============================================================
# HELPERS
# ============================================================

def to_numpy_float(values):
    """Convert a one-dimensional Awkward field to float64; None -> NaN."""
    values = ak.fill_none(values, np.nan)
    return np.asarray(ak.to_numpy(values), dtype=np.float64)


def to_numpy_bool(values):
    """Convert a one-dimensional Awkward mask to bool; None -> False."""
    values = ak.fill_none(values, False)
    return np.asarray(ak.to_numpy(values), dtype=np.bool_)


def fill_root_hist(hist, values, weights):
    """Bulk-fill a ROOT TH1 using finite values and weights only."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    valid = np.isfinite(values) & np.isfinite(weights)
    values = values[valid]
    weights = weights[valid]

    if values.size == 0:
        return

    hist.FillN(
        len(values),
        array("d", values),
        array("d", weights),
        1,
    )


def discover_files():
    files_by_process = {}

    for process in XSECS:
        directory = os.path.join(
            BASE_DIR,
            f"{process}_24SummerRun3",
        )
        paths = sorted(glob.glob(os.path.join(directory, "*.parquet")))

        if paths:
            files_by_process[process] = paths
        else:
            print(f"WARNING: No Parquet files found for {process}")

    if not files_by_process:
        raise RuntimeError(f"No Parquet files found under {BASE_DIR}")

    return files_by_process


def initialize_histograms():
    result = {}

    for var in VARIABLES:
        cfg = PLOT_CONFIG[var]
        xmin, xmax = cfg["range"]
        result[var] = []

        for istage, stage in enumerate(STAGE_NAMES):
            hist = ROOT.TH1D(
                f"h_{var}_{istage}",
                f"{var};{cfg['xlabel']};Expected events / bin",
                cfg["nbins"],
                xmin,
                xmax,
            )
            hist.Sumw2()
            hist.SetDirectory(0)
            result[var].append(hist)

    return result


def save_outputs(histograms, cutflow_yields, cutflow_sumw2, files_by_process):
    root_filename = os.path.join(OUTPUT_DIR, "cutflow_histograms.root")
    metadata_filename = os.path.join(OUTPUT_DIR, "cutflow_metadata.json")
    csv_filename = os.path.join(OUTPUT_DIR, "cutflow_yields.csv")

    h_cutflow = ROOT.TH1D(
        "h_cutflow",
        "Background Cutflow;Selection;Expected events",
        N_STAGES,
        0,
        N_STAGES,
    )
    h_cutflow.Sumw2()

    for i, stage in enumerate(STAGE_NAMES):
        h_cutflow.GetXaxis().SetBinLabel(i + 1, stage)
        h_cutflow.SetBinContent(i + 1, float(cutflow_yields[i]))
        h_cutflow.SetBinError(i + 1, float(np.sqrt(cutflow_sumw2[i])))

    input_files = {
        process: paths
        for process, paths in files_by_process.items()
    }

    metadata = {
        "processing_date": datetime.now().astimezone().isoformat(),
        "base_directory": BASE_DIR,
        "output_directory": OUTPUT_DIR,
        "luminosity_fb_inverse": LUMI_FB,
        "cross_sections_pb": XSECS,
        "weight_branch": WEIGHT_BRANCH,
        "weight_formula": (
            "weight * cross_section_pb * luminosity_fb_inverse * 1000"
        ),
        "weight_assumption": (
            "The weight branch already contains the normalized generator "
            "weight and scale factors; no additional sum_genw_presel "
            "division is applied."
        ),
        "batch_size": BATCH_SIZE,
        "input_files": input_files,
        "selections": [
            {"stage": stage, "mask": mask}
            for stage, mask in SELECTIONS
        ],
        "variables": VARIABLES,
        "plot_config": {
            var: {
                "range": list(cfg["range"]),
                "nbins": cfg["nbins"],
                "xlabel": cfg["xlabel"],
            }
            for var, cfg in PLOT_CONFIG.items()
        },
        "cutflow_yields": cutflow_yields.tolist(),
        "cutflow_sumw2": cutflow_sumw2.tolist(),
        "cutflow_uncertainties": np.sqrt(cutflow_sumw2).tolist(),
    }

    output_file = ROOT.TFile(root_filename, "RECREATE")
    if not output_file or output_file.IsZombie():
        raise RuntimeError(f"Could not create ROOT file: {root_filename}")

    output_file.cd()
    h_cutflow.Write()

    for var in VARIABLES:
        directory = output_file.mkdir(var)
        directory.cd()
        for hist in histograms[var]:
            hist.Write()

    output_file.cd()
    ROOT.TObjString(json.dumps(metadata, indent=4)).Write(
        "processing_metadata"
    )
    output_file.Close()

    with open(metadata_filename, "w") as stream:
        json.dump(metadata, stream, indent=4)

    with open(csv_filename, "w") as stream:
        stream.write("Selection,Yield,Statistical_uncertainty\n")
        for i, stage in enumerate(STAGE_NAMES):
            stream.write(
                f"{stage},{cutflow_yields[i]:.10f},"
                f"{np.sqrt(cutflow_sumw2[i]):.10f}\n"
            )

    return root_filename, metadata_filename, csv_filename


# ============================================================
# MAIN PROCESSING
# ============================================================

def main():
    files_by_process = discover_files()
    histograms = initialize_histograms()

    cutflow_yields = np.zeros(N_STAGES, dtype=np.float64)
    cutflow_sumw2 = np.zeros(N_STAGES, dtype=np.float64)

    required = list(dict.fromkeys(
        VARIABLES
        + [WEIGHT_BRANCH]
        + [mask for _, mask in SELECTIONS if mask is not None]
    ))

    print("\nBackground samples:")
    for process, paths in files_by_process.items():
        print(f"{process:20s}: {len(paths)} files")

    for process, paths in files_by_process.items():
        normalization = XSECS[process] * LUMI_FB * 1000.0

        print("\n" + "=" * 70)
        print(f"Processing background: {process}")
        print(f"Cross section: {XSECS[process]} pb")
        print(f"Luminosity: {LUMI_FB} fb^-1")
        print(f"Normalization multiplier: {normalization}")
        print("=" * 70)

        for ifile, filename in enumerate(paths, start=1):
            print(f"\nFile {ifile}/{len(paths)}: {os.path.basename(filename)}")

            parquet_file = pq.ParquetFile(filename)
            available = set(parquet_file.schema.names)
            missing = set(required) - available

            if missing:
                raise RuntimeError(
                    f"Missing branches in {filename}:\n{sorted(missing)}"
                )

            file_events = 0

            for ibatch, batch in enumerate(
                parquet_file.iter_batches(
                    batch_size=BATCH_SIZE,
                    columns=required,
                ),
                start=1,
            ):
                events = ak.from_arrow(batch)
                n_events = len(events)
                file_events += n_events

                # The weight branch is already normalized at production level.
                weights = to_numpy_float(events[WEIGHT_BRANCH])
                weights = weights * normalization

                values_dict = {
                    var: to_numpy_float(events[var])
                    for var in VARIABLES
                }

                for istage, (_, mask_name) in enumerate(SELECTIONS):
                    if mask_name is None:
                        mask = np.ones(n_events, dtype=np.bool_)
                    else:
                        mask = to_numpy_bool(events[mask_name])

                    if len(mask) != n_events:
                        raise RuntimeError(
                            f"Invalid mask length for {mask_name} in {filename}"
                        )

                    selected_weights = weights[mask]
                    valid_weights = np.isfinite(selected_weights)
                    selected_weights = selected_weights[valid_weights]

                    if selected_weights.size == 0:
                        continue

                    cutflow_yields[istage] += np.sum(
                        selected_weights,
                        dtype=np.float64,
                    )
                    cutflow_sumw2[istage] += np.sum(
                        selected_weights * selected_weights,
                        dtype=np.float64,
                    )

                    for var in VARIABLES:
                        selected_values = values_dict[var][mask]
                        selected_values = selected_values[valid_weights]

                        fill_root_hist(
                            histograms[var][istage],
                            selected_values,
                            selected_weights,
                        )

                if ibatch % 10 == 0:
                    print(
                        f"  Processed {ibatch} batches "
                        f"({file_events:,} events in this file)"
                    )

            print(
                f"Completed {os.path.basename(filename)} "
                f"({file_events:,} events)"
            )

    print("\n" + "=" * 90)
    print("TOTAL BACKGROUND CUTFLOW")
    print("=" * 90)
    print(f"{'Selection':35s}{'Yield':>20s}{'Stat. uncertainty':>25s}")
    print("-" * 90)

    for i, stage in enumerate(STAGE_NAMES):
        print(
            f"{stage:35s}"
            f"{cutflow_yields[i]:20.6f}"
            f"{np.sqrt(cutflow_sumw2[i]):25.6f}"
        )

    print("=" * 90)

    root_file, metadata_file, csv_file = save_outputs(
        histograms,
        cutflow_yields,
        cutflow_sumw2,
        files_by_process,
    )

    print("\nPROCESSING COMPLETED")
    print(f"ROOT histograms : {root_file}")
    print(f"Metadata JSON   : {metadata_file}")
    print(f"Cutflow CSV     : {csv_file}")
    print(f"Final yield     : {cutflow_yields[-1]:.6f} "
          f"+/- {np.sqrt(cutflow_sumw2[-1]):.6f}")
    print("\nYou can now use plot_from_cache.py to redraw plots "
          "without rereading Parquet files.")


if __name__ == "__main__":
    main()
