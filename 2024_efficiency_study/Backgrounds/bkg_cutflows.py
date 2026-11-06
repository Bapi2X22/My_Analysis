
import os
import glob
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
    "Backgrounds/NTuples_2024_BKG_step_by_step_bool_cumulative/merged/"
)

OUTPUT_DIR = (
    "/eos/user/b/bbapi/www/Analysis_plots/BKG_cutflow/"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

LUMI = 109.0  # pb^-1

BATCH_SIZE = 100_000

WEIGHT_BRANCH = "weight"


# ============================================================
# ROOT CONFIGURATION
# ============================================================

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)
ROOT.gStyle.SetLegendBorderSize(0)


# ============================================================
# CROSS SECTIONS (pb)
# ============================================================

xsecs = {
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
# BACKGROUND FILE DISCOVERY
# ============================================================

FILES = {}

for process in xsecs:

    dirname = f"{process}_24SummerRun3"

    directory = os.path.join(BASE_DIR, dirname)

    files = sorted(
        glob.glob(os.path.join(directory, "*.parquet"))
    )

    if files:
        FILES[process] = files
    else:
        print(
            f"WARNING: No Parquet files found for {process}"
        )


print("\nBackground samples:")

for process, paths in FILES.items():
    print(f"{process:20s}: {len(paths)} files")


if not FILES:
    raise RuntimeError("No Parquet files found!")


# ============================================================
# CUMULATIVE SELECTIONS
# ============================================================

# Each mask is already cumulative.
# Therefore, DO NOT combine it with previous masks again.

SELECTIONS = [

    ("No selection", None),

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

STAGE_NAMES = [x[0] for x in SELECTIONS]

N_STAGES = len(SELECTIONS)


# ============================================================
# VARIABLES
# ============================================================

variables_list = [

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

    "mass"

]


# ============================================================
# PLOT CONFIGURATION
# ============================================================

plot_config = {

    # Lepton

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


    # Jets

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


    # Photons

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


    # Angular variables

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
        "range": (0, 3.2),
        "nbins": 16,
        "xlabel": "#Delta#phi between #gamma#gamma and MET",
    },

    "mass": {
        "range": (10, 70),
        "nbins": 60,
        "xlabel": "Diphoton invariant mass (m_{#gamma#gamma}) [GeV]",
    }

}


# ============================================================
# INITIALIZE ROOT HISTOGRAMS
# ============================================================

histograms = {}

for var in variables_list:

    cfg = plot_config[var]

    xmin, xmax = cfg["range"]
    nbins = cfg["nbins"]

    histograms[var] = []

    for i, stage in enumerate(STAGE_NAMES):

        h = ROOT.TH1D(
            f"h_{var}_{i}",
            f"{var};{cfg['xlabel']};Expected events / bin",
            nbins,
            xmin,
            xmax
        )

        h.Sumw2()
        h.SetDirectory(0)

        histograms[var].append(h)


# ============================================================
# CUT-FLOW YIELD ACCUMULATORS
# ============================================================

# Independent of histogram ranges.
# Includes all events passing each cumulative selection.

cutflow_yields = np.zeros(N_STAGES, dtype=np.float64)

cutflow_sumw2 = np.zeros(N_STAGES, dtype=np.float64)


# ============================================================
# HELPER: CONVERT AWKWARD ARRAY TO NUMPY
# ============================================================

def to_numpy_float(values):

    values = ak.fill_none(values, np.nan)

    return np.asarray(
        ak.to_numpy(values),
        dtype=np.float64
    )


def to_numpy_bool(values):

    values = ak.fill_none(values, False)

    return np.asarray(
        ak.to_numpy(values),
        dtype=np.bool_
    )


# ============================================================
# HELPER: FILL ROOT HISTOGRAM IN BULK
# ============================================================

def fill_root_hist(hist, values, weights):

    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    valid = (
        np.isfinite(values)
        & np.isfinite(weights)
    )

    values = values[valid]
    weights = weights[valid]

    if len(values) == 0:
        return

    # ROOT TH1::FillN
    # Use array('d') for reliable PyROOT conversion.

    hist.FillN(
        len(values),
        array("d", values),
        array("d", weights),
        1
    )


# ============================================================
# PROCESS EACH BACKGROUND
# ============================================================

for process, files in FILES.items():

    print("\n" + "=" * 65)
    print(f"Processing background: {process}")
    print("=" * 65)

    # weight branch already contains the normalized
    # generator weight and scale factors.
    #
    # Multiply only by cross section and luminosity.

    normalization = xsecs[process] * LUMI * 1000

    print("Cross section:", xsecs[process], "pb")
    print("Luminosity:", LUMI, "pb^-1")
    print("Normalization:", normalization)

    for ifile, filename in enumerate(files):

        print(
            f"\nFile {ifile + 1}/{len(files)}: "
            f"{os.path.basename(filename)}"
        )

        pf = pq.ParquetFile(filename)

        available = set(pf.schema.names)

        required = (
            variables_list
            + [WEIGHT_BRANCH]
            + [
                mask
                for _, mask in SELECTIONS
                if mask is not None
            ]
        )

        # Remove duplicate branch names while preserving order.

        required = list(dict.fromkeys(required))

        missing = set(required) - available

        if missing:

            raise RuntimeError(
                f"\nMissing branches in {filename}:\n"
                f"{sorted(missing)}\n"
            )

        # ----------------------------------------------------
        # LAZY BATCH READING
        # ----------------------------------------------------

        for ibatch, batch in enumerate(
            pf.iter_batches(
                batch_size=BATCH_SIZE,
                columns=required
            )
        ):

            events = ak.from_arrow(batch)

            n_events = len(events)

            # ------------------------------------------------
            # EVENT WEIGHTS
            # ------------------------------------------------

            weights = to_numpy_float(
                events[WEIGHT_BRANCH]
            )

            weights *= normalization

            # ------------------------------------------------
            # READ VARIABLES ONLY ONCE PER BATCH
            # ------------------------------------------------

            values_dict = {

                var: to_numpy_float(events[var])

                for var in variables_list

            }

            # ------------------------------------------------
            # LOOP OVER CUMULATIVE SELECTIONS
            # ------------------------------------------------

            for istage, (stage_name, mask_name) in enumerate(
                SELECTIONS
            ):

                # --------------------------------------------
                # Obtain cumulative selection mask
                # --------------------------------------------

                if mask_name is None:

                    mask = np.ones(
                        n_events,
                        dtype=np.bool_
                    )

                else:

                    mask = to_numpy_bool(
                        events[mask_name]
                    )

                if len(mask) != n_events:
                    raise RuntimeError(
                        f"Invalid mask length: {mask_name}"
                    )

                # --------------------------------------------
                # Selected event weights
                # --------------------------------------------

                selected_weights = weights[mask]

                valid_weights = np.isfinite(
                    selected_weights
                )

                selected_weights = selected_weights[
                    valid_weights
                ]

                if len(selected_weights) == 0:
                    continue

                # --------------------------------------------
                # Accumulate total weighted yield
                # --------------------------------------------

                cutflow_yields[istage] += np.sum(
                    selected_weights
                )

                cutflow_sumw2[istage] += np.sum(
                    selected_weights**2
                )

                # --------------------------------------------
                # Fill all ROOT histograms
                # --------------------------------------------

                for var in variables_list:

                    selected_values = values_dict[var][mask]

                    # Apply same valid-weight mask to values.

                    selected_values = selected_values[
                        valid_weights
                    ]

                    fill_root_hist(
                        histograms[var][istage],
                        selected_values,
                        selected_weights
                    )

            if (ibatch + 1) % 10 == 0:

                print(
                    f"  Processed {ibatch + 1} batches "
                    f"({(ibatch + 1) * BATCH_SIZE:,} events)"
                )

        print(
            f"Completed: {os.path.basename(filename)}"
        )


# ============================================================
# PRINT TOTAL BACKGROUND CUTFLOW
# ============================================================

print("\n")
print("=" * 85)
print("TOTAL BACKGROUND CUTFLOW")
print("=" * 85)

print(
    f"{'Selection':35s}"
    f"{'Yield':>18s}"
    f"{'Stat. uncertainty':>22s}"
)

print("-" * 85)

for i, stage in enumerate(STAGE_NAMES):

    uncertainty = np.sqrt(cutflow_sumw2[i])

    print(
        f"{stage:35s}"
        f"{cutflow_yields[i]:18.5f}"
        f"{uncertainty:22.5f}"
    )

print("=" * 85)


# ============================================================
# SAVE CUTFLOW ROOT HISTOGRAM
# ============================================================

h_cutflow = ROOT.TH1D(
    "h_cutflow",
    "Background Cutflow;Selection;Expected events",
    N_STAGES,
    0,
    N_STAGES
)

h_cutflow.Sumw2()

for i, stage in enumerate(STAGE_NAMES):

    h_cutflow.GetXaxis().SetBinLabel(
        i + 1,
        stage
    )

    h_cutflow.SetBinContent(
        i + 1,
        cutflow_yields[i]
    )

    h_cutflow.SetBinError(
        i + 1,
        np.sqrt(cutflow_sumw2[i])
    )


# ============================================================
# DRAW OVERLAY USING PYROOT
# ============================================================

COLORS = [

    ROOT.kBlack,
    ROOT.kRed + 1,
    ROOT.kBlue + 1,
    ROOT.kGreen + 2,
    ROOT.kMagenta + 1,
    ROOT.kOrange + 7,
    ROOT.kCyan + 2,
    ROOT.kViolet + 1,
    ROOT.kAzure + 1,
    ROOT.kPink + 7,
    ROOT.kTeal + 3,
    ROOT.kGray + 2,
    ROOT.kSpring + 5,
    ROOT.kOrange + 1,
    ROOT.kBlue - 7,
    ROOT.kRed - 7,
    ROOT.kGreen + 3

]


def draw_overlay(var):

    cfg = plot_config[var]

    canvas = ROOT.TCanvas(
        f"c_{var}",
        f"c_{var}",
        1100,
        850
    )

    canvas.SetLeftMargin(0.13)
    canvas.SetRightMargin(0.05)
    canvas.SetBottomMargin(0.13)
    canvas.SetTopMargin(0.08)

    legend = ROOT.TLegend(
        0.48, 0.55,
        0.93, 0.90
    )

    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextSize(0.024)
    legend.SetNColumns(2)

    hists = histograms[var]

    maximum = max(
        h.GetMaximum()
        for h in hists
    )

    first = True

    for i, (stage, h) in enumerate(
        zip(STAGE_NAMES, hists)
    ):

        h.SetLineColor(
            COLORS[i % len(COLORS)]
        )

        h.SetLineWidth(2)

        h.SetFillStyle(0)

        h.GetXaxis().SetTitle(
            cfg["xlabel"]
        )

        h.GetYaxis().SetTitle(
            "Expected events / bin"
        )

        h.GetXaxis().SetTitleSize(0.045)
        h.GetYaxis().SetTitleSize(0.045)

        h.GetXaxis().SetLabelSize(0.04)
        h.GetYaxis().SetLabelSize(0.04)

        h.GetYaxis().SetTitleOffset(1.35)

        h.SetMaximum(
            maximum * 1.35 if maximum > 0 else 1.0
        )

        h.SetMinimum(0)

        if first:

            h.Draw("HIST")
            first = False

        else:

            h.Draw("HIST SAME")

        legend.AddEntry(
            h,
            stage,
            "l"
        )

    legend.Draw()

    canvas.RedrawAxis()

    canvas.SaveAs(
        f"{OUTPUT_DIR}/{var}_cutflow_overlay.png"
    )

    canvas.SaveAs(
        f"{OUTPUT_DIR}/{var}_cutflow_overlay.pdf"
    )

    canvas.Close()


# ============================================================
# DRAW ALL VARIABLES
# ============================================================

print("\nDrawing ROOT overlays...")

for var in variables_list:

    print("Drawing:", var)

    draw_overlay(var)


# ============================================================
# SAVE ALL HISTOGRAMS TO ROOT FILE
# ============================================================

root_filename = os.path.join(
    OUTPUT_DIR,
    "cutflow_histograms.root"
)

output_file = ROOT.TFile(
    root_filename,
    "RECREATE"
)

# Save cutflow

output_file.cd()

h_cutflow.Write()

# Save variable histograms

for var in variables_list:

    directory = output_file.mkdir(var)

    directory.cd()

    for i, h in enumerate(histograms[var]):

        h.Write()

output_file.Close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)

print("PROCESSING COMPLETED")

print("=" * 65)

print("ROOT file:")
print(root_filename)

print("\nPlots:")
print(OUTPUT_DIR)

print("\nNumber of variables:", len(variables_list))
print("Number of selection stages:", N_STAGES)

print("\nFinal background yield:")

print(
    f"{cutflow_yields[-1]:.6f} "
    f"+/- {np.sqrt(cutflow_sumw2[-1]):.6f}"
)

print("=" * 65)
