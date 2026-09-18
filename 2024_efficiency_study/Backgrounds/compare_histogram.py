#!/usr/bin/env python3

import os
import argparse

import numpy as np
import awkward as ak
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

DIR_SKIM = "NTuples_2024_BKG_skim_new/merged"
DIR_RAW  = "NTuples_BKG_2024_with_MET/merged"

SAMPLE_SKIM = "TTto2L2Nu_24SummerRun3"
SAMPLE_RAW  = "TTto2L2Nu-24SummerRun3"

CATEGORY = "CAT1"

# Cross section and luminosity
XSEC = 405.87          # pb
LUMI = 109.0           # fb^-1
LUMI_PB = LUMI * 1000  # convert fb^-1 -> pb^-1

# Overall MC normalization
SCALE = XSEC * LUMI_PB


# ============================================================
# Binning
#
# Format:
#
# "variable": (number_of_bins, xmin, xmax)
#
# The bin width is calculated automatically.
# ============================================================

VARIABLES = {

    # --------------------------------------------------------
    # Photons
    # --------------------------------------------------------

    "pholead_pt":      (40, 0, 200),
    "phosublead_pt":   (40, 0, 150),

    "pholead_eta":     (40, -3.0, 3.0),
    "phosublead_eta":  (40, -3.0, 3.0),

    "pholead_phi":     (40, -3.2, 3.2),
    "phosublead_phi":  (40, -3.2, 3.2),

    # --------------------------------------------------------
    # Electron
    # --------------------------------------------------------

    "electron_pt":     (40, 0, 150),
    "electron_eta":    (40, -2.5, 2.5),
    "electron_phi":    (40, -3.2, 3.2),

    # --------------------------------------------------------
    # Muon
    # --------------------------------------------------------

    "muon_pt":         (40, 0, 150),
    "muon_eta":        (40, -2.5, 2.5),
    "muon_phi":        (40, -3.2, 3.2),

    # --------------------------------------------------------
    # Jets
    # --------------------------------------------------------

    "first_jet_pt":    (40, 0, 200),
    "first_jet_eta":   (40, -5, 5),
    "first_jet_phi":   (40, -3.2, 3.2),

    "second_jet_pt":   (40, 0, 150),
    "second_jet_eta":  (40, -5, 5),
    "second_jet_phi":  (40, -3.2, 3.2),

    # --------------------------------------------------------
    # Diphoton
    # --------------------------------------------------------

    "mass":            (50, 0, 100),
    "dipho_pt":        (40, 0, 250),

    # --------------------------------------------------------
    # MET
    # --------------------------------------------------------

    "PFMET_pt":        (40, 0, 200),
    "PuppiMET_pt":     (40, 0, 200),

    "PFMET_phi":       (40, -3.2, 3.2),
    "PuppiMET_phi":    (40, -3.2, 3.2),

    # --------------------------------------------------------
    # Event-level
    # --------------------------------------------------------

    "Njets":           (10, 0, 10),
    "n_bJets":         (6, -0.5, 5.5),
    "nPV":             (50, 0, 100),

    # --------------------------------------------------------
    # Angular variables
    # --------------------------------------------------------

    "delphi_gg":       (32, 0, 3.2),
    "delphi_bb":       (32, 0, 3.2),
    "delphi_bbgg":     (32, 0, 3.2),

    # --------------------------------------------------------
    # Lepton
    # --------------------------------------------------------

    "leppt":           (40, 0, 200),
    "lepeta":          (40, -2.5, 2.5),

    # --------------------------------------------------------
    # BDT
    # --------------------------------------------------------

    "BDT_score":       (40, -1, 1),
}


# ============================================================
# Axis labels
# ============================================================

VARIABLE_LABELS = {

    # Photons
    "pholead_pt":
        r"Leading photon $p_T$ [GeV]",

    "phosublead_pt":
        r"Subleading photon $p_T$ [GeV]",

    "pholead_eta":
        r"Leading photon $\eta$",

    "phosublead_eta":
        r"Subleading photon $\eta$",

    "pholead_phi":
        r"Leading photon $\phi$",

    "phosublead_phi":
        r"Subleading photon $\phi$",

    # Electron
    "electron_pt":
        r"Electron $p_T$ [GeV]",

    "electron_eta":
        r"Electron $\eta$",

    "electron_phi":
        r"Electron $\phi$",

    # Muon
    "muon_pt":
        r"Muon $p_T$ [GeV]",

    "muon_eta":
        r"Muon $\eta$",

    "muon_phi":
        r"Muon $\phi$",

    # Jets
    "first_jet_pt":
        r"Leading jet $p_T$ [GeV]",

    "first_jet_eta":
        r"Leading jet $\eta$",

    "first_jet_phi":
        r"Leading jet $\phi$",

    "second_jet_pt":
        r"Subleading jet $p_T$ [GeV]",

    "second_jet_eta":
        r"Subleading jet $\eta$",

    "second_jet_phi":
        r"Subleading jet $\phi$",

    # Diphoton
    "mass":
        r"$m_{\gamma\gamma}$ [GeV]",

    "dipho_pt":
        r"Diphoton $p_T$ [GeV]",

    # MET
    "PFMET_pt":
        r"PF MET $p_T$ [GeV]",

    "PuppiMET_pt":
        r"PUPPI MET $p_T$ [GeV]",

    "PFMET_phi":
        r"PF MET $\phi$",

    "PuppiMET_phi":
        r"PUPPI MET $\phi$",

    # Event
    "Njets":
        r"$N_{\mathrm{jets}}$",

    "n_bJets":
        r"$N_{b\mathrm{-jets}}$",

    "nPV":
        r"$N_{\mathrm{PV}}$",

    # Angular
    "delphi_gg":
        r"$\Delta\phi(\gamma,\gamma)$",

    "delphi_bb":
        r"$\Delta\phi(b,b)$",

    "delphi_bbgg":
        r"$\Delta\phi(bb,\gamma\gamma)$",

    # Lepton
    "leppt":
        r"Lepton $p_T$ [GeV]",

    "lepeta":
        r"Lepton $\eta$",

    # BDT
    "BDT_score":
        r"BDT score",
}


# ============================================================
# Load parquet
# ============================================================

def load_parquet(path, variable):

    print(f"Reading: {path}")

    data = ak.from_parquet(
        path,
        columns=[variable, "weight"]
    )

    x = ak.to_numpy(data[variable])
    w = ak.to_numpy(data["weight"])

    # --------------------------------------------------------
    # Total entries before masking
    # --------------------------------------------------------

    total_entries_before_mask = len(x)

    # --------------------------------------------------------
    # Remove:
    #
    #   -999 masked values
    #   NaN
    #   +inf
    #   -inf
    #
    # Use x > -900 rather than x != -999 so that other
    # possible negative mask values are also removed.
    # --------------------------------------------------------

    valid = (
        np.isfinite(x)
        & np.isfinite(w)
        & (x > -900)
    )

    x = x[valid]
    w = w[valid]

    masked_entries = (
        total_entries_before_mask
        - len(x)
    )

    print(
        f"  Total entries : "
        f"{total_entries_before_mask:,}"
    )

    print(
        f"  Valid entries : "
        f"{len(x):,}"
    )

    print(
        f"  Masked        : "
        f"{masked_entries:,}"
    )

    return x, w


# ============================================================
# Histogram
# ============================================================

def make_histogram(x, w, edges):

    hist, _ = np.histogram(
        x,
        bins=edges,
        weights=w
    )

    # Apply cross section * luminosity
    hist *= SCALE

    return hist


# ============================================================
# Statistics
# ============================================================

def get_statistics(
    x,
    w,
    xmin,
    xmax
):

    # --------------------------------------------------------
    # Underflow
    # --------------------------------------------------------

    underflow_mask = x < xmin

    # --------------------------------------------------------
    # Overflow
    # --------------------------------------------------------

    overflow_mask = x >= xmax

    # --------------------------------------------------------
    # In-range
    # --------------------------------------------------------

    inrange_mask = (
        (x >= xmin)
        & (x < xmax)
    )

    # --------------------------------------------------------
    # Total entries
    #
    # This is the number of valid, unmasked events.
    # --------------------------------------------------------

    total_entries = len(x)

    # --------------------------------------------------------
    # Weighted integral
    # --------------------------------------------------------

    integral = (
        np.sum(w[inrange_mask])
        * SCALE
    )

    # --------------------------------------------------------
    # Weighted underflow
    # --------------------------------------------------------

    underflow = (
        np.sum(w[underflow_mask])
        * SCALE
    )

    # --------------------------------------------------------
    # Weighted overflow
    # --------------------------------------------------------

    overflow = (
        np.sum(w[overflow_mask])
        * SCALE
    )

    # --------------------------------------------------------
    # Statistical uncertainty
    #
    # sigma = sqrt(sum(w^2)) * xsec * lumi
    # --------------------------------------------------------

    stat = (
        np.sqrt(
            np.sum(
                w[inrange_mask] ** 2
            )
        )
        * SCALE
    )

    return {
        "entries": total_entries,
        "integral": integral,
        "underflow": underflow,
        "overflow": overflow,
        "stat": stat,
    }


# ============================================================
# Statistics text
# ============================================================

def format_stats(stats):

    return (
        f"Entries   = {stats['entries']:,}\n"
        f"Integral  = {stats['integral']:.4g}\n"
        f"Underflow = {stats['underflow']:.4g}\n"
        f"Overflow  = {stats['overflow']:.4g}\n"
        f"Stat.     = {stats['stat']:.4g}"
    )


# ============================================================
# Plot
# ============================================================

def plot_comparison(
    variable,
    x_skim,
    w_skim,
    x_raw,
    w_raw,
    edges,
    output_dir
):

    xmin = edges[0]
    xmax = edges[-1]

    # --------------------------------------------------------
    # Calculate bin width automatically
    # --------------------------------------------------------

    bin_width = edges[1] - edges[0]

    # --------------------------------------------------------
    # Histograms
    # --------------------------------------------------------

    h_skim = make_histogram(
        x_skim,
        w_skim,
        edges
    )

    h_raw = make_histogram(
        x_raw,
        w_raw,
        edges
    )

    # --------------------------------------------------------
    # Bin centers
    # --------------------------------------------------------

    centers = (
        edges[:-1] + edges[1:]
    ) / 2

    # --------------------------------------------------------
    # Ratio
    # --------------------------------------------------------

    ratio = np.full(
        len(h_skim),
        np.nan,
        dtype=float
    )

    valid = h_raw != 0

    ratio[valid] = (
        h_skim[valid] / h_raw[valid]
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    stats_skim = get_statistics(
        x_skim,
        w_skim,
        xmin,
        xmax
    )

    stats_raw = get_statistics(
        x_raw,
        w_raw,
        xmin,
        xmax
    )

    # ========================================================
    # Figure
    # ========================================================

    fig, (ax, rax) = plt.subplots(
        2,
        1,
        figsize=(8, 8),
        sharex=True,
        gridspec_kw={
            "height_ratios": [3, 1],
            "hspace": 0.05
        }
    )

    # ========================================================
    # Main histogram
    # ========================================================

    # --------------------------------------------------------
    # Raw = filled histogram
    # --------------------------------------------------------

    ax.stairs(
        h_raw,
        edges,
        fill=True,
        alpha=0.35,
        linewidth=1.5,
        label="Raw"
    )

    # --------------------------------------------------------
    # Skim = points
    # No error bars
    # --------------------------------------------------------

    ax.plot(
        centers,
        h_skim,
        marker="o",
        linestyle="none",
        markersize=4,
        label="Skim"
    )

    # Connect skim points
    ax.step(
        edges[:-1],
        h_skim,
        where="post",
        linewidth=1.0,
        alpha=0.8
    )

    # --------------------------------------------------------
    # Log scale
    # --------------------------------------------------------

    ax.set_yscale("log")

    # --------------------------------------------------------
    # Y axis
    #
    # Bin width is already shown here.
    # --------------------------------------------------------

    ax.set_ylabel(
        f"Events / {bin_width:g}"
    )

    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=12
    )

    # ========================================================
    # CMS header
    # ========================================================

    # Use the same baseline for CMS and Simulation.
    # The small x difference makes them appear as one label.

    fig.text(
        0.105,
        0.925,
        "CMS",
        fontsize=25,
        fontweight="bold",
        ha="left",
        va="top"
    )

    fig.text(
        0.215,
        0.916,
        "Simulation",
        fontsize=18,
        style="italic",
        ha="left",
        va="top"
    )

    fig.text(
        0.95,
        0.920,
        "Work in progress (13.6 TeV)",
        fontsize=13,
        ha="right",
        va="top"
    )

    fig.text(
        0.45,
        0.87,
        "TTo2L2Nu",
        fontsize=12,
        style="italic",
        ha="left",
        va="top"
    )

    # ========================================================
    # Statistics
    # ========================================================

    raw_stats_text = (
        "Raw\n"
        + format_stats(stats_raw)
    )

    skim_stats_text = (
        "Skim\n"
        + format_stats(stats_skim)
    )

    # --------------------------------------------------------
    # Raw statistics
    #
    # Move upward and make more transparent.
    # --------------------------------------------------------

    ax.text(
        0.98,
        0.97,
        raw_stats_text,
        transform=ax.transAxes,
        fontsize=9.5,
        ha="right",
        va="top",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            edgecolor="black",
            alpha=0.45
        )
    )

    # --------------------------------------------------------
    # Skim statistics
    #
    # Move upward as well.
    # --------------------------------------------------------

    ax.text(
        0.98,
        0.75,
        skim_stats_text,
        transform=ax.transAxes,
        fontsize=9.5,
        ha="right",
        va="top",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            edgecolor="black",
            alpha=0.45
        )
    )

    # --------------------------------------------------------
    # Main plot grid
    # --------------------------------------------------------

    ax.grid(
        True,
        axis="y",
        which="both",
        alpha=0.25
    )

    # ========================================================
    # Ratio
    # ========================================================

    rax.step(
        edges[:-1],
        ratio,
        where="post",
        linewidth=1.5
    )

    rax.plot(
        centers,
        ratio,
        marker="o",
        linestyle="none",
        markersize=3
    )

    # --------------------------------------------------------
    # Reference line
    # --------------------------------------------------------

    rax.axhline(
        1.0,
        linestyle="--",
        linewidth=1
    )

    # --------------------------------------------------------
    # Ratio range
    # --------------------------------------------------------

    rax.set_ylim(
        0.98,
        1.02
    )

    # --------------------------------------------------------
    # Ratio labels
    # --------------------------------------------------------

    rax.set_ylabel(
        "Skim / Raw"
    )

    rax.set_xlabel(
        VARIABLE_LABELS.get(
            variable,
            variable
        )
    )

    # --------------------------------------------------------
    # Ratio grid
    # --------------------------------------------------------

    rax.grid(
        True,
        axis="y",
        alpha=0.25
    )

    # ========================================================
    # Save
    # ========================================================

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    outfile_png = os.path.join(
        output_dir,
        f"{variable}_{CATEGORY}.png"
    )

    outfile_pdf = os.path.join(
        output_dir,
        f"{variable}_{CATEGORY}.pdf"
    )

    fig.savefig(
        outfile_png,
        dpi=200,
        bbox_inches="tight"
    )

    fig.savefig(
        outfile_pdf,
        bbox_inches="tight"
    )

    plt.close(fig)

    # ========================================================
    # Print statistics
    # ========================================================

    print()
    print("=" * 70)
    print(f"Variable  : {variable}")
    print(f"Bin width : {bin_width:g}")
    print(f"Range     : [{xmin:g}, {xmax:g}]")
    print(f"Bins      : {len(edges) - 1}")
    print(f"Scale     : {SCALE:.6e}")
    print("=" * 70)

    print()
    print("RAW")
    print(
        f"  Total entries : "
        f"{stats_raw['entries']:,}"
    )
    print(
        f"  Integral      : "
        f"{stats_raw['integral']:.6g}"
    )
    print(
        f"  Underflow     : "
        f"{stats_raw['underflow']:.6g}"
    )
    print(
        f"  Overflow      : "
        f"{stats_raw['overflow']:.6g}"
    )
    print(
        f"  Stat.         : "
        f"{stats_raw['stat']:.6g}"
    )

    print()
    print("SKIM")
    print(
        f"  Total entries : "
        f"{stats_skim['entries']:,}"
    )
    print(
        f"  Integral      : "
        f"{stats_skim['integral']:.6g}"
    )
    print(
        f"  Underflow     : "
        f"{stats_skim['underflow']:.6g}"
    )
    print(
        f"  Overflow      : "
        f"{stats_skim['overflow']:.6g}"
    )
    print(
        f"  Stat.         : "
        f"{stats_skim['stat']:.6g}"
    )

    print()
    print(f"Saved: {outfile_png}")
    print(f"Saved: {outfile_pdf}")

# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Compare Raw and Skim merged parquet samples"
        )
    )

    parser.add_argument(
        "--variable",
        help="Variable to plot"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Plot all predefined variables"
    )

    parser.add_argument(
        "--category",
        default=CATEGORY,
        help="Category, e.g. CAT1"
    )

    parser.add_argument(
        "--output",
        default="comparison_plots",
        help="Output directory"
    )

    args = parser.parse_args()

    # ========================================================
    # Select variables
    # ========================================================

    if args.all:

        variables = list(
            VARIABLES.keys()
        )

    elif args.variable:

        variables = [
            args.variable
        ]

    else:

        parser.error(
            "Use --variable VARIABLE or --all"
        )

    # ========================================================
    # Loop over variables
    # ========================================================

    for variable in variables:

        if variable not in VARIABLES:

            print(
                f"WARNING: {variable} "
                f"is not defined in VARIABLES"
            )

            continue

        # ----------------------------------------------------
        # Get binning
        # ----------------------------------------------------

        nbins, xmin, xmax = VARIABLES[
            variable
        ]

        # ----------------------------------------------------
        # Construct bin edges
        # ----------------------------------------------------

        edges = np.linspace(
            xmin,
            xmax,
            nbins + 1
        )

        # Automatically calculated bin width
        bin_width = (
            edges[1] - edges[0]
        )

        # ----------------------------------------------------
        # File paths
        # ----------------------------------------------------

        skim_file = os.path.join(
            DIR_SKIM,
            SAMPLE_SKIM,
            f"{args.category}_merged.parquet"
        )

        raw_file = os.path.join(
            DIR_RAW,
            SAMPLE_RAW,
            f"{args.category}_merged.parquet"
        )

        # ----------------------------------------------------
        # Check files
        # ----------------------------------------------------

        if not os.path.exists(
            skim_file
        ):

            print(
                f"ERROR: Skim file does not exist:\n"
                f"{skim_file}"
            )

            continue

        if not os.path.exists(
            raw_file
        ):

            print(
                f"ERROR: Raw file does not exist:\n"
                f"{raw_file}"
            )

            continue

        # ----------------------------------------------------
        # Print configuration
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print(
            f"Variable  : {variable}"
        )
        print(
            f"Bins      : {nbins}"
        )
        print(
            f"Range     : {xmin} -> {xmax}"
        )
        print(
            f"Bin width : {bin_width:g}"
        )
        print(
            f"Xsec      : {XSEC} pb"
        )
        print(
            f"Lumi      : {LUMI} fb^-1"
        )
        print(
            f"Scale     : {SCALE:.6e}"
        )
        print("=" * 70)

        # ----------------------------------------------------
        # Load Skim
        # ----------------------------------------------------

        x_skim, w_skim = load_parquet(
            skim_file,
            variable
        )

        # ----------------------------------------------------
        # Load Raw
        # ----------------------------------------------------

        x_raw, w_raw = load_parquet(
            raw_file,
            variable
        )

        # ----------------------------------------------------
        # Plot
        # ----------------------------------------------------

        plot_comparison(
            variable,
            x_skim,
            w_skim,
            x_raw,
            w_raw,
            edges,
            args.output
        )


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    main()
