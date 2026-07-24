#!/usr/bin/env python3

import os
import re
import glob
import argparse

import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import pandas as pd

from tqdm import tqdm

# ============================================================
# CONSTANTS
# ============================================================

LUMI = 109.0 * 1000
SIG_XSEC = 1.457 * 0.33

BACKGROUND_XSECS = {
    "WGtoLNuG": 671.5,
    "TTG1Jets": 4.634,
    "TTto2L2Nu": 98.04,
    "TTtoLNu2Q": 405.87,
    "DYto2E10": 21140.0,
    "DYto2E50": 2124.08,
    "DYto2Mu10": 21190.0,
    "DYto2Mu50": 2124.08,
}


# ============================================================
# AMS
# ============================================================

def AMS(s, b):

    if b <= 0:
        return 0.0

    if s <= 0:
        return 0.0

    return np.sqrt(
        2.0 * (
            (s + b) * np.log(1.0 + s / b) - s
        )
    )


# ============================================================
# Read AMS window file
# ============================================================

def read_windows(fname):

    windows = {}

    with open(fname) as f:

        next(f)   # Skip header

        for line in f:

            if not line.strip():
                continue

            pieces = line.split()

            proc = pieces[0]

            # Match strings like WH(M=12 GeV)
            m = re.search(r"M=(\d+)", proc)

            if m is None:
                print(f"Skipping line: {line.strip()}")
                continue

            mass = int(m.group(1))

            low = float(pieces[5])
            high = float(pieces[6])

            windows[mass] = (low, high)

    return windows

# ============================================================
# Find all signal files
# ============================================================

def discover_signal_files(signal_dir):

    files = sorted(
        glob.glob(
            os.path.join(
                signal_dir,
                "WH-2024M*",
                "nominal",
                "CAT1_merged.parquet",
            )
        )
    )

    sig = {}

    for f in files:

        m = re.findall(
            r"WH-2024M(\d+)",
            f,
        )[0]

        sig[int(m)] = f

    return sig


# ============================================================
# Discover background files
# ============================================================

def discover_backgrounds(background_dir):

    out = []

    dirs = sorted(glob.glob(os.path.join(background_dir, "*")))

    for d in dirs:

        parquet = glob.glob(
            os.path.join(d, "*.parquet")
        )

        if len(parquet) == 0:
            continue

        sample = os.path.basename(d)

        xsec = None

        for key in BACKGROUND_XSECS:

            if key in sample:
                xsec = BACKGROUND_XSECS[key]

        if xsec is None:
            raise RuntimeError(
                f"Unknown sample {sample}"
            )

        out.append(
            {
                "sample": sample,
                "file": parquet[0],
                "xsec": xsec,
            }
        )

    return out

def make_mass_histogram(
        sig,
        backgrounds,
        masspoint,
        cut,
        low,
        high,
        output_dir,
):

    branch = f"bdt_{masspoint}"

    sig_mask = sig["bdt"] > cut

    sig_mass = sig["mass"][sig_mask]
    # sig_weight = (
    #     sig["weight"][sig_mask]
    #     * SIG_XSEC
    #     * LUMI
    # )

    SIGNAL_SCALE = 0.01

    sig_weight = (
        sig["weight"][sig_mask]
        * SIG_XSEC
        * LUMI
        * SIGNAL_SCALE
    )

    bkg_mass = []
    bkg_weight = []

    for bkg in backgrounds:

        mask = bkg[branch] > cut

        bkg_mass.append(bkg["mass"][mask])

        bkg_weight.append(
            bkg["weight"][mask]
            * bkg["xsec"]
            * LUMI
        )

    if len(bkg_mass):

        bkg_mass = np.concatenate(bkg_mass)
        bkg_weight = np.concatenate(bkg_weight)

    else:

        bkg_mass = np.array([])
        bkg_weight = np.array([])

    plt.figure(figsize=(8,6))

    bins = np.linspace(10,70,61)

    plt.hist(
        bkg_mass,
        bins=bins,
        weights=bkg_weight,
        histtype="stepfilled",
        alpha=0.5,
        label="Background",
    )

    plt.hist(
        sig_mass,
        bins=bins,
        weights=sig_weight,
        histtype="step",
        linewidth=2,
        label=f"Signal ({masspoint} GeV)",
    )

    plt.axvspan(
        low,
        high,
        color="red",
        alpha=0.15,
        label=r"$\pm2\sigma$ window",
    )

    plt.xlabel(r"$m_{\gamma\gamma}$ (GeV)")
    plt.ylabel("Expected Events")
    plt.legend()
    plt.tight_layout()

    # Linear scale
    plt.yscale("linear")
    plt.savefig(
        os.path.join(
            output_dir,
            f"M{masspoint}_BDT_{cut:.3f}_linear.png",
        ),
        dpi=200,
    )

    # Log scale
    plt.yscale("log")
    plt.savefig(
        os.path.join(
            output_dir,
            f"M{masspoint}_BDT_{cut:.3f}_log.png",
        ),
        dpi=200,
    )

    plt.close()


# ============================================================
# Load signal
# ============================================================

def load_signal(fname):

    print(f"Loading signal : {fname}")

    arr = ak.from_parquet(fname)

    return {
        "mass": ak.to_numpy(arr.mass),
        "bdt": ak.to_numpy(arr.bdt_score),
        "weight": ak.to_numpy(arr.weight),
    }


# ============================================================
# Load backgrounds
# ============================================================

def load_backgrounds(backgrounds):

    loaded = []

    print("\nLoading backgrounds...\n")

    for b in tqdm(backgrounds):

        arr = ak.from_parquet(b["file"])

        loaded.append(
            {
                "sample": b["sample"],
                "xsec": b["xsec"],
                "mass": ak.to_numpy(arr.mass),
                "weight": ak.to_numpy(arr.weight),

                "bdt_12": ak.to_numpy(arr.bdt_12),
                "bdt_15": ak.to_numpy(arr.bdt_15),
                "bdt_20": ak.to_numpy(arr.bdt_20),
                "bdt_25": ak.to_numpy(arr.bdt_25),
                "bdt_30": ak.to_numpy(arr.bdt_30),
                "bdt_35": ak.to_numpy(arr.bdt_35),
                "bdt_40": ak.to_numpy(arr.bdt_40),
                "bdt_45": ak.to_numpy(arr.bdt_45),
                "bdt_50": ak.to_numpy(arr.bdt_50),
                "bdt_55": ak.to_numpy(arr.bdt_55),
                "bdt_60": ak.to_numpy(arr.bdt_60),
            }
        )

    return loaded


# ============================================================
# Compute signal yield
# ============================================================

def signal_yield(sig, low, high, cut):

    mask = (
        (sig["mass"] >= low)
        &
        (sig["mass"] <= high)
        &
        (sig["bdt"] > cut)
    )

    if np.sum(mask) == 0:
        return 0.0

    return np.sum(
        sig["weight"][mask]
        * SIG_XSEC
        * LUMI
    )


# ============================================================
# Compute background yield
# ============================================================

def background_yield(backgrounds,
                     low,
                     high,
                     cut,
                     masspoint):

    total = 0.0

    branch = f"bdt_{masspoint}"

    for b in backgrounds:

        bdt = b[branch]

        mask = (
            (b["mass"] >= low)
            &
            (b["mass"] <= high)
            &
            (bdt > cut)
        )

        if np.sum(mask) == 0:
            continue

        total += np.sum(
            b["weight"][mask]
            * b["xsec"]
            * LUMI
        )

    return total


# ============================================================
# Scan one signal mass point
# ============================================================

# def scan_masspoint(
#         masspoint,
#         sig,
#         backgrounds,
#         window,
#         output_dir,
#         step=0.01,
# ):

def scan_masspoint(
        masspoint,
        sig,
        backgrounds,
        window,
        output_dir,
        step,
        min_background,
        plot_mode,
):

    low, high = window

    print("\n======================================")
    print(f"Mass point : {masspoint}")
    print(f"Window     : [{low:.3f}, {high:.3f}]")
    print("======================================")

    cuts = np.arange(0.0, 1.0001, step)

    rows = []

    best_cut = None
    best_s = 0.0
    best_b = 0.0
    best_ams = -1.0

    mass_hist_dir = os.path.join(
        output_dir,
        "Histograms",
        f"M{masspoint}",
    )

    os.makedirs(
        mass_hist_dir,
        exist_ok=True,
    )

    for cut in tqdm(cuts, leave=False):

        s = signal_yield(
            sig,
            low,
            high,
            cut,
        )

        b = background_yield(
            backgrounds,
            low,
            high,
            cut,
            masspoint,
        )

        b_total = background_yield(
            backgrounds,
            10.0,
            70.0,
            cut,
            masspoint,
        )

        # ams = AMS(s, b)

        if b_total < min_background:
            ams = 0.0
        else:
            ams = AMS(s, b)

        if plot_mode == "all":
            make_mass_histogram(
                sig=sig,
                backgrounds=backgrounds,
                masspoint=masspoint,
                cut=cut,
                low=low,
                high=high,
                output_dir=mass_hist_dir,
            )

        rows.append(
            {
                "cut": cut,
                "signal": s,
                "background": b,
                "AMS": ams,
            }
        )

        if ams > best_ams:

            best_ams = ams
            best_cut = cut
            best_s = s
            best_b = b

    ######################################################
    # Save CSV
    ######################################################

    if plot_mode == "best":
        make_mass_histogram(
            sig=sig,
            backgrounds=backgrounds,
            masspoint=masspoint,
            cut=best_cut,
            low=low,
            high=high,
            output_dir=mass_hist_dir,
        )

    df = pd.DataFrame(rows)

    csv_name = os.path.join(
        output_dir,
        f"mass{masspoint}_scan.csv",
    )

    df.to_csv(
        csv_name,
        index=False,
    )

    ######################################################
    # Plot
    ######################################################

    plt.figure(figsize=(7,5))

    plt.plot(
        df["cut"],
        df["AMS"],
        lw=2,
    )

    plt.axvline(
        best_cut,
        ls="--",
        lw=1.5,
    )

    plt.xlabel("BDT cut")
    plt.ylabel("AMS")

    plt.title(
        f"M={masspoint} GeV"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            f"mass{masspoint}_AMS.png",
        ),
        dpi=200,
    )

    plt.close()

    ######################################################
    # Print
    ######################################################

    print()

    print(
        f"Best cut : {best_cut:.3f}"
    )

    print(
        f"Signal   : {best_s:.5f}"
    )

    print(
        f"Background : {best_b:.5f}"
    )

    print(
        f"AMS : {best_ams:.5f}"
    )

    return {
        "Mass": masspoint,
        "BestCut": best_cut,
        "SignalYield": best_s,
        "BackgroundYield": best_b,
        "AMS": best_ams,
    }


# ============================================================
# Scan all masses
# ============================================================

# def run_scan(
#         signal_files,
#         backgrounds,
#         windows,
#         output_dir,
#         step,
# ):

def run_scan(
    signal_files,
    backgrounds,
    windows,
    output_dir,
    step,
    min_background,
    plot_mode,
):

    summary = []

    masses = sorted(signal_files.keys())

    for mass in masses:

        if mass not in windows:

            print(
                f"No AMS window for M={mass}"
            )
            continue

        sig = load_signal(
            signal_files[mass]
        )

        # result = scan_masspoint(
        #     masspoint=mass,
        #     sig=sig,
        #     backgrounds=backgrounds,
        #     window=windows[mass],
        #     output_dir=output_dir,
        #     step=step,
        # )

        result = scan_masspoint(
            masspoint=mass,
            sig=sig,
            backgrounds=backgrounds,
            window=windows[mass],
            output_dir=output_dir,
            step=step,
            min_background=min_background,
            plot_mode=plot_mode,
        )

        summary.append(result)

    ######################################################
    # Summary CSV
    ######################################################

    summary_df = pd.DataFrame(summary)

    summary_df = summary_df.sort_values(
        "Mass"
    )

    summary_df.to_csv(
        os.path.join(
            output_dir,
            "best_BDT_cuts.csv",
        ),
        index=False,
    )

    ######################################################
    # Summary text
    ######################################################

    with open(
        os.path.join(
            output_dir,
            "summary.txt",
        ),
        "w",
    ) as f:

        for _, row in summary_df.iterrows():

            f.write(
                f"Mass = {int(row['Mass'])}\n"
            )

            f.write(
                f"Best cut : {row['BestCut']:.3f}\n"
            )

            f.write(
                f"Signal : {row['SignalYield']:.6f}\n"
            )

            f.write(
                f"Background : {row['BackgroundYield']:.6f}\n"
            )

            f.write(
                f"AMS : {row['AMS']:.6f}\n"
            )

            f.write("\n")

    ######################################################
    # Best AMS vs Mass
    ######################################################

    plt.figure(figsize=(7,5))

    plt.plot(
        summary_df["Mass"],
        summary_df["AMS"],
        "o-",
        lw=2,
    )

    plt.xlabel(
        "Signal mass (GeV)"
    )

    plt.ylabel(
        "Best AMS"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            "best_AMS_vs_mass.png",
        ),
        dpi=200,
    )

    plt.close()

    return summary_df



# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="BDT optimization using AMS"
    )

    parser.add_argument(
        "--signal-dir",
        default="/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/NTuples_WH_2024_HDNA_presel_with_latest_BDT_score/merged_without_BDT",
        help="Signal directory",
    )

    parser.add_argument(
        "--background-dir",
        default="/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/NTuples_BKG_2024_HDNA_presel_pBDT_score_latest_pho15/merged",
        help="Background directory",
    )

    parser.add_argument(
        "--windows",
        default="AMS_ranges.txt",
        help="AMS window text file",
    )

    parser.add_argument(
        "--output",
        default="AMS_scan",
        help="Output directory",
    )

    parser.add_argument(
        "--step",
        type=float,
        default=0.01,
        help="BDT scan step",
    )

    parser.add_argument(
        "--min-background",
        type=float,
        default=0.0,
        help="Minimum expected background yield after all selections."
    )

    parser.add_argument(
        "--plot-mode",
        choices=["none", "best", "all"],
        default="best",
        help="Save mass histograms after BDT selection.",
    )

    args = parser.parse_args()

    os.makedirs(
        args.output,
        exist_ok=True,
    )

    hist_dir = os.path.join(
        args.output,
        "Histograms",
    )

    os.makedirs(
        hist_dir,
        exist_ok=True,
    )

    print("=" * 60)
    print("Reading AMS windows...")
    print("=" * 60)

    windows = read_windows(args.windows)

    print(f"Loaded {len(windows)} AMS windows")

    print("\n" + "=" * 60)
    print("Discovering signal samples...")
    print("=" * 60)

    signal_files = discover_signal_files(
        args.signal_dir
    )

    print(f"Found {len(signal_files)} signal samples")

    for m in sorted(signal_files):
        print(
            f"  M={m:2d} : {signal_files[m]}"
        )

    print("\n" + "=" * 60)
    print("Discovering backgrounds...")
    print("=" * 60)

    background_info = discover_backgrounds(
        args.background_dir
    )

    print(f"Found {len(background_info)} background samples\n")

    for b in background_info:

        print(
            f"{b['sample']:30s}"
            f"xsec = {b['xsec']}"
        )

    print("\n" + "=" * 60)
    print("Loading background files...")
    print("=" * 60)

    backgrounds = load_backgrounds(
        background_info
    )

    print("\n" + "=" * 60)
    print("Running BDT optimization")
    print("=" * 60)

    # summary = run_scan(
    #     signal_files=signal_files,
    #     backgrounds=backgrounds,
    #     windows=windows,
    #     output_dir=args.output,
    #     step=args.step,
    # )

    summary = run_scan(
        signal_files=signal_files,
        backgrounds=backgrounds,
        windows=windows,
        output_dir=args.output,
        step=args.step,
        min_background=args.min_background,
        plot_mode=args.plot_mode,
    )

    print("\n" + "=" * 60)
    print("Optimization finished")
    print("=" * 60)

    print(summary)

    print("\nOutput written to")
    print(args.output)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()