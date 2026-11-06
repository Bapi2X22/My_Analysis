#!/usr/bin/env python3
"""
Plot-only stage for the BBgg cumulative cutflow.

Reads cutflow_histograms.root and cutflow_metadata.json.
Does not read or process any Parquet files.
"""

import os
import json
import ROOT
ROOT.gROOT.ProcessLine(".L /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/live_stats.C")

# ============================================================
# USER CONTROLS
# ============================================================

OUTPUT_DIR = "/eos/user/b/bbapi/www/Analysis_plots/BKG_cutflow/"

ROOT_FILENAME = os.path.join(OUTPUT_DIR, "cutflow_histograms.root")
PLOT_ROOT_FILENAME = os.path.join(OUTPUT_DIR, "cutflow_plots.root")
METADATA_FILENAME = os.path.join(OUTPUT_DIR, "cutflow_metadata.json")
PLOT_OUTPUT_DIR = OUTPUT_DIR
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

# None = all variables/stages.
SELECTED_VARIABLES = None
SELECTED_STAGES = None

LOG_Y = True
Y_MAX_FACTOR = 1.35
Y_MAX_OVERRIDE = None

LEGEND_COLUMNS = 1

# Canvas layout: main histogram on the left, legend/integrals on the right.
CANVAS_WIDTH = 1200
CANVAS_HEIGHT = 900
MAIN_PAD_XMAX = 0.82

# CMS label settings.
LUMI_FB = 109.0
YEAR = "2024"
ENERGY = "13.6 TeV"
CMS_STATUS = "Simulation Preliminary"


# ============================================================
# ROOT STYLE
# ============================================================

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)
ROOT.gStyle.SetLegendBorderSize(0)

COLORS = [
    ROOT.kBlack, ROOT.kRed + 1, ROOT.kBlue + 1, ROOT.kGreen + 2,
    ROOT.kMagenta + 1, ROOT.kOrange + 7, ROOT.kCyan + 2,
    ROOT.kViolet + 1, ROOT.kAzure + 1, ROOT.kPink + 7,
    ROOT.kTeal + 3, ROOT.kGray + 2, ROOT.kSpring + 5,
    ROOT.kOrange + 1, ROOT.kBlue - 7, ROOT.kRed - 7,
    ROOT.kGreen + 3,
]


def CMS_label(pad, lumi=LUMI_FB, year=YEAR, energy=ENERGY,
              status=CMS_STATUS, x=0.13, y=0.90):
    """Draw CMS-style label in the coordinate system of the supplied pad."""
    pad.cd()
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextAngle(0)
    latex.SetTextColor(ROOT.kBlack)

    latex.SetTextFont(61)
    latex.SetTextSize(0.060)
    latex.DrawLatex(x, y, "CMS")

    if status:
        latex.SetTextFont(52)
        latex.SetTextSize(0.040)
        latex.DrawLatex(x + 0.115, y, status)

    latex.SetTextFont(42)
    latex.SetTextSize(0.035)
    latex.SetTextAlign(31)
    lumi_text = f"{year} ({lumi:g} fb^{{-1}}) at {energy}"
    latex.DrawLatex(0.97, y, lumi_text)
    return latex


def attach_live_stats(canvas, cut_histograms=None):
    canvas.cd()
    keep = []
    for h in (cut_histograms or []):
        c = h.Clone(h.GetName())
        c.SetDirectory(0)
        c.SetLineColor(0)
        c.SetLineWidth(0)
        c.SetFillStyle(0)
        c.Draw("HIST SAME")
        keep.append(c)
    live = ROOT.TExec("live_stats", "updateStats();")
    live.Draw()
    keep.append(live)
    return keep


def load_metadata(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found: {path}")
    with open(path) as stream:
        return json.load(stream)


def get_selected_variables(metadata):
    available = metadata["variables"]
    if SELECTED_VARIABLES is None:
        return available
    unknown = set(SELECTED_VARIABLES) - set(available)
    if unknown:
        raise ValueError(f"Unknown variables requested: {sorted(unknown)}")
    return SELECTED_VARIABLES


def get_selected_stages(metadata):
    available = [item["stage"] for item in metadata["selections"]]
    if SELECTED_STAGES is None:
        return list(range(len(available)))
    unknown = set(SELECTED_STAGES) - set(available)
    if unknown:
        raise ValueError(f"Unknown stages requested: {sorted(unknown)}")
    return [i for i, stage in enumerate(available) if stage in SELECTED_STAGES]


def move_edge_bins_to_flow_bins(hist):
    """
    Move the first regular bin into underflow and the last regular bin
    into overflow. Interior bins remain unchanged.

    Bin errors are combined in quadrature. This operation is performed
    only on plotting clones, never on the cached source histograms.
    """
    nbins = hist.GetNbinsX()
    if nbins < 2:
        return

    first_content = hist.GetBinContent(1)
    first_error = hist.GetBinError(1)
    under_content = hist.GetBinContent(0)
    under_error = hist.GetBinError(0)
    hist.SetBinContent(0, under_content + first_content)
    hist.SetBinError(0, (under_error**2 + first_error**2)**0.5)
    hist.SetBinContent(1, 0.0)
    hist.SetBinError(1, 0.0)

    last_content = hist.GetBinContent(nbins)
    last_error = hist.GetBinError(nbins)
    over_content = hist.GetBinContent(nbins + 1)
    over_error = hist.GetBinError(nbins + 1)
    hist.SetBinContent(nbins + 1, over_content + last_content)
    hist.SetBinError(nbins + 1, (over_error**2 + last_error**2)**0.5)
    hist.SetBinContent(nbins, 0.0)
    hist.SetBinError(nbins, 0.0)


def draw_statbox_manual_small(hist, x1, y1, x2, y2, color,
                              label=None, is_signal=False):
    """Draw a compact ROOT statbox containing the regular-bin integral."""
    stats = ROOT.TPaveText(x1, y1, x2, y2, "NDC")
    stats.SetName(f"stat_{hist.GetName()}") 
    stats.SetFillColor(0)
    stats.SetFillStyle(1001)    
    stats.SetBorderSize(1)
    stats.SetLineColor(color)
    stats.SetTextColor(color)
    stats.SetTextFont(42)
    stats.SetTextSize(0.09)

    nbins = hist.GetNbinsX()
    integral = hist.Integral(0, nbins+1)

    if label is not None:
        stats.AddText(label)

    if is_signal:
        stats.AddText(f"{integral:.2f}")
    else:
        stats.AddText(f"Integral = {integral:.2f}")

    stats.Draw()
    return stats


def draw_overlay(root_file, metadata, var, stage_indices, output_root_file):
    cfg = metadata["plot_config"][var]
    stages = [item["stage"] for item in metadata["selections"]]

    canvas = ROOT.TCanvas(f"c_{var}", f"c_{var}", CANVAS_WIDTH, CANVAS_HEIGHT)

    main_pad = ROOT.TPad(f"main_{var}", "main pad", 0.0, 0.0,
                         MAIN_PAD_XMAX, 1.0)
    side_pad = ROOT.TPad(f"side_{var}", "legend and integrals",
                         MAIN_PAD_XMAX, 0.0, 1.0, 1.0)
    main_pad.SetLeftMargin(0.15)
    main_pad.SetRightMargin(0.025)
    main_pad.SetBottomMargin(0.13)
    main_pad.SetTopMargin(0.12)
    side_pad.SetLeftMargin(0.00)
    side_pad.SetRightMargin(0.00)
    side_pad.SetTopMargin(0.12)
    side_pad.SetBottomMargin(0.08)
    main_pad.Draw()
    side_pad.Draw()

    histograms = []
    maximum = 0.0

    for index in stage_indices:
        hist_name = f"{var}/h_{var}_{index}"
        source = root_file.Get(hist_name)
        if not source:
            raise RuntimeError(f"Histogram not found in ROOT file: {hist_name}")

        clone = source.Clone(f"draw_{var}_{index}")
        clone.SetDirectory(0)
        move_edge_bins_to_flow_bins(clone)
        histograms.append((index, clone))
        maximum = max(maximum, clone.GetMaximum())

    if not histograms:
        print(f"Skipping {var}: no stages selected")
        canvas.Close()
        return

    y_max = (float(Y_MAX_OVERRIDE) if Y_MAX_OVERRIDE is not None
             else (maximum * Y_MAX_FACTOR if maximum > 0 else 1.0))

    main_pad.cd()
    if LOG_Y:
        main_pad.SetLogy(True)

    for position, (index, hist) in enumerate(histograms):
        hist.SetLineColor(COLORS[index % len(COLORS)])
        hist.SetLineWidth(1)
        hist.SetFillStyle(0)
        hist.GetXaxis().SetTitle(cfg["xlabel"])
        hist.GetYaxis().SetTitle("Expected events / bin")
        hist.GetXaxis().SetTitleSize(0.045)
        hist.GetYaxis().SetTitleSize(0.045)
        hist.GetXaxis().SetLabelSize(0.040)
        hist.GetYaxis().SetLabelSize(0.040)
        hist.GetYaxis().SetTitleOffset(1.45)
        hist.SetMinimum(0.01 if LOG_Y else 0.0)
        hist.SetMaximum(y_max)
        hist.Draw("HIST" if position == 0 else "HIST SAME")

    live = ROOT.TExec(f"live_stats_{var}", "updateStats();")
    live.Draw() 

    CMS_objects = CMS_label(main_pad)
    main_pad.RedrawAxis()

    canvas.Update()

    # Right-side panel: legend at the top, followed by a proper TPaveText
    # statbox for every selected cut.
    side_pad.cd()

    statbox_objects = []
    n_entries = len(histograms)
    box_top = 0.94
    box_bottom = 0.06
    available_height = box_top - box_bottom
    box_height = min(0.105, available_height / max(n_entries, 1))

    for position, (index, hist) in enumerate(histograms):
        y2 = box_top - position * box_height
        y1 = y2 - box_height * 0.88
        statbox = draw_statbox_manual_small(
            hist,
            0.005, y1, 0.98, y2,
            COLORS[index % len(COLORS)],
            label=stages[index],
            is_signal=False,
        )
        statbox_objects.append(statbox)

    # Keep ROOT-owned drawing objects alive until all output formats are saved.
    side_pad.Modified()

    canvas.cd()
    canvas.Modified()
    canvas.Update()

    png_path = os.path.join(PLOT_OUTPUT_DIR, f"{var}_cutflow_overlay.png")
    pdf_path = os.path.join(PLOT_OUTPUT_DIR, f"{var}_cutflow_overlay.pdf")
    canvas.SaveAs(png_path)
    canvas.SaveAs(pdf_path)
    output_root_file.cd()
    canvas.Write(f"c_{var}", ROOT.TObject.kOverwrite)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    canvas.Close()


def draw_cutflow(root_file, metadata, output_root_file):
    source = root_file.Get("h_cutflow")
    if not source:
        print("WARNING: h_cutflow not found; skipping cutflow plot.")
        return

    clone = source.Clone("h_cutflow_for_plot")
    clone.SetDirectory(0)

    canvas = ROOT.TCanvas("c_cutflow", "c_cutflow", 1200, 900)
    canvas.Modified()
    canvas.Update()
    pad = ROOT.TPad("cutflow_pad", "cutflow pad", 0.0, 0.0, 1.0, 1.0)
    pad.SetLeftMargin(0.20)
    pad.SetBottomMargin(0.25)
    pad.SetRightMargin(0.02)
    pad.SetTopMargin(0.12)
    pad.Draw()
    pad.cd()

    clone.SetLineWidth(2)
    clone.SetMarkerStyle(20)
    clone.SetMarkerSize(1.0)
    clone.GetYaxis().SetTitle("Expected events")
    clone.GetXaxis().SetLabelSize(0.030)
    clone.GetXaxis().LabelsOption("v")
    clone.SetMinimum(0)
    maximum = clone.GetMaximum()
    clone.SetMaximum(maximum * 1.25 if maximum > 0 else 1.0)
    clone.Draw("E1")
    cms_objects = CMS_label(pad)
    pad.RedrawAxis()

    canvas.SaveAs(os.path.join(PLOT_OUTPUT_DIR, "cutflow.png"))
    canvas.SaveAs(os.path.join(PLOT_OUTPUT_DIR, "cutflow.pdf"))
    output_root_file.cd()
    canvas.Write("c_cutflow", ROOT.TObject.kOverwrite)
    canvas.Close()


def main():
    if not os.path.exists(ROOT_FILENAME):
        raise FileNotFoundError(f"ROOT histogram file not found: {ROOT_FILENAME}")

    metadata = load_metadata(METADATA_FILENAME)

    print("Loaded processing metadata:")
    print(f"  Processing date : {metadata.get('processing_date', 'unknown')}")
    print(f"  Luminosity      : {metadata.get('luminosity_fb_inverse')} fb^-1")
    print(f"  Batch size      : {metadata.get('batch_size')}")
    print(f"  Weight branch   : {metadata.get('weight_branch')}")
    print(f"  ROOT cache      : {ROOT_FILENAME}")

    print("\nCached total background cutflow:")
    yields = metadata["cutflow_yields"]
    uncertainties = metadata["cutflow_uncertainties"]
    for item, value, error in zip(metadata["selections"], yields, uncertainties):
        print(f"{item['stage']:35s} {value:16.6f} +/- {error:.6f}")

    root_file = ROOT.TFile.Open(ROOT_FILENAME, "READ")
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"Could not open ROOT file: {ROOT_FILENAME}")

    output_root_file = ROOT.TFile.Open(
        PLOT_ROOT_FILENAME, "RECREATE"
    )

    if not output_root_file or output_root_file.IsZombie():
        raise RuntimeError(
            f"Could not create ROOT output file: {PLOT_ROOT_FILENAME}"
        )

    variables = get_selected_variables(metadata)
    stage_indices = get_selected_stages(metadata)

    print("\nDrawing cached histograms...")
    print("\nHistogram integral diagnostics:")

    for var in ["first_jet_eta", "first_jet_B",
                "second_jet_eta", "second_jet_B"]:

        print(f"\nVariable: {var}")

        for index in stage_indices:

            hist_name = f"{var}/h_{var}_{index}"
            hist = root_file.Get(hist_name)

            if not hist:
                print(f"  Missing: {hist_name}")
                continue

            nbins = hist.GetNbinsX()

            regular = hist.Integral(1, nbins)
            underflow = hist.GetBinContent(0)
            overflow = hist.GetBinContent(nbins + 1)
            total = hist.Integral(0, nbins + 1)

            print(
                f"  {metadata['selections'][index]['stage']:25s} "
                f"Regular = {regular:15.4f} "
                f"Underflow = {underflow:12.4f} "
                f"Overflow = {overflow:12.4f} "
                f"Total = {total:15.4f}"
            )
    for var in variables:
        print(f"Drawing {var}")
        draw_overlay(root_file, metadata, var, stage_indices, output_root_file)

    draw_cutflow(root_file, metadata, output_root_file)
    output_root_file.Write()
    output_root_file.Close()
    root_file.Close()
    print(f"\nAll plots saved under: {PLOT_OUTPUT_DIR}")


if __name__ == "__main__":
    main()