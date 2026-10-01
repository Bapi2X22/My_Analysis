import os
import numpy as np
import ROOT
from plot_config import PROCESS_COLORS
from config import DR_CUT
ROOT.gROOT.ProcessLine(".L /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/Overlap_removal/live_stats.C")

HIGH_CUT_PROCESSES = {
    "TTG1Jets",
    "WGtoLNuG",
    "DYGto2LG",
}

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def set_stack_ymax_log(stack, histograms):
    ymax = stack.GetMaximum()
    stack.SetMaximum(ymax * 1000.0)
    # stack.SetMinimum(max(ymax * 1e-5, 1e-3))

def set_overlay_ymax_1p4_log(histograms):
    max_height = 0.0
    for hist in histograms:
        nbins = hist.GetNbinsX()
        if nbins <= 1:
            continue
        hist_max = max(hist.GetBinContent(i) for i in range(1, nbins))
        max_height = max(max_height, hist_max)
    if max_height > 0:
        ymax = 100.0 * max_height
        for hist in histograms:
            hist.SetMaximum(ymax)

def CMS_label(pad,
            #   lumi="39.05 fb^{-1}",
              lumi=109.0,
              year="2024",
              energy="13.6 TeV",
              status="Simulation Preliminary",
              x=0.12,
              y=0.92):

    pad.cd()

    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextAngle(0)
    latex.SetTextColor(ROOT.kBlack)

    # ---- CMS (bold) ----
    latex.SetTextFont(61)
    latex.SetTextSize(0.06)
    latex.DrawLatex(x, y, "CMS")

    # ---- Status (italic) ----
    if status != "":
        latex.SetTextFont(52)
        latex.SetTextSize(0.045)
        latex.DrawLatex(x + 0.11, y, status)

    # ---- Lumi text (right aligned) ----
    latex.SetTextFont(42)
    latex.SetTextSize(0.045)
    latex.SetTextAlign(31)
    lumi_text = f"{year} ({lumi} fb^{{-1}})"
    latex.DrawLatex(0.88, y, lumi_text)

def darker_color(color):

    if color == ROOT.kBlue:
        return ROOT.kBlue + 3

    if color == ROOT.kGreen + 2:
        return ROOT.kGreen + 3

    if color == ROOT.kRed:
        return ROOT.kRed + 2

    return color

def draw_statbox_manual(hist, x1, y1, x2, y2, color, label=None, show_mean=True):

    stats = ROOT.TPaveText(x1, y1, x2, y2, "NDC")
    stats.SetName(f"stat_{hist.GetName()}")
    stats.SetFillColor(0)
    stats.SetBorderSize(1)
    stats.SetTextColor(color)
    stats.SetTextFont(42)
    stats.SetTextSize(0.02)

    nbins = hist.GetNbinsX()
    mean = hist.GetMean()
    underflow = hist.GetBinContent(0)
    overflow  = hist.GetBinContent(nbins + 1)

    integral = hist.Integral(1, nbins)

    if label is not None:
        stats.AddText(label)

    stats.AddText(f"Entries = {int(hist.GetEntries())}")
    if show_mean:
        stats.AddText(f"Mean = {mean:.2f}")
    stats.AddText(f"Overflow = {overflow:.2f}")
    stats.AddText(f"Integral = {integral:.2f}")

    stats.Draw()

    return stats

# def draw_statboxes_horizontal( histograms, processes, x1=0.32, x2=0.88, y1=0.72, y2=0.88, show_mean=True, label_suffix1=None, label_suffix2=False):
#     statboxes = []
#     n = len(histograms)
#     if n == 0:
#         return statboxes
#     x_start = x1
#     x_end = x2
#     total_width = x_end - x_start
#     gap = 0.01
#     box_width = (total_width - (n - 1) * gap) / n
#     for i, (hist, process) in enumerate(zip(histograms, processes)):
#         Label_suffix2 = ""
#         if label_suffix2:
#             if process in HIGH_CUT_PROCESSES:
#                 Label_suffix2 = f"After dR > {DR_CUT}"
#             else:
#                 Label_suffix2 = f"After dR < {DR_CUT}"
#         box_x1 = x_start + i * (box_width + gap)
#         box_x2 = box_x1 + box_width
#         color = PROCESS_COLORS.get( process, ROOT.kBlack)
#         label = process
#         if label_suffix2:
#             label = f"{process} {label_suffix1} + {Label_suffix2}"
#         else:
#             label = f"{process} {label_suffix1}"
#         stats = draw_statbox_manual( hist, box_x1, y1, box_x2, y2, color, label=label, show_mean=show_mean)
#         statboxes.append(stats)
#     return statboxes

# def draw_statboxes_horizontal(histograms,processes,x1=0.32,x2=0.88,y1=0.72,y2=0.88,show_mean=True,label_suffix1=None,label_suffix2=False):
#     statboxes = []
#     n = len(histograms)
#     if n == 0:
#         return statboxes
#     x_start = x1
#     x_end = x2
#     total_width = x_end - x_start
#     gap = 0.01
#     box_width = (total_width - (n - 1) * gap) / n
#     for i, (hist, process) in enumerate(zip(histograms, processes)):
#         label_suffix2_text = ""
#         if label_suffix2:
#             if process in HIGH_CUT_PROCESSES:
#                 label_suffix2_text = f"dR > {DR_CUT}"
#             else:
#                 label_suffix2_text = f"dR < {DR_CUT}"
#         box_x1 = x_start + i * (box_width + gap)
#         box_x2 = box_x1 + box_width
#         color = PROCESS_COLORS.get(process, ROOT.kBlack)
#         if label_suffix2:
#             label_parts = []
#         # label_parts = [process]
#         if label_suffix1:
#             label_parts.append(label_suffix1)
#         if label_suffix2_text:
#             label_parts.append(label_suffix2_text)
#         label = " + ".join(label_parts)
#         stats = draw_statbox_manual(hist,box_x1,y1,box_x2,y2,color,label=label,show_mean=show_mean)
#         statboxes.append(stats)
#     return statboxes

def draw_statboxes_horizontal(
    histograms,
    processes,
    x1=0.32,
    x2=0.88,
    y1=0.72,
    y2=0.88,
    show_mean=True,
    label_suffix1=None,
    label_suffix2=False,
):
    statboxes = []
    n = len(histograms)

    if n == 0:
        return statboxes

    x_start = x1
    x_end = x2
    total_width = x_end - x_start

    gap = 0.01
    box_width = (total_width - (n - 1) * gap) / n

    for i, (hist, process) in enumerate(zip(histograms, processes)):

        label_suffix2_text = ""

        if label_suffix2:
            if process in HIGH_CUT_PROCESSES:
                label_suffix2_text = f"dR > {DR_CUT}"
            else:
                label_suffix2_text = f"dR < {DR_CUT}"

            # No process name when label_suffix2=True
            label_parts = []

            if label_suffix1:
                label_parts.append(label_suffix1)

            if label_suffix2_text:
                label_parts.append(label_suffix2_text)

        else:
            # Include process name normally
            label_parts = [process]

            if label_suffix1:
                label_parts.append(label_suffix1)

        box_x1 = x_start + i * (box_width + gap)
        box_x2 = box_x1 + box_width

        color = PROCESS_COLORS.get(process, ROOT.kBlack)

        label = " + ".join(label_parts)

        stats = draw_statbox_manual(
            hist,
            box_x1,
            y1,
            box_x2,
            y2,
            color,
            label=label,
            show_mean=show_mean,
        )

        statboxes.append(stats)

    return statboxes

def make_cumulative_histograms(histograms,processes,group_name,plot_name):
    cumulative_histograms = []
    if not histograms:
        return cumulative_histograms
    nbins = histograms[0].GetNbinsX()
    cumulative = np.zeros(nbins, dtype=float)
    line_styles = [ROOT.kDashed,ROOT.kDotted,ROOT.kDashDotted]
    for i, (hist, process) in enumerate(zip(histograms, processes)):
        color = PROCESS_COLORS.get(process,ROOT.kBlack)
        cumulative_hist = hist.Clone(f"{process}_{plot_name}_cumulative_{group_name}_{i}")
        cumulative_hist.SetDirectory(0)
        cumulative_hist.Reset("ICES")
        for ibin in range(1, nbins + 1):
            current = hist.GetBinContent(ibin)
            cumulative[ibin - 1] += current
            if current > 0:
                cumulative_hist.SetBinContent(ibin,cumulative[ibin - 1])
            else:
                cumulative_hist.SetBinContent(ibin,0.0)
        cumulative_hist.SetFillStyle(0)
        cumulative_hist.SetLineColor(color)
        cumulative_hist.SetLineStyle(line_styles[i % len(line_styles)])
        cumulative_hist.SetLineWidth(3)
        cumulative_histograms.append(cumulative_hist)
    return cumulative_histograms

def merge_overflow_into_last_bin(hist):
    nbins = hist.GetNbinsX()
    overflow = hist.GetBinContent(nbins + 1)
    overflow_error = hist.GetBinError(nbins + 1)
    last = hist.GetBinContent(nbins)
    last_error = hist.GetBinError(nbins)
    hist.SetBinContent(nbins, last + overflow)
    hist.SetBinError(nbins, np.sqrt(last_error**2 + overflow_error**2))
    hist.SetBinContent(nbins + 1, 0.0)
    hist.SetBinError(nbins + 1, 0.0)
    return hist

def set_ymax_1p4(hist):
    nbins = hist.GetNbinsX()
    if nbins <= 1:
        return
    max_height = max(hist.GetBinContent(i) for i in range(1, nbins))
    if max_height > 0:
        hist.SetMaximum(1.4 * max_height)

def set_overlay_ymax_1p4(histograms):
    max_height = 0.0
    for hist in histograms:
        nbins = hist.GetNbinsX()
        if nbins <= 1:
            continue
        hist_max = max(hist.GetBinContent(i) for i in range(1, nbins))
        max_height = max(max_height, hist_max)
    if max_height > 0:
        ymax = 1.4 * max_height
        for hist in histograms:
            hist.SetMaximum(ymax)

def set_stack_ymax_1p4(stack, histograms):
    if not histograms:
        return
    nbins = histograms[0].GetNbinsX()
    if nbins <= 1:
        return
    max_height = 0.0
    for ibin in range(1, nbins):
        bin_height = sum(hist.GetBinContent(ibin) for hist in histograms)
        max_height = max(max_height, bin_height)
    if max_height > 0:
        stack.SetMaximum(1.4 * max_height)