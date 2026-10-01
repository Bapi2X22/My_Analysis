import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import ROOT
from glob import glob
import pyarrow.parquet as pq

ROOT.gROOT.SetBatch(True)   # (you are saving files → good practice)
ROOT.gStyle.SetOptStat(1110)
ROOT.gROOT.ForceStyle()

# Plot_dir = "/eos/user/b/bbapi/www/Analysis_plots/BDT/variables_check/Before_BDT/"
# Plot_dir_BDT = "/eos/user/b/bbapi/www/Analysis_plots/BDT/variables_check/After_BDT/"

Plot_dir_ele = "/eos/user/b/bbapi/www/Analysis_plots/DATA_MC_validation/Cutflow/Overlap_removed_btag/TTbar_treatment/ele/"
Plot_dir_mu = "/eos/user/b/bbapi/www/Analysis_plots/DATA_MC_validation/Cutflow/Overlap_removed_btag/TTbar_treatment/mu/"

# Plot_dir = "/eos/user/b/bbapi/www/Analysis_plots/BDT/variables_check_without_DY/Before_BDT/"
# Plot_dir_BDT = "/eos/user/b/bbapi/www/Analysis_plots/BDT/variables_check_without_DY/After_BDT/"

def CMS_label(pad,
            #   lumi="39.05 fb^{-1}",
              lumi=109.0,
              year="2024",
              energy="13.6 TeV",
              status="Simulation Preliminary",
              x=0.12,
              y=0.95):

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


def get_global_range(arrays, padding=0.05, invalid_value=-999.0):

    mins = []
    maxs = []

    for arr in arrays:

        arr = ak.to_numpy(ak.flatten(arr, axis=None))

        # Remove invalid entries
        arr = arr[arr != invalid_value]

        if len(arr) == 0:
            continue

        mins.append(np.min(arr))
        maxs.append(np.max(arr))

    if len(mins) == 0:
        return 0., 1.

    xmin = min(mins)
    xmax = max(maxs)

    width = xmax - xmin

    xmin -= padding * width
    xmax += padding * width

    return xmin, xmax


def get_mgg(one_photon, category_mask):

    cat_ph = one_photon[category_mask]

    sorted_ph = cat_ph[ak.argsort(cat_ph.pt, ascending=False)]
    padded_ph = ak.pad_none(sorted_ph, 2)

    lead = padded_ph[:, 0]
    sub  = padded_ph[:, 1]

    valid_mask = (
        ~ak.is_none(lead.pt) &
        ~ak.is_none(sub.pt)
    )

    lead = lead[valid_mask]
    sub  = sub[valid_mask]

    lead_p4 = ak.zip(
        {
            "pt": lead.pt,
            "eta": lead.eta,
            "phi": lead.phi,
            "mass": ak.zeros_like(lead.pt),
        },
        with_name="Momentum4D",
    )

    sub_p4 = ak.zip(
        {
            "pt": sub.pt,
            "eta": sub.eta,
            "phi": sub.phi,
            "mass": ak.zeros_like(sub.pt),
        },
        with_name="Momentum4D",
    )

    diphoton = lead_p4 + sub_p4

    return ak.to_numpy(diphoton.mass)


def draw_statbox_manual(hist, x1, y1, x2, y2, color, label=None):

    stats = ROOT.TPaveText(x1, y1, x2, y2, "NDC")
    stats.SetFillColor(0)
    stats.SetBorderSize(1)
    stats.SetTextColor(color)
    stats.SetTextFont(42)
    stats.SetTextSize(0.02)

    nbins = hist.GetNbinsX()

    underflow = hist.GetBinContent(0)
    overflow  = hist.GetBinContent(nbins + 1)

    integral = hist.Integral(1, nbins)

    if label is not None:
        stats.AddText(label)

    stats.AddText(f"Entries = {int(hist.GetEntries())}")
    stats.AddText(f"Underflow = {underflow:.2f}")
    stats.AddText(f"Overflow = {overflow:.2f}")
    stats.AddText(f"Integral = {integral:.2f}")

    stats.Draw()

    return stats


def draw_statbox_manual_small(hist, x1, y1, x2, y2, color, label=None):

    stats = ROOT.TPaveText(x1, y1, x2, y2, "NDC")
    stats.SetFillColor(0)
    stats.SetBorderSize(1)
    stats.SetTextColor(color)
    stats.SetTextFont(42)
    stats.SetTextSize(0.02)

    nbins = hist.GetNbinsX()

    underflow = hist.GetBinContent(0)
    overflow  = hist.GetBinContent(nbins + 1)

    integral = hist.Integral(1, nbins)

    if label is not None:
        stats.AddText(label)

    stats.AddText(f"Entries = {int(hist.GetEntries())}")
    stats.AddText(f"Integral = {integral:.2f}")

    stats.Draw()

    return stats

def draw_statbox_combined(hist_list, x1, y1, x2, y2, color, label=None):

    stats = ROOT.TPaveText(x1, y1, x2, y2, "NDC")
    stats.SetFillColor(0)
    stats.SetBorderSize(1)
    stats.SetTextColor(color)
    stats.SetTextFont(42)
    stats.SetTextSize(0.02)

    nbins = hist.GetNbinsX()

    # underflow = hist.GetBinContent(0)
    # overflow  = hist.GetBinContent(nbins + 1)

    # integral = hist.Integral(1, nbins)

    underflow = 0
    overflow = 0
    Integral = 0
    Entries = 0

    for h in hist_list:
        underflow += h.GetBinContent(0)
        overflow += h.GetBinContent(nbins + 1)
        Integral += h.Integral(1, nbins)
        Entries += h.GetEntries()

    if label is not None:
        stats.AddText(label)

    stats.AddText(f"Entries = {int(Entries)}")
    stats.AddText(f"Integral = {Integral:.2f}")

    stats.Draw()

    return stats

# # ==============================
# # INPUTS
# # ==============================
# lumi = 39.05  # fb^-1
# # lumi = 32.05  # fb^-1

# # cross sections in pb
# xsec0 = 671.5
# xsec1 = 4.634
# xsec2 = 98.04
# xsec3 = 405.87
# xsec4 = 2124.08
# xsec5 = 2124.08
# xsec6 = 21140.0
# xsec7 = 21190.0


# # variables_list = ['electron_eta', 'electron_phi', 'electron_pt','first_jet_eta', 'first_jet_phi', 'first_jet_pt', 'lepeta', 'leppt', 'muon_eta', 'muon_phi', 'muon_pt', 'pholead_ScEta', 'pholead_eta', 'pholead_mvaID', 'pholead_phi', 'pholead_pt', 'phosublead_ScEta', 'phosublead_eta','phosublead_mvaID', 'phosublead_phi', 'phosublead_pt', 'phosublead_superclusterEta','second_jet_eta', 'second_jet_phi', 'second_jet_pt']

# variables_list = ['electron_eta', 'electron_phi', 'electron_pt', 'muon_eta', 'muon_phi', 'muon_pt']

# plot_config = {
#     "electron_pt": {
#         "range": (0.0, 200.0),
#         "nbins": 40,          # bin width = 10 GeV
#     },
#     "muon_pt": {
#         "range": (0.0, 200.0),
#         "nbins": 40,          # bin width = 10 GeV
#     },
#     "electron_eta": {
#         "range": (-2.5, 2.5),
#         "nbins": 25,          # bin width = 0.2
#     },
#     "muon_eta": {
#         "range": (-2.5, 2.5),
#         "nbins": 25,          # bin width = 0.2
#     },
#     "electron_phi": {
#         "range": (-3.2, 3.2),
#         "nbins": 32,          # bin width = 0.2
#     },
#     "muon_phi": {
#         "range": (-3.2, 3.2),
#         "nbins": 32,          # bin width = 0.2
#     },
# }

# base_dir = "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/NTuples_2024_BKG_lepCorr/"

# base_dir += "/merged"

# dir_2L2Nu_cat1    = f"{base_dir}/TTto2L2Nu_24SummerRun3/CAT1_merged.parquet"
# dir_LNu2Q_cat1    = f"{base_dir}/TTtoLNu2Q_24SummerRun3/CAT1_merged.parquet"
# dir_G1Jets_cat1   = f"{base_dir}/TTG1Jets_24SummerRun3/CAT1_merged.parquet"
# dir_WGtoLNuG_cat1 = f"{base_dir}/WGtoLNuG_24SummerRun3/CAT1_merged.parquet"
# dir_DYto2Mu50_cat1 = f"{base_dir}/DYto2Mu50_24SummerRun3/CAT1_merged.parquet"
# dir_DYto2E50_cat1  = f"{base_dir}/DYto2E50_24SummerRun3/CAT1_merged.parquet"
# dir_DYto2Mu10_cat1 = f"{base_dir}/DYto2Mu10_24SummerRun3/CAT1_merged.parquet"
# dir_DYto2E10_cat1 = f"{base_dir}/DYto2E10_24SummerRun3/CAT1_merged.parquet"


# dir_2L2Nu_inclusive    = f"{base_dir}/TTto2L2Nu_24SummerRun3/Inclusive_merged.parquet"
# dir_LNu2Q_inclusive    = f"{base_dir}/TTtoLNu2Q_24SummerRun3/Inclusive_merged.parquet"
# dir_G1Jets_inclusive   = f"{base_dir}/TTG1Jets_24SummerRun3/Inclusive_merged.parquet"
# dir_WGtoLNuG_inclusive = f"{base_dir}/WGtoLNuG_24SummerRun3/Inclusive_merged.parquet"
# dir_DYto2Mu50_inclusive = f"{base_dir}/DYto2Mu50_24SummerRun3/Inclusive_merged.parquet"
# dir_DYto2E50_inclusive  = f"{base_dir}/DYto2E50_24SummerRun3/Inclusive_merged.parquet"
# dir_DYto2Mu10_inclusive = f"{base_dir}/DYto2Mu10_24SummerRun3/Inclusive_merged.parquet"
# dir_DYto2E10_inclusive = f"{base_dir}/DYto2E10_24SummerRun3/Inclusive_merged.parquet"

# # events_2L2Nu_cat1    = ak.from_parquet(dir_2L2Nu_cat1)
# # events_LNu2Q_cat1    = ak.from_parquet(dir_LNu2Q_cat1)
# # events_G1Jets_cat1   = ak.from_parquet(dir_G1Jets_cat1)
# # events_WGtoLNuG_cat1 = ak.from_parquet(dir_WGtoLNuG_cat1)
# # events_DYto2Mu50_cat1 = ak.from_parquet(dir_DYto2Mu50_cat1)
# # events_DYto2E50_cat1  = ak.from_parquet(dir_DYto2E50_cat1)

# file_pattern_ele_cat1 = "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_Data_skim_lepCorr/merged/Data_EGamma-Data-2024*/EGamma-Data-2024*_CAT1_merged.parquet"
# file_pattern_ele_inclusive = "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_Data_skim_lepCorr/merged/Data_EGamma-Data-2024*/EGamma-Data-2024*_Inclusive_merged.parquet"

# files_ele_cat1 = sorted(glob(file_pattern_ele_cat1))
# files_ele_inclusive = sorted(glob(file_pattern_ele_inclusive))
# # Data_ele = ak.from_parquet(files_ele)

# file_pattern_mu_cat1 = "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_Data_skim_lepCorr/merged/Data_Muon-Data-2024*/Muon-Data-2024*_CAT1_merged.parquet"
# file_pattern_mu_inclusive = "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_Data_skim_lepCorr/merged/Data_Muon-Data-2024*/Muon-Data-2024*_Inclusive_merged.parquet"

# files_mu_cat1 = sorted(glob(file_pattern_mu_cat1))
# files_mu_inclusive = sorted(glob(file_pattern_mu_inclusive))
# # Data_mu = ak.from_parquet(files_mu)

# # ==============================
# # Categories
# # ==============================
# categories = ["cat1", "Inclusive"]

# # ==============================
# # Processes (BACKGROUND)
# # ==============================
# processes = {
#     "DYto2Mu50": {
#         "events": {
#             "cat1": dir_DYto2Mu50_cat1,
#             "Inclusive": dir_DYto2Mu50_inclusive
#         },
#         "xsec": xsec4,
#         "color": ROOT.kCyan+1
#     },
#     "DYto2E50": {
#         "events": {
#             "cat1": dir_DYto2E50_cat1,
#             "Inclusive": dir_DYto2E50_inclusive
#         },
#         "xsec": xsec5,
#         "color": ROOT.kOrange+7
#     },
#     "DYto2Mu10": {
#         "events": {
#             "cat1": dir_DYto2Mu10_cat1,
#             "Inclusive": dir_DYto2Mu10_inclusive
#         },
#         "xsec": xsec7,
#         "color": ROOT.kCyan+1
#     },
#     "DYto2E10": {
#         "events": {
#             "cat1": dir_DYto2E10_cat1,
#             "Inclusive": dir_DYto2E10_inclusive
#         },
#         "xsec": xsec6,
#         "color": ROOT.kOrange+7
#     },
#     "WGtoLNuG": {
#         "events": {
#             "cat1": dir_WGtoLNuG_cat1,
#             "Inclusive": dir_WGtoLNuG_inclusive
#         },
#         "xsec": xsec0,
#         "color": ROOT.kMagenta+1
#     },
#     "TTG1Jets": {
#         "events": {
#             "cat1": dir_G1Jets_cat1,
#             "Inclusive": dir_G1Jets_inclusive
#         },
#         "xsec": xsec1,
#         "color": ROOT.kRed+1
#     },
#     "TTto2L2Nu": {
#         "events": {
#             "cat1": dir_2L2Nu_cat1,
#             "Inclusive": dir_2L2Nu_inclusive
#         },
#         "xsec": xsec2,
#         "color": ROOT.kBlue+1
#     },
#     "TTtoLNu2Q": {
#         "events": {
#             "cat1": dir_LNu2Q_cat1,
#             "Inclusive": dir_LNu2Q_inclusive
#         },
#         "xsec": xsec3,
#         "color": ROOT.kGreen+2
#     }
# }

# # ==============================
# # SIGNAL
# # ==============================
# signal_masses = [20, 35, 55]
# signal_xsec = 0.48081  # pb
# Br_frac = 0.1

# #Add different colours than background
# signal_colors = {
#     20: ROOT.kBlack,
#     35: ROOT.kViolet+1,
#     55: ROOT.kPink+7
# }


# # ==============================
# # HELPERS
# # ==============================
# def make_hist(name, obs_name):
#     h = ROOT.TH1F(
#         name,
#         f";{obs_name};Events",
#         plot_config[obs_name]["nbins"],
#         plot_config[obs_name]["range"][0],
#         plot_config[obs_name]["range"][1]
#     )
#     h.Sumw2()
#     h.SetStats(1)
#     return h


# def fill_hist(hist, values, weights, scale, overflow=False):

#     xmax = hist.GetXaxis().GetXmax()
#     eps = 1e-6  # stay inside the last bin

#     for x, w in zip(values, weights):

#         if overflow and x >= xmax:
#             x = xmax - eps

#         hist.Fill(float(x), float(scale * w))


# # ==============================
# # MAIN LOOP
# # ==============================

# channels = ["electron", "muon"]

# for channel in channels:

#     for obs_name in variables_list:

#         overflow = "pt" in obs_name.lower()

#         for cat in categories:

#             use_logy = True

#             obs_lower = obs_name.lower()

#             # Keep references → avoid segfault
#             bkg_hists = []

#             stack = ROOT.THStack(
#                 f"stack_{obs_name}_{cat}",
#                 f";{obs_name};Events"
#             )

#             legend = ROOT.TLegend(0.64, 0.70, 0.86, 0.90)
#             legend.SetFillStyle(0)
#             legend.SetBorderSize(0)

#             needed_columns = [
#                 obs_name,
#                 "weight",
#                 "electron_pt",
#                 "muon_pt",
#             ]

#             # ------------------------------
#             # BACKGROUND LOOP
#             # ------------------------------
#             for proc_name, proc in processes.items():

#                 pf = pq.ParquetFile(proc["events"][cat])

#                 h = make_hist(
#                     f"{proc_name}_{channel}_{obs_name}_{cat}", obs_name
#                 )

#                 for batch in pf.iter_batches(
#                         batch_size=200000,
#                         columns=needed_columns):

#                     events = ak.Array(batch.to_pydict())

#                     values = np.asarray(events[obs_name])
#                     weights = np.asarray(events.weight)

#                     ele_pt = np.asarray(events.electron_pt)
#                     mu_pt  = np.asarray(events.muon_pt)

#                     if channel == "electron":
#                         channel_mask = (ele_pt != -999.) & (mu_pt == -999.)
#                     else:   # muon
#                         channel_mask = (ele_pt == -999.) & (mu_pt != -999.)

#                     mask = (values != -999.0) & channel_mask

#                     values = values[mask]
#                     weights = weights[mask]

#                     scale = proc["xsec"] * lumi * 1000.0

#                     fill_hist(h, values, weights, scale, overflow = overflow)

#                 h.SetFillColor(proc["color"])
#                 h.SetLineColor(0)
#                 h.SetLineWidth(0)

#                 stack.Add(h)
#                 legend.AddEntry(h, proc_name, "f")

#                 bkg_hists.append(h)

#             hData = make_hist(
#                 f"data_{channel}_{obs_name}_{cat}", obs_name
#             )

#             # if channel == "electron":
#             #     data_files = files_ele
#             # else:
#             #     data_files = files_mu

#             if channel == "electron":
#                 data_files = files_ele_cat1 if cat == "cat1" else files_ele_inclusive
#             else:
#                 data_files = files_mu_cat1 if cat == "cat1" else files_mu_inclusive

#             needed_columns = [obs_name, "weight"]

#             import pyarrow.parquet as pq

#             for filename in data_files:

#                 pf = pq.ParquetFile(filename)

#                 for batch in pf.iter_batches(
#                         batch_size=200000,
#                         columns=needed_columns):

#                     events = ak.Array(batch.to_pydict())

#                     data_values = np.asarray(events[obs_name])
#                     data_weights = np.asarray(events.weight)

#                     mask = (data_values != -999.)

#                     fill_hist(
#                         hData,
#                         data_values[mask],
#                         data_weights[mask],
#                         scale=1.0,
#                         overflow=overflow,
#                     )

#             hData.SetMarkerStyle(20)
#             hData.SetMarkerColor(ROOT.kBlack)
#             hData.SetLineColor(ROOT.kBlack)

#             # ==============================
#             # DRAW
#             # ==============================
#             c = ROOT.TCanvas(f"c_{obs_name}_{cat}", "", 1000, 800)

#             pad1 = ROOT.TPad("pad1", "pad1", 0.0, 0.30, 1.0, 1.0)
#             pad2 = ROOT.TPad("pad2", "pad2", 0.0, 0.00, 1.0, 0.30)

#             pad1.SetBottomMargin(0.02)
#             pad1.SetLeftMargin(0.12)
#             pad1.SetRightMargin(0.15)

#             pad2.SetTopMargin(0.05)
#             pad2.SetBottomMargin(0.35)
#             pad2.SetLeftMargin(0.12)
#             pad2.SetRightMargin(0.15)
#             pad2.SetGridy()

#             if use_logy:
#                 pad1.SetLogy()

#             pad1.Draw()
#             pad2.Draw()

#             # c.SetLogy()

#             pad1.cd()

#             stack.Draw("hist")

#             stack.GetXaxis().SetLabelSize(0)
#             stack.GetXaxis().SetTitleSize(0)

#             if use_logy:
#                 stack.SetMinimum(0.1)
#                 stack.SetMaximum(stack.GetMaximum() * 10.0)
#             elif ("mass" in obs_name):
#                 stack.SetMaximum(stack.GetMaximum() * 1.65)
#             else:
#                 stack.SetMaximum(stack.GetMaximum() * 1.4)

#             hBkg = stack.GetStack().Last().Clone(f"hBkg_{channel}_{obs_name}_{cat}")

#             stat_boxes = []

#             hData.Draw("PE SAME")

#             # ratio = hData.Clone(f"ratio_{channel}_{obs_name}_{cat}")
#             # ratio.Divide(hBkg)

#             ratio = hData.Clone(f"ratio_{channel}_{obs_name}_{cat}")

#             ratio.SetStats(0)

#             for i in range(1, ratio.GetNbinsX()+1):
#                 d = hData.GetBinContent(i)
#                 m = hBkg.GetBinContent(i)

#                 if m > 0:
#                     ratio.SetBinContent(i, d/m)
#                     ratio.SetBinError(i, hData.GetBinError(i)/m)
#                 else:
#                     ratio.SetBinContent(i, 0)
#                     ratio.SetBinError(i, 0)

#             # ------------------------------
#             # DRAW STAT BOXES (AFTER DRAWING ALL HISTS)
#             # ------------------------------
#             y_top = 0.90
#             height = 0.13

#             # Background stat box
#             box = draw_statbox_manual(
#                 hBkg,
#                 0.85,
#                 y_top - height,
#                 0.99,
#                 y_top,
#                 ROOT.kBlue,
#                 label="Background"
#             )
#             stat_boxes.append(box)

#             # Data stat box
#             box = draw_statbox_manual(
#                 hData,
#                 0.85,
#                 y_top - 2*height,
#                 0.99,
#                 y_top - 1*height,
#                 ROOT.kBlack,
#                 label="Data"
#             )
#             stat_boxes.append(box)

#             # ------------------------------
#             # FINAL DRAWING
#             # ------------------------------
#             legend.Draw()

#             latex = ROOT.TLatex()
#             latex.SetNDC()
#             latex.SetTextSize(0.035)
#             latex.SetTextFont(42)
#             xrange = plot_config[obs_name]["range"]
#             xminG, xmaxG = xrange[0], xrange[1]
#             nbins = plot_config[obs_name]["nbins"]
#             latex.DrawLatex(0.15, 0.87, f"{obs_name}, Category {cat[-1]}, binWidth = {(xmaxG - xminG)/nbins:.2f}")

#             CMS_label(c, lumi = lumi )

#             pad2.cd()

#             ratio.SetTitle("")

#             ratio.GetYaxis().SetTitle("Data/MC")
#             ratio.GetYaxis().CenterTitle()

#             ratio.GetYaxis().SetNdivisions(505)
#             ratio.GetYaxis().SetTitleSize(0.09)
#             ratio.GetYaxis().SetTitleOffset(0.45)
#             ratio.GetYaxis().SetLabelSize(0.08)

#             ratio.GetXaxis().SetTitle(obs_name)
#             ratio.GetXaxis().SetTitleSize(0.12)
#             ratio.GetXaxis().SetLabelSize(0.10)

#             ratio.SetMinimum(0.0)
#             ratio.SetMaximum(2.0)

#             ratio.SetMarkerStyle(20)
#             ratio.SetLineColor(ROOT.kBlack)

#             ratio.Draw("PE")

#             lines = []

#             line = ROOT.TLine(
#                 ratio.GetXaxis().GetXmin(),
#                 1.0,
#                 ratio.GetXaxis().GetXmax(),
#                 1.0,
#             )
#             line.SetLineStyle(2)
#             line.SetLineWidth(2)
#             line.Draw("same")

#             lines.append(line)

#             c.Modified()
#             c.Update()

#             # ------------------------------
#             # SAVE
#             # ------------------------------
#             print(f"Saving {obs_name}, {cat}")
#             if channel == "electron":
#                 c.SaveAs(f"{Plot_dir_ele}/stacked_{obs_name}_{cat}_linear.png")
#                 c.SaveAs(f"{Plot_dir_ele}/stacked_{obs_name}_{cat}_linear.pdf")
#             else:
#                 c.SaveAs(f"{Plot_dir_mu}/stacked_{obs_name}_{cat}_linear.png")
#                 c.SaveAs(f"{Plot_dir_mu}/stacked_{obs_name}_{cat}_linear.pdf")
















# ==============================
# INPUTS
# ==============================

lumi = 39.05  # fb^-1
# lumi = 32.05  # fb^-1


# Cross sections in pb
xsecs = {
    "WGtoLNuG": 671.5,
    "WtoENu0J": 55850,
    "WtoENu1J": 9177,
    "WtoENu2J": 3474,
    "WtoMuNu0J": 55920,
    "WtoMuNu1J": 9202,
    "WtoMuNu2J": 3490,
    "TTG1Jets": 4.634,
    "TTto2L2Nu": 98.04,
    "TTtoLNu2Q": 405.87,
    "DYto2Mu10": 21190.0,
    "DYto2Mu50": 2124.08,
    "DYto2E10": 21140.0,
    "DYto2E50": 2124.08,
    "DYGto2LG4": 88.13,
    "DYGto2LG50": 126.7
}


# ==============================
# VARIABLES
# ==============================

variables_list = [
    "electron_eta",
    "electron_phi",
    "electron_pt",
    "muon_eta",
    "muon_phi",
    "muon_pt",
]


plot_config = {
    "electron_pt": {
        "range": (0.0, 200.0),
        "nbins": 40,
    },
    "muon_pt": {
        "range": (0.0, 200.0),
        "nbins": 40,
    },
    "electron_eta": {
        "range": (-2.5, 2.5),
        "nbins": 25,
    },
    "muon_eta": {
        "range": (-2.5, 2.5),
        "nbins": 25,
    },
    "electron_phi": {
        "range": (-3.2, 3.2),
        "nbins": 32,
    },
    "muon_phi": {
        "range": (-3.2, 3.2),
        "nbins": 32,
    },
}


# ==============================
# PATHS
# ==============================

bkg_base = (
    "/eos/user/b/bbapi/"
    "My_Analysis/2024_efficiency_study/"
    "Backgrounds/NTuples_2024_BKG_overlap_removal_final/merged"
)

data_base = (
    "/eos/user/b/bbapi/"
    "My_Analysis/2024_efficiency_study/"
    "Data/NTuples_2024_Data_skim_lepCorr/merged"
)


# ==============================
# CATEGORIES
# ==============================

# categories = ["CAT1", "Inclusive", "twopho"]
categories = ["CAT1", "Inclusive"]
channels = ["electron", "muon"]


# ==============================
# BACKGROUND CONFIGURATION
# ==============================

background_colors = {
    "DYto2Mu50": ROOT.kCyan + 1,
    "DYto2E50": ROOT.kOrange + 7,
    "DYto2Mu10": ROOT.kCyan + 1,
    "DYto2E10": ROOT.kOrange + 7,
    "WGtoLNuG": ROOT.kMagenta + 1,
    "TTG1Jets": ROOT.kRed + 1,
    "TTto2L2Nu": ROOT.kBlue + 1,
    "TTtoLNu2Q": ROOT.kGreen + 2,
    "DYGto2LG4": ROOT.kViolet + 1,
    "DYGto2LG50": ROOT.kViolet + 1,
    "WtoENu0J": ROOT.kGray + 1,
    "WtoENu1J": ROOT.kGray + 1,
    "WtoENu2J": ROOT.kGray + 1,
    "WtoMuNu0J": ROOT.kPink + 1,
    "WtoMuNu1J": ROOT.kPink + 1,
    "WtoMuNu2J": ROOT.kPink + 1
}


processes = {
    process: {
        "file": {
            category: (
                f"{bkg_base}/{process}_24SummerRun3/"
                f"{category}_merged.parquet"
            )
            for category in categories
        },
        "xsec": xsecs[process],
        "color": background_colors[process],
    }
    for process in xsecs
}


# ==============================
# DATA FILES
# ==============================

data_periods = {
    "electron": [
        "Data_EGamma-Data-2024E",
        "Data_EGamma-Data-2024F",
    ],
    "muon": [
        "Data_Muon-Data-2024E",
        "Data_Muon-Data-2024F",
    ],
}


def get_data_files(channel, category):
    """Return a flat list of data Parquet files."""

    channel_name = (
        "EGamma"
        if channel == "electron"
        else "Muon"
    )

    files = []

    for period in data_periods[channel]:

        pattern = (
            f"{data_base}/{period}/"
            f"{channel_name}-Data-2024*"
            f"_{category}_merged.parquet"
        )

        files.extend(glob(pattern))

    return sorted(files)


data_files = {
    channel: {
        category: get_data_files(channel, category)
        for category in categories
    }
    for channel in channels
}


# ==============================
# SIGNAL
# ==============================

signal_masses = [20, 35, 55]
signal_xsec = 0.48081  # pb
Br_frac = 0.1

signal_colors = {
    20: ROOT.kBlack,
    35: ROOT.kViolet + 1,
    55: ROOT.kPink + 7,
}


# ==============================
# HELPER FUNCTIONS
# ==============================

def make_hist(name, obs_name):
    """Create an empty histogram."""

    config = plot_config[obs_name]
    xmin, xmax = config["range"]

    hist = ROOT.TH1F(
        name,
        f";{obs_name};Events",
        config["nbins"],
        xmin,
        xmax,
    )

    hist.Sumw2()
    hist.SetStats(1)

    return hist


def fill_hist(hist, values, weights, scale=1.0, overflow=False):
    """Fill a ROOT histogram."""

    xmax = hist.GetXaxis().GetXmax()
    eps = 1e-6

    for value, weight in zip(values, weights):

        if overflow and value >= xmax:
            value = xmax - eps

        hist.Fill(
            float(value),
            float(scale * weight),
        )


def get_channel_mask(channel, electron_pt, muon_pt):
    """Return the electron/muon event selection."""

    if channel == "electron":
        return (
            (electron_pt != -999.)
            & (muon_pt == -999.)
        )

    return (
        (electron_pt == -999.)
        & (muon_pt != -999.)
    )


def fill_parquet_hist(
    hist,
    filenames,
    obs_name,
    columns,
    scale=1.0,
    channel=None,
    overflow=False,
):
    """
    Read one or more Parquet files in batches and fill a histogram.

    filenames may be either a single filename or a list of filenames.
    """

    if isinstance(filenames, str):
        filenames = [filenames]

    for filename in filenames:

        if not isinstance(filename, str):
            raise TypeError(
                "Each Parquet input must be a string path. "
                f"Received {type(filename).__name__}: {filename}"
            )

        pf = pq.ParquetFile(filename)

        for batch in pf.iter_batches(
            batch_size=200000,
            columns=columns,
        ):

            events = ak.Array(batch.to_pydict())

            values = np.asarray(events[obs_name])
            weights = np.asarray(events.weight)

            mask = values != -999.

            if channel is not None:

                electron_pt = np.asarray(
                    events.electron_pt
                )

                muon_pt = np.asarray(
                    events.muon_pt
                )

                mask &= get_channel_mask(
                    channel,
                    electron_pt,
                    muon_pt,
                )

            fill_hist(
                hist,
                values[mask],
                weights[mask],
                scale=scale,
                overflow=overflow,
            )


def make_ratio(data_hist, bkg_hist, name):
    """Create Data/MC ratio histogram."""

    ratio = data_hist.Clone(name)
    ratio.SetStats(0)

    for i in range(1, ratio.GetNbinsX() + 1):

        data = data_hist.GetBinContent(i)
        mc = bkg_hist.GetBinContent(i)

        if mc > 0:
            ratio.SetBinContent(i, data / mc)
            ratio.SetBinError(
                i,
                data_hist.GetBinError(i) / mc,
            )
        else:
            ratio.SetBinContent(i, 0)
            ratio.SetBinError(i, 0)

    return ratio


def draw_ratio(ratio, obs_name):
    """Configure and draw the Data/MC ratio."""

    ratio.SetTitle("")

    ratio.GetYaxis().SetTitle("Data/MC")
    ratio.GetYaxis().CenterTitle()

    ratio.GetYaxis().SetNdivisions(505)
    ratio.GetYaxis().SetTitleSize(0.09)
    ratio.GetYaxis().SetTitleOffset(0.45)
    ratio.GetYaxis().SetLabelSize(0.08)

    ratio.GetXaxis().SetTitle(obs_name)
    ratio.GetXaxis().SetTitleSize(0.12)
    ratio.GetXaxis().SetLabelSize(0.10)

    ratio.SetMinimum(0.0)
    ratio.SetMaximum(2.0)

    ratio.SetMarkerStyle(20)
    ratio.SetLineColor(ROOT.kBlack)

    ratio.Draw("PE")

    line = ROOT.TLine(
        ratio.GetXaxis().GetXmin(),
        1.0,
        ratio.GetXaxis().GetXmax(),
        1.0,
    )

    line.SetLineStyle(2)
    line.SetLineWidth(2)
    line.Draw("same")

    return line


def draw_plot(
    stack,
    h_data,
    channel,
    obs_name,
    category,
):
    """Draw and save the stacked MC plus data plot."""

    use_logy = True

    canvas = ROOT.TCanvas(
        f"c_{channel}_{obs_name}_{category}",
        "",
        1000,
        800,
    )

    pad1 = ROOT.TPad(
        f"pad1_{channel}_{obs_name}_{category}",
        "pad1",
        0.0,
        0.30,
        1.0,
        1.0,
    )

    pad2 = ROOT.TPad(
        f"pad2_{channel}_{obs_name}_{category}",
        "pad2",
        0.0,
        0.00,
        1.0,
        0.30,
    )

    pad1.SetBottomMargin(0.02)
    pad1.SetLeftMargin(0.12)
    pad1.SetRightMargin(0.15)

    pad2.SetTopMargin(0.05)
    pad2.SetBottomMargin(0.35)
    pad2.SetLeftMargin(0.12)
    pad2.SetRightMargin(0.15)
    pad2.SetGridy()

    if use_logy:
        pad1.SetLogy()

    pad1.Draw()
    pad2.Draw()

    pad1.cd()

    stack.Draw("hist")

    stack.GetXaxis().SetLabelSize(0)
    stack.GetXaxis().SetTitleSize(0)

    if use_logy:
        stack.SetMinimum(0.1)
        stack.SetMaximum(stack.GetMaximum() * 10.0)

    elif "mass" in obs_name.lower():
        stack.SetMaximum(stack.GetMaximum() * 1.65)

    else:
        stack.SetMaximum(stack.GetMaximum() * 1.4)

    h_bkg = stack.GetStack().Last().Clone(
        f"hBkg_{channel}_{obs_name}_{category}"
    )

    h_data.Draw("PE SAME")

    ratio = make_ratio(
        h_data,
        h_bkg,
        f"ratio_{channel}_{obs_name}_{category}",
    )

    # stat_boxes = []

    # y_top = 0.90
    # height = 0.13

    # stat_boxes.append(
    #     draw_statbox_manual(
    #         h_bkg,
    #         0.85,
    #         y_top - height,
    #         0.99,
    #         y_top,
    #         ROOT.kBlue,
    #         label="Background",
    #     )
    # )

    # stat_boxes.append(
    #     draw_statbox_manual(
    #         h_data,
    #         0.85,
    #         y_top - 2 * height,
    #         0.99,
    #         y_top - height,
    #         ROOT.kBlack,
    #         label="Data",
    #     )
    # )

    stat_boxes = []

    y_top = 0.90

    height = 0.12
    component_height = height / 2.0
    gap = 0.0


    # ---------------------------------------------------------
    # Total Background
    # ---------------------------------------------------------
    stat_boxes.append(
        draw_statbox_manual(
            h_bkg,
            0.85,
            y_top - height,
            0.99,
            y_top,
            ROOT.kBlue,
            label="Background",
        )
    )


    # ---------------------------------------------------------
    # Data
    # ---------------------------------------------------------
    stat_boxes.append(
        draw_statbox_manual(
            h_data,
            0.85,
            y_top - 2 * height,
            0.99,
            y_top - height,
            ROOT.kBlack,
            label="Data",
        )
    )


    # ---------------------------------------------------------
    # Individual background components
    # ---------------------------------------------------------
    hists = stack.GetHists()

    box_top = y_top - 2 * height - gap

    h_WtoE = []
    h_WtoMu = []
    h_DYG = []
    h_DYE = []
    h_DYMu = []

    for i in range(hists.GetSize()):

        h = hists.At(i)

        # Extract only the sample name
        suffix = f"_{channel}_{obs_name}_{category}"
        label = h.GetName().replace(suffix, "")

        if label in ["WtoENu0J", "WtoENu1J", "WtoENu2J"]:
            h_WtoE.append(h)
        if label in ["WtoMuNu0J", "WtoMuNu1J", "WtoMuNu2J"]:
            h_WtoMu.append(h)
        if label in ["DYto2E10", "DYto2E50"]:
            h_DYE.append(h)
        if label in ["DYto2Mu10", "DYto2Mu50"]:
            h_DYMu.append(h)
        if label in ["DYGto2LG4", "DYGto2LG50"]:
            h_DYG.append(h)

        box_bottom = box_top - component_height

        if label in ["WGtoLNuG", "TTto2L2Nu", "TTtoLNu2Q", "TTG1Jets"]:

            stat_boxes.append(
                draw_statbox_manual_small(
                    h,
                    0.85,
                    box_bottom,
                    0.99,
                    box_top,
                    h.GetFillColor(),
                    label=label,
                )
            )

            box_top = box_bottom - gap

    stat_boxes.append(draw_statbox_combined(h_WtoE, 0.85, box_bottom, 0.99, box_top, ROOT.kGray + 1, label="WtoE2J"))
    stat_boxes.append(draw_statbox_combined(h_WtoMu, 0.85, box_bottom-component_height, 0.99, box_bottom, ROOT.kPink + 1, label="WtoMu2J"))
    stat_boxes.append(draw_statbox_combined(h_DYG, 0.85, box_bottom-2*component_height, 0.99, box_bottom-component_height, ROOT.kViolet + 1, label="DYGto2LG"))
    stat_boxes.append(draw_statbox_combined(h_DYE, 0.85, box_bottom-3*component_height, 0.99, box_bottom-2*component_height, ROOT.kOrange + 7, label="DYto2E2J"))
    stat_boxes.append(draw_statbox_combined(h_DYMu, 0.85, box_bottom-4*component_height, 0.99, box_bottom-3*component_height, ROOT.kCyan + 1, label="DYto2Mu2J"))

    legend = ROOT.TLegend(
        0.64,
        0.70,
        0.86,
        0.90,
    )

    legend.SetFillStyle(0)
    legend.SetBorderSize(0)

    for hist in stack.GetHists():

        process_name = hist.GetName().split(
            f"_{channel}_{obs_name}_{category}"
        )[0]

        legend.AddEntry(
            hist,
            process_name,
            "f",
        )

    legend.Draw()

    latex = ROOT.TLatex()

    latex.SetNDC()
    latex.SetTextSize(0.035)
    latex.SetTextFont(42)

    xmin, xmax = plot_config[obs_name]["range"]
    nbins = plot_config[obs_name]["nbins"]

    bin_width = (xmax - xmin) / nbins

    latex.DrawLatex(
        0.15,
        0.87,
        (
            f"{obs_name}, Category {category}, "
            f"binWidth = {bin_width:.2f}"
        ),
    )

    CMS_label(
        canvas,
        lumi=lumi,
    )

    pad2.cd()

    line = draw_ratio(
        ratio,
        obs_name,
    )

    canvas._stat_boxes = stat_boxes
    canvas._line = line
    canvas._legend = legend

    canvas.Modified()
    canvas.Update()

    plot_dir = (
        Plot_dir_ele
        if channel == "electron"
        else Plot_dir_mu
    )

    print(
        f"Saving {channel}: "
        f"{obs_name}, {category}"
    )

    canvas.SaveAs(
        f"{plot_dir}/"
        f"stacked_{obs_name}_{category}_linear.png"
    )

    canvas.SaveAs(
        f"{plot_dir}/"
        f"stacked_{obs_name}_{category}_linear.pdf"
    )


# ==============================
# MAIN LOOP
# ==============================

for channel in channels:

    for obs_name in variables_list:

        overflow = "pt" in obs_name.lower()

        for category in categories:

            stack = ROOT.THStack(
                f"stack_{channel}_{obs_name}_{category}",
                f";{obs_name};Events",
            )

            mc_columns = [
                obs_name,
                "weight",
                "electron_pt",
                "muon_pt",
            ]

            for process_name, process in processes.items():

                hist = make_hist(
                    f"{process_name}_{channel}_{obs_name}_{category}",
                    obs_name,
                )

                scale = (
                    process["xsec"]
                    * lumi
                    * 1000.0
                )

                fill_parquet_hist(
                    hist=hist,
                    filenames=process["file"][category],
                    obs_name=obs_name,
                    columns=mc_columns,
                    scale=scale,
                    channel=channel,
                    overflow=overflow,
                )

                hist.SetFillColor(
                    process["color"]
                )
                hist.SetLineColor(0)
                hist.SetLineWidth(0)

                stack.Add(hist)

            h_data = make_hist(
                f"data_{channel}_{obs_name}_{category}",
                obs_name,
            )

            fill_parquet_hist(
                hist=h_data,
                filenames=data_files[channel][category],
                obs_name=obs_name,
                columns=[
                    obs_name,
                    "weight",
                ],
                scale=1.0,
                channel=None,
                overflow=overflow,
            )

            h_data.SetMarkerStyle(20)
            h_data.SetMarkerColor(ROOT.kBlack)
            h_data.SetLineColor(ROOT.kBlack)

            draw_plot(
                stack=stack,
                h_data=h_data,
                channel=channel,
                obs_name=obs_name,
                category=category,
            )
