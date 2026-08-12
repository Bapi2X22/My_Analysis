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

Plot_dir_ele = "/eos/user/b/bbapi/www/Analysis_plots/DATA_MC_validation/Cutflow/lepton_cut/ele/"
Plot_dir_mu = "/eos/user/b/bbapi/www/Analysis_plots/DATA_MC_validation/Cutflow/lepton_cut/mu/"

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

# ==============================
# INPUTS
# ==============================
lumi = 5.44  # fb^-1
# lumi = 32.05  # fb^-1

# cross sections in pb
xsec0 = 671.5
xsec1 = 4.634
xsec2 = 98.04
xsec3 = 405.87
xsec4 = 2124.08
xsec5 = 2124.08
xsec6 = 21140.0
xsec7 = 21190.0


# variables_list = ['electron_eta', 'electron_phi', 'electron_pt','first_jet_eta', 'first_jet_phi', 'first_jet_pt', 'lepeta', 'leppt', 'muon_eta', 'muon_phi', 'muon_pt', 'pholead_ScEta', 'pholead_eta', 'pholead_mvaID', 'pholead_phi', 'pholead_pt', 'phosublead_ScEta', 'phosublead_eta','phosublead_mvaID', 'phosublead_phi', 'phosublead_pt', 'phosublead_superclusterEta','second_jet_eta', 'second_jet_phi', 'second_jet_pt']

variables_list = ['electron_eta', 'electron_phi', 'electron_pt', 'muon_eta', 'muon_phi', 'muon_pt']

plot_config = {
    "electron_pt": {
        "range": (0.0, 200.0),
        "nbins": 40,          # bin width = 10 GeV
    },
    "muon_pt": {
        "range": (0.0, 200.0),
        "nbins": 40,          # bin width = 10 GeV
    },
    "electron_eta": {
        "range": (-2.5, 2.5),
        "nbins": 25,          # bin width = 0.2
    },
    "muon_eta": {
        "range": (-2.5, 2.5),
        "nbins": 25,          # bin width = 0.2
    },
    "electron_phi": {
        "range": (-3.2, 3.2),
        "nbins": 32,          # bin width = 0.2
    },
    "muon_phi": {
        "range": (-3.2, 3.2),
        "nbins": 32,          # bin width = 0.2
    },
}

base_dir = "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/NTuples_BKG_2024_DATA_MC_check_lepton_only_sel"

base_dir += "/merged"

dir_2L2Nu_cat1    = f"{base_dir}/TTto2L2Nu_24SummerRun3/CAT1_merged.parquet"
dir_LNu2Q_cat1    = f"{base_dir}/TTtoLNu2Q_24SummerRun3/CAT1_merged.parquet"
dir_G1Jets_cat1   = f"{base_dir}/TTG1Jets_24SummerRun3/CAT1_merged.parquet"
dir_WGtoLNuG_cat1 = f"{base_dir}/WGtoLNuG_24SummerRun3/CAT1_merged.parquet"
dir_DYto2Mu50_cat1 = f"{base_dir}/DYto2Mu50_24SummerRun3/CAT1_merged.parquet"
dir_DYto2E50_cat1  = f"{base_dir}/DYto2E50_24SummerRun3/CAT1_merged.parquet"
dir_DYto2Mu10_cat1 = f"{base_dir}/DYto2Mu10_24SummerRun3/CAT1_merged.parquet"
dir_DYto2E10_cat1 = f"{base_dir}/DYto2E10_24SummerRun3/CAT1_merged.parquet"

# events_2L2Nu_cat1    = ak.from_parquet(dir_2L2Nu_cat1)
# events_LNu2Q_cat1    = ak.from_parquet(dir_LNu2Q_cat1)
# events_G1Jets_cat1   = ak.from_parquet(dir_G1Jets_cat1)
# events_WGtoLNuG_cat1 = ak.from_parquet(dir_WGtoLNuG_cat1)
# events_DYto2Mu50_cat1 = ak.from_parquet(dir_DYto2Mu50_cat1)
# events_DYto2E50_cat1  = ak.from_parquet(dir_DYto2E50_cat1)

file_pattern_ele = "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_Data_validation_debug_F/merged/Data_EGamma-Data-2024*/EGamma-Data-2024*_CAT1_merged.parquet"

files_ele = sorted(glob(file_pattern_ele))
# Data_ele = ak.from_parquet(files_ele)

file_pattern_mu = "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_Data_validation_debug_F/merged/Data_Muon-Data-2024*/Muon-Data-2024*_CAT1_merged.parquet"

files_mu = sorted(glob(file_pattern_mu))
# Data_mu = ak.from_parquet(files_mu)

# ==============================
# Categories
# ==============================
categories = ["cat1"]

# ==============================
# Processes (BACKGROUND)
# ==============================
processes = {
    "DYto2Mu50": {
        "events": {
            "cat1": dir_DYto2Mu50_cat1
        },
        "xsec": xsec4,
        "color": ROOT.kCyan+1
    },
    "DYto2E50": {
        "events": {
            "cat1": dir_DYto2E50_cat1
        },
        "xsec": xsec5,
        "color": ROOT.kOrange+7
    },
    "DYto2Mu10": {
        "events": {
            "cat1": dir_DYto2Mu10_cat1
        },
        "xsec": xsec7,
        "color": ROOT.kCyan+1
    },
    "DYto2E10": {
        "events": {
            "cat1": dir_DYto2E10_cat1
        },
        "xsec": xsec6,
        "color": ROOT.kOrange+7
    },
    "WGtoLNuG": {
        "events": {
            "cat1": dir_WGtoLNuG_cat1
        },
        "xsec": xsec0,
        "color": ROOT.kMagenta+1
    },
    "TTG1Jets": {
        "events": {
            "cat1": dir_G1Jets_cat1
        },
        "xsec": xsec1,
        "color": ROOT.kRed+1
    },
    "TTto2L2Nu": {
        "events": {
            "cat1": dir_2L2Nu_cat1
        },
        "xsec": xsec2,
        "color": ROOT.kBlue+1
    },
    "TTtoLNu2Q": {
        "events": {
            "cat1": dir_LNu2Q_cat1
        },
        "xsec": xsec3,
        "color": ROOT.kGreen+2
    }
}

# ==============================
# SIGNAL
# ==============================
signal_masses = [20, 35, 55]
signal_xsec = 0.48081  # pb
Br_frac = 0.1

#Add different colours than background
signal_colors = {
    20: ROOT.kBlack,
    35: ROOT.kViolet+1,
    55: ROOT.kPink+7
}


# ==============================
# HELPERS
# ==============================
def make_hist(name, obs_name):
    h = ROOT.TH1F(
        name,
        f";{obs_name};Events",
        plot_config[obs_name]["nbins"],
        plot_config[obs_name]["range"][0],
        plot_config[obs_name]["range"][1]
    )
    h.Sumw2()
    h.SetStats(1)
    return h


def fill_hist(hist, values, weights, scale, overflow=False):

    xmax = hist.GetXaxis().GetXmax()
    eps = 1e-6  # stay inside the last bin

    for x, w in zip(values, weights):

        if overflow and x >= xmax:
            x = xmax - eps

        hist.Fill(float(x), float(scale * w))


# ==============================
# MAIN LOOP
# ==============================

channels = ["electron", "muon"]

for channel in channels:

    for obs_name in variables_list:

        overflow = "pt" in obs_name.lower()

        for cat in categories:

            use_logy = True

            obs_lower = obs_name.lower()

            # Keep references → avoid segfault
            bkg_hists = []

            stack = ROOT.THStack(
                f"stack_{obs_name}_{cat}",
                f";{obs_name};Events"
            )

            legend = ROOT.TLegend(0.64, 0.70, 0.86, 0.90)
            legend.SetFillStyle(0)
            legend.SetBorderSize(0)

            needed_columns = [
                obs_name,
                "weight",
                "electron_pt",
                "muon_pt",
            ]

            # ------------------------------
            # BACKGROUND LOOP
            # ------------------------------
            for proc_name, proc in processes.items():

                pf = pq.ParquetFile(proc["events"][cat])

                h = make_hist(
                    f"{proc_name}_{channel}_{obs_name}_{cat}", obs_name
                )

                for batch in pf.iter_batches(
                        batch_size=200000,
                        columns=needed_columns):

                    events = ak.Array(batch.to_pydict())

                    values = np.asarray(events[obs_name])
                    weights = np.asarray(events.weight)

                    ele_pt = np.asarray(events.electron_pt)
                    mu_pt  = np.asarray(events.muon_pt)

                    if channel == "electron":
                        channel_mask = (ele_pt != -999.) & (mu_pt == -999.)
                    else:   # muon
                        channel_mask = (ele_pt == -999.) & (mu_pt != -999.)

                    mask = (values != -999.0) & channel_mask

                    values = values[mask]
                    weights = weights[mask]

                    scale = proc["xsec"] * lumi * 1000.0

                    fill_hist(h, values, weights, scale, overflow = overflow)

                h.SetFillColor(proc["color"])
                h.SetLineColor(0)
                h.SetLineWidth(0)

                stack.Add(h)
                legend.AddEntry(h, proc_name, "f")

                bkg_hists.append(h)

            hData = make_hist(
                f"data_{channel}_{obs_name}_{cat}", obs_name
            )

            if channel == "electron":
                data_files = files_ele
            else:
                data_files = files_mu

            needed_columns = [obs_name, "weight"]

            import pyarrow.parquet as pq

            for filename in data_files:

                pf = pq.ParquetFile(filename)

                for batch in pf.iter_batches(
                        batch_size=200000,
                        columns=needed_columns):

                    events = ak.Array(batch.to_pydict())

                    data_values = np.asarray(events[obs_name])
                    data_weights = np.asarray(events.weight)

                    mask = (data_values != -999.)

                    fill_hist(
                        hData,
                        data_values[mask],
                        data_weights[mask],
                        scale=1.0,
                        overflow=overflow,
                    )

            hData.SetMarkerStyle(20)
            hData.SetMarkerColor(ROOT.kBlack)
            hData.SetLineColor(ROOT.kBlack)

            # ==============================
            # DRAW
            # ==============================
            c = ROOT.TCanvas(f"c_{obs_name}_{cat}", "", 1000, 800)

            pad1 = ROOT.TPad("pad1", "pad1", 0.0, 0.30, 1.0, 1.0)
            pad2 = ROOT.TPad("pad2", "pad2", 0.0, 0.00, 1.0, 0.30)

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

            # c.SetLogy()

            pad1.cd()

            stack.Draw("hist")

            stack.GetXaxis().SetLabelSize(0)
            stack.GetXaxis().SetTitleSize(0)

            if use_logy:
                stack.SetMinimum(0.1)
                stack.SetMaximum(stack.GetMaximum() * 10.0)
            elif ("mass" in obs_name):
                stack.SetMaximum(stack.GetMaximum() * 1.65)
            else:
                stack.SetMaximum(stack.GetMaximum() * 1.4)

            hBkg = stack.GetStack().Last().Clone(f"hBkg_{channel}_{obs_name}_{cat}")

            stat_boxes = []

            hData.Draw("PE SAME")

            # ratio = hData.Clone(f"ratio_{channel}_{obs_name}_{cat}")
            # ratio.Divide(hBkg)

            ratio = hData.Clone(f"ratio_{channel}_{obs_name}_{cat}")

            ratio.SetStats(0)

            for i in range(1, ratio.GetNbinsX()+1):
                d = hData.GetBinContent(i)
                m = hBkg.GetBinContent(i)

                if m > 0:
                    ratio.SetBinContent(i, d/m)
                    ratio.SetBinError(i, hData.GetBinError(i)/m)
                else:
                    ratio.SetBinContent(i, 0)
                    ratio.SetBinError(i, 0)

            # ------------------------------
            # DRAW STAT BOXES (AFTER DRAWING ALL HISTS)
            # ------------------------------
            y_top = 0.90
            height = 0.13

            # Background stat box
            box = draw_statbox_manual(
                hBkg,
                0.85,
                y_top - height,
                0.99,
                y_top,
                ROOT.kBlue,
                label="Background"
            )
            stat_boxes.append(box)

            # Data stat box
            box = draw_statbox_manual(
                hData,
                0.85,
                y_top - 2*height,
                0.99,
                y_top - 1*height,
                ROOT.kBlack,
                label="Data"
            )
            stat_boxes.append(box)

            # ------------------------------
            # FINAL DRAWING
            # ------------------------------
            legend.Draw()

            latex = ROOT.TLatex()
            latex.SetNDC()
            latex.SetTextSize(0.035)
            latex.SetTextFont(42)
            xrange = plot_config[obs_name]["range"]
            xminG, xmaxG = xrange[0], xrange[1]
            nbins = plot_config[obs_name]["nbins"]
            latex.DrawLatex(0.15, 0.87, f"{obs_name}, Category {cat[-1]}, binWidth = {(xmaxG - xminG)/nbins:.2f}")

            CMS_label(c, lumi = lumi )

            pad2.cd()

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

            lines = []

            line = ROOT.TLine(
                ratio.GetXaxis().GetXmin(),
                1.0,
                ratio.GetXaxis().GetXmax(),
                1.0,
            )
            line.SetLineStyle(2)
            line.SetLineWidth(2)
            line.Draw("same")

            lines.append(line)

            c.Modified()
            c.Update()

            # ------------------------------
            # SAVE
            # ------------------------------
            print(f"Saving {obs_name}, {cat}")
            if channel == "electron":
                c.SaveAs(f"{Plot_dir_ele}/stacked_{obs_name}_{cat}_linear.png")
                c.SaveAs(f"{Plot_dir_ele}/stacked_{obs_name}_{cat}_linear.pdf")
            else:
                c.SaveAs(f"{Plot_dir_mu}/stacked_{obs_name}_{cat}_linear.png")
                c.SaveAs(f"{Plot_dir_mu}/stacked_{obs_name}_{cat}_linear.pdf")