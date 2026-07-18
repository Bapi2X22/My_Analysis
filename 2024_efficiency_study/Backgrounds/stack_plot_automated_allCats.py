import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import ROOT
import argparse
import os
import shutil

ROOT.gROOT.SetBatch(True)   # (you are saving files → good practice)
ROOT.gStyle.SetOptStat(1110)
ROOT.gROOT.ForceStyle()

parser = argparse.ArgumentParser()

parser.add_argument(
    "--BDT",
    action="store_true",
    help="Use BDT-processed samples (merged instead of merged_old)"
)

args = parser.parse_args()

BDT = args.BDT

Plot_dir = "/eos/user/b/bbapi/www/Analysis_plots/BDT/variables_check_pBDT/10GeV/Before_BDT/"
Plot_dir_BDT = "/eos/user/b/bbapi/www/Analysis_plots/BDT/variables_check_pBDT/10GeV/After_BDT/"

# Plot_dir = "/eos/user/b/bbapi/www/Analysis_plots/BDT/variables_check_without_DY/Before_BDT/"
# Plot_dir_BDT = "/eos/user/b/bbapi/www/Analysis_plots/BDT/variables_check_without_DY/After_BDT/"

def CMS_label(pad,
              lumi="109 fb^{-1}",
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
    lumi_text = f"{year} ({lumi})"
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


def draw_statbox_manual(hist, x1, y1, x2, y2, color):

    stats = ROOT.TPaveText(x1, y1, x2, y2, "NDC")
    stats.SetFillColor(0)
    stats.SetBorderSize(1)
    stats.SetTextColor(color)
    stats.SetTextFont(42)
    stats.SetTextSize(0.02)

    nbins = hist.GetNbinsX()

    underflow = hist.GetBinContent(0)
    overflow  = hist.GetBinContent(nbins + 1)

    integral = hist.Integral(1, nbins)  # excludes under/overflow
    # If you want to include them:
    # integral = hist.Integral(0, nbins + 1)

    stats.AddText(f"Entries = {int(hist.GetEntries())}")
    stats.AddText(f"Underflow = {underflow:.2f}")
    stats.AddText(f"Overflow = {overflow:.2f}")
    stats.AddText(f"Integral = {integral:.2f}")

    stats.Draw()

    return stats


# ==============================
# INPUTS
# ==============================
lumi = 109  # fb^-1

# cross sections in pb
xsec0 = 671.5
xsec1 = 4.634
xsec2 = 98.04
xsec3 = 405.87
xsec4 = 2230.0
xsec5 = 2244.0
xsec6 = 21140.0
xsec7 = 21190.0


variables_list = ['Njets', 'delphi_bb', 'delphi_bbgg', 'delphi_gg', 'diff_first_jet_probb_probbb', 'dipho_pt', 'electron_eta', 'electron_phi', 'electron_pt', 'first_jet_B', 'first_jet_eta', 'first_jet_phi', 'first_jet_probb', 'first_jet_probbb', 'first_jet_pt', 'lepeta', 'leppt', 'mass','muon_eta', 'muon_phi', 'muon_pt', 'n_bJets', 'pT1_by_mbb', 'pT1_by_mgg', 'pT2_by_mbb', 'pT2_by_mgg', 'pholead_ScEta', 'pholead_ecalPFClusterIso', 'pholead_eta', 'pholead_hoe', 'pholead_mvaID', 'pholead_pfRelIso03_all_quadratic', 'pholead_pfRelIso03_chg_quadratic', 'pholead_phi', 'pholead_pt', 'pholead_r9', 'pholead_s4', 'pholead_sieie', 'pholead_sieip', 'pholead_sipip', 'pholead_superclusterEta', 'pholead_trkSumPtHollowConeDR03', 'pholead_trkSumPtSolidConeDR04', 'phosublead_ScEta', 'phosublead_ecalPFClusterIso', 'phosublead_eta', 'phosublead_hoe','phosublead_mvaID', 'phosublead_pfRelIso03_all_quadratic', 'phosublead_pfRelIso03_chg_quadratic', 'phosublead_phi', 'phosublead_pt', 'phosublead_r9', 'phosublead_s4', 'phosublead_sieie', 'phosublead_sieip', 'phosublead_sipip', 'phosublead_superclusterEta', 'phosublead_trkSumPtHollowConeDR03', 'phosublead_trkSumPtSolidConeDR04','second_jet_B', 'second_jet_eta', 'second_jet_phi', 'second_jet_probb', 'second_jet_probbb', 'second_jet_pt', 'BDT_score', 'mass_point']

base_dir = "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/NTuples_BKG_2024_HDNA_presel_pBDT_score_pho10/"

if BDT:
    base_dir += "/merged"
else:
    base_dir += "/merged_old"

# dir_2L2Nu_cat1    = f"{base_dir}/TTto2L2Nu-24SummerRun3/nominal/CAT1_merged.parquet"
# dir_LNu2Q_cat1    = f"{base_dir}/TTtoLNu2Q-24SummerRun3/nominal/CAT1_merged.parquet"
# dir_G1Jets_cat1   = f"{base_dir}/TTG1Jets-24SummerRun3/nominal/CAT1_merged.parquet"
# dir_WGtoLNuG_cat1 = f"{base_dir}/WGtoLNuG-24SummerRun3/nominal/CAT1_merged.parquet"
# dir_DYto2Mu50_cat1 = f"{base_dir}/DYto2Mu50-24SummerRun3/nominal/CAT1_merged.parquet"
# dir_DYto2E50_cat1  = f"{base_dir}/DYto2E50-24SummerRun3/nominal/CAT1_merged.parquet"

# events_2L2Nu_cat1    = ak.from_parquet(dir_2L2Nu_cat1)
# events_LNu2Q_cat1    = ak.from_parquet(dir_LNu2Q_cat1)
# events_G1Jets_cat1   = ak.from_parquet(dir_G1Jets_cat1)
# events_WGtoLNuG_cat1 = ak.from_parquet(dir_WGtoLNuG_cat1)
# events_DYto2Mu50_cat1 = ak.from_parquet(dir_DYto2Mu50_cat1)
# events_DYto2E50_cat1  = ak.from_parquet(dir_DYto2E50_cat1)


# dir_DYto2E10_cat1 = "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/NTuples_BKG_2024_HDNA_presel_official_full/merged/DYto2E10-24SummerRun3/nominal/CAT1_merged.parquet"
# dir_DYto2E10_cat2 = "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/NTuples_BKG_2024_HDNA_presel_official_full/merged/DYto2E10-24SummerRun3/nominal/CAT2_merged.parquet"
# dir_DYto2E10_cat3 = "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/NTuples_BKG_2024_HDNA_presel_official_full/merged/DYto2E10-24SummerRun3/nominal/CAT3_merged.parquet"
# events_DYto2E10_cat1 = ak.from_parquet(dir_DYto2E10_cat1)
# events_DYto2E10_cat2 = ak.from_parquet(dir_DYto2E10_cat2)
# events_DYto2E10_cat3 = ak.from_parquet(dir_DYto2E10_cat3)

# dir_DYto2Mu10_cat1 = "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/NTuples_BKG_2024_HDNA_presel_official_full/merged/DYto2Mu10-24SummerRun3/nominal/CAT1_merged.parquet"
# dir_DYto2Mu10_cat2 = "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/NTuples_BKG_2024_HDNA_presel_official_full/merged/DYto2Mu10-24SummerRun3/nominal/CAT2_merged.parquet"
# dir_DYto2Mu10_cat3 = "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/NTuples_BKG_2024_HDNA_presel_official_full/merged/DYto2Mu10-24SummerRun3/nominal/CAT3_merged.parquet"
# events_DYto2Mu10_cat1 = ak.from_parquet(dir_DYto2Mu10_cat1)
# events_DYto2Mu10_cat2 = ak.from_parquet(dir_DYto2Mu10_cat2)
# events_DYto2Mu10_cat3 = ak.from_parquet(dir_DYto2Mu10_cat3)


categories = [
    "M12", "M15", "M20", "M25", "M30",
    "M35", "M40", "M45", "M50", "M55", "M60", "CAT1"
]

process_dirs = {
    "TTto2L2Nu": "TTto2L2Nu-24SummerRun3",
    "TTtoLNu2Q": "TTtoLNu2Q-24SummerRun3",
    "TTG1Jets": "TTG1Jets-24SummerRun3",
    "WGtoLNuG": "WGtoLNuG-24SummerRun3",
    "DYto2Mu": "DYto2Mu50-24SummerRun3",
    "DYto2E": "DYto2E50-24SummerRun3",
}

process_events = {}

for proc, dirname in process_dirs.items():
    process_events[proc] = {}
    for cat in categories:
        path = f"{base_dir}/{dirname}/{cat}_merged.parquet"
        process_events[proc][cat] = ak.from_parquet(path)

# ==============================
# HISTOGRAM SETTINGS
# ==============================
nbins = 60
xmin = 10
xmax = 70

# ==============================
# Categories
# ==============================
cat_sig = ["cat1"]

# ==============================
# Processes (BACKGROUND)
# ==============================
processes = {
    "DYto2Mu": {
        "events": process_events["DYto2Mu"],
        "xsec": xsec4,
        "color": ROOT.kCyan+1
    },
    "DYto2E": {
        "events": process_events["DYto2E"],
        "xsec": xsec5,
        "color": ROOT.kOrange+7
    },
    "WGtoLNuG": {
        "events": process_events["WGtoLNuG"],
        "xsec": xsec0,
        "color": ROOT.kMagenta+1
    },
    "TTG1Jets": {
        "events": process_events["TTG1Jets"],
        "xsec": xsec1,
        "color": ROOT.kRed+1
    },
    "TTto2L2Nu": {
        "events": process_events["TTto2L2Nu"],
        "xsec": xsec2,
        "color": ROOT.kBlue+1
    },
    "TTtoLNu2Q": {
        "events": process_events["TTtoLNu2Q"],
        "xsec": xsec3,
        "color": ROOT.kGreen+2
    }
}

# ==============================
# SIGNAL
# ==============================
signal_masses = [20, 35, 55]
signal_xsec = 1.457*0.22  # pb
Br_frac = 0.01

#Add different colours than background
signal_colors = {
    20: ROOT.kBlack,
    35: ROOT.kViolet+1,
    55: ROOT.kPink+7
}

signal_samples = {}

for m in signal_masses:
    signal_samples[m] = {}
    for i, cat in enumerate(cat_sig, start=1):
        if BDT:
            path = f"/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/NTuples_WH_2024_HDNA_presel_pho10/merged/WH-2024M{m}/nominal/CAT{i}_merged.parquet"
        else:
            path = f"/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/NTuples_WH_2024_HDNA_presel_pho10/merged_old/WH-2024M{m}/nominal/CAT{i}_merged.parquet"
        signal_samples[m][cat] = ak.from_parquet(path)


# ==============================
# HELPERS
# ==============================
def make_hist(name, xmin, xmax):
    h = ROOT.TH1F(
        name,
        f";{name};Events",
        25,
        xmin,
        xmax
    )
    h.Sumw2()
    h.SetStats(1)
    return h


def fill_hist(hist, values, weights, scale):
    # ROOT.gStyle.SetOptStat(1111111)
    for x, w in zip(values, weights):
        hist.Fill(float(x), float(scale * w))


# ==============================
# MAIN LOOP
# ==============================
for obs_name in variables_list:

    for cat in categories:

        use_logy = False

        # ----------------------------------
        # Determine global histogram range
        # ----------------------------------
        arrays_for_range = []

        for proc_name, proc in processes.items():

            events = proc["events"][cat]

            if obs_name not in events.fields:
                continue

            arrays_for_range.append(events[obs_name])

        xminG, xmaxG = get_global_range(arrays_for_range)



        obs_lower = obs_name.lower()

        signal_boost = 1

        if "mass" in obs_lower:
            xminG, xmaxG = 0, 70

        elif "pt" in obs_lower:
            xminG, xmaxG = 0, 200
            use_logy = True
        if ("BDT_score" in obs_name) & (BDT==False):
            signal_boost = 10

        print(f"{obs_name} ({cat}) : [{xminG:.2f}, {xmaxG:.2f}]")

        # Keep references → avoid segfault
        bkg_hists = []
        signal_hists = []

        stack = ROOT.THStack(
            f"stack_{obs_name}_{cat}",
            f";{obs_name};Events"
        )

        legend = ROOT.TLegend(0.64, 0.70, 0.86, 0.90)
        legend.SetFillStyle(0)
        legend.SetBorderSize(0)

        # ------------------------------
        # BACKGROUND LOOP
        # ------------------------------
        for proc_name, proc in processes.items():

            events = proc["events"][cat]

            values = np.asarray(getattr(events, obs_name))
            # obs_func = observables[obs_name]["func"]
            # values = np.asarray(obs_func(events))
            # weights = np.asarray(getattr(events, "weight"))

            # Original weights
            # ----------------------------------------

            weights = np.asarray(getattr(events, "weight"))

            mask = (values != -999.0)

            values = values[mask]
            weights = weights[mask]

            # ----------------------------------------
            # Replace negative weights by positive
            # and renormalize
            # ----------------------------------------

            sum_original = np.sum(weights)

            abs_weights = np.abs(weights)

            sum_abs = np.sum(abs_weights)

            if sum_abs > 0:
                renorm = abs(sum_original) / sum_abs
            else:
                renorm = 1.0

            weights = abs_weights * renorm

            scale = proc["xsec"] * lumi * 1000.0

            h = make_hist(
                f"{proc_name}_{obs_name}_{cat}_{id(events)}",
                xminG,
                xmaxG
            )

            fill_hist(h, values, weights, scale)

            h.SetFillColor(proc["color"])
            h.SetLineColor(ROOT.kBlack)

            stack.Add(h)
            legend.AddEntry(h, proc_name, "f")

            bkg_hists.append(h)

        # ==============================
        # DRAW
        # ==============================
        c = ROOT.TCanvas(f"c_{obs_name}_{cat}", "", 1000, 700)
        if use_logy:
            c.SetLogy()
        c.SetLeftMargin(0.12)
        c.SetRightMargin(0.15)
        # c.SetLogy()

        stack.Draw("hist")
        if use_logy:
            stack.SetMinimum(0.1)
        if (BDT) and ("mass" in obs_name):
            stack.SetMaximum(stack.GetMaximum() * 1.65)
        else:
            stack.SetMaximum(stack.GetMaximum() * 1.4)

        stat_boxes = []
        signal_hists = []

        # ------------------------------
        # DRAW SIGNALS
        # ------------------------------
        for i, m in enumerate(signal_masses):

            sig_events = signal_samples[m][cat_sig[0]]

            values = np.asarray(getattr(sig_events, obs_name))
            # obs_func = observables[obs_name]["func"]
            # values = np.asarray(obs_func(sig_events))
            weights = np.asarray(getattr(sig_events, "weight"))

            mask = (values != -999.0)

            values = values[mask]
            weights = weights[mask]

            scale = signal_xsec * lumi * 1000.0 * Br_frac * signal_boost

            h_sig = make_hist(
                f"sig_M{m}_{obs_name}_{cat}_{id(sig_events)}",
                xminG,
                xmaxG
            )

            fill_hist(h_sig, values, weights, scale)

            h_sig.SetLineColor(signal_colors[m])
            h_sig.SetLineWidth(3)
            h_sig.SetFillStyle(0)
            h_sig.SetLineStyle(2) 

            h_sig.Draw("hist same")

            legend.AddEntry(h_sig, f"WH M{m} (0.32 pb, Br = {Br_frac})", "l")

            signal_hists.append(h_sig)

        # ------------------------------
        # DRAW STAT BOXES (AFTER DRAWING ALL HISTS)
        # ------------------------------
        y_top = 0.90
        height = 0.1

        for i, h_sig in enumerate(signal_hists):

            box = draw_statbox_manual(
                h_sig,
                0.85,
                y_top - (i+1)*height,
                0.99,
                y_top - i*height,
                h_sig.GetLineColor()
            )

            stat_boxes.append(box)   # prevent garbage collection

        # ------------------------------
        # FINAL DRAWING
        # ------------------------------
        legend.Draw()

        latex = ROOT.TLatex()
        latex.SetNDC()
        latex.SetTextSize(0.035)
        latex.SetTextFont(42)
        latex.DrawLatex(0.15, 0.87, f"{obs_name}, {cat} GeV MHypothesis")

        CMS_label(c)

        c.Modified()
        c.Update()

        # ------------------------------
        # SAVE
        # ------------------------------
        print(f"Saving {obs_name}, {cat}")
        # if BDT:
        #     c.SaveAs(f"{Plot_dir_BDT}/stacked_{obs_name}_{cat}_linear.png")
        #     c.SaveAs(f"{Plot_dir_BDT}/stacked_{obs_name}_{cat}_linear.pdf")
        # else:
        #     c.SaveAs(f"{Plot_dir}/stacked_{obs_name}_{cat}_linear.png")
        #     c.SaveAs(f"{Plot_dir}/stacked_{obs_name}_{cat}_linear.pdf")


        if BDT:
            outdir = os.path.join(Plot_dir_BDT, cat)
            parent_dir = Plot_dir_BDT
        else:
            outdir = os.path.join(Plot_dir, cat)
            parent_dir = Plot_dir

        # Create category directory
        os.makedirs(outdir, exist_ok=True)

        # Copy index.php
        index_src = os.path.join(parent_dir, "index.php")
        index_dst = os.path.join(outdir, "index.php")

        if os.path.exists(index_src) and not os.path.exists(index_dst):
            shutil.copy2(index_src, index_dst)

        # Copy res directory
        res_src = os.path.join(parent_dir, "res")
        res_dst = os.path.join(outdir, "res")

        if os.path.exists(res_src) and not os.path.exists(res_dst):
            shutil.copytree(res_src, res_dst)

        # Save plots
        c.SaveAs(os.path.join(outdir, f"stacked_{obs_name}_linear.png"))
        c.SaveAs(os.path.join(outdir, f"stacked_{obs_name}_linear.pdf"))