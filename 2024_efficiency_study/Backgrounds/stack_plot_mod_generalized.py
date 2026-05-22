import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import ROOT

ROOT.gROOT.SetBatch(True)   # (you are saving files → good practice)
ROOT.gStyle.SetOptStat(1110)
ROOT.gROOT.ForceStyle()


Plot_dir = "/eos/user/b/bbapi/www/Background_study_2024/stack_plots/updated_presel/"

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

dir_2L2Nu_cat1 = "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/NTuples_BKG_2024_HDNA_presel_final_good_selections/merged/TTto2L2Nu-24SummerRun3/nominal/diphoton/CAT1_merged.parquet"
dir_2L2Nu_cat2 = "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/NTuples_BKG_2024_HDNA_presel_final_good_selections/merged/TTto2L2Nu-24SummerRun3/nominal/diphoton/CAT2_merged.parquet"
dir_2L2Nu_cat3 = "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/NTuples_BKG_2024_HDNA_presel_final_good_selections/merged/TTto2L2Nu-24SummerRun3/nominal/diphoton/CAT3_merged.parquet"
events_2L2Nu_cat1 = ak.from_parquet(dir_2L2Nu_cat1)
events_2L2Nu_cat2 = ak.from_parquet(dir_2L2Nu_cat2)
events_2L2Nu_cat3 = ak.from_parquet(dir_2L2Nu_cat3)

dir_LNu2Q_cat1 = "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/NTuples_BKG_2024_HDNA_presel_final_good_selections/merged/TTtoLNu2Q-24SummerRun3/nominal/diphoton/CAT1_merged.parquet"
dir_LNu2Q_cat2 = "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/NTuples_BKG_2024_HDNA_presel_final_good_selections/merged/TTtoLNu2Q-24SummerRun3/nominal/diphoton/CAT2_merged.parquet"
dir_LNu2Q_cat3 = "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/NTuples_BKG_2024_HDNA_presel_final_good_selections/merged/TTtoLNu2Q-24SummerRun3/nominal/diphoton/CAT3_merged.parquet"
events_LNu2Q_cat1 = ak.from_parquet(dir_LNu2Q_cat1)
events_LNu2Q_cat2 = ak.from_parquet(dir_LNu2Q_cat2)
events_LNu2Q_cat3 = ak.from_parquet(dir_LNu2Q_cat3)

dir_G1Jets_cat1 = "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/NTuples_BKG_2024_HDNA_presel_final_good_selections/merged/TTG1Jets-24SummerRun3/nominal/diphoton/CAT1_merged.parquet"
dir_G1Jets_cat2 = "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/NTuples_BKG_2024_HDNA_presel_final_good_selections/merged/TTG1Jets-24SummerRun3/nominal/diphoton/CAT2_merged.parquet"
dir_G1Jets_cat3 = "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/NTuples_BKG_2024_HDNA_presel_final_good_selections/merged/TTG1Jets-24SummerRun3/nominal/diphoton/CAT3_merged.parquet"
events_G1Jets_cat1 = ak.from_parquet(dir_G1Jets_cat1)
events_G1Jets_cat2 = ak.from_parquet(dir_G1Jets_cat2)
events_G1Jets_cat3 = ak.from_parquet(dir_G1Jets_cat3)

dir_WGtoLNuG_cat1 = "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/NTuples_BKG_2024_HDNA_presel_final_good_selections/merged/WGtoLNuG-24SummerRun3/nominal/diphoton/CAT1_merged.parquet"
dir_WGtoLNuG_cat2 = "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/NTuples_BKG_2024_HDNA_presel_final_good_selections/merged/WGtoLNuG-24SummerRun3/nominal/diphoton/CAT2_merged.parquet"
dir_WGtoLNuG_cat3 = "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/NTuples_BKG_2024_HDNA_presel_final_good_selections/merged/WGtoLNuG-24SummerRun3/nominal/diphoton/CAT3_merged.parquet"
events_WGtoLNuG_cat1 = ak.from_parquet(dir_WGtoLNuG_cat1)
events_WGtoLNuG_cat2 = ak.from_parquet(dir_WGtoLNuG_cat2)
events_WGtoLNuG_cat3 = ak.from_parquet(dir_WGtoLNuG_cat3)


# mgg_bkg1_cat1 = events_G1Jets_cat1.mass
# mgg_bkg1_cat2 = events_G1Jets_cat2.mass
# mgg_bkg1_cat3 = events_G1Jets_cat3.mass
# mgg_bkg2_cat1 = events_2L2Nu_cat1.mass
# mgg_bkg2_cat2 = events_2L2Nu_cat2.mass
# mgg_bkg2_cat3 = events_2L2Nu_cat3.mass
# mgg_bkg3_cat1 = events_LNu2Q_cat1.mass
# mgg_bkg3_cat2 = events_LNu2Q_cat2.mass
# mgg_bkg3_cat3 = events_LNu2Q_cat3.mass

# w_bkg1_cat1 = events_G1Jets_cat1.weight
# w_bkg1_cat2 = events_G1Jets_cat2.weight
# w_bkg1_cat3 = events_G1Jets_cat3.weight
# w_bkg2_cat1 = events_2L2Nu_cat1.weight
# w_bkg2_cat2 = events_2L2Nu_cat2.weight
# w_bkg2_cat3 = events_2L2Nu_cat3.weight
# w_bkg3_cat1 = events_LNu2Q_cat1.weight
# w_bkg3_cat2 = events_LNu2Q_cat2.weight
# w_bkg3_cat3 = events_LNu2Q_cat3.weight

# mgg_bkg1_cat1 = events_G1Jets_cat1.mass
# mgg_bkg1_cat2 = events_G1Jets_cat2.mass
# mgg_bkg1_cat3 = events_G1Jets_cat3.mass
# mgg_bkg2_cat1 = events_2L2Nu_cat1.mass
# mgg_bkg2_cat2 = events_2L2Nu_cat2.mass
# mgg_bkg2_cat3 = events_2L2Nu_cat3.mass
# mgg_bkg3_cat1 = events_LNu2Q_cat1.mass
# mgg_bkg3_cat2 = events_LNu2Q_cat2.mass    
# mgg_bkg3_cat3 = events_LNu2Q_cat3.mass

# ==============================
# HISTOGRAM SETTINGS
# ==============================
nbins = 60
xmin = 10
xmax = 70


import ROOT
import awkward as ak
import numpy as np

# ==============================
# Observables

observables = {
    "mass": {
        "func": lambda ev: ev.mass,
        "x_title": "m_{#gamma#gamma} [GeV]",
        "nbins": 70,
        "xmin": 10,
        "xmax": 70
    },
    "pholead_pt": {
        "func": lambda ev: ev.pholead_pt,
        "x_title": "Lead p_{T} [GeV]",
        "nbins": 50,
        "xmin": 0,
        "xmax": 200
    },
    "phosublead_pt": {
        "func": lambda ev: ev.phosublead_pt,
        "x_title": "Sublead p_{T} [GeV]",
        "nbins": 50,
        "xmin": 0,
        "xmax": 200
    },
    "pholead_eta": {
        "func": lambda ev: ev.pholead_eta,
        "x_title": "Lead #eta",
        "nbins": 50,
        "xmin": -6,
        "xmax": 6
    },
    "phosublead_eta": {
        "func": lambda ev: ev.phosublead_eta,
        "x_title": "Sublead #eta",
        "nbins": 50,
        "xmin": -6,
        "xmax": 6
    },
    "pholead_mvaID": {
        "func": lambda ev: ev.pholead_mvaID,
        "x_title": "Lead photon mvaID",
        "nbins": 50,
        "xmin": -1,
        "xmax": 1
    },
    "phosublead_mvaID": {
        "func": lambda ev: ev.phosublead_mvaID,
        "x_title": "Sublead photon mvaID",
        "nbins": 50,
        "xmin": -1,
        "xmax": 1
    },

    # --- NEW DERIVED VARIABLES ---

    "pt_ratio": {
        "func": lambda ev: ak.where(
            ev.pholead_pt != 0,
            ev.phosublead_pt / ev.pholead_pt,
            0
        ),
        "x_title": "p_{T}^{sublead} / p_{T}^{lead}",
        "nbins": 50,
        "xmin": 0,
        "xmax": 1
    },

    "deltaR_pho": {
        "func": lambda ev: np.sqrt(
            (ev.pholead_eta - ev.phosublead_eta)**2 +
            np.arctan2(
                np.sin(ev.pholead_phi - ev.phosublead_phi),
                np.cos(ev.pholead_phi - ev.phosublead_phi)
            )**2
        ),
        "x_title": "#DeltaR(lead, sublead)",
        "nbins": 50,
        "xmin": 0,
        "xmax": 6
    }
}

# ==============================
# Categories
# ==============================
categories = ["cat1", "cat2", "cat3"]

# ==============================
# Processes (BACKGROUND)
# ==============================
processes = {
    "WGtoLNuG": {
        "events": {
            "cat1": events_WGtoLNuG_cat1,
            "cat2": events_WGtoLNuG_cat2,
            "cat3": events_WGtoLNuG_cat3,
        },
        "xsec": xsec0,
        "color": ROOT.kMagenta+1
    },
    "TTG1Jets": {
        "events": {
            "cat1": events_G1Jets_cat1,
            "cat2": events_G1Jets_cat2,
            "cat3": events_G1Jets_cat3,
        },
        "xsec": xsec1,
        "color": ROOT.kRed+1
    },
    "TTto2L2Nu": {
        "events": {
            "cat1": events_2L2Nu_cat1,
            "cat2": events_2L2Nu_cat2,
            "cat3": events_2L2Nu_cat3,
        },
        "xsec": xsec2,
        "color": ROOT.kBlue+1
    },
    "TTtoLNu2Q": {
        "events": {
            "cat1": events_LNu2Q_cat1,
            "cat2": events_LNu2Q_cat2,
            "cat3": events_LNu2Q_cat3,
        },
        "xsec": xsec3,
        "color": ROOT.kGreen+2
    }
}

# ==============================
# SIGNAL
# ==============================
signal_masses = [15, 35, 55]
signal_xsec = 1.53  # pb
Br_frac = 0.01

signal_colors = {
    15: ROOT.kYellow+1,
    35: ROOT.kOrange+7,
    55: ROOT.kCyan+1
}

signal_samples = {}

for m in signal_masses:
    signal_samples[m] = {}
    for i, cat in enumerate(categories, start=1):
        path = f"/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/NTuples_WH_2024_HDNA_presel_final_good_selections/merged/WH-2024M{m}/nominal/diphoton/CAT{i}_merged.parquet"
        signal_samples[m][cat] = ak.from_parquet(path)

# ==============================
# GLOBALS
# ==============================
lumi = lumi
Plot_dir = Plot_dir


# ==============================
# HELPERS
# ==============================
def make_hist(name, cfg):
    h = ROOT.TH1F(
        name,
        f";{cfg['x_title']};Events",
        cfg["nbins"],
        cfg["xmin"],
        cfg["xmax"]
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
for obs_name, obs_cfg in observables.items():

    for cat in categories:

        # Keep references → avoid segfault
        bkg_hists = []
        signal_hists = []

        stack = ROOT.THStack(
            f"stack_{obs_name}_{cat}",
            f";{obs_cfg['x_title']};Events"
        )

        legend = ROOT.TLegend(0.64, 0.70, 0.86, 0.90)
        legend.SetFillStyle(0)
        legend.SetBorderSize(0)

        # ------------------------------
        # BACKGROUND LOOP
        # ------------------------------
        for proc_name, proc in processes.items():

            events = proc["events"][cat]

            # values = np.asarray(getattr(events, obs_name))
            obs_func = observables[obs_name]["func"]
            values = np.asarray(obs_func(events))
            weights = np.asarray(getattr(events, "weight"))

            scale = proc["xsec"] * lumi * 1000.0

            h = make_hist(
                f"{proc_name}_{obs_name}_{cat}_{id(events)}",
                obs_cfg
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
        c.SetLeftMargin(0.12)
        c.SetRightMargin(0.15)
        # c.SetLogy()

        stack.Draw("hist")
        # stack.SetMinimum(0.1)
        # stack.SetMaximum(10000)

        # ------------------------------
        # SIGNAL OVERLAY
        # ------------------------------

        # ROOT.gStyle.SetOptStat(1111111)
        # stat_boxes = []
        # for i, m in enumerate(signal_masses):

        #     sig_events = signal_samples[m][cat]

        #     values = np.asarray(getattr(sig_events, obs_name))
        #     weights = np.asarray(getattr(sig_events, "weight"))

        #     scale = signal_xsec * lumi * 1000.0

        #     h_sig = make_hist(
        #         f"sig_M{m}_{obs_name}_{cat}_{id(sig_events)}",
        #         obs_cfg
        #     )

        #     fill_hist(h_sig, values, weights, scale)

        #     h_sig.SetLineColor(signal_colors[m])
        #     h_sig.SetLineWidth(3)
        #     h_sig.SetFillStyle(0)

        #     h_sig.SetStats(1)

        #     print(ROOT.gPad.GetListOfPrimitives())
        #     print(h_sig.GetListOfFunctions())

        #     h_sig.Draw("hist same")
        #     # h_sig.SetStats(1)

        #     y_top = 0.90
        #     height = 0.1

        #     # draw_statbox_manual(
        #     #     h_sig,
        #     #     0.78,
        #     #     y_top - (i+1)*height,
        #     #     0.98,
        #     #     y_top - i*height,
        #     #     signal_colors[m]
        #     # )
        # box = draw_statbox_manual(
        #     h_sig,
        #     0.78,
        #     y_top - (i+1)*height,
        #     0.98,
        #     y_top - i*height,
        #     signal_colors[m]
        # )

        # stat_boxes.append(box)   

        # c.Update()

        # # move_statbox_incremental(h_sig, c, i)

        # # draw_statbox_manual(h_sig, x1, y1, x2, y2, color)

        # # c.Update()

        # # move_statbox_incremental(h_sig, c, i)

        # legend.AddEntry(h_sig, f"WH M{m} (1.53 pb)", "l")

        # signal_hists.append(h_sig)

        # legend.Draw()

        # # ------------------------------
        # # TEXT
        # # ------------------------------
        # latex = ROOT.TLatex()
        # latex.SetNDC()
        # latex.SetTextSize(0.035)
        # latex.SetTextFont(42)
        # latex.DrawLatex(0.15, 0.87, f"{obs_name}, Category {cat[-1]}")

        # CMS_label(c)

        # c.Modified()
        # c.Update()

        stat_boxes = []
        signal_hists = []

        # ------------------------------
        # DRAW SIGNALS
        # ------------------------------
        for i, m in enumerate(signal_masses):

            sig_events = signal_samples[m][cat]

            # values = np.asarray(getattr(sig_events, obs_name))
            obs_func = observables[obs_name]["func"]
            values = np.asarray(obs_func(sig_events))
            weights = np.asarray(getattr(sig_events, "weight"))

            scale = signal_xsec * lumi * 1000.0 * Br_frac

            h_sig = make_hist(
                f"sig_M{m}_{obs_name}_{cat}_{id(sig_events)}",
                obs_cfg
            )

            fill_hist(h_sig, values, weights, scale)

            h_sig.SetLineColor(signal_colors[m])
            h_sig.SetLineWidth(3)
            h_sig.SetFillStyle(0)

            h_sig.Draw("hist same")

            legend.AddEntry(h_sig, f"WH M{m} (1.53 pb)", "l")

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
        latex.DrawLatex(0.15, 0.87, f"{obs_name}, Category {cat[-1]}")

        CMS_label(c)

        c.Modified()
        c.Update()

        # ------------------------------
        # SAVE
        # ------------------------------
        print(f"Saving {obs_name}, {cat}")
        c.SaveAs(f"{Plot_dir}/stacked_{obs_name}_{cat}_linear.png")
        c.SaveAs(f"{Plot_dir}/stacked_{obs_name}_{cat}_linear.pdf")