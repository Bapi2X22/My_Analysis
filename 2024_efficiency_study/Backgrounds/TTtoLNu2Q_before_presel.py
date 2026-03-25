import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import ROOT

Plot_dir = "/eos/user/b/bbapi/www/Background_study_2024/TTtoLNu2Q/"

def draw_side_statboxes(canvas, hists,
                        x1=0.80, x2=0.98,
                        y_top=0.92,
                        box_height=0.18,
                        gap=0.02):
    """
    Draw stacked statboxes on the right side of a canvas.

    Parameters
    ----------
    canvas : ROOT.TCanvas
    hists  : list of ROOT.TH1
    x1,x2  : horizontal NDC range
    y_top  : top starting NDC position
    box_height : height of each box
    gap    : vertical gap between boxes
    """

    # Temporary canvas to force stat creation
    tmp = ROOT.TCanvas("tmp_stats", "", 1, 1)

    orig_stats = {}

    for h in hists:
        h.Draw("hist")
        tmp.Update()

        st = h.GetListOfFunctions().FindObject("stats")
        if not st:
            raise RuntimeError(f"Stats not created for {h.GetName()}")

        orig_stats[h] = st.Clone(f"stats_clone_{h.GetName()}")

    tmp.Close()

    # Disable automatic stat boxes
    ROOT.gStyle.SetOptStat(0)

    canvas.cd()
    canvas.SetRightMargin(0.2)

    # Draw stacked statboxes
    for i, (hst, st) in enumerate(orig_stats.items()):

        st.SetParent(canvas)

        y2 = y_top - i * (box_height + gap)
        y1 = y2 - box_height

        st.SetX1NDC(x1)
        st.SetX2NDC(x2)
        st.SetY1NDC(y1)
        st.SetY2NDC(y2)

        st.SetTextColor(hst.GetLineColor())
        st.SetTextFont(62)   # avoid TTF bug
        st.SetFillStyle(0)
        st.SetBorderSize(1)

        st.Draw()

    canvas.Modified()
    canvas.Update()


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
    lumi_text = f"({year}_{lumi})"
    latex.DrawLatex(0.88, y, lumi_text)


import math

def merge_overflow(
    hist,
    add_underflow=False,
    add_overflow=True,
    label_overflow=True,
    float_format=".0f"
):
    """
    Merge overflow/underflow into visible bins.
    Optionally relabel last bin as ≥ upper_edge.

    Parameters
    ----------
    hist : ROOT.TH1
    add_underflow : bool
    add_overflow : bool
    label_overflow : bool
        If True, rename last bin to ≥ upper_edge
    float_format : str
        Formatting for upper edge (e.g. ".0f", ".1f")
    """

    n_bins = hist.GetNbinsX()

    # --- Underflow → first bin
    if add_underflow:
        underflow = hist.GetBinContent(0)
        first = hist.GetBinContent(1)

        hist.SetBinContent(1, first + underflow)

        err0 = hist.GetBinError(0)
        err1 = hist.GetBinError(1)
        hist.SetBinError(1, math.sqrt(err0**2 + err1**2))

        hist.SetBinContent(0, 0)
        hist.SetBinError(0, 0)

    # --- Overflow → last bin
    if add_overflow:
        overflow = hist.GetBinContent(n_bins + 1)
        last = hist.GetBinContent(n_bins)

        hist.SetBinContent(n_bins, last + overflow)

        errN = hist.GetBinError(n_bins)
        errOv = hist.GetBinError(n_bins + 1)
        hist.SetBinError(n_bins, math.sqrt(errN**2 + errOv**2))

        hist.SetBinContent(n_bins + 1, 0)
        hist.SetBinError(n_bins + 1, 0)

        # # --- Relabel last bin
        # if label_overflow:
        #     upper_edge = hist.GetXaxis().GetBinUpEdge(n_bins)
        #     label = f"#geq {format(upper_edge, float_format)}"
        #     hist.GetXaxis().SetBinLabel(n_bins, label)

import glob
import os

# -----------------------------
# Directory containing parquet files
# -----------------------------
parquet_dir = "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/NTuples_bbgg_LNu2Q_wpresel/TTtoLNu2Q_24SummerRun3/nominal/diphoton/"
files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))

print(f"Found {len(files)} parquet files")

# -----------------------------
# Create histogram ONCE
# -----------------------------
ROOT.gStyle.SetOptStat(1111111)

h = ROOT.TH1F(
    "h_n_ph_raw",
    ";Number of photons per event (before preselection);Events",
    8, 0, 8
)

h_lead = ROOT.TH1F("h_lead", ";Photon p_{T} [GeV];Events", 50, 0, 200)
h_sub  = ROOT.TH1F("h_sub",  ";Photon p_{T} [GeV];Events", 50, 0, 200)

h_nbjet = ROOT.TH1F(
    "h_n_j_raw",
    ";Number of b-Jets per event (before preselection);Events",
    10, 0, 10
)

h_lead_bjet = ROOT.TH1F("h_lead_bjet", ";b-Jets p_{T} [GeV];Events", 100, 0, 500)
h_sub_bjet  = ROOT.TH1F("h_sub_bjet",  ";b-Jets p_{T} [GeV];Events", 100, 0, 500)

# -----------------------------
# Loop over parquet files
# -----------------------------
for f in files:
    print(f"Processing {f}")

    a = ak.from_parquet(f)

    # select non-zero photons
    n_ph_raw = ak.to_numpy(a.n_photons[a.n_photons != 0])
    n_bjets_raw = ak.to_numpy(a.n_jets[a.n_jets != 0])
    lead_pho_pt = ak.to_numpy(a.pholead_pt[~ak.is_none(a.pholead_pt)])
    sublead_pho_pt = ak.to_numpy(a.phosublead_pt[~ak.is_none(a.phosublead_pt)])
    lead_bjet_pt = ak.to_numpy(a.first_jet_pt[a.first_jet_pt!=-999])
    sublead_bjet_pt = ak.to_numpy(a.second_jet_pt[a.second_jet_pt!=-999])

    # fill histogram
    for v in n_ph_raw:
        h.Fill(v)
    
    for v in n_bjets_raw:
        h_nbjet.Fill(v)

    for pt in lead_pho_pt:
        h_lead.Fill(pt)

    for pt in sublead_pho_pt:
        h_sub.Fill(pt)

    for pt in lead_bjet_pt:
        h_lead_bjet.Fill(pt)

    for pt in sublead_bjet_pt:
        h_sub_bjet.Fill(pt)

print("Done filling histogram.")

c = ROOT.TCanvas("c_n_ph_raw", "c_n_ph_raw", 800, 600)
c.SetLogy()

ROOT.gStyle.SetOptStat(1111111)

h.Draw("HIST")
h.SetMinimum(0.1)

c.Update()

stats = h.FindObject("stats")
if stats:
    stats.SetTextSize(0.035)     # enlarge text
    stats.SetX1NDC(0.60)         # left
    stats.SetX2NDC(0.88)         # right
    stats.SetY1NDC(0.60)         # bottom
    stats.SetY2NDC(0.88)         # top
    stats.SetBorderSize(1)

latex = ROOT.TLatex()
latex.SetNDC()              # normalized (0–1) coords
latex.SetTextSize(0.05)
latex.SetTextFont(42)
latex.DrawLatex(0.68, 0.55, f"TTtoLNu2Q")

CMS_label(c)

# -----------------------------
# Save
# -----------------------------
c.Update()
c.SaveAs(Plot_dir+f"n_photons_per_event_raw_TTtoLNu2Q.png")



c = ROOT.TCanvas("c", "c", 700, 600)
c.SetLeftMargin(0.12)

max_val = max(h_lead.GetMaximum(), h_sub.GetMaximum())
h_lead.SetMaximum(1.1 * max_val)   # 20% headroom
h_lead.SetMinimum(0)      

h_lead.SetLineColor(ROOT.kRed)
h_lead.SetLineWidth(2)

h_sub.SetLineColor(ROOT.kBlue)
h_sub.SetLineWidth(2)

ROOT.gStyle.SetOptStat(1111111)

h_lead.Draw("HIST")
h_sub.Draw("HIST SAME")

c.Update()

draw_side_statboxes(c, [h_lead, h_sub])

leg = ROOT.TLegend(0.50, 0.75, 0.78, 0.88)
leg.SetBorderSize(0)
leg.AddEntry(h_lead, f"Leading #gamma", "l")
leg.AddEntry(h_sub, f"Subleading #gamma", "l")
leg.Draw()

latex = ROOT.TLatex()
latex.SetNDC()              # normalized (0–1) coords
latex.SetTextSize(0.04)
latex.SetTextFont(42)
latex.DrawLatex(0.60, 0.55, f"TTtoLNu2Q")

latex = ROOT.TLatex()
latex.SetNDC()              # normalized (0–1) coords
latex.SetTextSize(0.04)
latex.SetTextFont(42)
latex.DrawLatex(0.50, 0.45, "Before preselection")

CMS_label(c)

c.SaveAs(Plot_dir+f"photons_pt_lead_sublead_raw_TTtoLNu2Q.png")



# -----------------------------
# Draw
# -----------------------------
c = ROOT.TCanvas("c_n_j_raw", "c_n_j_raw", 800, 600)
c.SetLogy()

ROOT.gStyle.SetOptStat(1111111)

h_nbjet.Draw("HIST")
h_nbjet.SetMinimum(0.1)

c.Update()

stats = h_nbjet.FindObject("stats")
if stats:
    stats.SetTextSize(0.035)     # enlarge text
    stats.SetX1NDC(0.60)         # left
    stats.SetX2NDC(0.88)         # right
    stats.SetY1NDC(0.60)         # bottom
    stats.SetY2NDC(0.88)         # top
    stats.SetBorderSize(1)

latex = ROOT.TLatex()
latex.SetNDC()              # normalized (0–1) coords
latex.SetTextSize(0.05)
latex.SetTextFont(42)
latex.DrawLatex(0.68, 0.55, f"TTtoLNu2Q")

CMS_label(c)

# -----------------------------
# Save
# -----------------------------
c.Update()
c.SaveAs(Plot_dir+f"n_jets_per_event_raw_TTtoLNu2Q.png")

c = ROOT.TCanvas("c", "c", 700, 600)
# c.SetLogy()
c.SetLeftMargin(0.12)

max_val = max(h_lead_bjet.GetMaximum(), h_sub_bjet.GetMaximum())
h_lead_bjet.SetMaximum(1.1 * max_val)   # 20% headroom
h_lead_bjet.SetMinimum(0)      

h_lead_bjet.SetLineColor(ROOT.kRed)
h_lead_bjet.SetLineWidth(2)

h_sub_bjet.SetLineColor(ROOT.kBlue)
h_sub_bjet.SetLineWidth(2)

# merge_overflow(h_lead_bjet)
# merge_overflow(h_sub_bjet)

ROOT.gStyle.SetOptStat(1111111)

h_lead_bjet.Draw("HIST")
h_sub_bjet.Draw("HIST SAME")

c.Update()

draw_side_statboxes(c, [h_lead_bjet, h_sub_bjet])

leg = ROOT.TLegend(0.50, 0.75, 0.78, 0.88)
leg.SetBorderSize(0)
leg.AddEntry(h_lead_bjet, "Leading b-Jet", "l")
leg.AddEntry(h_sub_bjet, "Subleading b-Jet", "l")
leg.Draw()

latex = ROOT.TLatex()
latex.SetNDC()              # normalized (0–1) coords
latex.SetTextSize(0.04)
latex.SetTextFont(42)
latex.DrawLatex(0.60, 0.55, f"TTtoLNu2Q")

latex = ROOT.TLatex()
latex.SetNDC()              # normalized (0–1) coords
latex.SetTextSize(0.04)
latex.SetTextFont(42)
latex.DrawLatex(0.50, 0.45, "Before preselection")

CMS_label(c)

c.SaveAs(Plot_dir+f"jets_pt_lead_sublead_raw_TTtoLNu2Q.png")
