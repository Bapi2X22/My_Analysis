import ROOT
import awkward as ak
import numpy as np
import glob
import os

# ROOT.gStyle.SetOptStat(0)

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

# -------------------------------
# CONFIG
# -------------------------------

base_dir = "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/NTuples_BKG_2024_HDNA_"

corrections = [
    "Pileup",
    "SF_photon_ID",
    "PreselSF",
    "ElectronVetoSF",
    "TriggerSF",
    "PartonShower"
]

sample_path = "TTG1Jets_24SummerRun3/nominal/diphoton/*.parquet"

output_dir = "/eos/user/b/bbapi/www/Systematics_study/Backgrounds/TTG1Jets/weight_corrections/HDNA_tightercut/"
os.makedirs(output_dir, exist_ok=True)

# -------------------------------
# Helper: Make histogram
# -------------------------------

def make_hist(name, title, bins, xmin, xmax):
    h = ROOT.TH1F(name, title, bins, xmin, xmax)
    h.Sumw2()
    return h


# -------------------------------
# Helper: Fill histogram
# -------------------------------

def fill_hist(hist, values, weights):
    for v, w in zip(values, weights):
        hist.Fill(v, w)


# -------------------------------
# Helper: Draw with ratio
# -------------------------------

def draw_ratio(h_nom, h_var, title, outname):

    c = ROOT.TCanvas("c", "", 800, 800)

    # Pads
    pad1 = ROOT.TPad("pad1", "", 0, 0.3, 1, 1)
    pad2 = ROOT.TPad("pad2", "", 0, 0, 1, 0.3)

    pad1.SetBottomMargin(0.02)
    pad1.SetRightMargin(0.2)
    pad2.SetTopMargin(0.05)
    pad2.SetBottomMargin(0.3)
    pad2.SetRightMargin(0.2)

    pad1.Draw()
    pad2.Draw()

    # -----------------------
    # Top pad
    # -----------------------
    pad1.cd()

    h_nom.SetLineColor(ROOT.kBlue)
    h_var.SetLineColor(ROOT.kRed)

    h_nom.SetLineWidth(2)
    h_var.SetLineWidth(2)

    h_nom.SetTitle(title)

    h_nom.GetXaxis().SetLabelSize(0)

    h_nom.Draw("HIST")
    h_var.Draw("HIST SAME")

    leg = ROOT.TLegend(0.6, 0.7, 0.8, 0.9)
    leg.AddEntry(h_nom, "Corrected", "l")
    leg.AddEntry(h_var, "Uncorrected", "l")
    leg.Draw()

    ROOT.gStyle.SetOptStat(111111) 

    draw_side_statboxes(c, [h_nom, h_var])

    latex = ROOT.TLatex()
    latex.SetNDC(True)          # normalized coordinates (0–1)
    latex.SetTextFont(42)
    latex.SetTextSize(0.035)

    latex.DrawLatex(0.52, 0.50, "TTG1Jets(2024)")

    # -----------------------
    # Ratio pad
    # -----------------------
    pad2.cd()

    ratio = h_nom.Clone("ratio")
    ratio.Divide(h_var)

    ratio.SetTitle("")
    ratio.GetYaxis().SetTitle("Ratio")
    ratio.GetXaxis().SetTitle(h_nom.GetXaxis().GetTitle())

    ratio.GetYaxis().SetNdivisions(505)
    ratio.GetYaxis().SetTitleSize(0.08)
    ratio.GetYaxis().SetTitleOffset(0.5)
    ratio.GetYaxis().SetLabelSize(0.07)

    ratio.GetXaxis().SetTitleSize(0.1)
    ratio.GetXaxis().SetLabelSize(0.08)

    ratio.SetMaximum(1.5)
    ratio.SetMinimum(0.5)

    ratio.Draw("P")

    # Line at 1
    line = ROOT.TLine(
        ratio.GetXaxis().GetXmin(), 1,
        ratio.GetXaxis().GetXmax(), 1
    )
    line.SetLineStyle(2)
    line.Draw()

    c.SaveAs(outname)


# -------------------------------
# MAIN LOOP
# -------------------------------

for corr in corrections:

    print(f"Processing: {corr}")

    directory = base_dir + corr + "/" + sample_path

    files = glob.glob(directory)

    if len(files) == 0:
        print(f"Skipping {corr}, no files found")
        continue

    a = ak.from_parquet(files)

    # -----------------------
    # Weights (normalized)
    # -----------------------
    w_corr = a.weight / ak.sum(a.weight)
    w_nom  = a.genWeight / ak.sum(a.genWeight)

    # Convert to numpy
    w_corr = ak.to_numpy(w_corr)
    w_nom  = ak.to_numpy(w_nom)

    # Variables
    lead_pt     = ak.to_numpy(a.pholead_pt)
    sublead_pt  = ak.to_numpy(a.phosublead_pt)
    mass        = ak.to_numpy(a.mass)

    # -----------------------
    # Histograms
    # -----------------------

    ROOT.gStyle.SetOptStat(111111)

    # Lead pT
    h_lead_corr = make_hist("h_lead_corr", ";Lead pT [GeV];Events", 100, 0, 200)
    ROOT.gStyle.SetOptStat(111111)
    h_lead_nom  = make_hist("h_lead_nom",  ";Lead pT [GeV];Events", 100, 0, 200)
    ROOT.gStyle.SetOptStat(111111)

    fill_hist(h_lead_corr, lead_pt, w_corr)
    ROOT.gStyle.SetOptStat(111111)
    fill_hist(h_lead_nom,  lead_pt, w_nom)

    draw_ratio(
        h_lead_corr, h_lead_nom,
        f"{corr} Lead pT",
        f"{output_dir}/{corr}_lead_pt.png"
    )

    # -----------------------

    ROOT.gStyle.SetOptStat(111111)

    # Sublead pT
    h_sub_corr = make_hist("h_sub_corr", ";Sublead pT [GeV];Events", 100, 0, 200)
    h_sub_nom  = make_hist("h_sub_nom",  ";Sublead pT [GeV];Events", 100, 0, 200)

    fill_hist(h_sub_corr, sublead_pt, w_corr)

    ROOT.gStyle.SetOptStat(111111)

    fill_hist(h_sub_nom,  sublead_pt, w_nom)

    ROOT.gStyle.SetOptStat(111111)

    draw_ratio(
        h_sub_corr, h_sub_nom,
        f"{corr} Sublead pT",
        f"{output_dir}/{corr}_sublead_pt.png"
    )

    # -----------------------

    ROOT.gStyle.SetOptStat(111111)

    # Mass
    h_mass_corr = make_hist("h_mass_corr", ";m_{#gamma#gamma} [GeV];Events", 100, 0, 200)
    ROOT.gStyle.SetOptStat(111111)
    h_mass_nom  = make_hist("h_mass_nom",  ";m_{#gamma#gamma} [GeV];Events", 100, 0, 200)

    fill_hist(h_mass_corr, mass, w_corr)
    ROOT.gStyle.SetOptStat(111111)
    fill_hist(h_mass_nom,  mass, w_nom)

    draw_ratio(
        h_mass_corr, h_mass_nom,
        f"{corr} Diphoton Mass",
        f"{output_dir}/{corr}_mass.png"
    )
