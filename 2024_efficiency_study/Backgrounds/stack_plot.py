import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import ROOT


Plot_dir = "/eos/user/b/bbapi/www/Background_study_2024/stack_plots/same_range/"

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



# ==============================
# INPUTS
# ==============================
lumi = 109  # fb^-1

# cross sections in pb
xsec1 = 4.1
xsec2 = 98.04
xsec3 = 405.87

dir_2L2Nu = '/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/NTuples_bbgg_2L2Nu/TTto2L2Nu_24SummerRun3/nominal/diphoton/'
events_2L2Nu = ak.from_parquet(dir_2L2Nu + '*.parquet')
dir_LNu2Q = '/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/NTuples_bbgg_LNu2Q/TTtoLNu2Q_24SummerRun3/nominal/diphoton/'
events_LNu2Q = ak.from_parquet(dir_LNu2Q + '*.parquet')
dir_G1Jets = '/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/NTuples_bbgg_TTG1Jets/TTG1Jets_24SummerRun3/nominal/diphoton/'
events_G1Jets = ak.from_parquet(dir_G1Jets + '*.parquet')

cat1_2L2Nu = events_2L2Nu.n_jets >= 2
cat2_2L2Nu = (events_2L2Nu.n_jets == 1) & (events_2L2Nu.first_jet_probbb > events_2L2Nu.first_jet_probb)
cat3_2L2Nu = (events_2L2Nu.n_jets == 1) & (events_2L2Nu.first_jet_probb > events_2L2Nu.first_jet_probbb)

cat1_LNu2Q = events_LNu2Q.n_jets >= 2
cat2_LNu2Q = (events_LNu2Q.n_jets == 1) & (events_LNu2Q.first_jet_probbb > events_LNu2Q.first_jet_probb)
cat3_LNu2Q = (events_LNu2Q.n_jets == 1) & (events_LNu2Q.first_jet_probb > events_LNu2Q.first_jet_probbb)

cat1_G1Jets = events_G1Jets.n_jets >= 2
cat2_G1Jets = (events_G1Jets.n_jets == 1) & (events_G1Jets.first_jet_probbb > events_G1Jets.first_jet_probb)
cat3_G1Jets = (events_G1Jets.n_jets == 1) & (events_G1Jets.first_jet_probb > events_G1Jets.first_jet_probbb)

mgg_bkg1 = events_G1Jets.mass
mgg_bkg2 = events_2L2Nu.mass
mgg_bkg3 = events_LNu2Q.mass

mgg_bkg1_cat1 = events_G1Jets.mass[cat1_G1Jets]
mgg_bkg1_cat2 = events_G1Jets.mass[cat2_G1Jets]
mgg_bkg1_cat3 = events_G1Jets.mass[cat3_G1Jets]
mgg_bkg2_cat1 = events_2L2Nu.mass[cat1_2L2Nu]
mgg_bkg2_cat2 = events_2L2Nu.mass[cat2_2L2Nu]
mgg_bkg2_cat3 = events_2L2Nu.mass[cat3_2L2Nu]
mgg_bkg3_cat1 = events_LNu2Q.mass[cat1_LNu2Q]
mgg_bkg3_cat2 = events_LNu2Q.mass[cat2_LNu2Q]
mgg_bkg3_cat3 = events_LNu2Q.mass[cat3_LNu2Q]

# number of generated events
Ngen1 = 571416
Ngen2 = 470123263
Ngen3 = 484475057
Ngen1_cat1 = 571416
Ngen1_cat2 = 571416
Ngen1_cat3 = 571416
Ngen2_cat1 = 470123263
Ngen2_cat2 = 470123263
Ngen2_cat3 = 470123263
Ngen3_cat1 = 484475057
Ngen3_cat2 = 484475057
Ngen3_cat3 = 484475057

# ==============================
# HISTOGRAM SETTINGS
# ==============================
nbins = 70
xmin = 10
xmax = 70

h1 = ROOT.TH1F("h1", ";m_{#gamma#gamma} [GeV];Events", nbins, xmin, xmax)
h2 = ROOT.TH1F("h2", "", nbins, xmin, xmax)
h3 = ROOT.TH1F("h3", "", nbins, xmin, xmax)
h1_cat1 = ROOT.TH1F("h1_cat1", ";m_{#gamma#gamma} [GeV];Events", nbins, xmin, xmax)
h1_cat2 = ROOT.TH1F("h1_cat2", ";m_{#gamma#gamma} [GeV];Events", nbins, xmin, xmax)
h1_cat3 = ROOT.TH1F("h1_cat3", ";m_{#gamma#gamma} [GeV];Events", nbins, xmin, xmax)
h2_cat1 = ROOT.TH1F("h2_cat1", ";m_{#gamma#gamma} [GeV];Events", nbins, xmin, xmax)
h2_cat2 = ROOT.TH1F("h2_cat2", ";m_{#gamma#gamma} [GeV];Events", nbins, xmin, xmax)
h2_cat3 = ROOT.TH1F("h2_cat3", ";m_{#gamma#gamma} [GeV];Events", nbins, xmin, xmax)
h3_cat1 = ROOT.TH1F("h3_cat1", ";m_{#gamma#gamma} [GeV];Events", nbins, xmin, xmax)
h3_cat2 = ROOT.TH1F("h3_cat2", ";m_{#gamma#gamma} [GeV];Events", nbins, xmin, xmax)
h3_cat3 = ROOT.TH1F("h3_cat3", ";m_{#gamma#gamma} [GeV];Events", nbins, xmin, xmax)

# h1.SetMinimum(0.01)
# h2.SetMinimum(0.01)
# h3.SetMinimum(0.01)
# h1_cat1.SetMinimum(0.01)
# h1_cat2.SetMinimum(0.01)
# h1_cat3.SetMinimum(0.01)
# h2_cat1.SetMinimum(0.01)
# h2_cat2.SetMinimum(0.01)
# h2_cat3.SetMinimum(0.01)
# h3_cat1.SetMinimum(0.01)
# h3_cat2.SetMinimum(0.01)
# h3_cat3.SetMinimum(0.01)    

# ==============================
# SCALE TO LUMI
# ==============================
w1 = xsec1 * lumi * 1000.0 / Ngen1
w2 = xsec2 * lumi * 1000.0 / Ngen2
w3 = xsec3 * lumi * 1000.0 / Ngen3

w1_cat1 = xsec1 * lumi * 1000.0 / Ngen1_cat1
w1_cat2 = xsec1 * lumi * 1000.0 / Ngen1_cat2
w1_cat3 = xsec1 * lumi * 1000.0 / Ngen1_cat3
w2_cat1 = xsec2 * lumi * 1000.0 / Ngen2_cat1
w2_cat2 = xsec2 * lumi * 1000.0 / Ngen2_cat2
w2_cat3 = xsec2 * lumi * 1000.0 / Ngen2_cat3
w3_cat1 = xsec3 * lumi * 1000.0 / Ngen3_cat1
w3_cat2 = xsec3 * lumi * 1000.0 / Ngen3_cat2
w3_cat3 = xsec3 * lumi * 1000.0 / Ngen3_cat3

# h1.Scale(w1)
# h2.Scale(w2)
# h3.Scale(w3)
# h1_cat1.Scale(w1_cat1)
# h1_cat2.Scale(w1_cat2)
# h1_cat3.Scale(w1_cat3)
# h2_cat1.Scale(w2_cat1)
# h2_cat2.Scale(w2_cat2)
# h2_cat3.Scale(w2_cat3)
# h3_cat1.Scale(w3_cat1)
# h3_cat2.Scale(w3_cat2)
# h3_cat3.Scale(w3_cat3)

# ==============================
# FILL HISTOGRAMS
# ==============================
for x in mgg_bkg1:
    h1.Fill(x, w1)

for x in mgg_bkg2:
    h2.Fill(x, w2)

for x in mgg_bkg3:
    h3.Fill(x, w3)

for x in mgg_bkg1_cat1:
    h1_cat1.Fill(x, w1_cat1)
for x in mgg_bkg1_cat2:
    h1_cat2.Fill(x, w1_cat2)
for x in mgg_bkg1_cat3:
    h1_cat3.Fill(x, w1_cat3)
for x in mgg_bkg2_cat1:
    h2_cat1.Fill(x, w2_cat1)
for x in mgg_bkg2_cat2:
    h2_cat2.Fill(x, w2_cat2)
for x in mgg_bkg2_cat3:
    h2_cat3.Fill(x, w2_cat3)
for x in mgg_bkg3_cat1:
    h3_cat1.Fill(x, w3_cat1)
for x in mgg_bkg3_cat2:
    h3_cat2.Fill(x, w3_cat2)
for x in mgg_bkg3_cat3:
    h3_cat3.Fill(x, w3_cat3)

# ==============================
# STYLE
# ==============================
h1.SetFillColor(ROOT.kRed+1)
h2.SetFillColor(ROOT.kBlue+1)
h3.SetFillColor(ROOT.kGreen+2)

h1.SetLineColor(ROOT.kBlack)
h2.SetLineColor(ROOT.kBlack)
h3.SetLineColor(ROOT.kBlack)

h1_cat1.SetFillColor(ROOT.kRed+1)
h1_cat2.SetFillColor(ROOT.kRed+1)
h1_cat3.SetFillColor(ROOT.kRed+1)
h2_cat1.SetFillColor(ROOT.kBlue+1)
h2_cat2.SetFillColor(ROOT.kBlue+1)
h2_cat3.SetFillColor(ROOT.kBlue+1)
h3_cat1.SetFillColor(ROOT.kGreen+2)
h3_cat2.SetFillColor(ROOT.kGreen+2)
h3_cat3.SetFillColor(ROOT.kGreen+2)
h1_cat1.SetLineColor(ROOT.kBlack)
h1_cat2.SetLineColor(ROOT.kBlack)
h1_cat3.SetLineColor(ROOT.kBlack)
h2_cat1.SetLineColor(ROOT.kBlack)
h2_cat2.SetLineColor(ROOT.kBlack)
h2_cat3.SetLineColor(ROOT.kBlack)
h3_cat1.SetLineColor(ROOT.kBlack)
h3_cat2.SetLineColor(ROOT.kBlack)
h3_cat3.SetLineColor(ROOT.kBlack)   

# ==============================
# STACK
# ==============================
stack = ROOT.THStack("stack", ";m_{#gamma#gamma} [GeV];Events")
stack_cat1 = ROOT.THStack("stack_cat1", ";m_{#gamma#gamma} [GeV];Events")
stack_cat2 = ROOT.THStack("stack_cat2", ";m_{#gamma#gamma} [GeV];Events")
stack_cat3 = ROOT.THStack("stack_cat3", ";m_{#gamma#gamma} [GeV];Events")

stack.Add(h1)
stack.Add(h2)
stack.Add(h3)
stack_cat1.Add(h1_cat1)
stack_cat1.Add(h2_cat1)
stack_cat1.Add(h3_cat1)
stack_cat2.Add(h1_cat2)
stack_cat2.Add(h2_cat2)
stack_cat2.Add(h3_cat2)
stack_cat3.Add(h1_cat3)
stack_cat3.Add(h2_cat3)
stack_cat3.Add(h3_cat3)

# ==============================
# DRAW
# ==============================
c = ROOT.TCanvas("c","",800,700)
c.SetLeftMargin(0.12)
c.SetLogy()
stack.Draw("hist")
stack.SetMinimum(0.1)
stack.SetMaximum(1000)

leg = ROOT.TLegend(0.70,0.75,0.92,0.91)
leg.AddEntry(h1, "TTG1Jets", "f")
leg.AddEntry(h2, "TTto2L2Nu", "f")
leg.AddEntry(h3, "TTtoLNu2Q", "f")
leg.SetFillStyle(0) 
leg.SetBorderSize(0)
leg.Draw()

latex = ROOT.TLatex()
latex.SetNDC()              # normalized (0–1) coords
latex.SetTextSize(0.03)
latex.SetTextFont(42)
latex.DrawLatex(0.15, 0.87, "Category inclusive")

c.Update()

CMS_label(c)

c.SaveAs(Plot_dir + "stacked_mgg_inclusive.png")


c_cat1 = ROOT.TCanvas("c_cat1","",800,700)
c_cat1.SetLeftMargin(0.12)
c_cat1.SetLogy()
stack_cat1.Draw("hist")
stack_cat1.SetMinimum(0.1)
stack_cat1.SetMaximum(1000)

c_cat1.Modified()
c_cat1.Update()

leg = ROOT.TLegend(0.70,0.75,0.92,0.91)
leg.AddEntry(h1_cat1, "TTG1Jets", "f")
leg.AddEntry(h2_cat1, "TTto2L2Nu", "f")
leg.AddEntry(h3_cat1, "TTtoLNu2Q", "f")
leg.SetFillStyle(0) 
leg.SetBorderSize(0)
leg.Draw()

latex = ROOT.TLatex()
latex.SetNDC()              # normalized (0–1) coords
latex.SetTextSize(0.03)
latex.SetTextFont(42)
latex.DrawLatex(0.15, 0.87, "Category 1")

c_cat1.Update()

CMS_label(c_cat1)

c_cat1.SaveAs(Plot_dir + "stacked_mgg_cat1.png")


c_cat2 = ROOT.TCanvas("c_cat2","",800,700)
c_cat2.SetLeftMargin(0.12)
c_cat2.SetLogy()
stack_cat2.Draw("hist")
stack_cat2.SetMinimum(0.1)
stack_cat2.SetMaximum(1000)

c_cat2.Modified()
c_cat2.Update()

leg = ROOT.TLegend(0.70,0.75,0.92,0.91)
leg.AddEntry(h1_cat2, "TTG1Jets", "f")
leg.AddEntry(h2_cat2, "TTto2L2Nu", "f")
leg.AddEntry(h3_cat2, "TTtoLNu2Q", "f")
leg.SetFillStyle(0) 
leg.SetBorderSize(0)
leg.Draw()

latex = ROOT.TLatex()
latex.SetNDC()              # normalized (0–1) coords
latex.SetTextSize(0.03)
latex.SetTextFont(42)
latex.DrawLatex(0.15, 0.87, "Category 2")

c_cat2.Update()

CMS_label(c_cat2)

c_cat2.SaveAs(Plot_dir + "stacked_mgg_cat2.png")   

c_cat3 = ROOT.TCanvas("c_cat3","",800,700)
c_cat3.SetLeftMargin(0.12)
c_cat3.SetLogy()
stack_cat3.Draw("hist")
stack_cat3.SetMinimum(0.1)
stack_cat3.SetMaximum(1000)

c_cat3.Modified()
c_cat3.Update()

leg = ROOT.TLegend(0.70,0.75,0.92,0.91)
leg.AddEntry(h1_cat3, "TTG1Jets", "f")
leg.AddEntry(h2_cat3, "TTto2L2Nu", "f")
leg.AddEntry(h3_cat3, "TTtoLNu2Q", "f")
leg.SetFillStyle(0) 
leg.SetBorderSize(0)
leg.Draw()

latex = ROOT.TLatex()
latex.SetNDC()              # normalized (0–1) coords
latex.SetTextSize(0.03)
latex.SetTextFont(42)
latex.DrawLatex(0.15, 0.87, "Category 3")

c_cat3.Update()

CMS_label(c_cat3)

c_cat3.SaveAs(Plot_dir + "stacked_mgg_cat3.png")