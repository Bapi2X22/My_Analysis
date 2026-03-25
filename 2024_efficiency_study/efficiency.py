import ROOT
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt

from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

ROOT.gROOT.SetBatch(True)

# --------------------------------------------------
# Config
# --------------------------------------------------
mass_points = [15, 20, 30, 45, 60]

base_path = "/eos/user/b/bbapi/MC_contacts/2024_signal_samples_WH/CMSSW_15_0_15/src"
file_tmpl = base_path + "/WH_2024_M{}.root"

outdir = "/eos/user/b/bbapi/www/Analysis_plots/Overlay_plots/WH_2024"

# Cuts
MUON_PT_CUT, ELE_PT_CUT = 24.0, 30.0
PHO1_PT_CUT, PHO2_PT_CUT = 20.0, 15.0
BJET_PT_CUT = 20.0

ELE_PHO_ETA_MAX = 2.5
MU_JET_ETA_MAX  = 2.4
ETA_GAP_LO, ETA_GAP_HI = 1.44, 1.57

BTAG_B_CUT     = 0.1272
BTAG_PROBB_CUT = 0.38

# --------------------------------------------------
# Helper: run full preselection and record counts
# --------------------------------------------------
def run_presel(events):

    counts = {}

    # -------------------------
    # Stage 0: initial
    events_0 = events
    counts["No preselections"] = len(events_0)

    # -------------------------
    # Step 1: exactly one lepton
    good_mu = events_0.Muon.pt > MUON_PT_CUT
    good_el = events_0.Electron.pt > ELE_PT_CUT

    mask_1lep = (
        ak.num(events_0.Muon[good_mu]) +
        ak.num(events_0.Electron[good_el])
    ) == 1

    events_1lep = events_0[mask_1lep]
    counts["lep pT >(e: 30, mu: 24) GeV"] = len(events_1lep)

    # -------------------------
    # Step 2: leading photon pT > PHO1_PT_CUT
    mask_npho1 = ak.num(events_1lep.Photon) >= 1
    events_tmp = events_1lep[mask_npho1]

    photons = events_tmp.Photon[
        ak.argsort(events_tmp.Photon.pt, axis=1, ascending=False)
    ]

    mask_pho1 = photons.pt[:, 0] > PHO1_PT_CUT
    events_pho1 = events_tmp[mask_pho1]
    counts["leading pho pT > 20 GeV"] = len(events_pho1)

    # -------------------------
    # Step 3: subleading photon pT > PHO2_PT_CUT
    mask_npho2 = ak.num(events_pho1.Photon) >= 2
    events_tmp = events_pho1[mask_npho2]

    photons = events_tmp.Photon[
        ak.argsort(events_tmp.Photon.pt, axis=1, ascending=False)
    ]

    mask_pho2 = photons.pt[:, 1] > PHO2_PT_CUT
    events_pho2 = events_tmp[mask_pho2]
    counts["subleading pho pT > 15 GeV"] = len(events_pho2)

    # -------------------------
    # Step 4: ≥1 jet with pT > BJET_PT_CUT
    mask_jet = ak.num(
        events_pho2.Jet[events_pho2.Jet.pt > BJET_PT_CUT]
    ) >= 1

    events_jet = events_pho2[mask_jet]
    counts["Jet pT > 20 GeV"] = len(events_jet)

    # -------------------------
    # Step 5: eta cuts
    events_eta = events_jet[
        ak.all(
            (np.abs(events_jet.Electron.eta) < ELE_PHO_ETA_MAX) &
            ~((np.abs(events_jet.Electron.eta) > ETA_GAP_LO) &
            (np.abs(events_jet.Electron.eta) < ETA_GAP_HI)),
            axis=1
        )
        &
        ak.all(
            (np.abs(events_jet.Photon.eta) < ELE_PHO_ETA_MAX) &
            ~((np.abs(events_jet.Photon.eta) > ETA_GAP_LO) &
            (np.abs(events_jet.Photon.eta) < ETA_GAP_HI)),
            axis=1
        )
        &
        ak.all(np.abs(events_jet.Muon.eta) < MU_JET_ETA_MAX, axis=1)
        &
        ak.all(np.abs(events_jet.Jet.eta) < MU_JET_ETA_MAX, axis=1)
    ]

    counts["eta cuts"] = len(events_eta)

    # -------------------------
    # Step 6: categories
    jets = events_eta.Jet

    bjets_B = jets[jets.btagUParTAK4B > BTAG_B_CUT]
    bjets_probb = jets[jets.btagUParTAK4probbb > BTAG_PROBB_CUT]

    mask_cat1 = ak.num(bjets_B) >= 2
    mask_cat2 = (ak.num(bjets_probb) >= 1) & (~mask_cat1)
    mask_cat3 = (ak.num(bjets_B) == 1) & (~mask_cat1) & (~mask_cat2)

    events_cat1 = events_eta[mask_cat1]
    events_cat2 = events_eta[mask_cat2]
    events_cat3 = events_eta[mask_cat3]

    counts["cat1"] = len(events_cat1)
    counts["cat2"] = len(events_cat2)
    counts["cat3"] = len(events_cat3)



    return counts

# --------------------------------------------------
# Loop over mass points
# --------------------------------------------------
stage_names = [
    "No preselections",
    "lep pT >(e: 30, mu: 24) GeV",
    "leading pho pT > 20 GeV",
    "subleading pho pT > 15 GeV",
    "Jet pT > 20 GeV",
    "eta cuts",
    "cat1",
    "cat2",
    "cat3",
]

efficiencies = {stage: [] for stage in stage_names}

for m in mass_points:
    print(f"Processing M={m} GeV")

    factory = NanoEventsFactory.from_root(
        f"{file_tmpl.format(m)}:Events",
        schemaclass=NanoAODSchema,
    )
    events = ak.materialize(factory.events())

    counts = run_presel(events)
    n0 = counts["No preselections"]

    for stage in stage_names:
        eff = counts[stage] / n0 if n0 > 0 else 0.0
        efficiencies[stage].append(eff)


import ROOT
import numpy as np

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)

# ==================================================
# Canvas and pads
# ==================================================
c = ROOT.TCanvas("c_eff", "WH preselection efficiency vs mass", 1600, 800)

# Main plot pad
# pad_main = ROOT.TPad("pad_main", "", 0.0, 0.0, 0.78, 1.0)
pad_main = ROOT.TPad("pad_main", "", 0.0, 0.0, 0.72, 1.0)
pad_main.SetLogy()
pad_main.SetLeftMargin(0.12)
pad_main.SetRightMargin(0.02)
pad_main.SetBottomMargin(0.12)
pad_main.SetTopMargin(0.08)
pad_main.Draw()

# Legend pad
# pad_legend = ROOT.TPad("pad_legend", "", 0.78, 0.0, 1.0, 1.0)
pad_legend = ROOT.TPad("pad_legend", "", 0.72, 0.0, 1.0, 1.0)
pad_legend.SetFillStyle(0)
pad_legend.SetBorderSize(0)
pad_legend.Draw()

# ==================================================
# Main plot
# ==================================================
pad_main.cd()

xmin, xmax = min(mass_points) - 5, max(mass_points) + 5

frame = ROOT.TH1F("frame", "", 100, xmin, xmax)
frame.SetMinimum(0.01)
frame.SetMaximum(120)

frame.GetXaxis().SetTitle("Mass point [GeV]")
frame.GetYaxis().SetTitle("Efficiency [%]")
# frame.GetYaxis().SetTitleOffset(1.05)
frame.GetYaxis().SetTitleOffset(1.35)


frame.Draw("AXIS")

# Hide default Y-axis labels and ticks
frame.GetYaxis().SetLabelSize(0)
frame.GetYaxis().SetTickLength(0)

# --------------------------------------------------
# Custom log-scale Y ticks (CMS style)
# --------------------------------------------------
y_ticks = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]

tick_text = ROOT.TLatex()
tick_text.SetTextFont(42)
tick_text.SetTextSize(0.04)
tick_text.SetTextAlign(32)

for y in y_ticks:
    line = ROOT.TLine(xmin, y, xmax, y)
    line.SetLineStyle(3)
    line.SetLineColor(ROOT.kGray + 1)
    line.Draw()

    tick_text.DrawLatex(xmin - 0.5, y, f"{y:g}")

# --------------------------------------------------
# Draw graphs
# --------------------------------------------------
colors = [
    ROOT.kBlue + 1,
    ROOT.kRed + 1,
    ROOT.kGreen + 2,
    ROOT.kMagenta + 1,
    ROOT.kOrange + 7,
    ROOT.kCyan + 2,
    ROOT.kViolet,
    ROOT.kGray + 2,
]

markers = [20, 21, 22, 23, 33, 34, 29, 47]
linestyles = [1, 1, 1, 1, 2, 2, 2, 2]

graphs = []

for i, stage in enumerate(stage_names[1:]):
    x = np.array(mass_points, dtype="float64")
    y = np.array([100.0 * e for e in efficiencies[stage]], dtype="float64")

    g = ROOT.TGraph(len(x), x, y)
    g.SetLineColor(colors[i % len(colors)])
    g.SetMarkerColor(colors[i % len(colors)])
    g.SetMarkerStyle(markers[i % len(markers)])
    g.SetLineStyle(linestyles[i % len(linestyles)])
    g.SetLineWidth(2)
    g.SetMarkerSize(1.3)

    g.Draw("LP SAME")
    graphs.append((g, stage))

# --------------------------------------------------
# CMS text
# --------------------------------------------------
cms = ROOT.TLatex()
cms.SetNDC()
cms.SetTextFont(52)
cms.SetTextSize(0.038)
cms.DrawLatex(0.15, 0.88, "CMS Simulation Preliminary")

extra = ROOT.TLatex()
extra.SetNDC()
extra.SetTextFont(42)
extra.SetTextSize(0.038)
extra.DrawLatex(0.68, 0.16, "Run 3 (13.6 TeV)")

title = ROOT.TLatex()
title.SetNDC()
title.SetTextFont(42)
title.SetTextSize(0.046)
title.SetTextAlign(22)
title.DrawLatex(0.55, 0.975, "WH preselection efficiency vs mass")

# ==================================================
# Legend
# ==================================================
pad_legend.cd()

# leg = ROOT.TLegend(0.02, 0.20, 0.95, 0.90)
leg = ROOT.TLegend(0.01, 0.18, 0.98, 0.90)
leg.SetBorderSize(0)
leg.SetFillStyle(0)
leg.SetTextSize(0.055)

for g, stage in graphs:
    leg.AddEntry(g, stage, "lp")

leg.Draw()

# ==================================================
# Save outputs
# ==================================================
c.Print(f"{outdir}/presel_efficiency_vs_mass.pdf")
c.Print(f"{outdir}/presel_efficiency_vs_mass.png")

# High-resolution raster
c.SetCanvasSize(2400, 1700)
c.Update()
c.Print(f"{outdir}/presel_efficiency_vs_mass_highres.png")

print("Saved: presel_efficiency_vs_mass.[pdf/png]")

