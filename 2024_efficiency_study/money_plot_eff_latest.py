import awkward as ak
import numpy as np
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import ROOT

ROOT.gStyle.SetOptStat(0)

diB_mask_thr = 0.40
Plot_dir = '/eos/user/b/bbapi/www/Analysis_plots/Jets/btag_wp_optimization/probbb_thr/'

def CMS_label(pad,
              lumi="109 fb^{-1}",
              year="2024",
              energy="13.6 TeV",
              status="Simulation Preliminary",
              x=0.12,
              y=0.91):

    pad.cd()

    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextAngle(0)
    latex.SetTextColor(ROOT.kBlack)

    # ---- CMS (bold) ----
    latex.SetTextFont(61)
    latex.SetTextSize(0.04)
    latex.DrawLatex(x, y, "CMS")

    # ---- Status (italic) ----
    if status != "":
        latex.SetTextFont(52)
        latex.SetTextSize(0.035)
        latex.DrawLatex(x + 0.11, y, status)

    # ---- Lumi text (right aligned) ----
    latex.SetTextFont(42)
    latex.SetTextSize(0.035)
    latex.SetTextAlign(31)
    lumi_text = f"({year}-{lumi})"
    latex.DrawLatex(0.88, y, lumi_text)

base_dir = "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/NTuples_WH_2024_HDNA_presel_official_full/merged_old/"

def get_category_efficiencies(M):


    dir = f"{base_dir}"+f"WH-2024M{M}/nominal/"

    CAT1_events = ak.from_parquet(dir+"CAT1_merged.parquet")
    CAT2_events = ak.from_parquet(dir+"CAT2_merged.parquet")
    CAT3_events = ak.from_parquet(dir+"CAT3_merged.parquet")


    eff_res = 100. * ak.sum(CAT1_events.weight)
    eff_mer = 100. * ak.sum(CAT2_events.weight)
    eff_one = 100. * ak.sum(CAT3_events.weight)

    return eff_res, eff_mer, eff_one


mass_points = [12, 15, 20, 25, 30, 35, 40, 45, 50, 60]

n = len(mass_points)

def make_empty_hist(name):
    return ROOT.TH1F(name, "", n, 0, n)

# pt = 10
h_res_10 = make_empty_hist("h_res_10")
h_mer_10 = make_empty_hist("h_mer_10")
h_one_10 = make_empty_hist("h_one_10")

# pt = 15
h_res_15 = make_empty_hist("h_res_15")
h_mer_15 = make_empty_hist("h_mer_15")
h_one_15 = make_empty_hist("h_one_15")

for i, m in enumerate(mass_points):

    r15, m15, o15 = get_category_efficiencies(m)

    bin_idx = i + 1

    h_res_15.SetBinContent(bin_idx, r15)
    h_mer_15.SetBinContent(bin_idx, m15)
    h_one_15.SetBinContent(bin_idx, o15)


    # ----------------------------
    # COLORS (same for both pt)
    # ----------------------------

    # Resolved
    h_res_15.SetFillColorAlpha(ROOT.kBlue-7, 0.6)

    # Merged
    h_mer_15.SetFillColorAlpha(ROOT.kRed-4, 0.6)

    # One-jet
    h_one_15.SetFillColorAlpha(ROOT.kGreen+2, 0.6)

    # for h in [h_res_15, h_mer_15, h_one_15]:
        # h.SetFillStyle(0)

    # total efficiency histograms
    h_tot_15 = h_res_15.Clone("h_tot_15")
    h_tot_15.Add(h_one_15)
    h_tot_15.Add(h_mer_15)


    # ----------------------------
    # BORDER STYLES
    # ----------------------------

    # Solid → pt = 10
    for h in [h_res_15, h_mer_15, h_one_15]:
        h.SetLineColor(ROOT.kBlack)
        h.SetLineWidth(2)
        h.SetLineStyle(1)

    for h in [h_res_15, 
              h_one_15, 
              h_mer_15]:
        h.GetXaxis().SetBinLabel(bin_idx, f"{m}")


stack_15 = ROOT.THStack("stack_15", "")
stack_15.Add(h_res_15)
stack_15.Add(h_one_15)
stack_15.Add(h_mer_15)

c = ROOT.TCanvas("c", "", 800, 800)

stack_15.Draw("HIST")
stack_15.SetMaximum(10.0)
stack_15.SetMinimum(0.0)

stack_15.GetYaxis().SetTitle("Efficiency [%]")
stack_15.GetYaxis().SetTitleOffset(1.3)
stack_15.GetXaxis().SetLabelSize(0)

CMS_label(c)

leg = ROOT.TLegend(0.65, 0.7, 0.88, 0.83)
leg.SetBorderSize(0)
leg.SetFillStyle(0)

leg.AddEntry(h_res_15, "Resolved)", "f")

leg.AddEntry(h_mer_15, "Merged)", "f")

leg.AddEntry(h_one_15, "One-b-Jet)", "f")


leg.Draw()

CMS_label(c)


c.SaveAs(Plot_dir+f"efficiency_comparison_latest.pdf")
c.SaveAs(Plot_dir+f"efficiency_comparison_latest.png")