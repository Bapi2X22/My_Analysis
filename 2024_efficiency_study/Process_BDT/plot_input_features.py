import uproot
import numpy as np
import awkward as ak
import pandas as pd
import matplotlib.pyplot as plt
import ROOT
import os

Plot_dir = "/eos/user/b/bbapi/www/Analysis_plots/BDT/features_plot/"

dataset_Bkg = uproot.open('merged_bkg_with_allCorr.root')['DiphotonTree'].arrays()

df_bkg = ak.to_dataframe(dataset_Bkg)

dataset_sig = uproot.open('merged_signal_with_allCorr.root')['DiphotonTree'].arrays()

df_sig = ak.to_dataframe(dataset_sig)

def plot_feature_distribution_ROOT(feature, df_sig, df_bkg,
                              bins=50,
                              xmin=None,
                              xmax=None):

    # Convert to numpy
    sig_vals = df_sig[feature].to_numpy()
    bkg_vals = df_bkg[feature].to_numpy()

    sig_wgts = df_sig["evt_wgt"].to_numpy()
    bkg_wgts = df_bkg["evt_wgt"].to_numpy()

    # Auto range
    if xmin is None:
        xmin = min(sig_vals.min(), bkg_vals.min())

    if xmax is None:
        xmax = max(sig_vals.max(), bkg_vals.max())

    # Histograms
    h_sig = ROOT.TH1F(
        "h_sig",
        "",
        bins,
        xmin,
        xmax
    )

    h_bkg = ROOT.TH1F(
        "h_bkg",
        "",
        bins,
        xmin,
        xmax
    )

    # Fill histograms
    for val, wgt in zip(sig_vals, sig_wgts):
        h_sig.Fill(val, wgt)

    for val, wgt in zip(bkg_vals, bkg_wgts):
        h_bkg.Fill(val, wgt)

    # Style
    ROOT.gStyle.SetOptStat(0)

    h_sig.SetLineColor(ROOT.kBlue)
    h_sig.SetFillColorAlpha(ROOT.kBlue, 0.35)
    h_sig.SetLineWidth(2)

    h_bkg.SetLineColor(ROOT.kRed)
    h_bkg.SetFillColorAlpha(ROOT.kRed, 0.35)
    h_bkg.SetLineWidth(2)

    # Axis labels
    h_bkg.GetXaxis().SetTitle(feature)
    h_bkg.GetYaxis().SetTitle("Weighted Events")

    # Canvas
    c = ROOT.TCanvas("c", "", 800, 700)
    c.SetLeftMargin(0.15)

    # Draw
    maxy = max(h_sig.GetMaximum(), h_bkg.GetMaximum())
    h_bkg.SetMaximum(maxy * 1.3)

    h_bkg.Draw("HIST")
    h_sig.Draw("HIST SAME")

    # Legend
    leg = ROOT.TLegend(0.65, 0.75, 0.88, 0.88)
    leg.AddEntry(h_sig, "Signal", "f")
    leg.AddEntry(h_bkg, "Background", "f")
    leg.SetBorderSize(0)
    leg.Draw()
    
    os.makedirs("Plots", exist_ok=True)

    # Save
    c.SaveAs(Plot_dir + f"{feature}.png")
    c.SaveAs(Plot_dir + f"{feature}.pdf")

    return c, h_sig, h_bkg


plot_feature_distribution_ROOT('delphi_bb', df_sig, df_bkg)
plot_feature_distribution_ROOT('Njets', df_sig, df_bkg)
plot_feature_distribution_ROOT('delphi_gg', df_sig, df_bkg)
plot_feature_distribution_ROOT('delphi_bbgg', df_sig, df_bkg)
plot_feature_distribution_ROOT('pT1_by_mbb', df_sig, df_bkg)
plot_feature_distribution_ROOT('pT2_by_mbb', df_sig, df_bkg)
plot_feature_distribution_ROOT('pT1_by_mgg', df_sig, df_bkg)
plot_feature_distribution_ROOT('pT2_by_mgg', df_sig, df_bkg)
plot_feature_distribution_ROOT('n_bJets', df_sig, df_bkg)
plot_feature_distribution_ROOT('leppt', df_sig, df_bkg)
plot_feature_distribution_ROOT('lepeta', df_sig, df_bkg)
plot_feature_distribution_ROOT("first_jet_eta",  df_sig, df_bkg)
plot_feature_distribution_ROOT("second_jet_eta",  df_sig, df_bkg)
plot_feature_distribution_ROOT("pholead_eta",  df_sig, df_bkg)
plot_feature_distribution_ROOT("phosublead_eta",  df_sig, df_bkg)
plot_feature_distribution_ROOT("first_jet_B",  df_sig, df_bkg)
plot_feature_distribution_ROOT("second_jet_B",  df_sig, df_bkg)
plot_feature_distribution_ROOT("pholead_mvaID",  df_sig, df_bkg)
plot_feature_distribution_ROOT("phosublead_mvaID",  df_sig, df_bkg)
plot_feature_distribution_ROOT("mass_point",  df_sig, df_bkg)