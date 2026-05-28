import ROOT
import numpy as np
import uproot

ROOT.gStyle.SetOptStat(0)

Plot_dir = "/eos/user/b/bbapi/www/Analysis_plots/BDT_features/"


f = uproot.open("merged_bkg_withMass.root")
f_sig = uproot.open("merged_signal.root")

tree = f["DiphotonTree"]
tree_sig = f_sig["DiphotonTree"]

arr = tree.arrays(library="ak") 
arr_sig = tree_sig.arrays(library="ak")

inputVars = [
    "first_jet_eta",
    "second_jet_eta",
    "n_bJets",
    "pholead_eta",
    "phosublead_eta",
    "first_jet_B",
    "second_jet_B",
    "pholead_mvaID",
    "phosublead_mvaID"
]



for var in inputVars:

    # Variable values
    x_bkg = np.asarray(getattr(arr, var))
    x_sig = np.asarray(getattr(arr_sig, var))

    # Event weights
    w_bkg = np.asarray(arr.evt_wgt)
    w_sig = np.asarray(arr_sig.evt_wgt)

    # Remove NaN/Inf entries
    mask_bkg = np.isfinite(x_bkg) & np.isfinite(w_bkg)
    mask_sig = np.isfinite(x_sig) & np.isfinite(w_sig)

    x_bkg = x_bkg[mask_bkg]
    w_bkg = w_bkg[mask_bkg]

    x_sig = x_sig[mask_sig]
    w_sig = w_sig[mask_sig]

    if len(x_bkg) == 0 or len(x_sig) == 0:
        print(f"Skipping {var}: empty array")
        continue

    # Histogram range
    xmin = min(np.min(x_bkg), np.min(x_sig))
    xmax = max(np.max(x_bkg), np.max(x_sig))

    if xmin == xmax:
        xmin -= 1
        xmax += 1

    nbins = 100

    h_bkg = ROOT.TH1F(f"h_bkg_{var}", "", nbins, xmin, xmax)
    h_sig = ROOT.TH1F(f"h_sig_{var}", "", nbins, xmin, xmax)

    h_bkg.Sumw2()
    h_sig.Sumw2()

    # Fill weighted histograms
    for x, w in zip(x_bkg, w_bkg):
        h_bkg.Fill(float(x), float(w))

    for x, w in zip(x_sig, w_sig):
        h_sig.Fill(float(x), float(w))

    # Normalize to unit area (shape comparison)
    if h_bkg.Integral() > 0:
        h_bkg.Scale(1.0 / h_bkg.Integral())

    if h_sig.Integral() > 0:
        h_sig.Scale(1.0 / h_sig.Integral())

    # Style
    h_bkg.SetLineColor(ROOT.kBlue + 1)
    h_bkg.SetLineWidth(2)

    h_sig.SetLineColor(ROOT.kRed + 1)
    h_sig.SetLineWidth(2)

    ymax = 1.3 * max(h_bkg.GetMaximum(), h_sig.GetMaximum())

    c = ROOT.TCanvas(f"c_{var}", var, 800, 700)

    h_bkg.SetTitle("")
    h_bkg.GetXaxis().SetTitle(var)
    h_bkg.GetYaxis().SetTitle("Normalized entries")
    h_bkg.SetMaximum(ymax)

    h_bkg.Draw("HIST")
    h_sig.Draw("HIST SAME")

    leg = ROOT.TLegend(0.65, 0.75, 0.88, 0.88)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.AddEntry(h_sig, "Signal", "l")
    leg.AddEntry(h_bkg, "Background", "l")
    leg.Draw()

    c.SaveAs(Plot_dir + f"{var}_sig_vs_bkg.png")

    print(f"Saved {var}_sig_vs_bkg.png")
