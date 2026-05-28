import awkward as ak
import matplotlib.pyplot as plt
import ROOT

Plot_dir = '/eos/user/b/bbapi/www/Analysis_plots/Signal_BKG_shape_overlay/'

file1 = '/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/NTuples_BKG_2024_HDNA_presel_final_good_selections/merged/TTG1Jets-24SummerRun3/nominal/diphoton/CAT1_merged.parquet'
file2 = '/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/NTuples_BKG_2024_HDNA_presel_final_good_selections/merged/TTto2L2Nu-24SummerRun3/nominal/diphoton/CAT1_merged.parquet'
file3 = '/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/NTuples_BKG_2024_HDNA_presel_final_good_selections/merged/TTtoLNu2Q-24SummerRun3/nominal/diphoton/CAT1_merged.parquet'
file4 = '/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/NTuples_BKG_2024_HDNA_presel_final_good_selections/merged/WGtoLNuG-24SummerRun3/nominal/diphoton/CAT1_merged.parquet'



ROOT.gStyle.SetOptStat(111111)

def plot_overlay_1d(
    bkg_values,
    sig_values_list,
    sig_labels,
    bkg_labels,
    bins=50,
    x_range=(0, 200),
    variable="Mass (GeV)",
    outfile="overlay.png",
    normalize=False,
    logy=False
):

    ROOT.gStyle.SetOptStat(0)

    # --------------------------------------------------
    # Background histogram
    # --------------------------------------------------
    h_bkg = ROOT.TH1F(
        "h_bkg",
        f";{variable};Events",
        bins,
        x_range[0],
        x_range[1]
    )

    for x in bkg_values:
        h_bkg.Fill(x)

    if normalize and h_bkg.Integral() > 0:
        h_bkg.Scale(1.0 / h_bkg.Integral())

    h_bkg.SetFillColorAlpha(ROOT.kGray + 1, 0.50)
    h_bkg.SetLineColor(ROOT.kBlack)
    h_bkg.SetLineWidth(2)

    # --------------------------------------------------
    # Signal histograms
    # --------------------------------------------------
    colors = [
        ROOT.kRed,
        ROOT.kBlue,
        ROOT.kGreen + 2
    ]

    sig_hists = []

    for i, values in enumerate(sig_values_list):

        hsig = ROOT.TH1F(
            f"h_sig_{i}",
            "",
            bins,
            x_range[0],
            x_range[1]
        )

        for x in values:
            hsig.Fill(x)

        if normalize and hsig.Integral() > 0:
            hsig.Scale(1.0 / hsig.Integral())

        hsig.SetLineColor(colors[i])
        hsig.SetLineWidth(3)
        hsig.SetFillStyle(0)

        sig_hists.append(hsig)

    # --------------------------------------------------
    # Determine maximum
    # --------------------------------------------------
    ymax = h_bkg.GetMaximum()

    for h in sig_hists:
        ymax = max(ymax, h.GetMaximum())

    h_bkg.SetMaximum(ymax * 1.4)

    # --------------------------------------------------
    # Canvas
    # --------------------------------------------------
    canvas = ROOT.TCanvas(
        "canvas",
        "canvas",
        1000,
        700
    )

    if logy:
        canvas.SetLogy()
        h_bkg.SetMinimum(0.1)
        h_bkg.SetMaximum(ymax * 100)

    # --------------------------------------------------
    # Draw
    # --------------------------------------------------
    h_bkg.Draw("HIST")

    for h in sig_hists:
        h.Draw("HIST SAME")

    # --------------------------------------------------
    # Legend
    # --------------------------------------------------
    legend = ROOT.TLegend(
        0.65,
        0.70,
        0.88,
        0.88
    )

    legend.SetBorderSize(0)
    legend.SetFillStyle(0)

    legend.AddEntry(
        h_bkg,
        bkg_labels,
        "f"
    )

    for h, label in zip(sig_hists, sig_labels):
        legend.AddEntry(
            h,
            label,
            "l"
        )

    legend.Draw()

    canvas.Modified()
    canvas.Update()

    canvas.SaveAs(Plot_dir + outfile)

    return h_bkg, sig_hists, canvas

import awkward as ak

base_dir = "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/NTuples_WH_2024_HDNA_presel_final_good_selections/merged"

# ==========================================================
# Load backgrounds
# ==========================================================

backgrounds = {
    "TTG1Jets"  : file1,
    "TTto2L2Nu" : file2,
    "TTtoLNu2Q" : file3,
    "WGtoLNuG"  : file4,
}

# ==========================================================
# Load signals
# ==========================================================

signal_masses = [15, 35, 50]

signals = {}

for m in signal_masses:

    infile = (
        f"{base_dir}/WH-2024M{m}/nominal/diphoton/CAT1_merged.parquet"
    )

    signals[m] = ak.from_parquet(infile)

# ==========================================================
# Produce overlays
# ==========================================================

for bkg_name, bkg_file in backgrounds.items():

    print(f"Processing {bkg_name}")

    bkg = ak.from_parquet(bkg_file)

    # ------------------------------------------
    # Background variables
    # ------------------------------------------

    bkg_pt_ratio = (
        bkg.phosublead_pt /
        bkg.pholead_pt
    )

    bkg_ptlead_mass = (
        bkg.pholead_pt /
        bkg.mass
    )

    bkg_ptsublead_mass = (
        bkg.phosublead_pt /
        bkg.mass
    )

    # ------------------------------------------
    # Signal variables
    # ------------------------------------------

    sig_pt_ratio = []
    sig_ptlead_mass = []
    sig_ptsublead_mass = []

    labels = []

    for m in signal_masses:

        sig = signals[m]

        sig_pt_ratio.append(
            sig.phosublead_pt /
            sig.pholead_pt
        )

        sig_ptlead_mass.append(
            sig.pholead_pt /
            sig.mass
        )

        sig_ptsublead_mass.append(
            sig.phosublead_pt /
            sig.mass
        )

        labels.append(f"WH M{m}")

    # ======================================================
    # pT ratio
    # ======================================================

    plot_overlay_1d(
        bkg_values=bkg_pt_ratio,
        sig_values_list=sig_pt_ratio,
        sig_labels=labels,
        bkg_labels=bkg_name,
        bins=50,
        x_range=(0,1),
        variable=
        "p_{T}^{#gamma}_{sublead}/p_{T}^{#gamma}_{lead}",
        outfile=f"{bkg_name}_pt_ratio_overlay.png",
        normalize=True,
        logy=False
    )

    # ======================================================
    # lead photon pT / mass
    # ======================================================

    plot_overlay_1d(
        bkg_values=bkg_ptlead_mass,
        sig_values_list=sig_ptlead_mass,
        sig_labels=labels,
        bkg_labels=bkg_name,
        bins=50,
        x_range=(0,5),
        variable=
        "p_{T}^{#gamma}_{lead}/m_{#gamma#gamma}",
        outfile=f"{bkg_name}_ptlead_over_mass_overlay.png",
        normalize=True,
        logy=False
    )

    # ======================================================
    # sublead photon pT / mass
    # ======================================================

    plot_overlay_1d(
        bkg_values=bkg_ptsublead_mass,
        sig_values_list=sig_ptsublead_mass,
        sig_labels=labels,
        bkg_labels=bkg_name,
        bins=50,
        x_range=(0,5),
        variable=
        "p_{T}^{#gamma}_{sublead}/m_{#gamma#gamma}",
        outfile=f"{bkg_name}_ptsublead_over_mass_overlay.png",
        normalize=True,
        logy=False
    )