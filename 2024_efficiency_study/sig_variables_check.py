import awkward as ak
import matplotlib.pyplot as plt
import ROOT

Plot_dir = '/eos/user/b/bbapi/www/Analysis_plots/Signal_shape_check/'


ROOT.gStyle.SetOptStat(111111)

def plot_1d(
    values,
    bins=50,
    x_range=(0, 80),
    label="TTG",
    color=ROOT.kBlue,
    outfile="hist.png",
    variable="Mass", 
    logy = False, 
    legend_left = False
):

    # Create histogram
    hist = ROOT.TH1F(
        "hist",
        f";{variable};Events",
        bins,
        x_range[0],
        x_range[1]
    )

    # Fill histogram
    for val in values:
        hist.Fill(val)

    # Styling
    hist.SetFillColorAlpha(color, 0.5)
    hist.SetLineColor(color)
    hist.SetLineWidth(2)

    # Canvas
    canvas = ROOT.TCanvas("canvas", "canvas", 1000, 600)

    # Leave space on right for statbox
    ROOT.gPad.SetRightMargin(0.18)

    # Draw
    if logy:
        canvas.SetLogy()
        hist.SetMinimum(0.1)  # Avoid log(0) issues
    hist.Draw("HIST")

    # Legend
    if legend_left:
        legend = ROOT.TLegend(0.12, 0.80, 0.23, 0.88)
    else:   
        legend = ROOT.TLegend(0.68, 0.80, 0.83, 0.88)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.AddEntry(hist, label, "f")
    legend.Draw()

    # Force statbox creation
    ROOT.gPad.Update()

    # Move statbox to empty right space
    stat = hist.FindObject("stats")

    if stat:
        stat.SetX1NDC(0.83)
        stat.SetX2NDC(0.98)

        stat.SetY1NDC(0.65)
        stat.SetY2NDC(0.90)

    # Refresh
    ROOT.gPad.Modified()
    ROOT.gPad.Update()

    # Save
    canvas.SaveAs(Plot_dir + outfile)

    return hist, canvas


def plot_2d(
    x_values,
    y_values,
    x_bins=50,
    x_range=(0, 80),
    y_bins=50,
    y_range=(0, 80),
    label="TTG",
    outfile="hist2d.png",
    x_variable="X",
    y_variable="Y",
    draw_option="COLZ"
):

    # Create histogram
    hist = ROOT.TH2F(
        "hist2d",
        f"{label};{x_variable};{y_variable}",
        x_bins,
        x_range[0],
        x_range[1],
        y_bins,
        y_range[0],
        y_range[1]
    )

    # Fill histogram
    for x, y in zip(x_values, y_values):
        hist.Fill(x, y)

    # Canvas
    canvas = ROOT.TCanvas("canvas2d", "canvas2d", 1200, 700)

    # Extra space on right
    ROOT.gPad.SetRightMargin(0.28)

    # Draw
    hist.Draw(draw_option)
    hist.GetYaxis().SetTitleSize(0.055)

    # Force stat box creation
    ROOT.gPad.Update()

    # Move stat box into empty right margin
    stat = hist.FindObject("stats")

    if stat:
        stat.SetX1NDC(0.81)
        stat.SetX2NDC(0.98)

        stat.SetY1NDC(0.65)
        stat.SetY2NDC(0.90)

    # Refresh
    ROOT.gPad.Modified()
    ROOT.gPad.Update()

    # Save
    canvas.SaveAs(Plot_dir + outfile)

    return hist, canvas

import awkward as ak

base_dir = "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/NTuples_WH_2024_HDNA_presel_final_good_selections/merged"

# Example mass points
masses = [12, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]

for mass in masses:

    print(f"Processing M{mass}")

    infile = (
        f"{base_dir}/WH-2024M{mass}/nominal/diphoton/CAT1_merged.parquet"
    )

    events = ak.from_parquet(infile)

    tag = f"M{mass}"

    # ----------------------------
    # Mass distributions
    # ----------------------------
    plot_1d(
        events.mass,
        bins=50,
        x_range=(0, 80),
        label=tag,
        outfile=f"{tag}_mass.png",
        variable="m_{#gamma#gamma} (GeV)"
    )

    plot_1d(
        events.mass,
        bins=50,
        x_range=(0, 500),
        label=tag,
        outfile=f"{tag}_mass_full.png",
        variable="m_{#gamma#gamma} (GeV)"
    )

    # ----------------------------
    # Photon pT distributions
    # ----------------------------
    plot_1d(
        events.pholead_pt,
        bins=50,
        x_range=(0, 200),
        label=tag,
        outfile=f"{tag}_lead_pho_pT.png",
        variable="p_{T}^{#gamma}_{lead} (GeV)"
    )

    plot_1d(
        events.phosublead_pt,
        bins=50,
        x_range=(0, 200),
        label=tag,
        outfile=f"{tag}_sublead_pho_pT.png",
        variable="p_{T}^{#gamma}_{sublead} (GeV)"
    )

    # log scale
    plot_1d(
        events.pholead_pt,
        bins=50,
        x_range=(0, 200),
        label=tag,
        outfile=f"{tag}_lead_pho_pT_logy.png",
        variable="p_{T}^{#gamma}_{lead} (GeV)",
        logy=True
    )

    plot_1d(
        events.phosublead_pt,
        bins=50,
        x_range=(0, 200),
        label=tag,
        outfile=f"{tag}_sublead_pho_pT_logy.png",
        variable="p_{T}^{#gamma}_{sublead} (GeV)",
        logy=True
    )

    # ----------------------------
    # pT ratio
    # ----------------------------
    pt_ratio = events.phosublead_pt / events.pholead_pt

    plot_1d(
        pt_ratio,
        bins=50,
        x_range=(0, 1),
        label=tag,
        outfile=f"{tag}_pho_pT_ratio.png",
        variable="p_{T}^{#gamma}_{sublead}/p_{T}^{#gamma}_{lead}",
        legend_left=True
    )

    # ----------------------------
    # 2D plots
    # ----------------------------
    plot_2d(
        events.mass,
        events.pholead_pt / events.mass,
        x_bins=50,
        x_range=(0, 80),
        y_bins=50,
        y_range=(0, 10),
        x_variable="m_{#gamma#gamma} [GeV]",
        y_variable="p_{T}^{#gamma}_{lead}/m_{#gamma#gamma}",
        label=tag,
        outfile=f"mass_vs_ptovergamma_lead_{tag}.png"
    )

    plot_2d(
        events.mass,
        events.phosublead_pt / events.mass,
        x_bins=50,
        x_range=(0, 80),
        y_bins=50,
        y_range=(0, 10),
        x_variable="m_{#gamma#gamma} [GeV]",
        y_variable="p_{T}^{#gamma}_{sublead}/m_{#gamma#gamma}",
        label=tag,
        outfile=f"mass_vs_ptovergamma_sublead_{tag}.png"
    )

    plot_2d(
        events.mass,
        pt_ratio,
        x_bins=50,
        x_range=(0, 80),
        y_bins=50,
        y_range=(0, 1),
        x_variable="m_{#gamma#gamma} [GeV]",
        y_variable="p_{T}^{#gamma}_{sublead}/p_{T}^{#gamma}_{lead}",
        label=tag,
        outfile=f"mass_vs_pt_ratio_{tag}.png"
    )