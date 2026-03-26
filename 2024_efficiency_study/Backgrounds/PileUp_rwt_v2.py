import numpy as np
import awkward as ak
import uproot
import ROOT
import os
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

ROOT.gROOT.SetBatch(False)  # Make sure canvases render before writing
ROOT.TH1.SetDefaultSumw2(True)

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


# --- Config ---
# parquet_file = "AllDatasets.parquet"
data_pu_files = {
    "2024": "/eos/user/b/bbapi/CMSSW_14_0_15/src/pileup_histos/pileup_2024_Golden.root",
}
output_root = "PUPlots_TTG1Jets.root"

out_dir_base = "/eos/user/b/bbapi/www/PileUp/Pileup_reweighting"

mass_points = ["TTG1Jets"]

# Load events once
# Events = ak.from_parquet(parquet_file)

file_raw = 'root://xrootd-cms.infn.it///store/mc/RunIII2024Summer24NanoAODv15/TTG-1Jets_TuneCP5_13p6TeV_amcatnloFXFXold-pythia8/NANOAODSIM/150X_mcRun3_2024_realistic_v2-v2/120000/31088ed2-e940-45a0-85af-a299e67d1a57.root'
factory = NanoEventsFactory.from_root(
    f"{file_raw}:Events",
    schemaclass=NanoAODSchema,
)
Events = factory.events()

# Open ROOT output file
fout = ROOT.TFile(output_root, "RECREATE")

Era = "2024"

for year, data_pu_file in data_pu_files.items():
    print(f"\n=== Processing {year} ===")

    out_dir_year = os.path.join(out_dir_base, year)
    os.makedirs(out_dir_year, exist_ok=True)

    # mask = np.char.find(ak.to_numpy(Events["dataset"]), f"UL{str(year)[2:]}") >= 0
    # Events_year = Events[mask]
    # mass_points = sorted(set(ak.to_numpy(Events_year["dataset"])))

    with uproot.open(data_pu_file) as f:
        data_vals, bin_edges = f["era2024/pileup"].to_numpy()
    data_pdf = data_vals / np.sum(data_vals)

    fout.mkdir(year)
    fout.cd(year)

    for mass in mass_points:
        print(f"  → {mass}")

        # M_MC = Events_year[ak.to_numpy(Events_year["dataset"]) == mass]
        # mc_nTrueInt = ak.to_numpy(ak.flatten(Events.Pileup.nTrueInt))
        mc_nTrueInt = ak.to_numpy(Events.Pileup.nTrueInt)

        mc_vals, _ = np.histogram(mc_nTrueInt, bins=bin_edges)
        mc_pdf = mc_vals / np.sum(mc_vals)

        eps = 1e-8
        pu_weights_per_bin = np.where(mc_pdf > eps, data_pdf / mc_pdf, 0.0)

        bin_idx = np.digitize(mc_nTrueInt, bin_edges) - 1
        bin_idx = np.clip(bin_idx, 0, len(pu_weights_per_bin) - 1)
        pu_weight = pu_weights_per_bin[bin_idx]
        mc_vals_rw, _ = np.histogram(mc_nTrueInt, bins=bin_edges, weights=pu_weight)
        mc_pdf_rw = mc_vals_rw / np.sum(mc_vals_rw)

        ratio_rw = np.where(data_pdf > eps, mc_pdf_rw / data_pdf, 0)

        # h_mc = ROOT.TH1F(f"h_mc_{mass}", "MC before reweighting;PU;Normalized events", len(bin_edges)-1, bin_edges)
        # h_data = ROOT.TH1F(f"h_data_{mass}", "Data;PU;Normalized events", len(bin_edges)-1, bin_edges)
        # h_mc_rw = ROOT.TH1F(f"h_mc_rw_{mass}", "MC after reweighting;PU;Normalized events", len(bin_edges)-1, bin_edges)
        # h_weights = ROOT.TH1F(f"h_weights_{mass}", "PU weights;PU;Weight", len(bin_edges)-1, bin_edges)
        # h_ratio = ROOT.TH1F(f"h_ratio_{mass}", "Ratio MC_rw/Data;PU;Ratio", len(bin_edges)-1, bin_edges)

        ROOT.gStyle.SetOptStat(1111111)

        h_mc = ROOT.TH1F(
            f"h_mc_{mass}",
            f";PU;Normalized events",
            len(bin_edges)-1, bin_edges
        )
        h_data = ROOT.TH1F(
            f"h_data_{mass}",
            f";PU;Normalized events",
            len(bin_edges)-1, bin_edges
        )
        h_mc_rw = ROOT.TH1F(
            f"h_mc_rw_{mass}",
            f";PU;Normalized events",
            len(bin_edges)-1, bin_edges
        )
        h_weights = ROOT.TH1F(
            f"h_weights_{mass}",
            f";PU;Weight",
            len(bin_edges)-1, bin_edges
        )
        h_ratio = ROOT.TH1F(
            f"h_ratio_{mass}",
            f";PU;Ratio(MC_rw/Data)",
            len(bin_edges)-1, bin_edges
        )



        for i in range(len(mc_pdf)):
            h_mc.SetBinContent(i+1, mc_pdf[i])
            h_data.SetBinContent(i+1, data_pdf[i])
            h_mc_rw.SetBinContent(i+1, mc_pdf_rw[i])
            h_weights.SetBinContent(i+1, pu_weights_per_bin[i])
            h_ratio.SetBinContent(i+1, ratio_rw[i])

        h_mc.SetLineColor(ROOT.kRed)
        h_mc.SetLineWidth(2)
        h_data.SetMarkerStyle(20)
        h_data.SetMarkerColor(ROOT.kBlue)
        h_data.SetLineColor(ROOT.kBlue)
        h_mc_rw.SetLineColor(ROOT.kGreen+2)
        h_mc_rw.SetLineWidth(2)
        h_weights.SetLineColor(ROOT.kBlack)
        h_ratio.SetLineColor(ROOT.kMagenta)

        # Canvas 1: Data vs MC before reweighting
        c1 = ROOT.TCanvas(f"c_before_{mass}", f"", 800, 600)
        ROOT.gStyle.SetOptStat(0)
        # Set max before drawing
        ymax = 1.1 * max(h_mc.GetMaximum(), h_data.GetMaximum())
        h_mc.SetMaximum(ymax)
        h_mc.Draw("HIST")
        h_data.Draw("EP SAME")
        leg1 = ROOT.TLegend(0.7, 0.15, 0.88, 0.25)
        leg1.SetTextSize(0.02)
        # leg1.AddEntry(h_mc, "MC", "l")
        leg1.AddEntry(h_mc, f"MC", "l")
        leg1.AddEntry(h_data, "Data", "p")
        leg1.Draw()

        CMS_label(c1)

        latex = ROOT.TLatex()
        latex.SetNDC()
        latex.SetTextSize(0.04)
        latex.SetTextFont(42)
        latex.DrawLatex(0.63, 0.8, mass)

        latex = ROOT.TLatex()
        latex.SetNDC()              # normalized (0–1) coords
        latex.SetTextSize(0.04)
        latex.SetTextFont(42)
        latex.DrawLatex(0.58, 0.70, "Before reweighting")

        ROOT.gStyle.SetOptStat(1111111)

        c1.Update()

        draw_side_statboxes(c1,[h_mc,h_data])

        c1.Update()

        # c1.Modified()
        # c1.Update()
        c1.Write()

        # Canvas 2: Data vs MC after reweighting
        c2 = ROOT.TCanvas(f"c_after_{mass}", f"", 800, 600)
        ROOT.gStyle.SetOptStat(0)
        h_mc_rw.Draw("HIST")
        h_data.Draw("EP SAME")
        leg2 = ROOT.TLegend(0.7, 0.15, 0.88, 0.25)
        leg2.SetTextSize(0.02)
        leg2.AddEntry(h_mc_rw, f"MC reweighted", "l")
        leg2.AddEntry(h_data, "Data", "p")
        leg2.Draw()

        CMS_label(c2)

        latex = ROOT.TLatex()
        latex.SetNDC()
        latex.SetTextSize(0.04)
        latex.SetTextFont(42)
        latex.DrawLatex(0.63, 0.8, mass)

        latex = ROOT.TLatex()
        latex.SetNDC()              # normalized (0–1) coords
        latex.SetTextSize(0.04)
        latex.SetTextFont(42)
        latex.DrawLatex(0.58, 0.70, "After reweighting")

        # ROOT.gStyle.SetOptStat(1111111)

        c2.Update()

        ROOT.gStyle.SetOptStat(1111111)

        draw_side_statboxes(c2,[h_mc_rw,h_data])

        c2.Update()

        # c2.Modified()
        # c2.Update()
        c2.Write()

        # Write extra histograms to ROOT file
        h_weights.Write()
        h_ratio.Write()

        # Save canvases as PDF and PNG
        c1.SaveAs(f"{out_dir_year}/c_before_{mass}_{Era}.pdf")
        c1.SaveAs(f"{out_dir_year}/c_before_{mass}_{Era}.png")

        c2.SaveAs(f"{out_dir_year}/c_after_{mass}_{Era}.pdf")
        c2.SaveAs(f"{out_dir_year}/c_after_{mass}_{Era}.png")

        # Save h_weights and h_ratio as images
        ROOT.gStyle.SetOptStat(0)
        c_w = ROOT.TCanvas(f"c_weights_{mass}", f"PU weights - {mass}", 800, 600)
        c_w.SetLogy()
        h_weights.Draw("HIST")
        h_weights.SetMinimum(0.1)
        h_weights.SetMaximum(100.0)
        CMS_label(c_w)
        c_w.Modified()
        c_w.Update()
        c_w.SaveAs(f"{out_dir_year}/h_weights_{mass}_{Era}.pdf")
        c_w.SaveAs(f"{out_dir_year}/h_weights_{mass}_{Era}.png")
        c_w.Close()

        c_r = ROOT.TCanvas(f"c_ratio_{mass}", f"", 800, 600)
        h_ratio.Draw("HIST")
        CMS_label(c_r)
        c_r.Modified()
        c_r.Update()
        c_r.SaveAs(f"{out_dir_year}/h_ratio_{mass}_{Era}.pdf")
        c_r.SaveAs(f"{out_dir_year}/h_ratio_{mass}_{Era}.png")
        c_r.Close()

fout.Close()
print(f"\nSaved PU plots and histograms for 2016, 2017, and 2018 in {output_root}")



