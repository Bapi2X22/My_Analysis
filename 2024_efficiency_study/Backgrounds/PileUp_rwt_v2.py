import numpy as np
import awkward as ak
import uproot
import ROOT
import os
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

ROOT.gROOT.SetBatch(False)  # Make sure canvases render before writing
ROOT.TH1.SetDefaultSumw2(True)

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

        h_mc = ROOT.TH1F(
            f"h_mc_{mass}",
            f"MC before reweighting - {mass};PU;Normalized events",
            len(bin_edges)-1, bin_edges
        )
        h_data = ROOT.TH1F(
            f"h_data_{mass}",
            f"Data - {mass};PU;Normalized events",
            len(bin_edges)-1, bin_edges
        )
        h_mc_rw = ROOT.TH1F(
            f"h_mc_rw_{mass}",
            f"MC after reweighting - {mass};PU;Normalized events",
            len(bin_edges)-1, bin_edges
        )
        h_weights = ROOT.TH1F(
            f"h_weights_{mass}",
            f"PU weights - {mass};PU;Weight",
            len(bin_edges)-1, bin_edges
        )
        h_ratio = ROOT.TH1F(
            f"h_ratio_{mass}",
            f"Ratio MC_rw/Data - {mass};PU;Ratio",
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
        c1 = ROOT.TCanvas(f"c_before_{mass}", f"Before reweighting - {mass}", 800, 600)
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
        c1.Modified()
        c1.Update()
        c1.Write()

        # Canvas 2: Data vs MC after reweighting
        c2 = ROOT.TCanvas(f"c_after_{mass}", f"After reweighting - {mass}", 800, 600)
        h_mc_rw.Draw("HIST")
        h_data.Draw("EP SAME")
        leg2 = ROOT.TLegend(0.7, 0.15, 0.88, 0.25)
        leg2.SetTextSize(0.02)
        leg2.AddEntry(h_mc_rw, f"MC reweighted", "l")
        leg2.AddEntry(h_data, "Data", "p")
        leg2.Draw()
        c2.Modified()
        c2.Update()
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
        c_w = ROOT.TCanvas(f"c_weights_{mass}", f"PU weights - {mass}", 800, 600)
        c_w.SetLogy()
        h_weights.Draw("HIST")
        h_weights.SetMinimum(0.1)
        h_weights.SetMaximum(100.0)
        c_w.Modified()
        c_w.Update()
        c_w.SaveAs(f"{out_dir_year}/h_weights_{mass}_{Era}.pdf")
        c_w.SaveAs(f"{out_dir_year}/h_weights_{mass}_{Era}.png")
        c_w.Close()

        c_r = ROOT.TCanvas(f"c_ratio_{mass}", f"Ratio MC_rw/Data - {mass}", 800, 600)
        h_ratio.Draw("HIST")
        c_r.Modified()
        c_r.Update()
        c_r.SaveAs(f"{out_dir_year}/h_ratio_{mass}_{Era}.pdf")
        c_r.SaveAs(f"{out_dir_year}/h_ratio_{mass}_{Era}.png")
        c_r.Close()

fout.Close()
print(f"\nSaved PU plots and histograms for 2016, 2017, and 2018 in {output_root}")



