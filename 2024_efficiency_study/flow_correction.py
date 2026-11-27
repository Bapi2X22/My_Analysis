import os
import ROOT
import awkward as ak

a=ak.from_parquet("/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/NTuples_2024_WH_rem_corr/merged/WH-2024M35/Resolved_merged.parquet")

MVA_VARIABLES = {

    "pholead_mvaID": {
        "compare_with": "pholead_mvaID_nano",
        "range": (-1, 1),
        "nbins": 40,
        "xlabel": "Leading photon MVA ID",
        "label1": "MVA ID",
        "label2": "NanoAOD MVA ID",
    },

    "phosublead_mvaID": {
        "compare_with": "phosublead_mvaID_nano",
        "range": (-1, 1),
        "nbins": 40,
        "xlabel": "Subleading photon MVA ID",
        "label1": "MVA ID",
        "label2": "NanoAOD MVA ID",
    },
}

PHOTON_VARIABLES = {

    # =========================
    # Leading photon
    # =========================
    "pholead_r9": {
        "range": (0, 1.2),
        "nbins": 30,
        "xlabel": "Leading photon R_{9}",
    },
    "pholead_s4": {
        "range": (0, 1.2),
        "nbins": 30,
        "xlabel": "Leading photon S_{4}",
    },
    "pholead_sieie": {
        "range": (0, 0.05),
        "nbins": 30,
        "xlabel": "Leading photon #sigma_{i#etai#eta}",
    },
    "pholead_sieip": {
        "range": (-0.005, 0.005),
        "nbins": 30,
        "xlabel": "Leading photon #sigma_{i#etai#phi}",
    },
    "pholead_hoe": {
        "range": (0, 0.2),
        "nbins": 30,
        "xlabel": "Leading photon H/E",
    },
    "pholead_etaWidth": {
        "range": (0, 0.1),
        "nbins": 30,
        "xlabel": "Leading photon #eta width",
    },
    "pholead_phiWidth": {
        "range": (0, 0.2),
        "nbins": 30,
        "xlabel": "Leading photon #phi width",
    },

    # Isolation
    "pholead_pfChargedIso": {
        "range": (0, 20),
        "nbins": 40,
        "xlabel": "Leading photon charged isolation [GeV]",
    },
    "pholead_pfChargedIsoWorstVtx": {
        "range": (0, 20),
        "nbins": 40,
        "xlabel": "Leading photon charged isolation (worst vertex) [GeV]",
    },
    "pholead_pfPhoIso03": {
        "range": (0, 20),
        "nbins": 40,
        "xlabel": "Leading photon photon isolation [GeV]",
    },
    "pholead_ecalPFClusterIso": {
        "range": (0, 20),
        "nbins": 40,
        "xlabel": "Leading photon ECAL PF cluster isolation [GeV]",
    },
    "pholead_hcalPFClusterIso": {
        "range": (0, 20),
        "nbins": 40,
        "xlabel": "Leading photon HCAL PF cluster isolation [GeV]",
    },

    # =========================
    # Subleading photon
    # =========================
    "phosublead_r9": {
        "range": (0, 1.2),
        "nbins": 30,
        "xlabel": "Subleading photon R_{9}",
    },
    "phosublead_s4": {
        "range": (0, 1.2),
        "nbins": 30,
        "xlabel": "Subleading photon S_{4}",
    },
    "phosublead_sieie": {
        "range": (0, 0.05),
        "nbins": 30,
        "xlabel": "Subleading photon #sigma_{i#etai#eta}",
    },
    "phosublead_sieip": {
        "range": (-0.005, 0.005),
        "nbins": 30,
        "xlabel": "Subleading photon #sigma_{i#etai#phi}",
    },
    "phosublead_hoe": {
        "range": (0, 0.2),
        "nbins": 30,
        "xlabel": "Subleading photon H/E",
    },
    "phosublead_etaWidth": {
        "range": (0, 0.1),
        "nbins": 30,
        "xlabel": "Subleading photon #eta width",
    },
    "phosublead_phiWidth": {
        "range": (0, 0.2),
        "nbins": 30,
        "xlabel": "Subleading photon #phi width",
    },

    # Isolation
    "phosublead_pfChargedIso": {
        "range": (0, 20),
        "nbins": 40,
        "xlabel": "Subleading photon charged isolation [GeV]",
    },
    "phosublead_pfChargedIsoWorstVtx": {
        "range": (0, 20),
        "nbins": 40,
        "xlabel": "Subleading photon charged isolation (worst vertex) [GeV]",
    },
    "phosublead_pfPhoIso03": {
        "range": (0, 20),
        "nbins": 40,
        "xlabel": "Subleading photon photon isolation [GeV]",
    },
    "phosublead_ecalPFClusterIso": {
        "range": (0, 20),
        "nbins": 40,
        "xlabel": "Subleading photon ECAL PF cluster isolation [GeV]",
    },
    "phosublead_hcalPFClusterIso": {
        "range": (0, 20),
        "nbins": 40,
        "xlabel": "Subleading photon HCAL PF cluster isolation [GeV]",
    },
}




def compare_raw_corrected(
    df,
    variables_config,
    weight_col="weight",
    output_dir="Raw_vs_Corrected",
    mode="raw",
):

    ROOT.gStyle.SetOptStat(0)
    ROOT.gROOT.SetBatch(True)

    os.makedirs(output_dir, exist_ok=True)

    for var, config in variables_config.items():

        # Select comparison branches
        if mode == "raw":

            if var.startswith("pholead_"):
                compare_var = var.replace(
                    "pholead_", "pholead_raw_", 1
                )
            elif var.startswith("phosublead_"):
                compare_var = var.replace(
                    "phosublead_", "phosublead_raw_", 1
                )
            else:
                continue

            label1 = "Corrected"
            label2 = "Raw"

        elif mode == "mva":

            compare_var = config["compare_with"]

            label1 = config.get("label1", "MVA ID")
            label2 = config.get("label2", "NanoAOD MVA ID")

        else:
            raise ValueError(
                "mode must be 'raw' or 'mva'"
            )

        # Check branches
        if var not in df.fields or compare_var not in df.fields:
            print(f"Skipping {var}: missing comparison branch")
            continue

        if weight_col not in df.fields:
            raise KeyError(f"Missing weight branch: {weight_col}")

        # Configuration
        x_min, x_max = config["range"]
        nBins = config["nbins"]
        xlabel = config["xlabel"]

        # Initialize histograms
        h_corr = ROOT.TH1F(
            f"h_corr_{var}", "", nBins, x_min, x_max
        )
        h_raw = ROOT.TH1F(
            f"h_raw_{var}", "", nBins, x_min, x_max
        )

        h_corr.Sumw2()
        h_raw.Sumw2()

        # Fill weighted histograms
        for corr, other, weight in zip(
            df[var],
            df[compare_var],
            df[weight_col],
        ):
            corr = float(corr)
            other = float(other)
            weight = float(weight)

            if not (
                ROOT.TMath.Finite(corr)
                and ROOT.TMath.Finite(other)
                and ROOT.TMath.Finite(weight)
            ):
                continue

            scale_factor = 1.457 * 36.33 * 1000

            h_corr.Fill(corr, weight * scale_factor)
            h_raw.Fill(other, weight * scale_factor)

            # corr = float(corr)
            # raw = float(raw)
            # weight = float(weight)

            # if not (
            #     ROOT.TMath.Finite(corr)
            #     and ROOT.TMath.Finite(raw)
            #     and ROOT.TMath.Finite(weight)
            # ):
            #     continue

            # h_corr.Fill(corr, weight)
            # h_raw.Fill(raw, weight)

        # Canvas
        canvas = ROOT.TCanvas(f"c_{var}", "", 800, 800)

        pad1 = ROOT.TPad(f"pad1_{var}", "", 0, 0.30, 1, 1)
        pad2 = ROOT.TPad(f"pad2_{var}", "", 0, 0, 1, 0.30)

        pad1.SetBottomMargin(0.02)
        pad1.SetLeftMargin(0.13)
        pad1.SetRightMargin(0.05)

        pad2.SetTopMargin(0.03)
        pad2.SetBottomMargin(0.32)
        pad2.SetLeftMargin(0.13)
        pad2.SetRightMargin(0.05)

        pad1.Draw()
        pad2.Draw()

        # Upper panel
        pad1.cd()
        pad1.SetLogy(True)

        h_corr.SetLineColor(ROOT.kBlue + 1)
        h_raw.SetLineColor(ROOT.kRed + 1)

        h_corr.SetLineWidth(2)
        h_raw.SetLineWidth(2)
        h_raw.SetLineStyle(2)

        h_corr.SetTitle("")
        h_corr.GetYaxis().SetTitle("Weighted events")
        h_corr.GetYaxis().SetTitleSize(0.045)
        h_corr.GetYaxis().SetLabelSize(0.04)
        h_corr.GetYaxis().SetTitleOffset(1.3)
        h_corr.GetXaxis().SetLabelSize(0)

        max_y = max(h_corr.GetMaximum(), h_raw.GetMaximum())
        h_corr.SetMinimum(0.001)
        h_corr.SetMaximum(max_y * 100)

        h_corr.Draw("HIST")
        h_raw.Draw("HIST SAME")

        legend = ROOT.TLegend(0.62, 0.76, 0.88, 0.88)
        legend.SetBorderSize(0)
        legend.SetFillStyle(0)
        legend.AddEntry(h_corr, label1, "l")
        legend.AddEntry(h_raw, label2, "l")
        legend.Draw()

        # Lower panel: Corrected / Raw
        pad2.cd()

        h_ratio = h_corr.Clone(f"h_ratio_{var}")
        h_ratio.Divide(h_raw)

        h_ratio.SetTitle("")
        h_ratio.SetLineColor(ROOT.kBlack)
        h_ratio.SetMarkerColor(ROOT.kBlack)
        h_ratio.SetMarkerStyle(20)
        h_ratio.SetMarkerSize(0.7)

        h_ratio.SetMinimum(0.5)
        h_ratio.SetMaximum(1.5)

        h_ratio.GetYaxis().SetTitle(f"{label1} / {label2}")
        h_ratio.GetXaxis().SetTitle(xlabel)

        h_ratio.GetYaxis().SetNdivisions(505)
        h_ratio.GetYaxis().SetTitleSize(0.10)
        h_ratio.GetYaxis().SetLabelSize(0.085)
        h_ratio.GetYaxis().SetTitleOffset(0.55)

        h_ratio.GetXaxis().SetTitleSize(0.12)
        h_ratio.GetXaxis().SetLabelSize(0.10)
        h_ratio.GetXaxis().SetTitleOffset(1.05)

        h_ratio.Draw("E1")

        line = ROOT.TLine(x_min, 1.0, x_max, 1.0)
        line.SetLineStyle(2)
        line.SetLineColor(ROOT.kRed + 1)
        line.Draw("SAME")

        # Save plots
        canvas.cd()
        canvas.Update()

        canvas.SaveAs(
            f"{output_dir}/{var}_raw_vs_corrected.png"
        )
        canvas.SaveAs(
            f"{output_dir}/{var}_raw_vs_corrected.pdf"
        )

        print(f"Saved: {var}")

        # Cleanup
        canvas.Close()

        del h_corr, h_raw, h_ratio, legend, line, pad1, pad2


compare_raw_corrected(
    df=a,
    variables_config=PHOTON_VARIABLES,
    weight_col="weight",
    output_dir="/eos/user/b/bbapi/www/Analysis_plots/Flow_corrections/Raw_vs_Corrected/",
)

compare_raw_corrected(
    df=a,
    variables_config=MVA_VARIABLES,
    weight_col="weight",
    output_dir="/eos/user/b/bbapi/www/Analysis_plots/Flow_corrections/Raw_vs_Corrected/",
    mode="mva",
)