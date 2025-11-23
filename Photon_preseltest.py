#!/usr/bin/env python3
import os
import awkward as ak
import ROOT
import numpy as np

ROOT.gStyle.SetOptStat(0)
ROOT.gROOT.SetBatch(True)

# ======================
# CONFIGURATION
# ======================

outdir = "/eos/user/b/bbapi/www/Analysis_plots/Overlay_plots/"


presel_path = '/eos/user/b/bbapi/My_Analysis/NTuples_pretest_full/'
common_path_M20 = 'M20-RunIISummer20UL18NanoAODv2/nominal/'

# Define all preselections (ordered)
presel_stages = {
    "Without_presel_trigger": "Without_presel_trigger/" + common_path_M20,
    "Without_presel": "Without_presel/" + common_path_M20,
    "pT_eta_cut": "pT_eta_cut/" + common_path_M20,
    "electron_mva": "electron_mva/" + common_path_M20,
    "electron_pf_iso": "electron_pf_iso/" + common_path_M20,
    "Muon_pf_iso": "Muon_pf_iso/" + common_path_M20,
    "one_electron": "one_electron/" + common_path_M20,
    "b_tagged": "b_tagged/" + common_path_M20,
    "photon_mva": "photon_mva/" + common_path_M20,
    "photon_pixel_seed": "photon_pixel_seed/" + common_path_M20,
    "dr_ele_pho": "dr_ele_pho/" + common_path_M20,
}

Log = True

# Variables to plot
plot_vars = [
    ("pholead_pt", 100, 0, 200, "Leading photon p_{T} [GeV]", Log),
    ("pholead_eta", 100, -3, 3, "Leading photon #eta", Log),
    ("pholead_phi", 64, -3.2, 3.2, "Leading photon #phi", Log),
    ("pholead_mvaID", 100, -1, 1, "Leading photon MVA ID", Log),
    ("pholead_sieie", 100, 0, 0.07, "Leading photon #sigma_{i#etai#eta}", Log),
    ("pholead_r9", 100, 0, 1.5, "Leading photon R9", Log),
    ("pholead_hoe", 100, 0, 0.6, "Leading photon H/E", Log),

    ("phosublead_pt", 100, 0, 200, "Subleading photon p_{T} [GeV]", Log),
    ("phosublead_eta", 100, -3, 3, "Subleading photon #eta", Log),
    ("phosublead_phi", 64, -3.2, 3.2, "Subleading photon #phi", Log),
    ("phosublead_mvaID", 100, -1, 1, "Subleading photon MVA ID", Log),
    ("phosublead_sieie", 100, 0, 0.07, "Subleading photon #sigma_{i#etai#eta}", Log),
    ("phosublead_r9", 100, 0, 1.5, "Subleading photon R9", Log),
    ("phosublead_hoe", 100, 0, 0.6, "Subleading photon H/E", Log),
    ("Nphotons", 10, 0, 10, "Number of photons per event", Log),

    ("dipho_mass", 200, 15, 25, "diphoton inv mass distribution", Log),
    ("dijet_mass", 200, 0, 50, "diJet inv mass distribution", Log),
    ("mass", 200, 0, 200, "bbgg inv mass distribution", Log),
]

# ======================
# FUNCTIONS
# ======================

def plot_overlay(varname, title, bins, arrays_dict, outroot, outdir, logy = True):
    nbins, xmin, xmax = bins
    c = ROOT.TCanvas(f"c_{varname}", "", 800, 700)
    legend = ROOT.TLegend(0.6, 0.7, 0.88, 0.88)
    colors = [ROOT.kRed, ROOT.kBlue, ROOT.kGreen+1, ROOT.kMagenta,
              ROOT.kOrange+1, ROOT.kCyan+1, ROOT.kPink+6, ROOT.kAzure+2,
              ROOT.kViolet+2, ROOT.kGray+2]

    hist_list = []
    max_y = 0

    # Loop over preselections
    for i, (name, arr) in enumerate(arrays_dict.items()):
        if arr is None:
            continue

        hname = f"h_{varname}_{name}"
        hist = ROOT.TH1F(hname, title, nbins, xmin, xmax)
        hist.SetMinimum(0.1)
        hist.SetLineColor(colors[i % len(colors)])
        hist.SetLineWidth(2)

        try:
            vals = ak.flatten(arr[varname], axis=None)
        except KeyError:
            print(f" {varname} not found in {name}, skipping.")
            continue

        for v in vals:
            hist.Fill(v)

        if hist.GetMaximum() > max_y:
            max_y = hist.GetMaximum()

        hist_list.append((hist, name))

    if logy:
        c.SetLogy()

    # Draw
    first = True
    for hist, name in hist_list:
        hist.GetXaxis().SetTitle(title)
        hist.GetYaxis().SetTitle("Events")
        hist.SetMaximum(1.3 * max_y)
        hist.SetTitle("")
        if first:
            hist.Draw("HIST")
            first = False
        else:
            hist.Draw("HIST SAME")
        legend.AddEntry(hist, name, "l")

    legend.Draw()
    c.Update()

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    c.SaveAs(os.path.join(outdir, f"{varname}.pdf"))
    c.SaveAs(os.path.join(outdir, f"{varname}.png"))

    outroot.cd()
    for hist, _ in hist_list:
        hist.Write()

    c.Close()


# ======================
# MAIN
# ======================

if __name__ == "__main__":
    print("Loading preselections...")
    arrays_dict = {}
#     for key, path in presel_stages.items():
#         full_path = os.path.join(presel_path, path)
#         if not os.path.exists(full_path):
#             print(f"Missing file: {full_path}")
#             arrays_dict[key] = None
#             continue
#         arrays_dict[key] = ak.from_parquet(full_path)
#         print(f"  → Loaded {key} ({len(arrays_dict[key])} events)")
    for key, path in presel_stages.items():
        full_path = os.path.join(presel_path, path)
        if not os.path.exists(full_path):
            print(f"Missing file: {full_path}")
            arrays_dict[key] = None
            continue

        arr = ak.from_parquet(full_path)
        arrays_dict[key] = arr

        # Count events with at least one leading photon (pholead_pt exists and has length>0)
        if "pholead_pt" in arr.fields:
            nevents_with_photons = len(arr)
        else:
            nevents_with_photons = 0

        # Optional: also print total rows (for debugging)
        total_rows = len(arr)
        print(f"  → Loaded {key}: {nevents_with_photons} events with ≥1 photon (total rows: {total_rows})")




    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outroot = ROOT.TFile(os.path.join(outdir, "overlay_plots.root"), "RECREATE")

    # ================
    # Number of photons per event
    # ================
    # print("Plotting photon multiplicity overlay...")
    # photon_counts = {}
    # for name, arr in arrays_dict.items():
    #     if arr is None:
    #         continue
    #     if "pholead_pt" in arr.fields:
    #         photon_counts[name] = ak.num(arr["pholead_pt"])
    #     else:
    #         print(f"No photon branch in {name}")
    #         photon_counts[name] = None

    # nbins, xmin, xmax = (10, 0, 10)
    # c = ROOT.TCanvas("c_nPhotons", "", 800, 700)
    # legend = ROOT.TLegend(0.6, 0.7, 0.88, 0.88)
    # colors = [ROOT.kRed, ROOT.kBlue, ROOT.kGreen+1, ROOT.kMagenta,
    #           ROOT.kOrange+1, ROOT.kCyan+1, ROOT.kPink+6, ROOT.kAzure+2,
    #           ROOT.kViolet+2, ROOT.kGray+2]
    # max_y = 0
    # hist_list = []
    # for i, (name, vals) in enumerate(photon_counts.items()):
    #     if vals is None:
    #         continue
    #     h = ROOT.TH1F(f"h_nPhotons_{name}", "Photon multiplicity", nbins, xmin, xmax)
    #     h.SetLineColor(colors[i % len(colors)])
    #     h.SetLineWidth(2)
    #     for v in ak.flatten(vals, axis=None):
    #         h.Fill(v)
    #     if h.GetMaximum() > max_y:
    #         max_y = h.GetMaximum()
    #     hist_list.append((h, name))
    # first = True
    # for h, name in hist_list:
    #     h.GetXaxis().SetTitle("Number of photons per event")
    #     h.GetYaxis().SetTitle("Events")
    #     h.SetMaximum(1.3 * max_y)
    #     h.SetTitle("")
    #     if first:
    #         h.Draw("HIST")
    #         first = False
    #     else:
    #         h.Draw("HIST SAME")
    #     legend.AddEntry(h, name, "l")
    # legend.Draw()
    # c.SaveAs(os.path.join(outdir, "nPhotons.pdf"))
    # c.SaveAs(os.path.join(outdir, "nPhotons.png"))
    # outroot.cd()
    # for h, _ in hist_list:
    #     h.Write()
    # c.Close()

    # ================
    # Loop over photon vars
    # ================
    for varname, nbins, xmin, xmax, title, logy in plot_vars:
        print(f"Plotting overlay for {varname} ...")
        plot_overlay(varname, title, (nbins, xmin, xmax), arrays_dict, outroot, outdir, logy)

    outroot.Close()
    print(f"\nAll overlay plots saved to {outdir}/ and overlay_plots.root\n")
