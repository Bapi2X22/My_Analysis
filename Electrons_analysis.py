#!/usr/bin/env python3
import os
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import ROOT
import coffea

# Make ROOT batch mode and clean styling
ROOT.gStyle.SetOptStat(0)
ROOT.gROOT.SetBatch(True)

def delta_r_manual(obj1, obj2):
    deta = obj1.eta[:, None] - obj2.eta
    dphi = np.abs(obj1.phi[:, None] - obj2.phi)
    dphi = ak.where(dphi > np.pi, 2 * np.pi - dphi, dphi)
    return np.sqrt(deta**2 + dphi**2)

def flatten_min_array(nested_array):
    """
    Take a nested/variable-length Awkward array (or list of lists),
    for each sublist:
      - take the smallest number ignoring None
      - skip entirely None sublists
    Return a flat 1D NumPy array of the results.
    """
    # Convert to awkward array if not already
    arr = ak.Array(nested_array)
    
    # Replace None with np.inf
    arr_filled = ak.fill_none(arr, np.inf)
    
    # Take minimum along the last axis
    min_per_list = ak.min(arr_filled, axis=-1)
    
    # Keep only finite entries (skip fully None sublists)
    min_per_list = min_per_list[min_per_list != np.inf]
    
    # Convert to flat NumPy array
    flat_array = ak.to_numpy(min_per_list)
    
    return flat_array


def plot_histogram(varname, values, bins, title, legend, outdir, outroot, log_scale = False):
    nbins, xmin, xmax = bins
    hname = f"h_{varname}"
    hist = ROOT.TH1F(hname, title, nbins, xmin, xmax)
    hist.GetXaxis().SetTitle(title)
    hist.GetYaxis().SetTitle("Events")

    # Convert Awkward → NumPy if needed
    if isinstance(values, ak.Array):
        values = ak.flatten(values, axis=None)
    values = np.asarray(values)

    for val in values:
        hist.Fill(val)

    # Make canvas
    c = ROOT.TCanvas(f"c_{varname}", "", 800, 700)
    hist.SetLineWidth(2)
    hist.SetLineColor(ROOT.kBlue + 1)
    hist.Draw("HIST")

    legend_box = ROOT.TLegend(0.65, 0.75, 0.88, 0.88)
    legend_box.AddEntry(hist, legend, "l")
    legend_box.Draw()

    if log_scale:
        c.SetLogy()

    # Save to files
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    pdf_path = os.path.join(outdir, f"{varname}.pdf")
    png_path = os.path.join(outdir, f"{varname}.png")
    c.SaveAs(pdf_path)
    c.SaveAs(png_path)

    outroot.cd()
    hist.Write()
    c.Close()


def main():


#------------------------------------------------------Define events------------------------------------------------------------------------------------------------------

    single = '/eos/user/b/bbapi/CMSSW_13_3_1_patch1/src/NanoAODProduction/NanoAODv13/test/M20-RunIISummer20UL18NanoAODv2/B7D21F37-159B-4F42-B1A8-97F4732749B9.root'

    factory = NanoEventsFactory.from_root(
        f"{single}:Events",
        schemaclass=NanoAODSchema,
    )
    events = factory.events()

    events = ak.materialize(events)

#------------------------------------------------------Do analysis---------------------------------------------------------------------------------------------------------

    # good_electrons = (
    #     (events.Electron.pt > 33.0)
    #     & (np.abs(events.Electron.eta) < 2.5)
    #     & (events.Electron.pfRelIso03_all < 0.15)
    # )

    electrons = events.Electron

    good_electrons = (
        (electrons.pt > 33.0) &
        (np.abs(electrons.eta) < 2.5) &  # keep within tracker acceptance
        ~((np.abs(electrons.eta) > 1.44) & (np.abs(electrons.eta) < 1.57)) &  # remove transition
        (electrons.mvaIso_WP80) &        # tight MVA ID
        (electrons.pfRelIso03_all < 0.15)  # isolation cut
    )

    one_ele = ak.num(events.Electron[good_electrons]) == 1

    selected_electrons = events.Electron[good_electrons]

    print("selected_electrons", len(selected_electrons[ak.num(selected_electrons.pt)>0]))

    empty_electrons = ak.Array([[]] * len(events))

    event_mask = one_ele

    filtered_electrons = ak.where(event_mask, selected_electrons, empty_electrons)

    electrons_pt = ak.flatten(filtered_electrons.pt)
    electrons_eta = ak.flatten(filtered_electrons.eta)
    electrons_phi = ak.flatten(filtered_electrons.phi)
    electrons_mvaIso = ak.flatten(filtered_electrons.mvaIso)
    electrons_sieie = ak.flatten(filtered_electrons.sieie)
    electrons_r9 = ak.flatten(filtered_electrons.r9)
    electrons_hoe = ak.flatten(filtered_electrons.hoe)
    electrons_eta_in = ak.flatten(filtered_electrons.DeltaEtaInSC)
    electrons_phi_in = ak.flatten(filtered_electrons.DeltaPhiInSC)
    electrons_missingHits = ak.flatten(filtered_electrons.lostHits)
    electrons_pass_conv_veto = ak.flatten(filtered_electrons.convVeto)
    electrons_pfisorel = ak.flatten(filtered_electrons.pfRelIso03_all)
    nElectrons = ak.num(filtered_electrons)
    nElectrons = nElectrons[nElectrons> 0]  

    plot_list = [
        {"name": "Electron_pt_tight", "array": electrons_pt, "bins": (100, 0, 200), "title": "Electron p_{T} [GeV] after pT, eta, tight cut  and nElectron=1", "legend": "Electron pT"},
        {"name": "Electron_eta_tight", "array": electrons_eta, "bins": (100, 1, -1), "title": "Electron #eta after pT, eta, tight cut and nElectro=1", "legend": "Electron #eta"},
        {"name": "Electron_phi_tight", "array": electrons_phi, "bins": (100, 1, -1), "title": "Electron #phi after pT, eta, tight cut and nElectron=1", "legend": "Electron #phi"},
        {"name": "Electron_mvaIso_tight", "array": electrons_mvaIso, "bins": (100, 1, -1), "title": "Electron mvaIso after pT, eta, tight cut and nElectron=1", "legend": "Electron mvaIso"},
        {"name": "nElectron_tight", "array": nElectrons, "bins": (10, 0, 10), "title": "nElectrons per event after pT, eta, tight cut and nElectron=1", "legend": "nElectrons"},
        {"name": "Electron_sieie_tight", "array": electrons_sieie, "bins": (100, 0, 0.07), "title": "Electron #sigma_{i#eta i#eta} after pT, eta, tight cut and nElectron=1", "legend": "Electron #sigma_{i#eta i#eta}", "log_scale": True},
        {"name": "Electron_r9_tight", "array": electrons_r9, "bins": (100, 0, 6), "title": "Electron r9 after pT, eta, tight cut and nElectron=1", "legend": "Electron r9", "log_scale": True},
        {"name": "Electron_hoe_tight", "array": electrons_hoe, "bins": (100, 0, 10), "title": "Electron H/E after pT, eta, tight cut and nElectron=1", "legend": "Electron H/E", "log_scale": True},
        {"name": "Electron_eta_in_tight", "array": electrons_eta_in, "bins": (100, -0.05, 0.05), "title": "Electron #Delta#eta_{in} after pT, eta, tight cut and nElectron=1", "legend": "Electron #Delta#eta_{in}"},
        {"name": "Electron_phi_in_tight", "array": electrons_phi_in, "bins": (100, -0.2, 0.2), "title": "Electron #Delta#phi_{in} after pT, eta, tight cut and nElectron=1", "legend": "Electron #Delta#phi_{in}"},
        {"name": "Electron_missingHits_tight", "array": electrons_missingHits, "bins": (10, 0, 10), "title": "Electron missing hits after pT, eta, tight cut and nElectron=1", "legend": "Electron missing hits"},
        {"name": "Electron_pass_conv_veto_tight", "array": electrons_pass_conv_veto, "bins": (2, 0, 2), "title": "Electron pass conversion veto after pT, eta, tight cut and nElectron=1", "legend": "Electron pass conversion veto"},
        {"name": "Electron_pfisorel_tight", "array": electrons_pfisorel, "bins": (100, 0, 0.2), "title": "Electron pfisorel after pT, eta, tight cut and nElectron=1", "legend": "Electron pfisorel"},
    ]



#------------------------------------------------------Plots----------------------------------------------------------------------------------------------------------------

    # ======================================================
    # Output directory
    # ======================================================
    outdir = "/eos/user/b/bbapi/www/Analysis_plots/Electrons/After_cuts/"
    outroot_path = os.path.join(outdir, "Plots.root")

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    outroot = ROOT.TFile(outroot_path, "RECREATE")

    # ======================================================
    # Plot all arrays
    # ======================================================
    for cfg in plot_list:
        name = cfg["name"]
        arr = cfg["array"]
        log = cfg["log_scale"] if "log_scale" in cfg else False
        print(f"Plotting {name} ...")
        plot_histogram(name, arr, cfg["bins"], cfg["title"], cfg["legend"], outdir, outroot, log_scale=log)

    outroot.Close()

    print(f"\All array plots saved to:")
    print(f"   → ROOT file: {outroot_path}")
    print(f"   → PDF/PNG:   {outdir}/")


if __name__ == "__main__":
    main()