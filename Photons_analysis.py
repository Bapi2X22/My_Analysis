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

    gen = events.GenPart

    gen_photons = gen[(gen.pdgId == 22) & (gen.status == 1)]

    mother_idx = gen_photons.genPartIdxMother
    from_a_mask = gen[mother_idx].pdgId == 35
    gen_photons_from_a = gen_photons[from_a_mask]

    photons = events.Photon 

    abs_eta = np.abs(photons.eta)

    # Barrel–endcap transition exclusion (1.442 ≤ |η| ≤ 1.566)
    valid_eta = (abs_eta <= 2.5) & ~((abs_eta >= 1.442) & (abs_eta <= 1.566))

    # Barrel vs Endcap ID cuts
    is_barrel = abs_eta < 1.442
    is_endcap = (abs_eta > 1.566) & (abs_eta < 2.5)

    # Apply region-specific MVA thresholds
    barrel_cut = is_barrel & (photons.mvaID > -0.02)
    endcap_cut = is_endcap & (photons.mvaID > -0.26)

    # Combine everything
    good_photons = (
        (photons.pt > 10)
        & valid_eta
        & (barrel_cut | endcap_cut)
        & (~photons.pixelSeed)
    )

    # good_photons = (
    #     (photons.pt > 10)
    #     & (np.abs(photons.eta) < 2.5)
    #     & (photons.mvaID > -0.02)
    # )
    selected_photons = photons[good_photons]
    at_least_two_photons = ak.num(selected_photons) >= 2

    # dR_before = delta_r_manual(photons, gen_photons_from_a)
    # --- sort photons by descending pT ---
    ordered_photons_before = photons[ak.argsort(photons.pt, axis=1, ascending=False)]

    # # --- leading and subleading photons ---
    # lead_photon = ak.firsts(ordered_photons)
    # sublead_photon = ak.firsts(ordered_photons[:, 1])

    # pad each event to have at least 2 entries
    ordered_photons_padded_before = ak.pad_none(ordered_photons_before, 2)  

    # leading and subleading photons
    lead_photon_before = ak.firsts(ordered_photons_padded_before)      # first photon
    sublead_photon_before = ordered_photons_padded_before[:, 1] 

    dR_before_lead = delta_r_manual(lead_photon_before, gen_photons_from_a)
    dR_before_sublead = delta_r_manual(sublead_photon_before, gen_photons_from_a)

    empty_photons = ak.Array([[]] * len(events))

    event_mask = at_least_two_photons

    filtered_photons = ak.where(event_mask, selected_photons, empty_photons)

    # --- sort photons by descending pT ---
    ordered_photons = filtered_photons[ak.argsort(filtered_photons.pt, axis=1, ascending=False)]

    # # --- leading and subleading photons ---
    # lead_photon = ak.firsts(ordered_photons)
    # sublead_photon = ak.firsts(ordered_photons[:, 1])

    # pad each event to have at least 2 entries
    ordered_photons_padded = ak.pad_none(ordered_photons, 2)  

    # leading and subleading photons
    lead_photon = ak.firsts(ordered_photons_padded)      # first photon
    sublead_photon = ordered_photons_padded[:, 1] 

    dR_after_lead = delta_r_manual(lead_photon, gen_photons_from_a)
    dR_after_sublead = delta_r_manual(sublead_photon, gen_photons_from_a)

    photons_pt = ak.flatten(filtered_photons.pt)
    photons_eta = ak.flatten(filtered_photons.eta)
    photons_phi = ak.flatten(filtered_photons.phi)
    photons_sieie = ak.flatten(filtered_photons.sieie)
    photons_r9 = ak.flatten(filtered_photons.r9)
    photons_hoe = ak.flatten(filtered_photons.hoe)
    photons_mvaID = ak.flatten(filtered_photons.mvaID)
    # photons_pt = lead_photon.pt
    # photons_eta = lead_photon.eta
    # photons_phi = lead_photon.phi
    # photons_sieie = lead_photon.sieie
    # photons_r9 = lead_photon.r9
    # photons_hoe = lead_photon.hoe
    # photons_mvaID = lead_photon.mvaID
    dR_before_lead = flatten_min_array(dR_before_lead)
    dR_before_sublead = flatten_min_array(dR_before_sublead)
    dR_after_lead = flatten_min_array(dR_after_lead)
    dR_after_sublead = flatten_min_array(dR_after_sublead)
    nPhotons = ak.num(filtered_photons)
    nPhotons = nPhotons[nPhotons> 0]

    plot_list = [
        {"name": "Photon_pt_pixel", "array": photons_pt, "bins": (100, 0, 200), "title": "Photon p_{T} [GeV] after pT, eta, mvaID cut, pixel and nPhoton>=2", "legend": "Photon pT"},
        {"name": "Photon_eta_pixel", "array": photons_eta, "bins": (100, 1, -1), "title": "Photon #eta after pT, eta, mvaID cut, pixel and nPhoton>2", "legend": "Photon #eta"},
        {"name": "Photon_phi_pixel", "array": photons_phi, "bins": (100, 1, -1), "title": "Photon #phi after pT, eta, mvaID cut, pixel and nPhoton>=2", "legend": "Photon #phi"},
        {"name": "Photon_mvaID_pixel", "array": photons_mvaID, "bins": (100, 1, -1), "title": "Photon mvaID after pT, eta, mvaID cut, pixel and nPhoton>=2", "legend": "Photon mvaID"},
        {"name": "Photon_sieie_pixel", "array": photons_sieie, "bins": (100, 0, 0.07), "title": "Photon #sigma_{i#eta i#eta} after pT, eta, pT, eta, mvaID cut, pixel and nPhoton>=2", "legend": "Photon #sigma_{i#eta i#eta}", "log_scale": True},
        {"name": "Photon_r9_pixel", "array": photons_r9, "bins": (100, 0, 1.5), "title": "Photon r9 after pT, eta, pT, eta, mvaID cut, pixel and nPhoton>=2", "legend": "Photon r9", "log_scale": True},
        {"name": "Photon_hoe_pixel", "array": photons_hoe, "bins": (100, 0, 0.6), "title": "Photon H/E after pT, eta, pT, eta, mvaID cut, pixel and nPhoton>=2", "legend": "Photon H/E", "log_scale": True},
        {"name": "dR_before_lead", "array": dR_before_lead, "bins": (100, 0, 10), "title": "dR(lead pho, gen pho) before", "legend": "dR_before_lead"},
        {"name": "dR_before_sublead", "array": dR_before_sublead, "bins": (100, 0, 10), "title": "dR(sublead pho, gen pho) before", "legend": "dR_before_sublead"},
        {"name": "dR_after_lead", "array": dR_after_lead, "bins": (100, 0, 10), "title": "dR(lead pho, gen pho) after pT, eta, mvaID cut, pixel and nPhoton>=2", "legend": "dR_after_lead"},
        {"name": "dR_after_sublead", "array": dR_after_sublead, "bins": (100, 0, 10), "title": "dR(sublead pho, gen pho) after pT, eta, mvaID cut, pixel and nPhoton>=2", "legend": "dR_after_sublead"},
        {"name": "dR_before_lead_log", "array": dR_before_lead, "bins": (100, 0, 10), "title": "dR(lead pho, gen pho) before", "legend": "dR_before_lead", "log_scale": True},
        {"name": "dR_before_sublead_log", "array": dR_before_sublead, "bins": (100, 0, 10), "title": "dR(sublead pho, gen pho) before", "legend": "dR_before_sublead", "log_scale": True},
        {"name": "dR_after_lead_log", "array": dR_after_lead, "bins": (100, 0, 10), "title": "dR(lead pho, gen pho) after pT, eta, mvaID cut, pixel and nPhoton>=2", "legend": "dR_after_lead", "log_scale": True},
        {"name": "dR_after_sublead_log", "array": dR_after_sublead, "bins": (100, 0, 10), "title": "dR(sublead pho, gen pho) after pT, eta, mvaID cut, pixel and nPhoton>=2", "legend": "dR_after_sublead", "log_scale": True},
        {"name": "nPhoton_pixel", "array": nPhotons, "bins": (10, 0, 10), "title": "nPhotons per event after pT, eta, mvaID cut, pixel and nPhoton>=2", "legend": "nPhotons"},
    ]



#------------------------------------------------------Plots----------------------------------------------------------------------------------------------------------------

    # ======================================================
    # Output directory
    # ======================================================
    outdir = "/eos/user/b/bbapi/www/Analysis_plots/After_cuts/"
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
