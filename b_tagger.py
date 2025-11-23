#!/usr/bin/env python3
import argparse
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from array import array
import ROOT
import os
import re
import sys

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetTextFont(42)

# ----------------------------------------
# ΔR helper
# ----------------------------------------
def delta_r_manual(obj1, obj2):
    deta = obj1.eta[:, None] - obj2.eta
    dphi = np.abs(obj1.phi[:, None] - obj2.phi)
    dphi = ak.where(dphi > np.pi, 2 * np.pi - dphi, dphi)
    return np.sqrt(deta**2 + dphi**2)

def photon_preselections_ggH_BBGG(
    photons: ak.Array,
    events: ak.Array,
    electron_veto=True,
    revert_electron_veto=False,
    year="2022",
    IsFlag=False):
    """
    Apply full preselection on leptons, jets, and photons.
    Finally return only photons from events that pass all criteria.
    """

    # print("Number of events before preselection:", len(events))
    print("Year:", year)

    #-------------------------
    # b-tagging working point
    #-------------------------
    if year == "2022":
        wp_medium = 0.2783
    elif year == "2022EE":
        wp_medium = 0.2783
    else:
        raise ValueError(f"Unknown year {year}")

    gen = events.GenPart

    bquarks = gen[abs(gen.pdgId) == 5]
    mother_idx = bquarks.genPartIdxMother
    from_a_mask = gen[mother_idx].pdgId == 35
    bquarks_from_a = bquarks[from_a_mask]

    # ------------------------
    # Jet selection
    # ------------------------
    good_jets = (
        (events.Jet.pt > 20.0)
        & (np.abs(events.Jet.eta) < 2.4)
        # & (events.Jet.btagDeepFlavB > wp_medium)
    )
    selected_bjets = events.Jet[good_jets]
    at_least_two_bjets = ak.num(selected_bjets) >= 2

    # ------------------------
    # Photon selection
    # ------------------------
    abs_eta = np.abs(photons.eta)

    # Barrel–endcap transition exclusion (1.442 ≤ |η| ≤ 1.566)
    valid_eta = (abs_eta <= 2.5) & ~((abs_eta >= 1.442) & (abs_eta <= 1.566))

    # Barrel vs Endcap ID cuts
    is_barrel = abs_eta < 1.442
    is_endcap = (abs_eta > 1.566) & (abs_eta < 2.5)

    # Apply region-specific MVA thresholds
    barrel_cut = is_barrel & (photons.mvaID > -0.02)
    endcap_cut = is_endcap & (photons.mvaID > -0.26)

    # Base photon selection
    good_photons = (
        (photons.pt > 10)
        & valid_eta
        & (barrel_cut | endcap_cut)
        & (~photons.pixelSeed)
    )
    selected_photons = photons[good_photons]

    # Sort photons by pT in descending order
    selected_photons = selected_photons[ak.argsort(selected_photons.pt, ascending=False)]

    # ------------------------
    # Lead / Sublead cuts
    # ------------------------
    # Define masks for events that have at least 2 photons
    has_two_photons = ak.num(selected_photons) >= 2

    # Safe indexing: fill with -inf if photon is missing
    lead_pt = ak.fill_none(ak.firsts(selected_photons.pt), -999)
    sublead_pt = ak.fill_none(ak.pad_none(selected_photons.pt, 2)[:, 1], -999)

    lead_cut = lead_pt > 30
    sublead_cut = sublead_pt > 18

    # Event must have 2 photons and satisfy both pT cuts
    photon_event_mask = has_two_photons & lead_cut & sublead_cut
    # photon_event_mask = has_two_photons & lead_cut

    # ------------------------
    # Combine jet + photon criteria
    # ------------------------
    event_mask = at_least_two_bjets & photon_event_mask
    # event_mask = photon_event_mask

    # Prepare empty arrays for masked-out events
    empty_photons = ak.Array([[]] * len(events))
    empty_bjets = ak.Array([[]] * len(events))
    empty_bquarks = ak.Array([[]] * len(events))

    # Keep only events passing full selection
    filtered_photons = ak.where(event_mask, selected_photons, empty_photons)
    filtered_jets = ak.where(event_mask, selected_bjets, empty_bjets)
    gen_bquark = ak.where(event_mask, bquarks_from_a, empty_bquarks)

    filtered_photons = filtered_photons[ak.num(filtered_photons.pt)>0]
    filtered_jets = filtered_jets[ak.num(filtered_jets.pt)>0]
    gen_bquark = gen_bquark[ak.num(gen_bquark.pt)>0]

    # print(f"Events passing full preselection: {ak.sum(event_mask)}")

    return filtered_photons, filtered_jets, gen_bquark


def extract_info_from_path(input_file):
    # Extract just the file name and directory name
    filename = os.path.basename(input_file)
    dirname = os.path.basename(os.path.dirname(input_file))
    combined_name = f"{dirname}_{filename}"

    # --- Extract mass point ---
    m = re.search(r'(?:M|_)(\d+)', combined_name)
    mass_point = f"M{m.group(1)}" if m else "UnknownMass"

    # --- Extract year (e.g. UL16APV, UL18, 2022, 2023, etc.) ---
    y = re.search(r'(UL)?(16|17|18|2016|2017|2018|2022|2023|2024|2025)(APV)?', combined_name, re.IGNORECASE)
    if y:
        prefix = "20" if len(y.group(2)) == 2 else ""
        apv = "APV" if y.group(3) else ""
        year = f"{prefix}{y.group(2)}{apv}"
    else:
        year = "UnknownYear"

    # --- Energy label ---
    if any(x in year for x in ["2016", "2017", "2018"]):
        energy_label = "13 TeV"
    else:
        energy_label = "13.6 TeV"

    return mass_point, year, energy_label
import ROOT
import os

def plot_njets(jets, outdir="./plots", outname="NJets", title="Jet Multiplicity", logy=False):
    """
    Plot the number of jets per event.

    Parameters
    ----------
    jets : ak.Array
        Awkward Array containing jets per event (e.g. from events.Jet).
    outdir : str, optional
        Output directory for saving the plot.
    outname : str, optional
        Base name for the output file (without extension).
    title : str, optional
        Histogram title.
    logy : bool, optional
        If True, sets y-axis to log scale.
    """
    import awkward as ak
    import numpy as np

    # if not os.path.exists(outdir):
    #     os.makedirs(outdir)

    # --- Compute number of jets per event ---
    nJets = ak.num(jets)
    nJets_np = ak.to_numpy(nJets)

    # --- Create ROOT histogram ---
    nbins = 10
    xmin, xmax = 0, 10
    h_njets = ROOT.TH1F("h_njets", f"{title};N_{jets};Events", nbins, xmin, xmax)

    for n in nJets_np:
        h_njets.Fill(n)

    # --- Style ---
    h_njets.SetLineWidth(2)
    h_njets.SetLineColor(ROOT.kBlue + 1)
    h_njets.SetFillColorAlpha(ROOT.kBlue - 9, 0.35)

    # --- Draw ---
    c = ROOT.TCanvas("c_njets", "", 800, 700)
    if logy:
        c.SetLogy()
    h_njets.Draw("HIST")
    h_njets.SetTitle(title)
    h_njets.GetXaxis().SetTitle("Number of jets")
    h_njets.GetYaxis().SetTitle("Events")

    output_dir = os.path.dirname(os.path.abspath(outdir))

    # --- Save ---
    outpath_png = os.path.join(output_dir, f"{outname}.png")
    outpath_pdf = os.path.join(output_dir, f"{outname}.pdf")
    c.SaveAs(outpath_png)
    c.SaveAs(outpath_pdf)
    # print(f"Saved plot: {outpath_png}")
    return h_njets


def draw_performance_curves(taggers, effs, mistags, input_file, output):
    # Create output directory if not present
    os.makedirs(os.path.dirname(output), exist_ok=True)

    mass_point, year, energy_label = extract_info_from_path(input_file)

    # Colors and styles
    colors = [ROOT.kGreen+2, ROOT.kBlack, ROOT.kBlue, ROOT.kRed, ROOT.kMagenta]
    markers = [20, 21, 22, 23, 33]

    c = ROOT.TCanvas("c", "", 700, 550)
    c.SetLogy()  # log scale on Y

    c.SetTopMargin(0.08)
    c.SetLeftMargin(0.13)
    c.SetRightMargin(0.04)
    c.SetBottomMargin(0.12)


    graphs = []

    for i, tagger in enumerate(taggers):
        g = ROOT.TGraph(len(effs[i]), array('f', effs[i]), array('f', mistags[i]))
        g.SetLineColor(colors[i % len(colors)])
        g.SetLineWidth(2)
        g.SetMarkerStyle(markers[i % len(markers)])
        g.SetMarkerColor(colors[i % len(colors)])
        graphs.append(g)

    # Draw first graph with axis
    graphs[0].Draw("AL")
    title = f"B-tagging Performance ({mass_point}, {year})"
    graphs[0].SetTitle(title)
    graphs[0].GetXaxis().SetTitle("Efficiency (fraction of events kept)")
    graphs[0].GetYaxis().SetTitle("Mistagging rate")
    graphs[0].GetYaxis().SetTitleOffset(1.2)
    graphs[0].GetXaxis().SetTitleOffset(1.1)

    # Overlay the rest
    for g in graphs[1:]:
        g.Draw("L SAME")

    # --- CMS Label ---
# --- CMS Label Block ---
    texCMS = ROOT.TLatex()
    texCMS.SetNDC()
    texCMS.SetTextFont(61)  # bold font
    texCMS.SetTextSize(0.06)
    texCMS.DrawLatex(0.13, 0.88, "CMS")

    texSim = ROOT.TLatex()
    texSim.SetNDC()
    texSim.SetTextFont(52)  # italic
    texSim.SetTextSize(0.045)
    texSim.DrawLatex(0.23, 0.88, "Simulation")

    # --- Energy label (right-aligned) ---
    texEnergy = ROOT.TLatex()
    texEnergy.SetNDC()
    texEnergy.SetTextFont(42)
    texEnergy.SetTextSize(0.045)
    texEnergy.SetTextAlign(31)  # right aligned
    texEnergy.DrawLatex(0.94, 0.88, energy_label)  # e.g. "√s = 13 TeV"

    # # --- Additional note ---
    # texNote = ROOT.TLatex()
    # texNote.SetNDC()
    # texNote.SetTextFont(42)
    # texNote.SetTextSize(0.04)
    # texNote.SetTextAlign(13)  # left-top alignment
    # texNote.DrawLatex(0.13, 0.82, "No kinematic cut is applied")

  # e.g. "√s = 13 TeV" or "√s = 13.6 TeV"

    # --- Axes titles ---
    graphs[0].GetXaxis().SetTitle("Efficiency (fraction of events kept)")
    graphs[0].GetYaxis().SetTitle("Mistagging rate")
    graphs[0].GetXaxis().SetTitleSize(0.045)
    graphs[0].GetYaxis().SetTitleSize(0.045)
    graphs[0].GetXaxis().SetLabelSize(0.04)
    graphs[0].GetYaxis().SetLabelSize(0.04)
    graphs[0].GetYaxis().SetTitleOffset(1.3)

    # --- Legend ---
    leg = ROOT.TLegend(0.55, 0.15, 0.85, 0.35)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextFont(42)
    leg.SetTextSize(0.04)
    for g, tagger in zip(graphs, taggers):
        leg.AddEntry(g, tagger, "l")
    leg.Draw()

    # --- Optional grid ---
    c.SetGridy(True)
    c.SetLogy()
    c.SetLeftMargin(0.12)
    c.SetBottomMargin(0.12)


    # Save
    c.SaveAs(output)
    print(f"Plot saved → {output}")


# ----------------------------------------
# Efficiency calculator for one threshold
# ----------------------------------------
def compute_eff(events, tagger, thr):
    gen = events.GenPart
    gen_b = gen[(abs(gen.pdgId) == 5) & (gen.status == 23)]

    mother_idx = gen_b.genPartIdxMother
    valid_mom = mother_idx >= 0
    idx_safe = ak.where(valid_mom, mother_idx, 0)
    mom_pdgid = gen[idx_safe].pdgId
    from_a_mask = valid_mom & (abs(mom_pdgid) == 35)

    gen_b_from_a = gen_b[from_a_mask]
    gen_b_from_a = gen_b_from_a[ak.argsort(gen_b_from_a.pt, axis=1, ascending=False)]

    lead_b = ak.firsts(gen_b_from_a)
    sublead_b = gen_b_from_a[:, 1]

    jets = events.Jet
    tagged_jets = jets[jets[tagger] > thr]

    jets_padded_before = ak.pad_none(jets, 2)
    dR_before_lead = delta_r_manual(lead_b, jets_padded_before)
    dR_before_sublead = delta_r_manual(sublead_b, jets_padded_before)
    dR_before = np.minimum(dR_before_lead, dR_before_sublead)
    dR_clean_before = ak.drop_none(dR_before)

    count_total_valid_before = ak.sum(ak.num(dR_clean_before))

    jets_padded = ak.pad_none(tagged_jets, 2)
    dR_after_lead = delta_r_manual(lead_b, jets_padded)
    dR_after_sublead = delta_r_manual(sublead_b, jets_padded)
    dR_after = np.minimum(dR_after_lead, dR_after_sublead)
    dR_clean = ak.drop_none(dR_after)

    count_below_0p4 = ak.sum(dR_clean < 0.4)
    count_total_valid = ak.sum(ak.num(dR_clean))

    success_rate = (count_below_0p4 / count_total_valid) if count_total_valid > 0 else 0
    efficiency = (count_total_valid / count_total_valid_before) if count_total_valid_before > 0 else 0

    return float(success_rate), float(efficiency)


def compute_eff_miss_tag_rate(events, tagger, thr):

    gen = events.GenPart

    gen_b = gen[(abs(gen.pdgId) == 5) & (gen.status == 23)]

    # 2) Require valid mother and mother pdgId == 35 (A)
    mother_idx  = gen_b.genPartIdxMother
    valid_mom   = mother_idx >= 0
    idx_safe    = ak.where(valid_mom, mother_idx, 0)          # avoid negative indexing
    mom_pdgid   = gen[idx_safe].pdgId
    from_a_mask = valid_mom & (abs(mom_pdgid) == 35)

    # Keep only b’s from A (still preserves event count; events with no matches → empty lists)
    gen_b_from_a = gen_b[from_a_mask]

    gen_b_from_a = gen_b_from_a[ak.argsort(gen_b_from_a.pt, axis=1, ascending=False)]

    # leading and subleading jets
    lead_b = ak.firsts(gen_b_from_a)      # first jet
    sublead_b = gen_b_from_a[:, 1] 

    jets = events.Jet

    # tagged_jets = jets[jets[tagger] > threshold]  

    dR_after_lead = delta_r_manual(lead_b, jets)
    dR_after_sublead = delta_r_manual(sublead_b, jets)

    lead_min_idx = ak.argmin(dR_after_lead, axis=1, keepdims=False)
    sublead_min_idx = ak.argmin(dR_after_sublead, axis=1, keepdims=False)


    dr_after_lead = dR_after_lead[ak.local_index(dR_after_lead) == lead_min_idx]
    dr_after_sublead = dR_after_sublead[ak.local_index(dR_after_lead) == sublead_min_idx]


    conflict = lead_min_idx == sublead_min_idx



    keep_first = dr_after_lead < dr_after_sublead
    conflict_b = ak.broadcast_arrays(keep_first, conflict)[1]

    # Step 2: safely fill with NaN (avoids None-related TypeErrors)
    dr1_new = ak.where(conflict_b, ak.where(keep_first, dr_after_lead, np.nan), dr_after_lead)
    dr2_new = ak.where(conflict_b, ak.where(~keep_first, dr_after_sublead, np.nan), dr_after_sublead)


    dr_combined = ak.concatenate([dr1_new, dr2_new], axis=1)
    mask_valid = ~ak.is_none(dr_combined) & (dr_combined == dr_combined)  # remove None and NaN
    dr_combined = dr_combined[mask_valid]


    jet_idx = ak.local_index(jets)

    lead_idx_valid = (lead_min_idx >= 0) & (lead_min_idx < ak.num(jets))
    sublead_idx_valid = (sublead_min_idx >= 0) & (sublead_min_idx < ak.num(jets))

    lead_min_idx  = ak.mask(lead_min_idx, lead_idx_valid)
    sublead_min_idx = ak.mask(sublead_min_idx, sublead_idx_valid)

    lead_single = ak.singletons(lead_min_idx)
    sublead_single = ak.singletons(sublead_min_idx)

    matched_idx = ak.fill_none(
        ak.concatenate([lead_single, sublead_single], axis=1),
        -1
    )

    mask_valid = matched_idx != -1
    matched_idx = ak.mask(matched_idx, mask_valid)

    # 🔸 Remove duplicates per event (safe for all Awkward versions)
    unique_idx = ak.Array([list(dict.fromkeys(ev)) for ev in ak.to_list(matched_idx)])

    matched_jets = jets[unique_idx]

    all_tagger_score = ak.flatten(matched_jets[tagger])

    all_dr_values = ak.flatten(ak.drop_none(dr_combined))

    # mask for valid dr values
    mask_valid = (all_dr_values == all_dr_values)  # removes NaNs

    # mask for ΔR < 0.4
    mask_dr = (all_dr_values < 0.4) & mask_valid

    mask_dr_false = (all_dr_values > 0.4) & mask_valid

    # mask for tagger threshold (example: > 0.5)
    mask_tag = (all_tagger_score > thr) & mask_valid

    # jets that pass both
    mask_both = mask_dr & mask_tag

    mask_both_false = mask_dr_false & mask_tag

    # flatten everything to count
    N_total = ak.sum(mask_dr)
    N_total_unmatched = ak.sum(mask_dr_false)
    N_pass  = ak.sum(mask_both)
    N_pass_false  = ak.sum(mask_both_false)

    efficiency = N_pass / N_total if N_total > 0 else float("nan")
    miss_tag_rate = N_pass_false/N_total_unmatched

    return efficiency, miss_tag_rate


def compute_eff(events, tagger, thr):
    gen = events.GenPart
    gen_b = gen[(abs(gen.pdgId) == 5) & (gen.status == 23)]

    mother_idx = gen_b.genPartIdxMother
    valid_mom = mother_idx >= 0
    idx_safe = ak.where(valid_mom, mother_idx, 0)
    mom_pdgid = gen[idx_safe].pdgId
    from_a_mask = valid_mom & (abs(mom_pdgid) == 35)

    gen_b_from_a = gen_b[from_a_mask]
    gen_b_from_a = gen_b_from_a[ak.argsort(gen_b_from_a.pt, axis=1, ascending=False)]

    lead_b = ak.firsts(gen_b_from_a)
    sublead_b = gen_b_from_a[:, 1]

    jets = events.Jet
    tagged_jets = jets[jets[tagger] > thr]

    jets_padded_before = ak.pad_none(jets, 2)
    dR_before_lead = delta_r_manual(lead_b, jets_padded_before)
    dR_before_sublead = delta_r_manual(sublead_b, jets_padded_before)
    dR_before = np.minimum(dR_before_lead, dR_before_sublead)
    dR_clean_before = ak.drop_none(dR_before)

    count_total_valid_before = ak.sum(ak.num(dR_clean_before))

    jets_padded = ak.pad_none(tagged_jets, 2)
    dR_after_lead = delta_r_manual(lead_b, jets_padded)
    dR_after_sublead = delta_r_manual(sublead_b, jets_padded)
    dR_after = np.minimum(dR_after_lead, dR_after_sublead)
    dR_clean = ak.drop_none(dR_after)

    count_below_0p4 = ak.sum(dR_clean < 0.4)
    count_total_valid = ak.sum(ak.num(dR_clean))

    success_rate = (count_below_0p4 / count_total_valid) if count_total_valid > 0 else 0
    efficiency = (count_total_valid / count_total_valid_before) if count_total_valid_before > 0 else 0

    return float(success_rate), float(efficiency)


def compute_eff_miss_tag_rate_after_sel(events, tagger, thr, outdir):

    photons = events.Photon

    photons, jets , gen_b_from_a = photon_preselections_ggH_BBGG(photons, events)

    plot_njets(jets, outdir)

    gen_b_from_a = gen_b_from_a[ak.argsort(gen_b_from_a.pt, axis=1, ascending=False)]

    # leading and subleading jets
    lead_b = ak.firsts(gen_b_from_a)      # first jet
    sublead_b = gen_b_from_a[:, 1] 

    # tagged_jets = jets[jets[tagger] > threshold]  

    dR_after_lead = delta_r_manual(lead_b, jets)
    dR_after_sublead = delta_r_manual(sublead_b, jets)

    lead_min_idx = ak.argmin(dR_after_lead, axis=1, keepdims=False)
    sublead_min_idx = ak.argmin(dR_after_sublead, axis=1, keepdims=False)


    dr_after_lead = dR_after_lead[ak.local_index(dR_after_lead) == lead_min_idx]
    dr_after_sublead = dR_after_sublead[ak.local_index(dR_after_lead) == sublead_min_idx]


    conflict = lead_min_idx == sublead_min_idx



    keep_first = dr_after_lead < dr_after_sublead
    conflict_b = ak.broadcast_arrays(keep_first, conflict)[1]

    # Step 2: safely fill with NaN (avoids None-related TypeErrors)
    dr1_new = ak.where(conflict_b, ak.where(keep_first, dr_after_lead, np.nan), dr_after_lead)
    dr2_new = ak.where(conflict_b, ak.where(~keep_first, dr_after_sublead, np.nan), dr_after_sublead)


    dr_combined = ak.concatenate([dr1_new, dr2_new], axis=1)
    mask_valid = ~ak.is_none(dr_combined) & (dr_combined == dr_combined)  # remove None and NaN
    dr_combined = dr_combined[mask_valid]


    jet_idx = ak.local_index(jets)

    lead_idx_valid = (lead_min_idx >= 0) & (lead_min_idx < ak.num(jets))
    sublead_idx_valid = (sublead_min_idx >= 0) & (sublead_min_idx < ak.num(jets))

    lead_min_idx  = ak.mask(lead_min_idx, lead_idx_valid)
    sublead_min_idx = ak.mask(sublead_min_idx, sublead_idx_valid)

    lead_single = ak.singletons(lead_min_idx)
    sublead_single = ak.singletons(sublead_min_idx)

    matched_idx = ak.fill_none(
        ak.concatenate([lead_single, sublead_single], axis=1),
        -1
    )

    mask_valid = matched_idx != -1
    matched_idx = ak.mask(matched_idx, mask_valid)

    # 🔸 Remove duplicates per event (safe for all Awkward versions)
    unique_idx = ak.Array([list(dict.fromkeys(ev)) for ev in ak.to_list(matched_idx)])

    matched_jets = jets[unique_idx]

    all_tagger_score = ak.flatten(matched_jets[tagger])

    all_dr_values = ak.flatten(ak.drop_none(dr_combined))

    # mask for valid dr values
    mask_valid = (all_dr_values == all_dr_values)  # removes NaNs

    # mask for ΔR < 0.4
    mask_dr = (all_dr_values < 0.4) & mask_valid

    mask_dr_false = (all_dr_values > 0.4) & mask_valid

    # mask for tagger threshold (example: > 0.5)
    mask_tag = (all_tagger_score > thr) & mask_valid

    # jets that pass both
    mask_both = mask_dr & mask_tag

    mask_both_false = mask_dr_false & mask_tag

    # flatten everything to count
    N_total = ak.sum(mask_dr)
    N_total_unmatched = ak.sum(mask_dr_false)
    N_pass  = ak.sum(mask_both)
    N_pass_false  = ak.sum(mask_both_false)

    efficiency = N_pass / N_total if N_total > 0 else float("nan")
    miss_tag_rate = N_pass_false/N_total_unmatched

    print("Total number of events: ", len(jets))
    print("Total number of dR matched Jets: ", N_total)
    print("Total number of dR matched Jets with tagger passed: ", N_pass)

    return efficiency, miss_tag_rate



def main():
    parser = argparse.ArgumentParser(description="Scan b-tag efficiency vs mistag rate using NanoEventsFactory (PyROOT version)")
    parser.add_argument("--input", required=True, help="Path to NanoAOD file")
    parser.add_argument("--taggers", required=True, nargs="+", help="Jet tagger branches (space-separated, e.g. --taggers btagCSVV2 DeepFlavB)")
    parser.add_argument("--steps", type=int, default=200, help="Number of threshold steps between 0 and 1")
    parser.add_argument("--output", default="btag_eff_scan.root", help="Output plot filename")
    args = parser.parse_args()

    print(f"Loading file: {args.input}")
    
    output_dir = os.path.dirname(os.path.abspath(args.output))

    log_path = os.path.join(output_dir, os.path.splitext(os.path.basename(args.output))[0] + ".log")

    # Redirect stdout and stderr to the log file
    sys.stdout = open(log_path, "w")
    sys.stderr = sys.stdout

    print(f"==== BTag efficiency scan started ====")
    print(f"Input file : {args.input}")
    print(f"Taggers    : {args.taggers}")
    print(f"Steps      : {args.steps}")
    print(f"Output ROOT: {args.output}")
    print(f"Log file   : {log_path}\n")

    # --- Load events ---
    print(f"Loading file: {args.input}")

    factory = NanoEventsFactory.from_root(f"{args.input}:Events", schemaclass=NanoAODSchema)
    events = factory.events()
    events = ak.materialize(events)

    thresholds = np.linspace(0.0, 0.99, args.steps)

    all_effs, all_mistags = [], []

    for tagger in args.taggers:
        print(f"\nRunning scan for tagger: {tagger}")
        effs, mistags = [], []
        for thr in thresholds:
            # e, m = compute_eff_miss_tag_rate(events, tagger, thr)
            e, m = compute_eff_miss_tag_rate_after_sel(events, tagger, thr, args.output)
            effs.append(e)
            mistags.append(m)
            print(f"thr={thr:.2f} : eff={e:.3f}, mistag={m:.3f}")
        all_effs.append(effs)
        all_mistags.append(mistags)

    draw_performance_curves(args.taggers, all_effs, all_mistags, args.input, args.output)


if __name__ == "__main__":
    main()
