import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import ROOT
import glob
import os


sample_dirs = {

    # Normal samples
    "DYGto2LG": [
        "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/Skimmer/DYGto2LG50_24SummerRun3"
    ],

    "DYto2E": [
        "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/Skimmer/DYto2E50_24SummerRun3"
    ],

    "DYto2Mu": [
        "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/Skimmer/DYto2Mu50_24SummerRun3"
    ],

    "TTto2L2Nu": [
        "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/Skimmer/TTto2L2Nu_24SummerRun3"
    ],

    "TTtoLNu2Q": [
        "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/Skimmer/TTtoLNu2Q_24SummerRun3"
    ],

    "TTG1Jets": [
        "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/Skimmer/TTG1Jets_24SummerRun3"
    ],

    "WGtoLNuG": [
        "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/Skimmer/WGtoLNuG_24SummerRun3"
    ],

    # W + jets → electron
    "WtoE": [
        "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/Skimmer/WtoENu0J_24SummerRun3",
        "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/Skimmer/WtoENu1J_24SummerRun3",
        "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/Skimmer/WtoENu2J_24SummerRun3",
    ],

    # W + jets → muon
    "WtoMu": [
        "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/Skimmer/WtoMuNu0J_24SummerRun3",
        "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/Skimmer/WtoMuNu1J_24SummerRun3",
        "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/Skimmer/WtoMuNu2J_24SummerRun3",
    ],
}

selected_dirs = {

    "DYGto2LG": [
        "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/DYGto2LG50_24SummerRun3_selected"
    ],

    "DYto2E": [
        "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/DYto2E50_24SummerRun3_selected"
    ],

    "DYto2Mu": [
        "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/DYto2Mu50_24SummerRun3_selected"
    ],

    "TTto2L2Nu": [
        "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/TTto2L2Nu_24SummerRun3_selected"
    ],

    "TTtoLNu2Q": [
        "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/TTtoLNu2Q_24SummerRun3_selected2"
    ],

    "TTG1Jets": [
        "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/TTG1Jets_24SummerRun3_selected"
    ],

    "WGtoLNuG": [
        "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/WGtoLNuG_24SummerRun3_selected"
    ],
}

# ======================================================================
# W + Jets
# ======================================================================

WJets_dirs = {

    "WtoE": {

        "0J": "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/WtoENu0J_24SummerRun3_selected",

        "1J": "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/WtoENu1J_24SummerRun3_selected",

        "2J": "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/WtoENu2J_24SummerRun3_selected"
    },

    "WtoMu": {

        "0J": "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/WtoMuNu0J_24SummerRun3_selected",

        "1J": "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/WtoMuNu1J_24SummerRun3_selected",

        "2J": "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/WtoMuNu2J_24SummerRun3_selected"
    },
}

cross_sections = {

    "DYGto2LG": 126.7,
    "DYto2E": 2124.08,
    "DYto2Mu": 2124.08,
    "TTto2L2Nu": 98.04,
    "TTtoLNu2Q": 405.87,
    "TTG1Jets": 4.634,
    "WGtoLNuG": 671.5,


    "WtoE": {
        "0J": 55850,
        "1J": 9177,
        "2J": 3474,
    },

    "WtoMu": {
        "0J": 55920,
        "1J": 9202,
        "2J": 3490,
    },
}

sum_genweights = {

    "DYGto2LG": 6.411912138e+10,
    "DYto2E": 8.687530658e+12,
    "DYto2Mu": 8.7529833e+12,
    "TTto2L2Nu": 3.782010699e+10,
    "TTtoLNu2Q": 1.619674372e+11,
    "TTG1Jets": 164785934.4,
    "WGtoLNuG": 5.627469099e+11,
}

WJets_sum_genweights = {

    "WtoE": {

        "0J": 2.985935891e+13,
        "1J": 1.81602151e+13,
        "2J": 7.039157502e+12,
    },

    "WtoMu": {

        "0J": 2.896817149e+13,
        "1J": 1.58752659e+13,
        "2J": 7.709247872e+12,
    },
}

# n_files = {
#     "DYGto2LG": 30,
#     "DYto2E": 100,
#     "DYto2Mu": 100,
#     "TTto2L2Nu": 10,
#     "TTtoLNu2Q": 10,
#     "TTG1Jets": 10,
#     "WGtoLNuG": 100,
#     "WtoE": 100,
#     "WtoMu": 100,
# }

n_files = {
    "DYGto2LG": 30,
    "DYto2E": 100,
    "DYto2Mu": 100,
    "TTto2L2Nu": 10,
    "TTtoLNu2Q": 10,
    "TTG1Jets": 10,
    "WGtoLNuG": 100,
    "WtoE": 100,
    "WtoMu": 100,
}

# n_files = {
#     "DYGto2LG": 2,
#     "DYto2E": 2,
#     "DYto2Mu": 2,
#     "TTto2L2Nu": 2,
#     "TTtoLNu2Q": 2,
#     "TTG1Jets": 2,
#     "WGtoLNuG": 2,
#     "WtoE": 2,
#     "WtoMu": 2,
# }

def get_root_files(directory, n_files=None):

    root_files = sorted(glob.glob(os.path.join(directory, "*.root")))

    if len(root_files) == 0:
        print(f"WARNING: No ROOT files found in:")
        print(directory)
        return []

    if n_files is not None:
        root_files = root_files[:n_files]

    return root_files

def is_allowed_parent(pdgId):
    apdg = abs(pdgId)

    # Quarks: d, u, s, c, b, t
    is_quark = (apdg >= 1) & (apdg <= 6)

    # Gluon
    is_gluon = apdg == 21

    # Leptons: e, nu_e, mu, nu_mu, tau, nu_tau
    is_lepton = (
        (apdg == 11) |
        (apdg == 13) |
        (apdg == 15))

    # Bosons: gamma, Z, W, Higgs
    is_boson = (
        (apdg == 22) |
        (apdg == 23) |
        (apdg == 24) |
        (apdg == 25)
    )

    return is_quark | is_gluon | is_lepton | is_boson

def photon_parentage_ok(gen_pho, gen, max_depth=30):

    mother_idx = gen_pho.genPartIdxMother

    # Start with all photons considered valid
    history_ok = ak.ones_like(mother_idx, dtype=bool)

    # Photons with a valid mother need to be followed
    active = mother_idx >= 0

    current_idx = mother_idx

    for _ in range(max_depth):

        if not ak.any(active):
            break

        # Prevent -1 from being used as an index
        safe_idx = ak.where(active, current_idx, 0)

        ancestor = gen[safe_idx]

        ancestor_pdgId = ancestor.pdgId

        # Is this ancestor an allowed particle?
        allowed = is_allowed_parent(ancestor_pdgId)

        # If an active ancestor is NOT allowed,
        # reject this photon.
        history_ok = history_ok & ak.where(active, allowed, True)

        # Move to the next mother only if current ancestor
        # was allowed and has a valid mother.
        next_idx = ancestor.genPartIdxMother

        active = (active & allowed & (next_idx >= 0))

        current_idx = next_idx

    return history_ok

def photon_min_dr(events, photon_pt_min=10.0, photon_eta_max=3.0, other_pt_min=5.0):
    gen = events.GenPart
    apdg = abs(gen.pdgId)

    is_photon = apdg == 22
    photon_kin = (gen.pt > photon_pt_min) & (abs(gen.eta) < photon_eta_max)

    gen_idx = ak.local_index(gen, axis=1)

    photon_daughter_matrix = gen.genPartIdxMother[:, :, None] == gen_idx[:, None, :]

    daughter_is_photon = apdg[:, None, :] == 22

    has_photon_daughter = ak.any(photon_daughter_matrix & daughter_is_photon, axis=2)

    is_last_photon = is_photon & ~has_photon_daughter

    gen_pho = gen[is_last_photon & photon_kin]

    pho_history = photon_parentage_ok(gen_pho, gen)
    gen_pho = gen_pho[pho_history]

    gen_pho_pt = gen_pho.pt

    n_pho = ak.num(gen_pho)

    is_quark = (apdg >= 1) & (apdg <= 6)

    is_gluon = apdg == 21

    is_lepton = ((apdg == 11) | (apdg == 13) | (apdg == 15))

    is_boson = ((apdg == 23) | (apdg == 24) | (apdg == 25))

    is_other = is_quark | is_gluon | is_lepton | is_boson

    other_kin = gen.pt > other_pt_min
    other = gen[is_other & other_kin]

    pairs = ak.cartesian({"pho": gen_pho, "part": other}, axis=1, nested=True)

    pho = pairs["pho"]
    part = pairs["part"]

    deta = pho.eta - part.eta

    dphi = np.arctan2(np.sin(pho.phi - part.phi), np.cos(pho.phi - part.phi))

    dr = np.sqrt(deta**2 + dphi**2)

    min_dr = ak.min(dr, axis=2)

    min_dr_tmp = ak.fill_none(min_dr, np.inf)

    pass_dr_high = min_dr_tmp > 0.15
    pass_dr_low = min_dr_tmp < 0.15
    n_pho_after_dr_high = ak.sum(pass_dr_high, axis=1)
    n_pho_after_dr_low = ak.sum(pass_dr_low, axis=1)
    keep_high = (n_pho == 0) | (n_pho_after_dr_high > 0)
    keep_low = (n_pho == 0) | (n_pho_after_dr_low > 0)

    gen_pho_pt_after_dr = gen_pho_pt[pass_dr_high]

    n_pho_high = n_pho_after_dr_high[keep_high]
    n_pho_low = n_pho_after_dr_low[keep_low]

    genweight = events.genWeight

    genweight_high = genweight[keep_high]
    genweight_low  = genweight[keep_low]

    return dr, min_dr, n_pho, n_pho_high, n_pho_low, genweight_high, genweight_low, gen_pho_pt, gen_pho_pt_after_dr


def fake_only_photons_event_count(events, photon_pt_min=10.0, photon_eta_max=3.0, other_pt_min=5.0):
    gen = events.GenPart
    apdg = abs(gen.pdgId)

    is_photon = apdg == 22
    photon_kin = (gen.pt > photon_pt_min) & (abs(gen.eta) < photon_eta_max)

    gen_idx = ak.local_index(gen, axis=1)

    photon_daughter_matrix = gen.genPartIdxMother[:, :, None] == gen_idx[:, None, :]

    daughter_is_photon = apdg[:, None, :] == 22

    has_photon_daughter = ak.any(photon_daughter_matrix & daughter_is_photon, axis=2)

    is_last_photon = is_photon & ~has_photon_daughter

    gen_pho = gen[is_last_photon & photon_kin]

    pho_history = photon_parentage_ok(gen_pho, gen)
    gen_pho = gen_pho[pho_history]

    n_pho = ak.num(gen_pho)

    at_least_one_real_pho = n_pho > 0

    n_remaining_events = len(events[~at_least_one_real_pho])

    return n_remaining_events

def calculate_min_dr_from_directories(
    directories,
    n_files=3,
    photon_pt_min=10.0,
    photon_eta_max=3.0,
    other_pt_min=5.0,
):

    all_min_dr = []
    all_npho = []
    all_gen_pho_pt = []

    total_events = 0
    total_photons = 0

    for directory in directories:

        root_files = get_root_files(directory, n_files=n_files)

        print("\n" + "=" * 100)
        print("Directory:")
        print(directory)
        print("Number of files selected:", len(root_files))
        print("=" * 100)

        for i, file in enumerate(root_files):

            print(f"\nProcessing file {i+1}/{len(root_files)}:")
            print(os.path.basename(file))

            factory = NanoEventsFactory.from_root(f"{file}:Events", schemaclass=NanoAODSchema)

            events = factory.events()

            n_events = len(events)

            print("Events:", n_events)

            _, min_dr, npho, _, _, _, _, gen_pho_pt, gen_pho_pt_after_dr = photon_min_dr(events, photon_pt_min=photon_pt_min, photon_eta_max=photon_eta_max, other_pt_min=other_pt_min)

            min_dr_flat = ak.to_numpy(ak.flatten(min_dr, axis=None))

            min_dr_flat = min_dr_flat[np.isfinite(min_dr_flat)]

            gen_pho_pt_flat = ak.to_numpy(ak.flatten(gen_pho_pt, axis=None))

            all_gen_pho_pt.append(gen_pho_pt_flat)

            print("Valid photons:", len(min_dr_flat))

            all_min_dr.append(min_dr_flat)
            all_npho.append(npho)

            total_events += n_events
            total_photons += len(min_dr_flat)

            # Release memory
            del events
            del factory

    if len(all_min_dr) > 0:
        all_min_dr = np.concatenate(all_min_dr)
        all_npho = np.concatenate(all_npho)
        all_gen_pho_pt = np.concatenate(all_gen_pho_pt)
    else:
        all_min_dr = np.array([], dtype=float)

    print("\n" + "#" * 100)
    print("FINAL SAMPLE SUMMARY")
    print("#" * 100)

    print("Total events:", total_events)
    print("Total photons:", total_photons)

    return all_min_dr, all_npho, all_gen_pho_pt

def merge_overflow_into_last_bin(hist):

    n_bins = hist.GetNbinsX()
    last_content = hist.GetBinContent(n_bins)
    overflow_content = hist.GetBinContent(n_bins + 1)
    last_error = hist.GetBinError(n_bins)
    overflow_error = hist.GetBinError(n_bins + 1)

    hist.SetBinContent(n_bins, last_content + overflow_content)

    hist.SetBinError(n_bins, (last_error**2 + overflow_error**2)**0.5)
    hist.SetBinContent(n_bins + 1, 0.0)
    hist.SetBinError(n_bins + 1, 0.0)


# ROOT.gStyle.SetOptStat(111111)
ROOT.gStyle.SetOptStat("eiou")

def plot_photon_min_dr_root(min_dr_all, nphos_all, gen_pho_pt_all=None, weights_all=None, wgts_all=None, allnpho_high=None, allnpho_low=None, allgenw_high=None, 
    allgenw_low = None, gen_pho_pt_after_dr=None, gen_pho_pt_after_dr_weights=None, gen_pho_pt_weights=None,
    photon_pt_min=10.0, photon_eta_max=3.0, other_pt_min=5.0, dr_max=1.0, bins=50, label="DYGto2LG", 
    output_dir="/eos/user/b/bbapi/www/Analysis_plots/Overlap_removal/test/"):

    ROOT.gStyle.SetOptStat("eiou")
    root_dir = os.path.join(output_dir, "ROOT")
    png_dir = os.path.join(output_dir, "PNG")
    pdf_dir = os.path.join(output_dir, "PNG")
    os.makedirs(root_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)

    hist_name = f"h_min_dr_{label}"
    hist_npho_name = f"h_npho_{label}"
    hist_norm_name = f"h_min_dr_{label}_normalized"
    hist_npho_name_norm = f"h_npho_{label}_normalized"
    hist_weighted_name = f"h_min_dr_{label}_weighted"
    hist_npho_name_weighted = f"h_npho_{label}_weighted"
    hist_npho_high_name_weighted = f"h_nphohigh_{label}_weighted"
    hist_npho_low_name_weighted = f"h_npholow_{label}_weighted"

    hist_genpho_pt = ROOT.TH1D(f"h_genpho_pt_{label}", "", 50, 0.0, 200.0)

    hist_genpho_pt_weighted = ROOT.TH1D(f"h_genpho_pt_{label}_weighted", "", 50, 0.0, 200.0)

    hist_genpho_pt_after_dr_weighted = ROOT.TH1D(f"h_genpho_pt_{label}_after_dr_weighted", "", 50, 0.0, 200.0)

    # ROOT input files: unweighted
    if gen_pho_pt_all is not None and gen_pho_pt_weights is None:
        for pt in gen_pho_pt_all:
            hist_genpho_pt.Fill(float(pt))

    merge_overflow_into_last_bin(hist_genpho_pt)

    # Parquet: weighted, before DR cut
    if gen_pho_pt_all is not None and gen_pho_pt_weights is not None:
        for pt, weight in zip(gen_pho_pt_all, gen_pho_pt_weights):
            hist_genpho_pt_weighted.Fill(float(pt), float(weight))

    merge_overflow_into_last_bin(hist_genpho_pt_weighted)

    # Parquet: weighted, after DR cut
    if gen_pho_pt_after_dr is not None and gen_pho_pt_after_dr_weights is not None:
        for pt, weight in zip(gen_pho_pt_after_dr, gen_pho_pt_after_dr_weights):
            hist_genpho_pt_after_dr_weighted.Fill(float(pt), float(weight))
    
    merge_overflow_into_last_bin(hist_genpho_pt_after_dr_weighted)
    
    hist = ROOT.TH1D(hist_name, "", bins, 0.0, dr_max)
    if weights_all is None:
        for value in min_dr_all:
            hist.Fill(float(value))

    hist.SetStats(True)
    overflow = hist.GetBinContent(bins + 1)
    underflow = hist.GetBinContent(0)

    print("\n" + "=" * 80)
    print(f"Histogram: {label}")
    print("=" * 80)
    print("Unweighted integral:", hist.Integral())
    print("Underflow:", underflow)
    print("Overflow:", overflow)

    # Merge overflow into last visible bin
    merge_overflow_into_last_bin(hist)

    print("Unweighted integral after overflow merge:", hist.Integral())

    hist_npho = ROOT.TH1D(hist_npho_name, "", 4, 0.0, 4)
    hist_npho.SetStats(True)
    if wgts_all is None:
        for value in nphos_all:
            hist_npho.Fill(float(value))

    hist_norm = hist.Clone(hist_norm_name)
    hist_norm.SetStats(True)
    hist_norm.SetDirectory(0)
    merge_overflow_into_last_bin(hist_norm)
    integral = hist_norm.Integral()
    if integral > 0:
        hist_norm.Scale(1.0 / integral)

    hist_weighted = ROOT.TH1D(hist_weighted_name, "", bins, 0.0, dr_max)
    hist_weighted.SetStats(True)
    weighted_overflow = hist_weighted.GetBinContent(bins + 1)
    weighted_underflow = hist_weighted.GetBinContent(0)
    print("Weighted integral before overflow merge:", hist_weighted.Integral())
    print("Weighted underflow:", weighted_underflow)
    print("Weighted overflow:", weighted_overflow)
    if weights_all is not None:
        for value, weight in zip(min_dr_all, weights_all):
            hist_weighted.Fill(float(value), float(weight))

    merge_overflow_into_last_bin(hist_weighted)
    print("Weighted integral after overflow merge:", hist_weighted.Integral())

    hist_npho_norm = hist_npho.Clone(hist_npho_name_norm)
    hist_npho_norm.SetStats(True)
    hist_npho_norm.SetDirectory(0)
    integral = hist_npho_norm.Integral()
    if integral > 0:
        hist_npho_norm.Scale(1.0 / integral)

    hist_npho_weighted = ROOT.TH1D(hist_npho_name_weighted, "", 4, 0.0, 4)
    hist_npho_weighted.SetStats(True)
    if wgts_all is not None:
        for value, weight in zip(nphos_all, wgts_all):
            hist_npho_weighted.Fill(float(value), float(weight))

    hist_nphohigh_weighted = ROOT.TH1D(hist_npho_high_name_weighted, "", 4, 0.0, 4)
    hist_nphohigh_weighted.SetStats(True)
    if allnpho_high is not None:
        for value, weight in zip(allnpho_high, allgenw_high):
            hist_nphohigh_weighted.Fill(float(value), float(weight))

    hist_npholow_weighted = ROOT.TH1D(hist_npho_low_name_weighted, "", 4, 0.0, 4)
    hist_npholow_weighted.SetStats(True)
    if allnpho_low is not None:
        for value, weight in zip(allnpho_low, allgenw_low):
            hist_npholow_weighted.Fill(float(value), float(weight))

    weighted_overflow = hist_weighted.GetBinContent(bins + 1)
    weighted_underflow = hist_weighted.GetBinContent(0)

    print("Weighted integral:", hist_weighted.Integral())
    print("Weighted underflow:", weighted_underflow)
    print("Weighted overflow:", weighted_overflow)

    for h in [hist, hist_norm, hist_weighted]:
        h.SetLineWidth(2)
        h.GetXaxis().SetTitle("#Delta R_{min}(#gamma, GenPart)")
        h.SetTitle(f"Photon minimum #Delta R (p_{{T}}^{{#gamma}} > {photon_pt_min} GeV, p_{{T}}^{{GenPart}} > {other_pt_min} GeV)")

    for h in [hist_npho, hist_npho_norm, hist_npho_weighted, hist_nphohigh_weighted, hist_npholow_weighted]:
        h.SetLineWidth(2)
        h.GetXaxis().SetTitle("Number of isolated prompt photon per event")
        h.SetTitle(f"Photon minimum #Delta R (p_{{T}}^{{#gamma}} > {photon_pt_min} GeV, p_{{T}}^{{GenPart}} > {other_pt_min} GeV)")

    for h in [hist_genpho_pt, hist_genpho_pt_weighted, hist_genpho_pt_after_dr_weighted]:
        h.SetLineWidth(2)
        h.GetXaxis().SetTitle("p_{T}^{gen #gamma} [GeV]")

    hist.GetYaxis().SetTitle("Number of photons")
    hist_npho.GetYaxis().SetTitle("Number of events")
    hist_norm.GetYaxis().SetTitle("Normalized entries")
    hist_weighted.GetYaxis().SetTitle("Expected photons")
    hist_npho_norm.GetYaxis().SetTitle("Normalized entries")
    hist_npho_weighted.GetYaxis().SetTitle("Expected events")
    hist_nphohigh_weighted.GetYaxis().SetTitle("Expected events")
    hist_npholow_weighted.GetYaxis().SetTitle("Expected events")
    hist_genpho_pt.GetYaxis().SetTitle("Number of photons")
    hist_genpho_pt_weighted.GetYaxis().SetTitle("Expected photons")
    hist_genpho_pt_after_dr_weighted.GetYaxis().SetTitle("Expected photons")

    root_file_path = os.path.join(root_dir, "photon_min_dr_test.root")
    root_file = ROOT.TFile(root_file_path, "UPDATE")
    if weights_all is None:
        hist.Write(hist_name, ROOT.TObject.kOverwrite)
        hist_norm.Write(hist_norm_name, ROOT.TObject.kOverwrite)
    if weights_all is not None:
        hist_weighted.Write(hist_weighted_name, ROOT.TObject.kOverwrite)   
    if wgts_all is None: 
        hist_npho.Write(hist_npho_name, ROOT.TObject.kOverwrite)
        hist_npho_norm.Write(hist_npho_name_norm, ROOT.TObject.kOverwrite)
    if wgts_all is not None:
        hist_npho_weighted.Write(hist_npho_name_weighted, ROOT.TObject.kOverwrite)
    if allgenw_high is not None:
        hist_nphohigh_weighted.Write(hist_npho_high_name_weighted, ROOT.TObject.kOverwrite)
        hist_npholow_weighted.Write(hist_npho_low_name_weighted, ROOT.TObject.kOverwrite)
    if gen_pho_pt_all is not None and gen_pho_pt_weights is None:
        hist_genpho_pt.Write(f"h_genpho_pt_{label}", ROOT.TObject.kOverwrite)
    if gen_pho_pt_all is not None and gen_pho_pt_weights is not None:
        hist_genpho_pt_weighted.Write(f"h_genpho_pt_{label}_weighted", ROOT.TObject.kOverwrite)
        hist_genpho_pt_after_dr_weighted.Write(f"h_genpho_pt_{label}_after_dr_weighted", ROOT.TObject.kOverwrite)
    root_file.Close()

    print("\nROOT file saved to:")
    print(root_file_path)

    def make_canvas(h, normalized=False, weighted=False, nphoton=False, nPho_high=False, nPho_low=False, pt=False, pt_after_dr=False):
        if pt_after_dr and weighted:
            suffix = "_genpho_pt_after_dr_weighted"
        elif pt and weighted:
            suffix = "_genpho_pt_weighted"
        elif pt:
            suffix = "_genpho_pt"
        elif nphoton and normalized:
            suffix = "_npho_normalized"
        elif nphoton and weighted:
            suffix = "_npho_weighted"
        elif nphoton:
            suffix = "_npho"
        elif nPho_high:
            suffix = "_nphohigh"
        elif nPho_low:
            suffix = "_npholow"
        elif normalized:
            suffix = "_normalized"
        elif weighted:
            suffix = "_weighted"
        else:
            suffix = ""
        if nphoton:
            canvas = ROOT.TCanvas(f"canvas_{label}{suffix}", "Number of prompt isolated photons per event", 800, 600)
        else: 
            canvas = ROOT.TCanvas(f"canvas_{label}{suffix}", "Photon minimum Delta R", 800, 600)
        canvas.SetBottomMargin(0.15)
        canvas.SetLeftMargin(0.15)
        h.Draw("HIST")
        if not pt and not pt_after_dr:
            line = ROOT.TLine(0.2, 0, 0.2, h.GetMaximum() * 1.05)
            line.SetLineStyle(2)
            line.SetLineWidth(2)
            line.Draw()
        canvas.Update()
        return canvas
    
    if weights_all is None:
        canvas = make_canvas(hist)
        canvas.SaveAs(os.path.join(png_dir, f"{label}.png"))
        canvas.SaveAs(os.path.join(pdf_dir, f"{label}.pdf"))

        canvas_norm = make_canvas(hist_norm, normalized=True)
        canvas_norm.SaveAs(os.path.join(png_dir, f"{label}_normalized.png"))
        canvas_norm.SaveAs(os.path.join(pdf_dir, f"{label}_normalized.pdf"))

    if weights_all is not None:
        canvas_weighted = make_canvas(hist_weighted, weighted=True)
        canvas_weighted.SaveAs(os.path.join(png_dir, f"{label}_weighted.png"))
        canvas_weighted.SaveAs(os.path.join(pdf_dir, f"{label}_weighted.pdf"))

    if wgts_all is None:
        canvas_npho = make_canvas(hist_npho, nphoton=True)
        canvas_npho.SaveAs(os.path.join(png_dir, f"{label}_npho.png"))
        canvas_npho.SaveAs(os.path.join(pdf_dir, f"{label}_npho.pdf"))
        canvas_npho_norm = make_canvas(hist_npho_norm, nphoton=True, normalized=True)
        canvas_npho_norm.SaveAs(os.path.join(png_dir, f"{label}_npho_normalized.png"))
        canvas_npho_norm.SaveAs(os.path.join(pdf_dir, f"{label}_npho_normalized.pdf"))

    if wgts_all is not None:
        canvas_npho_weighted = make_canvas(hist_npho_weighted, nphoton=True, weighted=True)
        canvas_npho_weighted.SaveAs(os.path.join(png_dir, f"{label}_npho_weighted.png"))
        canvas_npho_weighted.SaveAs(os.path.join(pdf_dir, f"{label}_npho_weighted.pdf"))

    if allgenw_high is not None:
        canvas_npho_high_weighted = make_canvas(hist_nphohigh_weighted, nPho_high=True)
        canvas_npho_high_weighted.SaveAs(os.path.join(png_dir, f"{label}_npho_high_weighted.png"))
        canvas_npho_high_weighted.SaveAs(os.path.join(pdf_dir, f"{label}_npho_high_weighted.pdf"))
        canvas_npho_low_weighted = make_canvas(hist_npholow_weighted, nPho_low=True)
        canvas_npho_low_weighted.SaveAs(os.path.join(png_dir, f"{label}_npho_low_weighted.png"))
        canvas_npho_low_weighted.SaveAs(os.path.join(pdf_dir, f"{label}_npho_low_weighted.pdf"))

    # Gen-photon pT plots
    if gen_pho_pt_all is not None:
        canvas_genpho_pt = make_canvas(hist_genpho_pt, pt=True)
        canvas_genpho_pt.SaveAs(os.path.join(png_dir, f"{label}_genpho_pt.png"))
        canvas_genpho_pt.SaveAs(os.path.join(pdf_dir, f"{label}_genpho_pt.pdf"))

    if gen_pho_pt_weights is not None:
        canvas_genpho_pt_weighted = make_canvas(hist_genpho_pt_weighted, weighted=True, pt=True)
        canvas_genpho_pt_weighted.SaveAs(os.path.join(png_dir, f"{label}_genpho_pt_weighted.png"))
        canvas_genpho_pt_weighted.SaveAs(os.path.join(pdf_dir, f"{label}_genpho_pt_weighted.pdf"))

    if gen_pho_pt_after_dr_weights is not None:
        canvas_genpho_pt_after_dr = make_canvas(hist_genpho_pt_after_dr_weighted, weighted=True, pt=True, pt_after_dr=True)
        canvas_genpho_pt_after_dr.SaveAs(os.path.join(png_dir, f"{label}_genpho_pt_after_dr_weighted.png"))
        canvas_genpho_pt_after_dr.SaveAs(os.path.join(pdf_dir, f"{label}_genpho_pt_after_dr_weighted.pdf"))

    print("PNG/PDF plots saved in:", output_dir)

    return hist, hist_norm, hist_weighted, hist_npho, hist_npho_norm, hist_npho_weighted, hist_nphohigh_weighted, hist_npholow_weighted,
    hist_genpho_pt, hist_genpho_pt_weighted, hist_genpho_pt_after_dr_weighted 



output_dir = "/eos/user/b/bbapi/www/Analysis_plots/Overlap_removal/test/"

for label, directories in sample_dirs.items():

    print("\n\n")
    print("*" * 100)
    print(f"PROCESSING SAMPLE: {label}")
    print("*" * 100)

    min_dr_all, nphos_all, gen_pho_pt_all = calculate_min_dr_from_directories(
        directories,
        n_files=n_files[label],
        photon_pt_min=10.0,
        photon_eta_max=3.0,
        other_pt_min=5.0,
    )

    plot_photon_min_dr_root(
        min_dr_all,
        nphos_all, gen_pho_pt_all=gen_pho_pt_all,
        photon_pt_min=10.0,
        other_pt_min=5.0,
        dr_max=1.0,
        bins=50,
        label=label,
        output_dir=output_dir,
    )


def calculate_weighted_min_dr_parquet(
    selected_directory,
    xsec_pb,
    lumi_fb,
    sum_genweight,
    photon_pt_min=10.0,
    photon_eta_max=3.0,
    other_pt_min=5.0,
):

    parquet_files = sorted(glob.glob(os.path.join(selected_directory, "*.parquet")))

    all_min_dr = []
    all_weights = []
    all_nphos = []
    all_wgt = []
    total_fake_events = []
    all_npho_high = []
    all_npho_low = []
    all_genw_high = []
    all_genw_low = []
    all_gen_pho_pt = []
    all_gen_pho_pt_weights = []
    all_gen_pho_pt_after_dr = []
    all_gen_pho_pt_after_dr_weights = []

    print("\n" + "=" * 100)
    print("Selected directory:")
    print(selected_directory)
    print("Number of Parquet files:", len(parquet_files))
    print("Cross section [pb]:", xsec_pb)
    print("Sum genWeight:", sum_genweight)
    print("=" * 100)

    normalization = (xsec_pb * lumi_fb * 1000.0 / sum_genweight)

    print("Normalization factor:", normalization)

    for i, file in enumerate(parquet_files):

        print(f"\n[{i+1}/{len(parquet_files)}] " f"{os.path.basename(file)}")
        events = ak.from_parquet(file)
        print("Selected events:", len(events))

        fake_events = fake_only_photons_event_count(events)

        genweight = events.genWeight

        _, min_dr, npho, npho_high, npho_low, genw_high, genw_low, gen_pho_pt, gen_pho_pt_after_dr = photon_min_dr(events, photon_pt_min=photon_pt_min, photon_eta_max=photon_eta_max, other_pt_min=other_pt_min)

        photon_genweight = ak.broadcast_arrays(min_dr, genweight)[1]

        photon_pt_weight = ak.broadcast_arrays(gen_pho_pt, genweight)[1]

        photon_pt_weight_after_dr = ak.broadcast_arrays(gen_pho_pt_after_dr, genweight)[1]

        min_dr_flat = ak.to_numpy(ak.flatten(min_dr, axis=None))

        weight_flat = ak.to_numpy(ak.flatten(photon_genweight, axis=None))

        wgt_flat = ak.to_numpy(genweight)

        valid = (np.isfinite(min_dr_flat) & np.isfinite(weight_flat))

        min_dr_flat = min_dr_flat[valid]
        weight_flat = weight_flat[valid]

        weight_flat = (weight_flat * normalization)

        wgt_flat = (wgt_flat * normalization)

        genw_high_flat = (genw_high * normalization)

        genw_low_flat = (genw_low * normalization)

        gen_pho_pt_flat = ak.to_numpy(ak.flatten(gen_pho_pt, axis=None))

        gen_pho_pt_weight_flat = ak.to_numpy(ak.flatten(photon_pt_weight, axis=None))

        gen_pho_pt_after_dr_flat = ak.to_numpy(ak.flatten(gen_pho_pt_after_dr, axis=None))

        gen_pho_pt_weight_after_dr_flat = ak.to_numpy(ak.flatten(photon_pt_weight_after_dr, axis=None))

        gen_pho_pt_weight_flat *= normalization

        gen_pho_pt_weight_after_dr_flat *= normalization

        all_min_dr.append(min_dr_flat)

        all_weights.append(weight_flat)

        all_wgt.append(wgt_flat)

        all_nphos.append(npho)

        all_npho_high.append(npho_high)

        all_npho_low.append(npho_low)

        all_genw_high.append(genw_high_flat)

        all_genw_low.append(genw_low_flat)

        total_fake_events.append(fake_events)

        all_gen_pho_pt.append(gen_pho_pt_flat)
        all_gen_pho_pt_weights.append(gen_pho_pt_weight_flat)

        all_gen_pho_pt_after_dr.append(gen_pho_pt_after_dr_flat)
        all_gen_pho_pt_after_dr_weights.append(gen_pho_pt_weight_after_dr_flat)

        print("Selected photons:", len(min_dr_flat))

        print("Weighted yield:", np.sum(weight_flat))

        del events

    # --------------------------------------------------------------
    # Combine files
    # --------------------------------------------------------------

    if len(all_min_dr) > 0:

        all_min_dr = np.concatenate(all_min_dr)

        all_weights = np.concatenate(all_weights)

        all_nphos = np.concatenate(all_nphos)

        all_wgt = np.concatenate(all_wgt)

        all_npho_high = np.concatenate(all_npho_high)

        all_npho_low = np.concatenate(all_npho_low)

        all_genw_high = np.concatenate(all_genw_high)

        all_genw_low = np.concatenate(all_genw_low)

        all_gen_pho_pt = np.concatenate(all_gen_pho_pt)
        all_gen_pho_pt_weights = np.concatenate(all_gen_pho_pt_weights)

        all_gen_pho_pt_after_dr = np.concatenate(all_gen_pho_pt_after_dr)
        all_gen_pho_pt_after_dr_weights = np.concatenate(all_gen_pho_pt_after_dr_weights)

        total_fake_events = np.array(total_fake_events)

    else:

        all_min_dr = np.array([], dtype=float)

        all_weights = np.array([], dtype=float)

    print("\nTotal weighted yield:")
    print(np.sum(all_weights))

    print("Sample: ", selected_directory)
    print("Total fake events: ", np.sum(total_fake_events))

    return all_min_dr, all_weights, all_wgt, all_nphos, all_npho_high, all_npho_low, all_genw_high, all_genw_low, all_gen_pho_pt_after_dr, all_gen_pho_pt_after_dr_weights, all_gen_pho_pt, all_gen_pho_pt_weights

normal_results = {}

lumi_fb = 109.0

for sample, directories in selected_dirs.items():

    print("\n\n")
    print("*" * 100)
    print(f"PROCESSING {sample}")
    print("*" * 100)

    # There is one selected directory
    selected_directory = directories[0]

    min_dr, weights, wgts_all, nphos_all, allnpho_high, allnpho_low, allgenw_high, allgenw_low, all_gen_pho_pt_after_dr, all_gen_pho_pt_after_dr_weights, all_gen_pho_pt, all_gen_pho_pt_weights = calculate_weighted_min_dr_parquet(
        selected_directory=selected_directory,
        xsec_pb=cross_sections[sample],
        lumi_fb=lumi_fb,
        sum_genweight=sum_genweights[sample],
        photon_pt_min=10.0,
        photon_eta_max=3.0,
        other_pt_min=5.0,
    )

    normal_results[sample] = (min_dr, weights, wgts_all, nphos_all, allnpho_high, allnpho_low, allgenw_high, allgenw_low, all_gen_pho_pt_after_dr, all_gen_pho_pt_after_dr_weights, all_gen_pho_pt, all_gen_pho_pt_weights)

    plot_photon_min_dr_root(
        min_dr,
        nphos_all, all_gen_pho_pt,
        weights,
        wgts_all, allnpho_high, allnpho_low, allgenw_high, allgenw_low, all_gen_pho_pt_after_dr, all_gen_pho_pt_after_dr_weights, all_gen_pho_pt_weights,
        photon_pt_min=10.0,
        photon_eta_max=3.0,
        other_pt_min=5.0,
        dr_max=1.0,
        bins=50,
        label=sample,
        output_dir=output_dir,
    )

# for lepton in ["WtoE", "WtoMu"]:
#     results = []
#     for jet_bin in ["0J", "1J", "2J"]:
#         print("\n\n" + "*" * 100)
#         print(f"PROCESSING {lepton} {jet_bin}")
#         print("*" * 100)
#         selected_directory = WJets_dirs[lepton][jet_bin]
#         xsec = cross_sections[lepton][jet_bin]
#         sum_genw = WJets_sum_genweights[lepton][jet_bin]
#         min_dr, weights, wgts,  nphos, allnpho_high, allnpho_low, allgenw_high, allgenw_low, all_gen_pho_pt_after_dr, all_gen_pho_pt_after_dr_weights, all_gen_pho_pt, all_gen_pho_pt_after_dr_weights = calculate_weighted_min_dr_parquet(selected_directory, xsec, lumi_fb, sum_genw, photon_pt_min=10.0, photon_eta_max=3.0, other_pt_min=5.0)
#         results.append((min_dr, wgts, weights, nphos, allnpho_high, allnpho_low, allgenw_high, allgenw_low))
#     min_dr_all = np.concatenate([x[0] for x in results]) if results else np.array([])
#     weights_all = np.concatenate([x[1] for x in results]) if results else np.array([])
#     wgts_all = np.concatenate([x[2] for x in results]) if results else np.array([])
#     nphos_all = np.concatenate([x[3] for x in results]) if results else np.array([])
#     nphos_high_all = np.concatenate([x[4] for x in results]) if results else np.array([])
#     nphos_low_all = np.concatenate([x[5] for x in results]) if results else np.array([])
#     allgenw_high_all = np.concatenate([x[6] for x in results]) if results else np.array([])
#     allgenw_low_all = np.concatenate([x[7] for x in results]) if results else np.array([])

#     plot_photon_min_dr_root(min_dr_all, nphos_all, weights_all, wgts_all, nphos_high_all, nphos_low_all, allgenw_high_all, allgenw_low_all, photon_pt_min=10.0, photon_eta_max=3.0, other_pt_min=5.0, dr_max=1.0, bins=50, label=lepton, output_dir=output_dir)


for lepton in ["WtoE", "WtoMu"]:

    results = []

    for jet_bin in ["0J", "1J", "2J"]:

        print("\n\n" + "*" * 100)
        print(f"PROCESSING {lepton} {jet_bin}")
        print("*" * 100)

        selected_directory = WJets_dirs[lepton][jet_bin]
        xsec = cross_sections[lepton][jet_bin]
        sum_genw = WJets_sum_genweights[lepton][jet_bin]

        min_dr, weights, wgts, nphos, allnpho_high, allnpho_low, allgenw_high, allgenw_low, all_gen_pho_pt_after_dr, all_gen_pho_pt_after_dr_weights, all_gen_pho_pt, all_gen_pho_pt_weights = calculate_weighted_min_dr_parquet(selected_directory, xsec, lumi_fb, sum_genw, photon_pt_min=10.0, photon_eta_max=3.0, other_pt_min=5.0)

        results.append((min_dr, wgts, weights, nphos, allnpho_high, allnpho_low, allgenw_high, allgenw_low, all_gen_pho_pt_after_dr, all_gen_pho_pt_after_dr_weights, all_gen_pho_pt, all_gen_pho_pt_weights))

    min_dr_all = np.concatenate([x[0] for x in results]) if results else np.array([])
    weights_all = np.concatenate([x[1] for x in results]) if results else np.array([])
    wgts_all = np.concatenate([x[2] for x in results]) if results else np.array([])
    nphos_all = np.concatenate([x[3] for x in results]) if results else np.array([])
    nphos_high_all = np.concatenate([x[4] for x in results]) if results else np.array([])
    nphos_low_all = np.concatenate([x[5] for x in results]) if results else np.array([])
    allgenw_high_all = np.concatenate([x[6] for x in results]) if results else np.array([])
    allgenw_low_all = np.concatenate([x[7] for x in results]) if results else np.array([])
    all_gen_pho_pt_after_dr_all = np.concatenate([x[8] for x in results]) if results else np.array([])
    all_gen_pho_pt_after_dr_weights_all = np.concatenate([x[9] for x in results]) if results else np.array([])
    all_gen_pho_pt_all = np.concatenate([x[10] for x in results]) if results else np.array([])
    all_gen_pho_pt_weights_all = np.concatenate([x[11] for x in results]) if results else np.array([])

    plot_photon_min_dr_root(min_dr_all, nphos_all, all_gen_pho_pt_all, weights_all, wgts_all, nphos_high_all, nphos_low_all, allgenw_high_all, allgenw_low_all, gen_pho_pt_after_dr=all_gen_pho_pt_after_dr_all, gen_pho_pt_after_dr_weights=all_gen_pho_pt_after_dr_weights_all, gen_pho_pt_weights=all_gen_pho_pt_weights_all, photon_pt_min=10.0, photon_eta_max=3.0, other_pt_min=5.0, dr_max=1.0, bins=50, label=lepton, output_dir=output_dir)
