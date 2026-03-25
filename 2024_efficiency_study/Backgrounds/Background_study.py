import glob
import awkward as ak
import numpy as np
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

# -------------------------------------------------
# Collect files
# -------------------------------------------------
# files = glob.glob(
#     "/eos/user/b/bbapi/MC_contacts/TTG1Jets/CMSSW_15_0_15/src/TTGJets_NanoAOD_*.root"
# )

Plot_dir = "/eos/user/b/bbapi/www/Background_study_2024/TTto2L2Nu/"

with open("/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/TTto2L2Nu_2024.text", "r") as f:
    files = [line.strip() for line in f if line.strip()]

print(f"Found {len(files)} files")

def delta_r_manual(obj1, obj2):
    deta = obj1.eta[:, None] - obj2.eta
    dphi = np.abs(obj1.phi[:, None] - obj2.phi)
    dphi = ak.where(dphi > np.pi, 2 * np.pi - dphi, dphi)
    return np.sqrt(deta**2 + dphi**2)


def photon_preselections(
    photons: ak.Array,
    jets: ak.Array,
    electrons: ak.Array,
    muons: ak.Array,
    events: ak.Array,
    electron_veto=True,
    revert_electron_veto=False,
    year="2018",
    wp_medium=0.2783,
    IsFlag=False):
    """
    Apply full preselection on leptons, jets, and photons.
    Finally return only photons from events that pass all criteria.
    """

    print("Number of events before preselection:", len(events))

    # ------------------------
    # Lepton selection
    # ------------------------
    if year.startswith("2016"):
        ele_pt_cut, mu_pt_cut = 27, 26
    elif year == "2017":
        ele_pt_cut, mu_pt_cut = 33, 29
    elif year == "2018":
        # ele_pt_cut, mu_pt_cut = 33, 26
        ele_pt_cut, mu_pt_cut = 33, 26
    elif year == "2024":
        # ele_pt_cut, mu_pt_cut = 33, 26
        ele_pt_cut, mu_pt_cut = 30, 24

    else:
        raise ValueError(f"Unknown year {year}")

    # electrons = events.Electron

    good_electrons = (
        (electrons.pt > ele_pt_cut) &
        (np.abs(electrons.eta) < 2.5) &  # keep within tracker acceptance
        ~((np.abs(electrons.eta) > 1.44) & (np.abs(electrons.eta) < 1.57)) &  # remove transition
        (electrons.mvaIso_WP80) &        # tight MVA ID
        (electrons.pfRelIso03_all < 0.15)  # isolation cut
    )

    good_muons = (
        (muons.pt > mu_pt_cut)
        & (np.abs(muons.eta) < 2.4)
        & (muons.pfRelIso03_all < 0.15)
    )

    one_ele = ak.num(electrons[good_electrons]) == 1
    one_mu = ak.num(muons[good_muons]) == 1
    lepton_channel_mask = one_ele | one_mu
    # lepton_channel_mask = one_mu

    selected_electrons = electrons[good_electrons]
    print("selected_electrons", len(selected_electrons[ak.num(selected_electrons.pt)>0]))
    selected_muons = muons[good_muons]
    print("selected_muons", len(selected_muons[ak.num(selected_muons.pt)>0]))
    selected_leptons = ak.concatenate([selected_electrons, selected_muons], axis=1)
    print("selected_leptons", len(selected_leptons[ak.num(selected_leptons.pt)>0]))
    print("selected Electrons", len(selected_leptons[ak.num(selected_leptons[abs(selected_leptons.pdgId)==11])>0]))
    print("selected Muons", len(selected_leptons[ak.num(selected_leptons[abs(selected_leptons.pdgId)==13])>0]))

    # ------------------------
    # Jet selection
    # ------------------------
    good_jets = (
        (jets.pt > 20)
        & (np.abs(jets.eta) < 2.4)
        & (jets.btagUParTAK4B > 0.1272)
    )
    selected_bjets = jets[good_jets] 
    print("selected_b_jets: ", selected_bjets)
    at_least_one_bjets = ak.num(selected_bjets) >= 1
    # at_least_two_bjets = ak.num(selected_bjets) >= 2

    # keep top 2 by DeepJet score
    # top2_bjets = selected_jets[ak.argsort(selected_jets.btagDeepFlavB, ascending=False)][:, :2]

    # ------------------------
    # Photon selection (from photon_preselection output)
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

    # Combine everything
    good_photons = (
        (photons.pt > 10)
        & valid_eta
        & (barrel_cut | endcap_cut)
        & (~photons.pixelSeed)
    )
    selected_photons = photons[good_photons]
    at_least_two_photons = ak.num(selected_photons) >= 2

    dr = delta_r_manual(selected_leptons, selected_photons)
    dr_mask = ak.all(ak.all(dr > 0.4, axis=-1), axis=-1)

    # ΔR between electrons and photons
    dr_electrons = delta_r_manual(selected_electrons, selected_photons)

    # ΔR between muons and photons
    dr_muons = delta_r_manual(selected_muons, selected_photons)


    event_mask = lepton_channel_mask & at_least_one_bjets & at_least_two_photons & dr_mask

    # ------------------------
    # Apply mask — keep length same, empties for failed events
    # ------------------------
    empty_photons = ak.Array([[]] * len(events))
    empty_bjets = ak.Array([[]] * len(events))
    empty_leptons = ak.Array([[]] * len(events))

    filtered_photons = ak.where(event_mask, selected_photons, empty_photons)
    filtered_jets = ak.where(event_mask, selected_bjets, empty_bjets)
    filtered_leptons = ak.where(event_mask, selected_leptons, empty_leptons)

    return event_mask, filtered_photons, filtered_jets, filtered_leptons


def delta_r_manual(obj1, obj2):
    deta = obj1.eta[:, None] - obj2.eta
    dphi = np.abs(obj1.phi[:, None] - obj2.phi)
    dphi = ak.where(dphi > np.pi, 2 * np.pi - dphi, dphi)
    return np.sqrt(deta**2 + dphi**2)

def object_preselections(
    photons: ak.Array,
    jets: ak.Array,
    electrons: ak.Array,
    muons: ak.Array,
    year="2018",
):
    """
    Apply object-level preselection only.
    Returns selected photons, bjets, and leptons.
    No event-level requirements applied here.
    """

    # ------------------------
    # Year-dependent lepton pT cuts
    # ------------------------
    if year.startswith("2016"):
        ele_pt_cut, mu_pt_cut = 27, 26
    elif year == "2017":
        ele_pt_cut, mu_pt_cut = 33, 29
    elif year == "2018":
        ele_pt_cut, mu_pt_cut = 33, 26
    elif year == "2024":
        ele_pt_cut, mu_pt_cut = 30, 24
    else:
        raise ValueError(f"Unknown year {year}")

    # ========================
    # ELECTRONS
    # ========================
    good_electrons = (
        (electrons.pt > ele_pt_cut) &
        (np.abs(electrons.eta) < 2.5) &
        ~((np.abs(electrons.eta) > 1.44) & (np.abs(electrons.eta) < 1.57)) &
        (electrons.mvaIso_WP80) &
        (electrons.pfRelIso03_all < 0.15)
    )

    selected_electrons = electrons[good_electrons]

    # ========================
    # MUONS
    # ========================
    good_muons = (
        (muons.pt > mu_pt_cut) &
        (np.abs(muons.eta) < 2.4) &
        (muons.pfRelIso03_all < 0.15)
    )

    selected_muons = muons[good_muons]

    # Combine leptons (object level only)
    selected_leptons = ak.concatenate(
        [selected_electrons, selected_muons],
        axis=1
    )

    # ========================
    # BJETS
    # ========================
    good_jets = (
        (jets.pt > 20) &
        (np.abs(jets.eta) < 2.4) &
        (jets.btagUParTAK4B > 0.1272)
    )

    selected_bjets = jets[good_jets]

    # ========================
    # PHOTONS
    # ========================
    abs_eta = np.abs(photons.eta)

    valid_eta = (
        (abs_eta <= 2.5) &
        ~((abs_eta >= 1.442) & (abs_eta <= 1.566))
    )

    is_barrel = abs_eta < 1.442
    is_endcap = (abs_eta > 1.566) & (abs_eta < 2.5)

    barrel_cut = is_barrel & (photons.mvaID > -0.02)
    endcap_cut = is_endcap & (photons.mvaID > -0.26)

    good_photons = (
        (photons.pt > 10) &
        valid_eta &
        (barrel_cut | endcap_cut) &
        (~photons.pixelSeed)
    )

    selected_photons = photons[good_photons]

    return selected_photons, selected_bjets, selected_leptons, selected_electrons, selected_muons

# -------------------------------------------------
# Prepare containers
# -------------------------------------------------
all_photons = []
all_jets    = []
all_electrons = []
all_muons = []
all_event_masks = []

# -------------------------------------------------
# Loop over files
# -------------------------------------------------
for file in files[:30]:

    print(f"Processing: {file}")

    factory = NanoEventsFactory.from_root(
        f"{file}:Events",
        schemaclass=NanoAODSchema,
    )

    events = factory.events()

    # Run your preselection
    # evt_mask, fp, fj, fl = photon_preselections(
    #     events.Photon,
    #     events.Jet,
    #     events.Electron,
    #     events.Muon,
    #     events,
    #     year="2024"
    # )

    fp, fj, fl, fe, fm = object_preselections(
        events.Photon,
        events.Jet,
        events.Electron,
        events.Muon,
        year="2024"
    )

    one_ele = ak.num(fe) == 1
    one_mu = ak.num(fm) == 1
    lepton_channel_mask = one_ele | one_mu
    at_least_one_b = ak.num(fj) >= 1
    at_least_two_pho = ak.num(fp) >= 2
    dr = delta_r_manual(fl, fp)
    dr_mask = ak.all(ak.all(dr > 0.4, axis=-1), axis=-1)

    event_mask = lepton_channel_mask & at_least_one_b & at_least_two_pho & dr_mask

    # Store
    all_photons.append(fp)
    all_jets.append(fj)
    # all_leptons.append(fl)
    all_electrons.append(fe)
    all_muons.append(fm)
    all_event_masks.append(event_mask)

# -------------------------------------------------
# Merge across all files
# -------------------------------------------------
merged_photons = ak.concatenate(all_photons, axis=0)
merged_jets    = ak.concatenate(all_jets, axis=0)
# merged_leptons = ak.concatenate(all_leptons, axis=0)
merged_electrons = ak.concatenate(all_electrons, axis=0)
merged_muons = ak.concatenate(all_muons, axis=0)
merged_event_masks = ak.concatenate(all_event_masks, axis=0)

print("Total events after preselection:", len(merged_photons))

output_file = "TTto2L2Nu_2024_preselected.parquet"

ak.to_parquet(
    {
        "photons": merged_photons,
        "jets": merged_jets,
        "electrons": merged_electrons,
        "muons": merged_muons,
        "event_masks": merged_event_masks,
    },
    output_file,
)

print(f"Saved to {output_file}")


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

import math

def merge_overflow(
    hist,
    add_underflow=False,
    add_overflow=True,
    label_overflow=True,
    float_format=".0f"
):
    """
    Merge overflow/underflow into visible bins.
    Optionally relabel last bin as ≥ upper_edge.

    Parameters
    ----------
    hist : ROOT.TH1
    add_underflow : bool
    add_overflow : bool
    label_overflow : bool
        If True, rename last bin to ≥ upper_edge
    float_format : str
        Formatting for upper edge (e.g. ".0f", ".1f")
    """

    n_bins = hist.GetNbinsX()

    # --- Underflow → first bin
    if add_underflow:
        underflow = hist.GetBinContent(0)
        first = hist.GetBinContent(1)

        hist.SetBinContent(1, first + underflow)

        err0 = hist.GetBinError(0)
        err1 = hist.GetBinError(1)
        hist.SetBinError(1, math.sqrt(err0**2 + err1**2))

        hist.SetBinContent(0, 0)
        hist.SetBinError(0, 0)

    # --- Overflow → last bin
    if add_overflow:
        overflow = hist.GetBinContent(n_bins + 1)
        last = hist.GetBinContent(n_bins)

        hist.SetBinContent(n_bins, last + overflow)

        errN = hist.GetBinError(n_bins)
        errOv = hist.GetBinError(n_bins + 1)
        hist.SetBinError(n_bins, math.sqrt(errN**2 + errOv**2))

        hist.SetBinContent(n_bins + 1, 0)
        hist.SetBinError(n_bins + 1, 0)

        # # --- Relabel last bin
        # if label_overflow:
        #     upper_edge = hist.GetXaxis().GetBinUpEdge(n_bins)
        #     label = f"#geq {format(upper_edge, float_format)}"
        #     hist.GetXaxis().SetBinLabel(n_bins, label)






