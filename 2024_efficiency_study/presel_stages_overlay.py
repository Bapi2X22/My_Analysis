import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import ROOT

Plot_dir = '/eos/user/b/bbapi/www/Analysis_plots/Jets/btag_wp_optimization/probbb_thr/'

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


import ROOT
import awkward as ak
import numpy as np

ROOT.gStyle.SetOptStat(0)

def plot_variable(obj_array, index, masks, title,
                         xmin=0, xmax=200, nbins=50, Mass=None, total_events=None):

    canvas = ROOT.TCanvas(title, title, 800, 700)
    canvas.SetLeftMargin(0.12)
    canvas.SetLogy()
    legend = ROOT.TLegend(0.6, 0.65, 0.88, 0.88)
    legend.SetBorderSize(0)

    colors = [
        ROOT.kBlack,
        ROOT.kBlue,
        ROOT.kRed,
        ROOT.kGreen+2,
        ROOT.kMagenta,
        ROOT.kOrange+1,
    ]

    hist_list = []

    for i, (label, mask) in enumerate(masks):

        # Defensive multiplicity protection
        multiplicity_mask = ak.num(obj_array) > index
        final_mask = mask & multiplicity_mask

        if ak.sum(final_mask) == 0:
            continue

        # pts = obj_array[final_mask].pt[:, index]
        # pts = ak.to_numpy(ak.flatten(pts))
        pts = obj_array[final_mask].pt[:, index]
        pts = ak.to_numpy(pts)

        if len(pts) == 0:
            continue

        hist = ROOT.TH1F(
            f"h_{title}_{i}",
            f";{title} p_{{T}} [GeV];Events",
            nbins,
            xmin,
            xmax,
        )

        hist.SetMinimum(0.1)

        hist.SetLineColor(colors[i])
        hist.SetLineWidth(2)

        for pt in pts:
            hist.Fill(pt)

        n_events = len(pts)

        hist_list.append(hist)
        legend.AddEntry(hist, f"{label} ({n_events}, {n_events/total_events*100:.1f}%)", "l")

    # Draw
    for i, hist in enumerate(hist_list):
        if i == 0:
            hist.Draw("HIST")
        else:
            hist.Draw("HIST SAME")

    legend.Draw()
    canvas.Update()
    canvas.Draw()

    CMS_label(canvas)

    latex = ROOT.TLatex()
    latex.SetNDC()              # normalized (0–1) coords
    latex.SetTextSize(0.04)
    latex.SetTextFont(42)
    latex.DrawLatex(0.25, 0.85, f"M_{{A}} = {Mass} GeV")

    canvas.SaveAs(Plot_dir+f"{title.replace(' ', '_')}_{Mass}.png")

    return canvas


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

def run_for_mass(M):

    diB_mask_thr = 0.40
    Plot_dir = '/eos/user/b/bbapi/www/Analysis_plots/Jets/btag_wp_optimization/probbb_thr/'
    print(f"Running analysis for M = {M}")

    file = f"/eos/user/b/bbapi/MC_contacts/2024_signal_samples_WH/CMSSW_15_0_15/src/WH_2024_M{M}.root"

    factory = NanoEventsFactory.from_root(
        f"{file}:Events",
        schemaclass=NanoAODSchema,
    )

    events = factory.events()

    selected_photons, selected_bjets, selected_leptons, selected_electrons, selected_muons = object_preselections(
        events.Photon,
        events.Jet,
        events.Electron,
        events.Muon,
        year="2024"
    )

    one_ele = ak.num(selected_electrons) == 1
    one_mu = ak.num(selected_muons) == 1
    lepton_channel_mask = one_ele | one_mu
    at_least_one_b = ak.num(selected_bjets) >= 1
    at_least_two_pho = ak.num(selected_photons) >= 2
    dr = delta_r_manual(selected_leptons, selected_photons)
    dr_mask = ak.all(ak.all(dr > 0.4, axis=-1), axis=-1)

    event_mask = lepton_channel_mask & at_least_one_b & at_least_two_pho & dr_mask


    mask1 = lepton_channel_mask
    mask2 = mask1 & (ak.num(selected_photons) >= 1)
    mask3 = mask2 & (ak.num(selected_photons) >= 2)
    mask4 = mask3 & (ak.num(selected_bjets) >= 1)
    mask5 = mask4 & (ak.num(selected_bjets) >= 2)
    mask6 = mask5 & dr_mask

    masks = [
        ("1 lepton", mask1),
        ("#geq 1 #gamma", mask2),
        ("#geq 2 #gamma", mask3),
        ("#geq 1 b-jet", mask4),
        ("#geq 2 b-jets", mask5),
        ("#DeltaR > 0.4", mask6),
    ]


    plot_variable(selected_photons, 0, masks, "Leading Photon", Mass=M, total_events=len(events))
    plot_variable(selected_photons, 1, masks, "Subleading Photon", Mass=M, total_events=len(events))
    plot_variable(selected_bjets, 0, masks, "Leading b-jet", Mass=M, total_events=len(events))
    plot_variable(selected_bjets, 1, masks, "Subleading b-jet", Mass=M, total_events=len(events))


if __name__ == "__main__":

    masses = [15, 20, 25, 30, 35, 40, 45, 50, 60]

    for M in masses:
        run_for_mass(M)


