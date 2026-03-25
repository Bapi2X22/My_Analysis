import awkward as ak
import numpy as np
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import ROOT

ROOT.gStyle.SetOptStat(0)

diB_mask_thr = 0.40
Plot_dir = '/eos/user/b/bbapi/www/Background_study_2024/'

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
    latex.SetTextSize(0.04)
    latex.DrawLatex(x, y, "CMS")

    # ---- Status (italic) ----
    if status != "":
        latex.SetTextFont(52)
        latex.SetTextSize(0.035)
        latex.DrawLatex(x + 0.08, y, status)

    # ---- Lumi text (right aligned) ----
    latex.SetTextFont(42)
    latex.SetTextSize(0.035)
    latex.SetTextAlign(31)
    lumi_text = f"({year}-{lumi})"
    latex.DrawLatex(0.92, y, lumi_text)

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
    ph_thr = 15,
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
    barrel_cut = is_barrel & (photons.mvaID > 0.0439603)
    endcap_cut = is_endcap & (photons.mvaID > -0.249526)

    # Combine everything
    good_photons = (
        (photons.pt > ph_thr)
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

def build_fraction_hist(name, f1, f2, f3):

    h = ROOT.TH1F(
        name,
        ";Category;Fraction of Events",
        3, 0, 3
    )

    h.SetBinContent(1, f1)
    h.SetBinContent(2, f2)
    h.SetBinContent(3, f3)

    h.GetXaxis().SetBinLabel(1, "#geq 2 bJets")
    h.GetXaxis().SetBinLabel(2, "probbb > probb")
    h.GetXaxis().SetBinLabel(3, "probb > probbb")

    return h

def get_category_fractions(file, pt_thr):

    factory = NanoEventsFactory.from_root(
        f"{file}:Events",
        schemaclass=NanoAODSchema,
    )
    events = factory.events()

    total_gen = len(events)

    evt_mask, fp, fj, fl = photon_preselections(
        events.Photon, events.Jet,
        events.Electron, events.Muon,
        events, ph_thr=pt_thr, year='2024'
    )

    events = events[evt_mask]
    fj = fj[evt_mask]

    mask_jet = ak.num(fj.pt) > 0
    fj = fj[mask_jet]

    n_bjets = ak.num(fj.pt)

    n_total = len(n_bjets) 

    resolved = n_bjets >= 2

    merged = ((n_bjets == 1) &
              (fj.btagUParTAK4probbb[:,0] > fj.btagUParTAK4probb[:,0]))

    one_jet = ((n_bjets == 1) &
               (fj.btagUParTAK4probb[:,0] > fj.btagUParTAK4probbb[:,0]))

    n_cat1 = int(ak.sum(resolved))
    n_cat2 = int(ak.sum(merged))
    n_cat3 = int(ak.sum(one_jet))

    f_cat1 = round(n_cat1 / n_total,2)
    f_cat2 = round(n_cat2 / n_total,2)
    f_cat3 = round(n_cat3 / n_total,2)

    return f_cat1, f_cat2, f_cat3, n_cat1, n_cat2, n_cat3


def plot_fraction_comparison(file, label):

    # --- Get fractions ---
    f10 = get_category_fractions(file, 10)
    f15 = get_category_fractions(file, 15)

    f1_10, f2_10, f3_10, n1_10, n2_10, n3_10 = f10
    f1_15, f2_15, f3_15, n1_15, n2_15, n3_15 = f15

    # --- Histograms ---
    h10 = build_fraction_hist("h10", n1_10, n2_10, n3_10)
    h15 = build_fraction_hist("h15", n1_15, n2_15, n3_15)

    h10.SetFillColor(ROOT.kAzure-9)
    h10.SetLineColor(ROOT.kBlack)

    h15.SetFillColor(ROOT.kOrange-3)
    h15.SetLineColor(ROOT.kBlack)

    h10.GetXaxis().SetLabelSize(0)
    h15.GetXaxis().SetLabelSize(0)

    # --- Canvas with pads ---
    c = ROOT.TCanvas("c", "", 700, 700)

    pad1 = ROOT.TPad("pad1","",0,0.3,1,1)
    pad2 = ROOT.TPad("pad2","",0,0,1,0.3)

    pad1.SetBottomMargin(0.02)
    pad1.SetLeftMargin(0.12)
    pad2.SetLeftMargin(0.12)
    pad1.SetRightMargin(0.025)
    pad2.SetRightMargin(0.025)
    pad2.SetTopMargin(0.05)
    pad2.SetBottomMargin(0.35)

    pad1.Draw()
    pad2.Draw()

    # -------------------------
    # Top pad
    # -------------------------
    pad1.cd()

    # h10.SetMaximum(1.0)

    max_val = max(h10.GetMaximum(), h15.GetMaximum())
    h10.SetMaximum(max_val * 1.25)  

    h10.Draw("HIST")
    h15.Draw("HIST SAME")

    CMS_label(pad1)

    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.04)
    latex.DrawLatex(0.17,0.85,"After preselection")
    latex.DrawLatex(0.80,0.85,label)

    leg = ROOT.TLegend(0.75,0.72,0.95,0.82)
    leg.AddEntry(h10,"p_{T} > 10 GeV","f")
    leg.AddEntry(h15,"p_{T} > 15 GeV","f")
    leg.SetFillColor(0)
    leg.SetBorderSize(0)
    leg.Draw()

    # -------------------------
    # Ratio pad
    # -------------------------
    pad2.cd()

    h_ratio = h15.Clone("ratio")
    h_ratio.Divide(h10)

    h_ratio.GetYaxis().SetTitle(f" N(p_{{T}}>15) / N(p_{{T}}>10)")
    h_ratio.GetYaxis().SetTitleSize(0.07)
    h_ratio.GetYaxis().SetTitleOffset(0.6)

    h_ratio.SetMaximum(1.0)
    h_ratio.SetMinimum(0)

    h_ratio.GetYaxis().SetLabelSize(0.06)

    h_ratio.GetXaxis().SetTitleSize(0.12)
    h_ratio.GetXaxis().SetLabelSize(0.10)

    h_ratio.Draw("HIST")


    c.SaveAs(Plot_dir+label+f"/category_fraction_{label}.pdf")
    c.SaveAs(Plot_dir+label+f"/category_fraction_{label}.png")


backgrounds = {
    "TTG1Jets": "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/skimmed_TTG1Jets/TTG1Jets_preselected.root",
    "TTto2L2Nu": "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/skimmed_TTto2L2Nu/TTto2L2Nu_preselected.root",
    "TTtoLNu2Q": "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/skimmed/TTtoLNu2Q_preselected.root"
}

for label, file in backgrounds.items():
    plot_fraction_comparison(file, label)

