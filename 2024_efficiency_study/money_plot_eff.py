import awkward as ak
import numpy as np
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import ROOT

ROOT.gStyle.SetOptStat(0)

diB_mask_thr = 0.40
Plot_dir = '/eos/user/b/bbapi/www/Analysis_plots/Jets/btag_wp_optimization/probbb_thr/'

def CMS_label(pad,
              lumi="109 fb^{-1}",
              year="2024",
              energy="13.6 TeV",
              status="Simulation Preliminary",
              x=0.12,
              y=0.94):

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
        latex.DrawLatex(x + 0.11, y, status)

    # ---- Lumi text (right aligned) ----
    latex.SetTextFont(42)
    latex.SetTextSize(0.035)
    latex.SetTextAlign(31)
    lumi_text = f"({year}-{lumi})"
    latex.DrawLatex(0.88, y, lumi_text)

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

def get_category_efficiencies(file, pt_thr):

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

    resolved = n_bjets >= 2

    merged = ((n_bjets == 1) &
              (fj.btagUParTAK4probbb[:,0] > fj.btagUParTAK4probb[:,0]))

    one_jet = ((n_bjets == 1) &
               (fj.btagUParTAK4probb[:,0] > fj.btagUParTAK4probbb[:,0]))

    eff_res = 100. * ak.sum(resolved) / total_gen
    eff_mer = 100. * ak.sum(merged) / total_gen
    eff_one = 100. * ak.sum(one_jet) / total_gen

    return eff_res, eff_mer, eff_one


mass_points = [15, 20, 25, 30, 35, 40, 45, 50, 60]

n = len(mass_points)

def make_empty_hist(name):
    return ROOT.TH1F(name, "", n, 0, n)

# pt = 10
h_res_10 = make_empty_hist("h_res_10")
h_mer_10 = make_empty_hist("h_mer_10")
h_one_10 = make_empty_hist("h_one_10")

# pt = 15
h_res_15 = make_empty_hist("h_res_15")
h_mer_15 = make_empty_hist("h_mer_15")
h_one_15 = make_empty_hist("h_one_15")

for i, m in enumerate(mass_points):

    file = f'/eos/user/b/bbapi/MC_contacts/2024_signal_samples_WH/CMSSW_15_0_15/src/WH_2024_M{m}.root'

    r10, m10, o10 = get_category_efficiencies(file, 10)
    r15, m15, o15 = get_category_efficiencies(file, 15)

    bin_idx = i + 1

    h_res_10.SetBinContent(bin_idx, r10)
    h_mer_10.SetBinContent(bin_idx, m10)
    h_one_10.SetBinContent(bin_idx, o10)

    h_res_15.SetBinContent(bin_idx, r15)
    h_mer_15.SetBinContent(bin_idx, m15)
    h_one_15.SetBinContent(bin_idx, o15)

    # ----------------------------
    # COLORS (same for both pt)
    # ----------------------------

    # Resolved
    h_res_10.SetFillColorAlpha(ROOT.kBlue-7, 0.6)

    # Merged
    h_mer_10.SetFillColorAlpha(ROOT.kRed-4, 0.6)

    # One-jet
    h_one_10.SetFillColorAlpha(ROOT.kGreen+2, 0.6)

    for h in [h_res_15, h_mer_15, h_one_15]:
        h.SetFillStyle(0)

    # total efficiency histograms
    h_tot_10 = h_res_10.Clone("h_tot_10")
    h_tot_10.Add(h_mer_10)
    h_tot_10.Add(h_one_10)

    h_tot_15 = h_res_15.Clone("h_tot_15")
    h_tot_15.Add(h_mer_15)
    h_tot_15.Add(h_one_15)

    h_ratio_res = h_res_15.Clone("h_ratio_res")
    h_ratio_res.Divide(h_res_10)

    h_ratio_mer = h_mer_15.Clone("h_ratio_mer")
    h_ratio_mer.Divide(h_mer_10)

    h_ratio_one = h_one_15.Clone("h_ratio_one")
    h_ratio_one.Divide(h_one_10)


    # ----------------------------
    # BORDER STYLES
    # ----------------------------

    # Solid → pt = 10
    for h in [h_res_10, h_mer_10, h_one_10]:
        h.SetLineColor(ROOT.kBlack)
        h.SetLineWidth(2)
        h.SetLineStyle(1)

    # Dashed → pt = 15
    for h in [h_res_15, h_mer_15, h_one_15]:
        h.SetLineColor(ROOT.kBlack)
        h.SetLineWidth(2)
        h.SetLineStyle(2)

    for h in [h_res_10, h_res_15,
              h_mer_10, h_mer_15,
              h_one_10, h_one_15]:
        h.GetXaxis().SetBinLabel(bin_idx, f"{m}")


stack_10 = ROOT.THStack("stack_10", "")
stack_10.Add(h_res_10)
stack_10.Add(h_mer_10)
stack_10.Add(h_one_10)

c = ROOT.TCanvas("c", "", 800, 800)

pad1 = ROOT.TPad("pad1","",0,0.30,1,1)
pad2 = ROOT.TPad("pad2","",0,0.00,1,0.30)

pad1.SetBottomMargin(0.02)
pad2.SetTopMargin(0.05)
pad2.SetBottomMargin(0.35)

pad1.Draw()
pad2.Draw()

pad1.cd()

stack_10.Draw("HIST")
stack_10.SetMaximum(10.0)
stack_10.SetMinimum(0.0)

stack_10.GetYaxis().SetTitle("Efficiency [%]")
stack_10.GetYaxis().SetTitleOffset(1.3)
stack_10.GetXaxis().SetLabelSize(0)

CMS_label(c)

leg = ROOT.TLegend(0.65, 0.8, 0.88, 0.93)
leg.SetBorderSize(0)
leg.SetFillStyle(0)

leg.AddEntry(h_res_10, "Resolved (p_{T} > 10 GeV)", "f")

leg.AddEntry(h_mer_10, "Merged (p_{T} > 10 GeV)", "f")

leg.AddEntry(h_one_10, "One-jet (p_{T} > 10 GeV)", "f")

leg.Draw()

CMS_label(c)

pad2.cd()

h_ratio_res.SetLineColor(ROOT.kBlue+2)
h_ratio_mer.SetLineColor(ROOT.kRed+1)
h_ratio_one.SetLineColor(ROOT.kGreen+3)

for h in [h_ratio_res, h_ratio_mer, h_ratio_one]:
    h.SetLineWidth(2)
    h.SetMarkerStyle(20)

h_ratio_res.SetMinimum(0.0)
h_ratio_res.SetMaximum(1.0)

h_ratio_res.GetYaxis().SetTitle(f"#epsilon(p_{{T}}>15) / #epsilon(p_{{T}}>10)")
h_ratio_res.GetYaxis().SetTitleSize(0.07)
h_ratio_res.GetYaxis().SetLabelSize(0.06)
h_ratio_res.GetYaxis().SetTitleOffset(0.6)

h_ratio_res.GetXaxis().SetTitle("m_{A} [GeV]")
h_ratio_res.GetXaxis().SetTitleSize(0.10)
h_ratio_res.GetXaxis().SetLabelSize(0.09)

h_ratio_res.Draw("E1")
h_ratio_mer.Draw("E1 SAME")
h_ratio_one.Draw("E1 SAME")


c.SaveAs(Plot_dir+f"efficiency_comparison_newId.pdf")
c.SaveAs(Plot_dir+f"efficiency_comparison_newId.png")