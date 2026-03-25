import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import ROOT
# vector.register_awkward()

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


def get_mgg(one_photon, category_mask):

    cat_ph = one_photon[category_mask]

    sorted_ph = cat_ph[ak.argsort(cat_ph.pt, ascending=False)]
    padded_ph = ak.pad_none(sorted_ph, 2)

    lead = padded_ph[:, 0]
    sub  = padded_ph[:, 1]

    valid_mask = (
        ~ak.is_none(lead.pt) &
        ~ak.is_none(sub.pt)
    )

    lead = lead[valid_mask]
    sub  = sub[valid_mask]

    lead_p4 = ak.zip(
        {
            "pt": lead.pt,
            "eta": lead.eta,
            "phi": lead.phi,
            "mass": ak.zeros_like(lead.pt),
        },
        with_name="Momentum4D",
    )

    sub_p4 = ak.zip(
        {
            "pt": sub.pt,
            "eta": sub.eta,
            "phi": sub.phi,
            "mass": ak.zeros_like(sub.pt),
        },
        with_name="Momentum4D",
    )

    diphoton = lead_p4 + sub_p4

    return ak.to_numpy(diphoton.mass)

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

    evt_mask, fp, fj, fl = photon_preselections(events.Photon, events.Jet, events.Electron, events.Muon, events, year='2024')

    gen = events[evt_mask].GenPart

    bquarks = gen[abs(gen.pdgId) == 5]
    mother_idx = bquarks.genPartIdxMother
    from_a_mask = gen[mother_idx].pdgId == 35
    bquarks_from_a = bquarks[from_a_mask]

    fj = fj[ak.num(fj.pt)>0]

    # -----------------------------
    # Compute multiplicity per event
    # -----------------------------
    n_fj = ak.to_numpy(ak.num(fj.pt))   # number of fj per event

    # -----------------------------
    # Create histogram
    # -----------------------------
    ROOT.gStyle.SetOptStat(1111111)

    h = ROOT.TH1F(
        "h_nfj",
        ";Number of b-jets per event;Events",
        5, 0, 5
    )

    for v in n_fj:
        h.Fill(v)

    # -----------------------------
    # Draw
    # -----------------------------
    c = ROOT.TCanvas("c_nfj", "c_nfj", 800, 600)
    c.SetLogy()
    h.Draw("HIST")
    h.SetMinimum(0.1)

    CMS_label(c)

    c.Update()

    stats = h.FindObject("stats")
    if stats:
        stats.SetTextSize(0.035)     # enlarge text
        stats.SetX1NDC(0.60)         # left
        stats.SetX2NDC(0.88)         # right
        stats.SetY1NDC(0.60)         # bottom
        stats.SetY2NDC(0.88)         # top
        stats.SetBorderSize(1)

    latex = ROOT.TLatex()
    latex.SetNDC()              # normalized (0–1) coords
    latex.SetTextSize(0.03)
    latex.SetTextFont(42)
    latex.DrawLatex(0.12, 0.85, f"M_{{A}} = {M} GeV")

    # -----------------------------
    # Save
    # -----------------------------
    c.Update()
    c.SaveAs(Plot_dir+f"n_fatjets_per_event_{M}.png")


    one_jet = fj[ak.num(fj.pt)==1]


    # =========================
    # Build histograms
    # =========================
    vals = {
        "btagUParTAK4B":      one_jet.btagUParTAK4B,
        "btagUParTAKprobb":   one_jet.btagUParTAK4probb,
        "btagUParTAKprobbb":  one_jet.btagUParTAK4probbb,
    }

    ROOT.gStyle.SetOptStat(111111)
    ROOT.gStyle.SetStatFontSize(0.030)

    hists = {}
    colors = [ROOT.kRed, ROOT.kBlue, ROOT.kGreen+2]

    for (name, arr), col in zip(vals.items(), colors):
        h = ROOT.TH1F(name, name, 50, 0, 1)
        h.SetDirectory(0)                 # prevent ROOT ownership
        for x in ak.to_numpy(arr):
            h.Fill(x)
        h.SetLineColor(col)
        h.SetLineWidth(2)
        h.SetMinimum(0.1)
        hists[name] = h

    # =========================
    # FORCE statbox creation (once)
    # =========================
    tmp = ROOT.TCanvas("tmp_stats", "", 1, 1)
    for h in hists.values():
        h.Draw("hist")
        tmp.Update()
    tmp.Close()

    # =========================
    # Extract original statboxes
    # =========================
    orig_stats = {}
    for h in hists.values():
        st = h.GetListOfFunctions().FindObject("stats")
        if not st:
            raise RuntimeError(f"Stats not created for {h.GetName()}")
        orig_stats[h] = st.Clone(f"orig_stats_{h.GetName()}")

    # =========================
    # DISABLE ROOT automatic stat drawing (CRITICAL)
    # =========================
    ROOT.gStyle.SetOptStat(0)

    # =========================
    # Real canvas + overlay
    # =========================
    c = ROOT.TCanvas("c", "", 700, 600)
    c.SetRightMargin(0.20)
    c.SetLogy()

    first = True
    for h in hists.values():
        h.SetTitle("")
        h.GetXaxis().SetTitle("btagUParTAK4 tagger score")
        h.GetXaxis().SetTitleSize(0.045)
        h.GetXaxis().SetLabelSize(0.04)

        if first:
            h.Draw("hist")
            first = False
        else:
            h.Draw("hist same")

    # =========================
    # Legend
    # =========================
    leg = ROOT.TLegend(0.35, 0.65, 0.68, 0.85)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    for name, h in hists.items():
        leg.AddEntry(h, name, "l")
    leg.Draw()

    # =========================
    # Text
    # =========================
    # latex = ROOT.TLatex()
    # latex.SetNDC()
    # latex.SetTextFont(42)

    # latex.SetTextSize(0.04)
    # latex.DrawLatex(0.17, 0.85, f"M_{{A}} = {M} GeV")

    # latex.SetTextSize(0.045)
    # latex.DrawLatex(0.25, 0.93, "btagUParTAK4 distribution")

    # =========================
    # Draw ONLY cloned statboxes
    # =========================
    x1, x2 = 0.80, 0.98
    y_top  = 0.92
    h_box  = 0.18
    gap    = 0.02

    for i, (hst, st) in enumerate(orig_stats.items()):
        st.SetParent(c)

        y2 = y_top - i * (h_box + gap)
        y1 = y2 - h_box

        st.SetX1NDC(x1)
        st.SetX2NDC(x2)
        st.SetY1NDC(y1)
        st.SetY2NDC(y2)

        st.SetTextColor(hst.GetLineColor())
        st.SetTextFont(62)     # avoid TTF bug
        st.SetFillStyle(0)
        st.SetBorderSize(1)

        st.Draw()
    
    CMS_label(c)

    latex = ROOT.TLatex()
    latex.SetNDC()              # normalized (0–1) coords
    latex.SetTextSize(0.04)
    latex.SetTextFont(42)
    latex.DrawLatex(0.17, 0.85, f"M_{{A}} = {M} GeV")

    # =========================
    # Finalize
    # =========================
    c.Modified()
    c.Update()
    c.SaveAs(Plot_dir + f"btag_overlay_{M}.png")

    bquarks_from_a = bquarks_from_a[ak.num(fj.pt)==1]

    one_jet = one_jet[:, 0]

    dR = delta_r_manual(one_jet[:, None], bquarks_from_a)

    match_mask = dR < 0.4

    n_b_matched = ak.sum(match_mask, axis=1)

    dib_mask = one_jet.btagUParTAK4probbb > diB_mask_thr

    # Counts (convert awkward sums to python ints if needed)
    count_0b = int(ak.sum(n_b_matched == 0))
    count_1b = int(ak.sum(n_b_matched == 1))
    count_2b = int(ak.sum(n_b_matched == 2))

    # Create histogram with 3 bins
    h = ROOT.TH1F(
        "h_nb",
        ";Number of Matched b-quark (UParTAK4B medium, Nb-Jet == 1);Number of b-Jets",
        3, 0, 3
    )

    # Fill bins manually
    h.SetBinContent(1, count_0b)
    h.SetBinContent(2, count_1b)
    h.SetBinContent(3, count_2b)

    # Set bin labels
    h.GetXaxis().SetBinLabel(1, "0 b")
    h.GetXaxis().SetBinLabel(2, "1 b")
    h.GetXaxis().SetBinLabel(3, "2 b")

    h.GetXaxis().SetLabelSize(0.08) 
    h.GetXaxis().SetTitleOffset(1.3) 
    # Style
    h.SetFillColor(ROOT.kAzure + 1)
    h.SetLineColor(ROOT.kBlack)
    h.SetStats(0)

    # Draw
    c = ROOT.TCanvas("c", "", 800, 600)
    c.SetLeftMargin(0.15)
    h.Draw("BAR")

    latex = ROOT.TLatex()
    latex.SetNDC()              # normalized (0–1) coords
    latex.SetTextSize(0.04)
    latex.SetTextFont(42)
    latex.DrawLatex(0.17, 0.85, f"M_{{A}} = {M} GeV")

    CMS_label(c)

    c.Draw()

    c.SaveAs(Plot_dir + f"Nb_{M}.png")

    ROOT.gStyle.SetOptStat(0)

    def frac_nb(mask):
        nb = n_b_matched[mask]
        tot = len(dR)
        return [ak.sum(nb == n) / tot for n in [0, 1, 2]]

    before = frac_nb(ak.ones_like(dib_mask, dtype=bool))
    after  = frac_nb(dib_mask)

    c = ROOT.TCanvas("c", "c", 800, 600)
    c.SetBottomMargin(0.15)

    h_before = ROOT.TH1F("h_before", "", 3, 0, 3)
    h_after  = ROOT.TH1F("h_after",  "", 3, 0, 3)

    # Fill bins
    for i in range(3):
        h_before.SetBinContent(i+1, before[i])
        h_after.SetBinContent(i+1, after[i])


    # Colors & style
    h_before.SetFillColor(ROOT.kBlue-9)
    h_after.SetFillColor(ROOT.kRed-9)

    h_before.SetLineColor(ROOT.kBlue+1)
    h_after.SetLineColor(ROOT.kRed+1)

    h_before.SetBarWidth(0.35)
    h_after.SetBarWidth(0.35)

    h_before.SetBarOffset(0.15)
    h_after.SetBarOffset(0.55)

    # Axis labels
    h_before.GetYaxis().SetTitle("Fraction")
    h_before.GetXaxis().SetTitle(f"Number of Matched b-quark before and after UParTAK4probbb>{diB_mask_thr}")
    h_before.GetYaxis().SetRangeUser(0, 1)
    h_before.GetXaxis().SetLabelSize(0)
    h_before.GetXaxis().SetTitleOffset(1.4)


    h_before.Draw("bar")
    h_after.Draw("bar same")

    latex = ROOT.TLatex()
    latex.SetNDC()              # normalized (0–1) coords
    latex.SetTextSize(0.04)
    latex.SetTextFont(42)
    latex.DrawLatex(0.17, 0.85, f"M_{{A}} = {M} GeV")

    CMS_label(c)


    labels = ["0 b", "1 b", "2 b"]

    latex = ROOT.TLatex()
    latex.SetTextAlign(22)
    latex.SetTextSize(0.04)

    for i, lab in enumerate(labels):
        latex.DrawLatex(i + 0.5, -0.05, lab)


    leg = ROOT.TLegend(0.6, 0.75, 0.88, 0.88)
    leg.AddEntry(h_before, "Before UParTAK4probbb", "f")
    leg.AddEntry(h_after,  "After UParTAK4probbb",  "f")
    leg.Draw()


    c.Update()
    c.SaveAs(Plot_dir+f"dibjet_effect_nb_{M}.png")

    # -----------------------------
    # Compute fractions vs threshold
    # -----------------------------
    thresholds = np.linspace(0.0, 0.9, 21)

    frac_0b, frac_1b, frac_2b = [], [], []

    for thr in thresholds:
        mask = one_jet.btagUParTAK4probbb > thr
        nb = n_b_matched[mask]
        tot = len(nb)
        
        if tot == 0:
            frac_0b.append(0.0)
            frac_1b.append(0.0)
            frac_2b.append(0.0)
        else:
            frac_0b.append(float(ak.sum(nb == 0) / tot))
            frac_1b.append(float(ak.sum(nb == 1) / tot))
            frac_2b.append(float(ak.sum(nb == 2) / tot))

    # -----------------------------
    # Convert to numpy (TGraph needs this)
    # -----------------------------
    x  = np.array(thresholds, dtype="float64")
    y0 = np.array(frac_0b, dtype="float64")
    y1 = np.array(frac_1b, dtype="float64")
    y2 = np.array(frac_2b, dtype="float64")

    # -----------------------------
    # Build ROOT graphs
    # -----------------------------
    g0 = ROOT.TGraph(len(x), x, y0)
    g1 = ROOT.TGraph(len(x), x, y1)
    g2 = ROOT.TGraph(len(x), x, y2)

    for g, col in zip(
        [g2, g1, g0],
        [ROOT.kRed+1, ROOT.kBlue+1, ROOT.kGreen+2]
    ):
        g.SetMarkerStyle(20)
        g.SetMarkerSize(1.0)
        g.SetLineWidth(2)
        g.SetLineColor(col)
        g.SetMarkerColor(col)

    # -----------------------------
    # Draw
    # -----------------------------
    ROOT.gStyle.SetOptStat(0)

    c = ROOT.TCanvas("c", "c", 800, 600)

    mg = ROOT.TMultiGraph()
    mg.Add(g2, "LP")
    mg.Add(g1, "LP")
    mg.Add(g0, "LP")

    mg.Draw("A")
    mg.GetXaxis().SetTitle("UParTAK4probbb cut")
    mg.GetYaxis().SetTitle("Fraction of selected jets")
    mg.GetYaxis().SetRangeUser(0.0, 1.0)

    # -----------------------------
    # Legend
    # -----------------------------
    leg = ROOT.TLegend(0.55, 0.45, 0.88, 0.65)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)        # transparent
    leg.SetTextSize(0.03)
    leg.AddEntry(g2, "2 b matched", "lp")
    leg.AddEntry(g1, "1 b matched", "lp")
    leg.AddEntry(g0, "0 b matched", "lp")
    leg.Draw()

    # -----------------------------
    # Title
    # -----------------------------

    latex2 = ROOT.TLatex()
    latex2.SetNDC()              # normalized (0–1) coords
    latex2.SetTextSize(0.04)
    latex2.SetTextFont(42)
    latex2.DrawLatex(0.17, 0.75, f"M_{{A}} = {M} GeV")

    CMS_label(c)

    # -----------------------------
    # Finalize
    # -----------------------------
    c.Update()
    c.SaveAs(Plot_dir+f"gen_b_mult_vs_dibjet_threshold_{M}.png")

    # -----------------------------
    # Inputs
    # -----------------------------
    thresholds = np.linspace(0.0, 0.9, 21)

    # Purity = 2b fraction (already computed earlier)
    purity = frac_2b

    # -----------------------------
    # Efficiency computation
    # -----------------------------
    total_2b = ak.sum(n_b_matched == 2)
    efficiency = []

    for thr in thresholds:
        mask = (one_jet.btagUParTAK4probbb > thr) & (ak.flatten(n_b_matched) == 2)
        efficiency.append(float(ak.sum(mask) / total_2b))

    # -----------------------------
    # Convert to numpy (required by TGraph)
    # -----------------------------
    x = np.array(thresholds, dtype="float64")
    y_purity = np.array(purity, dtype="float64")
    y_eff    = np.array(efficiency, dtype="float64")

    # -----------------------------
    # Create graphs
    # -----------------------------
    g_purity = ROOT.TGraph(len(x), x, y_purity)
    g_eff    = ROOT.TGraph(len(x), x, y_eff)

    # Style
    g_purity.SetLineColor(ROOT.kRed+1)
    g_purity.SetMarkerColor(ROOT.kRed+1)
    g_purity.SetMarkerStyle(20)
    g_purity.SetLineWidth(2)

    g_eff.SetLineColor(ROOT.kBlue+1)
    g_eff.SetMarkerColor(ROOT.kBlue+1)
    g_eff.SetMarkerStyle(21)
    g_eff.SetLineWidth(2)

    # -----------------------------
    # Draw
    # -----------------------------
    ROOT.gStyle.SetOptStat(0)

    c = ROOT.TCanvas("c", "c", 800, 600)

    mg = ROOT.TMultiGraph()
    mg.Add(g_purity, "LP")
    mg.Add(g_eff, "LP")

    mg.Draw("A")
    mg.GetXaxis().SetTitle("UParTAK4probbb cut")
    mg.GetYaxis().SetTitle("Fraction")
    mg.GetYaxis().SetRangeUser(0.0, 1.1)

    # -----------------------------
    # Legend
    # -----------------------------
    leg = ROOT.TLegend(0.25, 0.20, 0.45, 0.38)
    leg.SetBorderSize(0)
    leg.AddEntry(g_purity, "Purity (2b fraction)", "lp")
    leg.AddEntry(g_eff,    "Efficiency (2b)",     "lp")
    leg.Draw()

    CMS_label(c)

    # -----------------------------
    # Title
    # -----------------------------

    latex = ROOT.TLatex()
    latex.SetNDC()              # normalized (0–1) coords
    latex.SetTextSize(0.04)
    latex.SetTextFont(42)
    latex.DrawLatex(0.17, 0.85, f"M_{{A}} = {M} GeV")

    # -----------------------------
    # Finalize
    # -----------------------------
    c.Update()
    c.SaveAs(Plot_dir+f"dibjet_purity_efficiency_{M}.png")

    # -----------------------------
    # Extract arrays
    # -----------------------------
    probb   = ak.to_numpy(one_jet.btagUParTAK4probb)
    probbb  = ak.to_numpy(one_jet.btagUParTAK4probbb)
    nb      = ak.to_numpy(n_b_matched)

    # Select 2b-matched jets
    mask_2b = (ak.flatten(nb) == 2)

    x = probb[mask_2b]
    y = probbb[mask_2b]

    # -----------------------------
    # Create TGraph (scatter)
    # -----------------------------
    g = ROOT.TGraph(len(x), x.astype("float64"), y.astype("float64"))

    g.SetTitle("")

    g.SetMarkerStyle(20)
    g.SetMarkerSize(0.4)                       # smaller points
    g.SetMarkerColorAlpha(ROOT.kBlue+2, 0.7)   # darker + less transparent
    g.SetLineColor(ROOT.kBlue+2)


    # -----------------------------
    # Draw
    # -----------------------------
    ROOT.gStyle.SetOptStat(0)

    c = ROOT.TCanvas("c", "c", 800, 600)
    c.SetRightMargin(0.02)

    g.Draw("AP")

    g.GetXaxis().SetTitle("probb")
    g.GetYaxis().SetTitle("probbb")
    g.GetXaxis().SetLimits(0.0, 1.0)
    g.GetYaxis().SetRangeUser(0.0, 1.0)

    # -----------------------------
    # Legend
    # -----------------------------
    leg = ROOT.TLegend(0.65, 0.85, 0.85, 0.75)
    leg.SetBorderSize(0)
    leg.SetTextSize(0.035)
    leg.AddEntry(g, "2 b matched", "p")
    leg.Draw()

    # -----------------------------
    # Title
    # -----------------------------

    latex = ROOT.TLatex()
    latex.SetNDC()              # normalized (0–1) coords
    latex.SetTextSize(0.04)
    latex.SetTextFont(42)
    latex.DrawLatex(0.67, 0.85, f"M_{{A}} = {M} GeV")

    CMS_label(c)

    # -----------------------------
    # Finalize
    # -----------------------------
    c.Update()
    c.SaveAs(Plot_dir + f"probb_vs_probbb_2b_{M}.png")


    # -----------------------------
    # Extract arrays
    # -----------------------------
    probb   = ak.to_numpy(one_jet.btagUParTAK4probb)
    probbb  = ak.to_numpy(one_jet.btagUParTAK4probbb)
    nb      = ak.to_numpy(n_b_matched)

    # Select 2b-matched jets
    mask_2b = (ak.flatten(nb) == 1)

    x = probb[mask_2b]
    y = probbb[mask_2b]

    # -----------------------------
    # Create TGraph (scatter)
    # -----------------------------
    g = ROOT.TGraph(len(x), x.astype("float64"), y.astype("float64"))

    g.SetTitle("")

    g.SetMarkerStyle(20)
    g.SetMarkerSize(0.4)                       # smaller points
    g.SetMarkerColorAlpha(ROOT.kBlue+2, 0.7)   # darker + less transparent
    g.SetLineColor(ROOT.kBlue+2)


    # -----------------------------
    # Draw
    # -----------------------------
    ROOT.gStyle.SetOptStat(0)

    c = ROOT.TCanvas("c", "c", 800, 600)
    c.SetRightMargin(0.02)

    g.Draw("AP")

    g.GetXaxis().SetTitle("probb")
    g.GetYaxis().SetTitle("probbb")
    g.GetXaxis().SetLimits(0.0, 1.0)
    g.GetYaxis().SetRangeUser(0.0, 1.0)

    # -----------------------------
    # Legend
    # -----------------------------
    leg = ROOT.TLegend(0.65, 0.85, 0.85, 0.75)
    leg.SetBorderSize(0)
    leg.SetTextSize(0.035)
    leg.AddEntry(g, "1 b matched", "p")
    leg.Draw()

    CMS_label(c)

    # -----------------------------
    # Title
    # -----------------------------

    latex = ROOT.TLatex()
    latex.SetNDC()              # normalized (0–1) coords
    latex.SetTextSize(0.04)
    latex.SetTextFont(42)
    latex.DrawLatex(0.67, 0.85, f"M_{{A}} = {M} GeV")

    # -----------------------------
    # Finalize
    # -----------------------------
    c.Update()
    c.SaveAs(Plot_dir + f"probb_vs_probbb_1b_{M}.png")

    # -----------------------------
    # Extract arrays
    # -----------------------------
    probb   = ak.to_numpy(one_jet.btagUParTAK4probb)
    probbb  = ak.to_numpy(one_jet.btagUParTAK4probbb)
    nb      = ak.to_numpy(n_b_matched)

    mask_2b = (ak.flatten(nb) == 2)

    x = probb[mask_2b]
    y = probbb[mask_2b]

    # -----------------------------
    # Create 2D histogram
    # -----------------------------
    # ROOT.gStyle.SetOptStat(0)
    # ROOT.gStyle.SetPalette(ROOT.kRedTemperature)  # close to "Reds" colormap

    c = ROOT.TCanvas("c", "c", 800, 600)
    c.SetRightMargin(0.2)

    # import array

    # stops  = array.array('d', [0.00, 0.50, 1.00])
    # red    = array.array('d', [1.00, 1.00, 0.60])
    # green  = array.array('d', [1.00, 0.40, 0.00])
    # blue   = array.array('d', [1.00, 0.40, 0.00])

    # ROOT.TColor.CreateGradientColorTable(
    #     len(stops), stops, red, green, blue, 255
    # )


    h2 = ROOT.TH2F(
        "h2",
        ";probb;probbb",
        20, 0.0, 1.0,
        20, 0.0, 1.0
    )

    for xi, yi in zip(x, y):
        h2.Fill(xi, yi)

    # -----------------------------
    # Draw
    # -----------------------------
    h2.Draw("COLZ")

    # -----------------------------
    # Title
    # -----------------------------

    CMS_label(c)

    latex = ROOT.TLatex()
    latex.SetNDC()              # normalized (0–1) coords
    latex.SetTextSize(0.04)
    latex.SetTextFont(42)
    latex.DrawLatex(0.57, 0.85, f"M_{{A}} = {M} GeV")

    # -----------------------------
    # Finalize
    # -----------------------------
    c.Update()
    c.SaveAs(Plot_dir+f"probb_vs_probbb_2b_colz_{M}.png")

    # -----------------------------
    # Extract arrays
    # -----------------------------
    probb   = ak.to_numpy(one_jet.btagUParTAK4probb)
    probbb  = ak.to_numpy(one_jet.btagUParTAK4probbb)
    nb      = ak.to_numpy(n_b_matched)

    mask_2b = (ak.flatten(nb) == 1)

    x = probb[mask_2b]
    y = probbb[mask_2b]

    # -----------------------------
    # Create 2D histogram
    # -----------------------------
    # ROOT.gStyle.SetOptStat(0)
    # ROOT.gStyle.SetPalette(ROOT.kRedTemperature)  # close to "Reds" colormap

    c = ROOT.TCanvas("c", "c", 800, 600)
    c.SetRightMargin(0.2)

    # import array

    # stops  = array.array('d', [0.00, 0.50, 1.00])
    # red    = array.array('d', [1.00, 1.00, 0.60])
    # green  = array.array('d', [1.00, 0.40, 0.00])
    # blue   = array.array('d', [1.00, 0.40, 0.00])

    # ROOT.TColor.CreateGradientColorTable(
    #     len(stops), stops, red, green, blue, 255
    # )


    h2 = ROOT.TH2F(
        "h2",
        ";probb;probbb",
        20, 0.0, 1.0,
        20, 0.0, 1.0
    )

    for xi, yi in zip(x, y):
        h2.Fill(xi, yi)

    # -----------------------------
    # Draw
    # -----------------------------
    h2.Draw("COLZ")

    # -----------------------------
    # Title
    # -----------------------------

    latex = ROOT.TLatex()
    latex.SetNDC()              # normalized (0–1) coords
    latex.SetTextSize(0.04)
    latex.SetTextFont(42)
    latex.DrawLatex(0.57, 0.85, f"M_{{A}} = {M} GeV")

    CMS_label(c)

    # -----------------------------
    # Finalize
    # -----------------------------
    c.Update()
    c.SaveAs(Plot_dir+f"probb_vs_probbb_1b_colz_{M}.png")

    def delta_r_bb(bquarks):
        deta = bquarks.eta[:, 0] - bquarks.eta[:, 1]
        dphi = np.abs(bquarks.phi[:, 0] - bquarks.phi[:, 1])
        dphi = ak.where(dphi > np.pi, 2*np.pi - dphi, dphi)
        return np.sqrt(deta**2 + dphi**2)

    dR_bb = delta_r_bb(bquarks_from_a)

    mask_2b     = (n_b_matched == 2)
    mask_good   = ak.flatten(mask_2b) & (one_jet.btagUParTAK4probbb > 0.7)
    mask_bad    = ak.flatten(mask_2b) & (one_jet.btagUParTAK4probbb < 0.2)

    dR_good = dR_bb[mask_good]
    dR_bad  = dR_bb[mask_bad]

    # -----------------------------
    # Convert to numpy
    # -----------------------------
    dR_good_np = ak.to_numpy(dR_good)
    dR_bad_np  = ak.to_numpy(dR_bad)

    # -----------------------------
    # Create histograms
    # -----------------------------
    ROOT.gStyle.SetOptStat(0)

    h_good = ROOT.TH1F("h_good", ";#DeltaR(b,b);Normalized entries", 30, 0.0, 0.6)
    h_bad  = ROOT.TH1F("h_bad",  ";#DeltaR(b,b);Normalized entries", 30, 0.0, 0.6)

    for v in dR_good_np:
        h_good.Fill(v)

    for v in dR_bad_np:
        h_bad.Fill(v)

    # Normalize to unit area
    if h_good.Integral() > 0:
        h_good.Scale(1.0 / h_good.Integral())
    if h_bad.Integral() > 0:
        h_bad.Scale(1.0 / h_bad.Integral())

    # -----------------------------
    # Styling
    # -----------------------------
    h_good.SetLineColor(ROOT.kBlue+1)
    h_good.SetLineWidth(2)

    h_bad.SetLineColor(ROOT.kRed+1)
    h_bad.SetLineWidth(2)
    h_bad.SetLineStyle(2)

    # -----------------------------
    # Draw
    # -----------------------------
    c = ROOT.TCanvas("c_dR", "c_dR", 800, 600)
    c.SetLeftMargin(0.15)
    c.SetRightMargin(0.02)

    h_good.SetMaximum(1.3 * max(h_good.GetMaximum(), h_bad.GetMaximum()))
    h_good.Draw("HIST")
    h_bad.Draw("HIST SAME")

    # -----------------------------
    # Legend and text
    # -----------------------------
    leg = ROOT.TLegend(0.50, 0.70, 0.83, 0.88)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)        # transparent
    leg.SetTextSize(0.03)
    leg.AddEntry(h_good, "2b matched, probbb > 0.7", "l")
    leg.AddEntry(h_bad,  "2b matched, probbb < 0.2", "l")
    leg.Draw()

    CMS_label(c)


    latex = ROOT.TLatex()
    latex.SetNDC()              # normalized (0–1) coords
    latex.SetTextSize(0.04)
    latex.SetTextFont(42)
    latex.DrawLatex(0.17, 0.85, f"M_{{A}} = {M} GeV")

    # -----------------------------
    # Save
    # -----------------------------
    c.Update()
    c.SaveAs(Plot_dir+f"deltaR_bb_good_vs_bad_{M}.png")


    # -----------------------------
    # Compute pT asymmetry
    # A_pT = min(pT1, pT2) / max(pT1, pT2)
    # -----------------------------
    pt = bquarks_from_a.pt   # shape: (nevents, 2)

    pt_min  = ak.min(pt, axis=1)
    pt_max  = ak.max(pt, axis=1)
    pt_asym = pt_min / pt_max

    # -----------------------------
    # Define tag regions
    # -----------------------------
    mask_2b   = (n_b_matched == 2)
    mask_good = ak.flatten(mask_2b) & (one_jet.btagUParTAK4probbb > 0.7)
    mask_bad  = ak.flatten(mask_2b) & (one_jet.btagUParTAK4probbb < 0.2)

    asym_good = ak.to_numpy(pt_asym[mask_good])
    asym_bad  = ak.to_numpy(pt_asym[mask_bad])

    # -----------------------------
    # Create histograms
    # -----------------------------
    ROOT.gStyle.SetOptStat(0)

    h_good = ROOT.TH1F(
        "h_asym_good",
        ";p_{T} asymmetry of bquarks p_{T}(sublead) / p_{T}(lead);Normalized entries",
        30, 0.0, 1.0
    )

    h_bad = ROOT.TH1F(
        "h_asym_bad",
        ";p_{T} asymmetry of bquarks p_{T}(sublead) / p_{T}(lead);Normalized entries",
        30, 0.0, 1.0
    )

    for v in asym_good:
        h_good.Fill(v)

    for v in asym_bad:
        h_bad.Fill(v)

    # Normalize
    if h_good.Integral() > 0:
        h_good.Scale(1.0 / h_good.Integral())
    if h_bad.Integral() > 0:
        h_bad.Scale(1.0 / h_bad.Integral())

    # -----------------------------
    # Styling
    # -----------------------------
    h_good.SetLineColor(ROOT.kBlue+1)
    h_good.SetLineWidth(2)

    h_bad.SetLineColor(ROOT.kRed+1)
    h_bad.SetLineWidth(2)
    h_bad.SetLineStyle(2)

    # -----------------------------
    # Draw
    # -----------------------------
    c = ROOT.TCanvas("c_ptasym", "c_ptasym", 800, 600)

    c.SetLeftMargin(0.15)
    c.SetRightMargin(0.02)

    h_good.SetMaximum(1.3 * max(h_good.GetMaximum(), h_bad.GetMaximum()))
    h_good.Draw("HIST")
    h_bad.Draw("HIST SAME")

    # -----------------------------
    # Legend and text
    # -----------------------------
    leg = ROOT.TLegend(0.55, 0.70, 0.88, 0.88)
    leg.SetBorderSize(0)
    leg.AddEntry(h_good, "2b matched, probbb > 0.7", "l")
    leg.AddEntry(h_bad,  "2b matched, probbb < 0.2", "l")
    leg.Draw()

    CMS_label(c)

    latex = ROOT.TLatex()
    latex.SetNDC()              # normalized (0–1) coords
    latex.SetTextSize(0.04)
    latex.SetTextFont(42)
    latex.DrawLatex(0.17, 0.85, f"M_{{A}} = {M} GeV")

    # -----------------------------
    # Save
    # -----------------------------
    c.Update()
    c.SaveAs(Plot_dir+f"pt_asymmetry_good_vs_bad_{M}.png")

    # -----------------------------
    # Event-level quantities
    # -----------------------------
    probb_evt  = one_jet.btagUParTAK4probb
    probbb_evt = one_jet.btagUParTAK4probbb
    nb_evt     = ak.flatten(n_b_matched)

    # Sanity check
    assert len(probb_evt) == len(probbb_evt) == len(nb_evt)

    # -----------------------------
    # Define categories
    # -----------------------------
    # cat1 = probbb_evt > probb_evt   # tagger prefers 2b
    # cat2 = probb_evt > probbb_evt   # tagger prefers 1b

    cat1 = (probbb_evt > probb_evt) & (probbb_evt > diB_mask_thr)   # tagger prefers 2b
    cat2 = (probb_evt > probbb_evt) | (probbb_evt < diB_mask_thr)   # tagger prefers 1b

    cat1_nothr = (probbb_evt > probb_evt)
    cat2_nothr = (probb_evt > probbb_evt)

    # -----------------------------
    # Count gen matching in each category
    # -----------------------------
    results = {
        "cat1_probbb_gt_probb": {
            "total": ak.sum(cat1),
            "1b_matched": ak.sum(cat1 & (nb_evt == 1)),
            "2b_matched": ak.sum(cat1 & (nb_evt == 2)),
        },
        "cat2_probb_gt_probbb": {
            "total": ak.sum(cat2),
            "1b_matched": ak.sum(cat2 & (nb_evt == 1)),
            "2b_matched": ak.sum(cat2 & (nb_evt == 2)),
        },
    }

    results_nothr = {
        "cat1_probbb_gt_probb": {
            "total": ak.sum(cat1_nothr),
            "1b_matched": ak.sum(cat1_nothr & (nb_evt == 1)),
            "2b_matched": ak.sum(cat1_nothr & (nb_evt == 2)),
        },
        "cat2_probb_gt_probbb": {
            "total": ak.sum(cat2_nothr),
            "1b_matched": ak.sum(cat2_nothr & (nb_evt == 1)),
            "2b_matched": ak.sum(cat2_nothr & (nb_evt == 2)),
        },
    }

    # -----------------------------
    # Print results (absolute + fractions)
    # -----------------------------
    for cat, vals in results.items():
        tot = vals["total"]
        print(f"\n{cat}")
        print(f"  Total events: {tot}")
        if tot > 0:
            print(f"  1b matched: {vals['1b_matched']}  ({vals['1b_matched']/tot:.3f})")
            print(f"  2b matched: {vals['2b_matched']}  ({vals['2b_matched']/tot:.3f})")


    ROOT.gStyle.SetOptStat(1111111)

    # -----------------------------
    # Select cat1 events
    # -----------------------------
    mass_cat1 = ak.to_numpy(one_jet.mass[cat1])

    # -----------------------------
    # Create histogram
    # -----------------------------
    h = ROOT.TH1F("h",
                ";Jet Mass (cat2: probbb > probb) GeV;Events",
                80, 0, 200)   # adjust range if needed

    # Fill histogram
    for m in mass_cat1:
        h.Fill(float(m))

    # -----------------------------
    # Draw
    # -----------------------------
    c = ROOT.TCanvas("c","Jet Mass cat1",800,600)

    h.SetLineColor(ROOT.kBlue+1)
    h.SetLineWidth(2)

    h.Draw("hist")

    latex = ROOT.TLatex()
    latex.SetNDC()              # normalized (0–1) coords
    latex.SetTextSize(0.04)
    latex.SetTextFont(42)
    latex.DrawLatex(0.22, 0.85, f"M_{{A}} = {M} GeV")

    CMS_label(c)

    c.Draw()

    c.Update()

    stats = h.FindObject("stats")
    if stats:
        stats.SetTextSize(0.035)     # enlarge text
        stats.SetX1NDC(0.60)         # left
        stats.SetX2NDC(0.88)         # right
        stats.SetY1NDC(0.60)         # bottom
        stats.SetY2NDC(0.88)         # top
        stats.SetBorderSize(1)

    c.SaveAs(Plot_dir+f"m_dib_{M}.png")


    # -----------------------------
    # Numbers from your study
    # (replace with your actual values if needed)
    # -----------------------------
    cat1_total = results["cat1_probbb_gt_probb"]["total"]
    cat1_1b    = results["cat1_probbb_gt_probb"]["1b_matched"]
    cat1_2b    = results["cat1_probbb_gt_probb"]["2b_matched"]

    cat2_total = results["cat2_probb_gt_probbb"]["total"]
    cat2_1b    = results["cat2_probb_gt_probbb"]["1b_matched"]
    cat2_2b    = results["cat2_probb_gt_probbb"]["2b_matched"]

    # Convert to fractions
    cat1_1b_frac = cat1_1b / cat1_total if cat1_total > 0 else 0
    cat1_2b_frac = cat1_2b / cat1_total if cat1_total > 0 else 0

    cat2_1b_frac = cat2_1b / cat2_total if cat2_total > 0 else 0
    cat2_2b_frac = cat2_2b / cat2_total if cat2_total > 0 else 0

    # -----------------------------
    # Create stacked histograms
    # -----------------------------
    ROOT.gStyle.SetOptStat(0)

    h_1b = ROOT.TH1F("h_1b", ";Category;Fraction of events", 2, 0, 2)
    h_2b = ROOT.TH1F("h_2b", ";Category;Fraction of events", 2, 0, 2)

    # Bin 1: probbb > probb
    h_1b.SetBinContent(1, cat1_1b_frac)
    h_2b.SetBinContent(1, cat1_2b_frac)

    # Bin 2: probb > probbb
    h_1b.SetBinContent(2, cat2_1b_frac)
    h_2b.SetBinContent(2, cat2_2b_frac)

    # Styling
    h_1b.SetFillColor(ROOT.kBlue+1)
    h_2b.SetFillColor(ROOT.kRed+1)

    h_1b.SetLineColor(ROOT.kBlack)
    h_2b.SetLineColor(ROOT.kBlack)

    # -----------------------------
    # Stack
    # -----------------------------
    stack = ROOT.THStack("stack", "")
    stack.Add(h_1b)
    stack.Add(h_2b)

    # -----------------------------
    # Draw
    # -----------------------------
    c = ROOT.TCanvas("c_cat", "c_cat", 800, 600)
    stack.Draw("BAR")
    stack.GetYaxis().SetRangeUser(0, 1.05)
    stack.GetYaxis().SetTitle("Fraction of events")

    # Custom x labels
    stack.GetXaxis().SetLabelSize(0)
    latex = ROOT.TLatex()
    latex.SetTextAlign(22)
    latex.SetTextSize(0.03)
    # latex.DrawLatex(0.5, -0.05, "probbb > probb")
    # latex.DrawLatex(1.5, -0.05, "probb > probbb")
    text1 = f"(probbb > probb) & probbb > {diB_mask_thr}"
    text2 = f"(probb > probbb) | probbb < {diB_mask_thr}"

    latex.DrawLatex(0.5, -0.05, text1)
    latex.DrawLatex(1.5, -0.05, text2)

    # -----------------------------
    # Legend
    # -----------------------------
    leg = ROOT.TLegend(0.25, 0.30, 0.48, 0.42)
    leg.SetBorderSize(0)
    leg.AddEntry(h_1b, "1b matched", "f")
    leg.AddEntry(h_2b, "2b matched", "f")
    leg.Draw()

    # -----------------------------
    # Title
    # -----------------------------
    # title = ROOT.TLatex()
    # title.SetNDC()
    # title.SetTextSize(0.045)
    # title.DrawLatex(0.15, 0.92, "Gen b-matching vs tagger preference")

    CMS_label(c)

    latex = ROOT.TLatex()
    latex.SetNDC()              # normalized (0–1) coords
    latex.SetTextSize(0.04)
    latex.SetTextFont(42)
    latex.DrawLatex(0.17, 0.65, f"M_{{A}} = {M} GeV")

    # -----------------------------
    # Save
    # -----------------------------
    c.Update()
    c.SaveAs(Plot_dir+f"category_gen_matching_{M}.png")


    # -----------------------------
    # Numbers from your study
    # (replace with your actual values if needed)
    # -----------------------------
    cat1_total = results_nothr["cat1_probbb_gt_probb"]["total"]
    cat1_1b    = results_nothr["cat1_probbb_gt_probb"]["1b_matched"]
    cat1_2b    = results_nothr["cat1_probbb_gt_probb"]["2b_matched"]

    cat2_total = results_nothr["cat2_probb_gt_probbb"]["total"]
    cat2_1b    = results_nothr["cat2_probb_gt_probbb"]["1b_matched"]
    cat2_2b    = results_nothr["cat2_probb_gt_probbb"]["2b_matched"]

    # Convert to fractions
    cat1_1b_frac = cat1_1b / cat1_total if cat1_total > 0 else 0
    cat1_2b_frac = cat1_2b / cat1_total if cat1_total > 0 else 0

    cat2_1b_frac = cat2_1b / cat2_total if cat2_total > 0 else 0
    cat2_2b_frac = cat2_2b / cat2_total if cat2_total > 0 else 0

    # -----------------------------
    # Create stacked histograms
    # -----------------------------
    ROOT.gStyle.SetOptStat(0)

    h_1b = ROOT.TH1F("h_1b", ";Category;Fraction of events", 2, 0, 2)
    h_2b = ROOT.TH1F("h_2b", ";Category;Fraction of events", 2, 0, 2)

    # Bin 1: probbb > probb
    h_1b.SetBinContent(1, cat1_1b_frac)
    h_2b.SetBinContent(1, cat1_2b_frac)

    # Bin 2: probb > probbb
    h_1b.SetBinContent(2, cat2_1b_frac)
    h_2b.SetBinContent(2, cat2_2b_frac)

    # Styling
    h_1b.SetFillColor(ROOT.kBlue+1)
    h_2b.SetFillColor(ROOT.kRed+1)

    h_1b.SetLineColor(ROOT.kBlack)
    h_2b.SetLineColor(ROOT.kBlack)

    # -----------------------------
    # Stack
    # -----------------------------
    stack = ROOT.THStack("stack", "")
    stack.Add(h_1b)
    stack.Add(h_2b)

    # -----------------------------
    # Draw
    # -----------------------------
    c = ROOT.TCanvas("c_cat", "c_cat", 800, 600)
    stack.Draw("BAR")
    stack.GetYaxis().SetRangeUser(0, 1.05)
    stack.GetYaxis().SetTitle("Fraction of events")

    # Custom x labels
    stack.GetXaxis().SetLabelSize(0)
    latex = ROOT.TLatex()
    latex.SetTextAlign(22)
    latex.SetTextSize(0.04)
    latex.DrawLatex(0.5, -0.05, "probbb > probb")
    latex.DrawLatex(1.5, -0.05, "probb > probbb")

    # -----------------------------
    # Legend
    # -----------------------------
    leg = ROOT.TLegend(0.25, 0.30, 0.48, 0.42)
    leg.SetBorderSize(0)
    leg.AddEntry(h_1b, "1b matched", "f")
    leg.AddEntry(h_2b, "2b matched", "f")
    leg.Draw()

    # -----------------------------
    # Title
    # -----------------------------
    # title = ROOT.TLatex()
    # title.SetNDC()
    # title.SetTextSize(0.045)
    # title.DrawLatex(0.15, 0.92, "Gen b-matching vs tagger preference")

    CMS_label(c)

    latex = ROOT.TLatex()
    latex.SetNDC()              # normalized (0–1) coords
    latex.SetTextSize(0.04)
    latex.SetTextFont(42)
    latex.DrawLatex(0.17, 0.65, f"M_{{A}} = {M} GeV")

    # -----------------------------
    # Save
    # -----------------------------
    c.Update()
    c.SaveAs(Plot_dir+f"category_gen_matching_nothr_{M}.png")


    n_bjets = ak.num(fj.pt)

    cat1 = n_bjets >= 2

    cat2 = (n_bjets == 1) & (fj.btagUParTAK4probbb[:,0] > fj.btagUParTAK4probb[:,0])

    cat3 = (n_bjets == 1) & (fj.btagUParTAK4probb[:,0] > fj.btagUParTAK4probbb[:,0])


    n_cat1 = int(ak.sum(cat1))
    n_cat2 = int(ak.sum(cat2))
    n_cat3 = int(ak.sum(cat3))

    print("Number of events in Cat1: ",n_cat1)
    print("Number of events in Cat2: ",n_cat2)
    print("Number of events in Cat3: ",n_cat3)

    one_photon = fp[ak.num(fp.pt)>0]

    mgg_cat1 = get_mgg(one_photon, cat1)    
    mgg_cat2 = get_mgg(one_photon, cat2)
    mgg_cat3 = get_mgg(one_photon, cat3)

    ROOT.gStyle.SetOptStat(111111)

    h1 = ROOT.TH1F("h1", ";m_{#gamma#gamma} (GeV);Events", 140, 0, 70)
    h2 = ROOT.TH1F("h2", ";m_{#gamma#gamma} (GeV);Events", 140, 0, 70)
    h3 = ROOT.TH1F("h3", ";m_{#gamma#gamma} (GeV);Events", 140, 0, 70)


    for val in mgg_cat1: h1.Fill(val)
    for val in mgg_cat2: h2.Fill(val)
    for val in mgg_cat3: h3.Fill(val)

    for h in [h1, h2, h3]:
        if h.Integral()>0:
            h.Scale(1/h.Integral())

    for h in [h1, h2, h3]:
        h.SetMinimum(0.001)

    h1.SetLineColor(ROOT.kBlue+1)
    h2.SetLineColor(ROOT.kRed+1)
    h3.SetLineColor(ROOT.kGreen+2)

    h1.SetLineWidth(2)
    h2.SetLineWidth(2)
    h3.SetLineWidth(2)
    c = ROOT.TCanvas(f"c_{M}", f"m_gg MA{M}", 800, 600)

    c.SetLogy()

    # Get maximum bin content among all histograms
    max_val = max(
        h1.GetMaximum(),
        h2.GetMaximum(),
        h3.GetMaximum()
    )

    # Set y-axis maximum to 1.2 × highest peak
    h1.SetMaximum(5.0 * max_val)

    h1.Draw("hist")
    h2.Draw("hist same")
    h3.Draw("hist same")

    leg = ROOT.TLegend(0.65,0.78,0.85,0.88)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)       # transparent fill
    leg.SetFillColor(0) 
    leg.AddEntry(h1, "Resolved", "l")
    leg.AddEntry(h2, "Merged", "l")
    leg.AddEntry(h3, "One-jet", "l")
    leg.Draw()

    CMS_label(c)

    draw_side_statboxes(c, [h1, h2, h3])

    latex = ROOT.TLatex()
    latex.SetNDC()              # normalized (0–1) coords
    latex.SetTextSize(0.03)
    latex.SetTextFont(42)
    latex.DrawLatex(0.65, 0.75, f"M_{{A}} = {M} GeV")

    latex = ROOT.TLatex()
    latex.SetNDC()              # normalized (0–1) coords
    latex.SetTextSize(0.04)
    latex.SetTextFont(42)
    latex.DrawLatex(0.15, 0.85, "After preselection (at least 1 b-Jet)")

    c.SaveAs(Plot_dir + f"m_gg_overlay_M{M}.png")

    print("Cat1 (>=2 b-Jets):", n_cat1)
    print("Cat2 (1 b-Jet & probbb > probb):", n_cat2)
    print("Cat3 (1 b-Jet & probb > probbbb):", n_cat3)


    n_total = len(n_bjets)   # or int(ak.sum(n_jets >= 1)) if already filtered

    f_cat1 = round(n_cat1 / n_total,2)
    f_cat2 = round(n_cat2 / n_total,2)
    f_cat3 = round(n_cat3 / n_total,2)

    # --------------------------------
    # Create histogram
    # --------------------------------
    h_frac = ROOT.TH1F(
        "h_frac",
        ";Category;Fraction of Events",
        3, 0, 3
    )

    h_frac.SetBinContent(1, f_cat1)
    h_frac.SetBinContent(2, f_cat2)
    h_frac.SetBinContent(3, f_cat3)

    # Bin labels
    h_frac.GetXaxis().SetBinLabel(1, "#geq 2 b-Jets")
    h_frac.GetXaxis().SetBinLabel(2, "1 b-Jet: probbb > probb")
    h_frac.GetXaxis().SetBinLabel(3, "1 b-Jet: probb > probbb")

    # Style
    h_frac.SetFillColor(ROOT.kAzure-9)
    h_frac.SetLineColor(ROOT.kBlack)
    h_frac.SetLineWidth(2)

    h_frac.GetXaxis().SetLabelSize(0.05)
    h_frac.GetYaxis().SetLabelSize(0.045)
    h_frac.GetYaxis().SetTitleSize(0.03)
    # h_frac.GetYaxis().SetTitleOffset(1.3)

    h_frac.SetMaximum(1.0)   # fractions go up to 1

    # --------------------------------
    # Draw
    # --------------------------------
    c = ROOT.TCanvas("c_frac", "c_frac", 800, 600)
    c.SetBottomMargin(0.18)
    c.SetLeftMargin(1.3)
    c.SetRightMargin(0.02)

    h_frac.SetMarkerSize(2.0);
    h_frac.Draw("HIST TEXT0")

    latex = ROOT.TLatex()
    latex.SetNDC()              # normalized (0–1) coords
    latex.SetTextSize(0.04)
    latex.SetTextFont(42)
    latex.DrawLatex(0.17, 0.85, f"M_{{A}} = {M} GeV")

    CMS_label(c)

    c.Update()
    c.SaveAs(Plot_dir+f"category_fractions_{M}.png")


    two_jets = fj[ak.num(fj.pt)>=2]

    # # Sort by btag score (highest first)
    # idx_btag = ak.argsort(two_jets.btagUParTAK4B, axis=1, ascending=False)
    # jets_btag_sorted = two_jets[idx_btag]

    # top2_bjets = jets_btag_sorted[:, :2]

    idx_pt = ak.argsort(two_jets.pt, axis=1, ascending=False)
    top2_sorted_by_pt = two_jets[idx_pt][:,:2]

    lead_bjet    = top2_sorted_by_pt[:, 0]
    sublead_bjet = top2_sorted_by_pt[:, 1]

    gen_all = events.GenPart

    bquarks_all = gen_all[abs(gen_all.pdgId) == 5]
    mother_idx_all = bquarks_all.genPartIdxMother
    from_a_mask_all = gen_all[mother_idx_all].pdgId == 35
    bquarks_from_a_all = bquarks_all[from_a_mask_all]

    two_bjet_bquark = bquarks_from_a_all[evt_mask][ak.num(fj.pt)>=2]

    # Sort pT per event in descending order
    genb_sorted = ak.sort(bquarks_from_a.pt, axis=1, ascending=False)

    # Leading and subleading
    pt_leadb = ak.to_numpy(genb_sorted[:, 0])
    pt_subleadb = ak.to_numpy(genb_sorted[:, 1])

    bquarks_from_a_all_sel = bquarks_from_a_all[ak.num(fj.pt)>0]

    bquarks_from_a_all_sel_idx = ak.argsort(bquarks_from_a_all_sel.pt, axis=1, ascending = False)
    bquarks_from_a_all_sel_sorted = bquarks_from_a_all_sel[bquarks_from_a_all_sel_idx]
    leadb_all = bquarks_from_a_all_sel_sorted[:,0]
    subleadb_all = bquarks_from_a_all_sel_sorted[:,1]

    bquarks_from_a_all_idx = ak.argsort(bquarks_from_a_all.pt, axis=1, ascending = False)
    bquarks_from_a_all_sorted = bquarks_from_a_all[bquarks_from_a_all_idx]
    leadb_all_before_sel = bquarks_from_a_all_sorted[:,0]
    subleadb_all_before_sel = bquarks_from_a_all_sorted[:,1]

    ratio = subleadb_all.pt / leadb_all.pt

    ratio_before_sel = subleadb_all_before_sel.pt / leadb_all_before_sel.pt

    dR_bb_all = delta_r_bb(bquarks_from_a_all_sel)

    dR_bb_all_before_sel = delta_r_bb(bquarks_from_a_all)


    ROOT.gStyle.SetOptStat(0)

    h_lead = ROOT.TH1F("h_lead", ";Gen b p_{T} [GeV];Events", 50, 0, 200)
    h_sub  = ROOT.TH1F("h_sub",  ";Gen b p_{T} [GeV];Events", 50, 0, 200)

    # Fill histograms
    for pt in leadb_all_before_sel.pt:
        h_lead.Fill(pt)

    for pt in subleadb_all_before_sel.pt:
        h_sub.Fill(pt)

    fraction_soft = np.sum(subleadb_all_before_sel.pt < 20) / len(leadb_all_before_sel.pt)
    print("Fraction with subleading pT < 20 GeV:", fraction_soft)


    c = ROOT.TCanvas("c", "c", 700, 600)
    c.SetLeftMargin(.12)

    max_val = max(h_lead.GetMaximum(), h_sub.GetMaximum())
    h_lead.SetMaximum(1.1 * max_val)   # 20% headroom
    h_lead.SetMinimum(0)      

    h_lead.SetLineColor(ROOT.kRed)
    h_lead.SetLineWidth(2)

    h_sub.SetLineColor(ROOT.kBlue)
    h_sub.SetLineWidth(2)

    h_lead.Draw("HIST")
    h_sub.Draw("HIST SAME")

    leg = ROOT.TLegend(0.60, 0.75, 0.88, 0.88)
    leg.SetBorderSize(0)
    leg.AddEntry(h_lead, "Leading b", "l")
    leg.AddEntry(h_sub, "Subleading b", "l")
    leg.Draw()

    latex = ROOT.TLatex()
    latex.SetNDC()              # normalized (0–1) coords
    latex.SetTextSize(0.04)
    latex.SetTextFont(42)
    latex.DrawLatex(0.60, 0.55, f"M_{{A}} = {M} GeV")

    # latex = ROOT.TLatex()
    # latex.SetNDC()              # normalized (0–1) coords
    # latex.SetTextSize(0.04)
    # latex.SetTextFont(42)
    # latex.DrawLatex(0.40, 0.45, "After preselection (at least 1 b-Jet)")

    CMS_label(c)

    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.03)
    latex.DrawLatex(0.15, 0.85, f"{fraction_soft*100:.1f}% of sublead b have p_{{T}} < 20 GeV")


    c.SaveAs(Plot_dir+f"genb_pt_lead_sublead_{M}.png")

    gen_photons = gen_all[(gen_all.pdgId == 22) & (gen_all.status == 1)]
    mother_idx = gen_photons.genPartIdxMother
    from_a_mask = gen_all[mother_idx].pdgId == 35
    photons_from_a = gen_photons[from_a_mask]

    photons_from_a_all_idx = ak.argsort(photons_from_a.pt, axis=1, ascending = False)
    photons_from_a_all_sorted = photons_from_a[photons_from_a_all_idx]

    mask_2pho = ak.num(photons_from_a_all_sorted) >= 2
    photons_2pho = photons_from_a_all_sorted[mask_2pho]

    leadpho_all_before_sel = photons_2pho[:,0]
    subleadpho_all_before_sel = photons_2pho[:,1]


    ROOT.gStyle.SetOptStat(0)

    h_lead = ROOT.TH1F("h_lead", ";Gen #gamma p_{T} [GeV];Events", 50, 0, 200)
    h_sub  = ROOT.TH1F("h_sub",  ";Gen #gamma p_{T} [GeV];Events", 50, 0, 200)

    # Fill histograms
    for pt in leadpho_all_before_sel.pt:
        h_lead.Fill(pt)

    for pt in subleadpho_all_before_sel.pt:
        h_sub.Fill(pt)

    # fraction_soft = np.sum(subleadpho_all_before_sel.pt < 20) / len(leadpho_all_before_sel.pt)
    # print("Fraction with subleading pT < 20 GeV:", fraction_soft)


    c = ROOT.TCanvas("c", "c", 700, 600)
    c.SetLeftMargin(.12)

    max_val = max(h_lead.GetMaximum(), h_sub.GetMaximum())
    h_lead.SetMaximum(1.1 * max_val)   # 20% headroom
    h_lead.SetMinimum(0)      

    h_lead.SetLineColor(ROOT.kRed)
    h_lead.SetLineWidth(2)

    h_sub.SetLineColor(ROOT.kBlue)
    h_sub.SetLineWidth(2)

    h_lead.Draw("HIST")
    h_sub.Draw("HIST SAME")

    leg = ROOT.TLegend(0.60, 0.75, 0.88, 0.88)
    leg.SetBorderSize(0)
    leg.AddEntry(h_lead, f"Leading #gamma", "l")
    leg.AddEntry(h_sub, f"Subleading #gamma", "l")
    leg.Draw()

    latex = ROOT.TLatex()
    latex.SetNDC()              # normalized (0–1) coords
    latex.SetTextSize(0.04)
    latex.SetTextFont(42)
    latex.DrawLatex(0.60, 0.55, f"M_{{A}} = {M} GeV")

    # latex = ROOT.TLatex()
    # latex.SetNDC()              # normalized (0–1) coords
    # latex.SetTextSize(0.04)
    # latex.SetTextFont(42)
    # latex.DrawLatex(0.40, 0.45, "After preselection (at least 1 b-Jet)")

    CMS_label(c)

    # latex = ROOT.TLatex()
    # latex.SetNDC()
    # latex.SetTextSize(0.03)
    # latex.DrawLatex(0.15, 0.85, f"{fraction_soft*100:.1f}% of sublead #gamma have p_{{T}} < 20 GeV")


    c.SaveAs(Plot_dir+f"genpho_pt_lead_sublead_{M}.png")

    h2 = ROOT.TH2F("h2",
                ";Leading p_{T};Subleading p_{T}",
                100, 0, 200,
                100, 0, 200)

    for l, s in zip(leadb_all_before_sel.pt, subleadb_all_before_sel.pt):
        h2.Fill(l, s)

    c = ROOT.TCanvas("c2", "", 700, 600)
    ROOT.gStyle.SetPalette(ROOT.kRainBow) 

    h2.Draw("COLZ")

    latex = ROOT.TLatex()
    latex.SetNDC()              # normalized (0–1) coords
    latex.SetTextSize(0.04)
    latex.SetTextFont(42)
    latex.DrawLatex(0.20, 0.55, f"M_{{A}} = {M} GeV")

    CMS_label(c)

    # latex = ROOT.TLatex()
    # latex.SetNDC()              # normalized (0–1) coords
    # latex.SetTextSize(0.04)
    # latex.SetTextFont(42)
    # latex.DrawLatex(0.20, 0.65, "After preselection (at least 1 b-Jet)")

    c.SaveAs(Plot_dir+f"pt_2D_{M}.png")


    # bquarks_from_a_all_sel = bquarks_from_a_all[ak.num(fj.pt)>0]

    # bquarks_from_a_all_sel_idx = ak.argsort(bquarks_from_a_all_sel.pt, axis=1, ascending = False)
    # bquarks_from_a_all_sel_sorted = bquarks_from_a_all_sel[bquarks_from_a_all_sel_idx]
    # leadb_all = bquarks_from_a_all_sel_sorted[:,0]
    # subleadb_all = bquarks_from_a_all_sel_sorted[:,1]

    # ratio = subleadb_all.pt / leadb_all.pt

    dR_bb_all = delta_r_bb(bquarks_from_a_all_sel)

    h_ratio = ROOT.TH1F("h_ratio", ";p_{T}^{sub}/p_{T}^{lead};Fraction", 40, 0, 1)

    for r in ratio:
        h_ratio.Fill(r)

    h_ratio.Scale(1.0 / h_ratio.Integral())

    c = ROOT.TCanvas("c_ratio", "", 700, 600)
    c.SetBottomMargin(0.14)
    h_ratio.SetLineWidth(2)
    h_ratio.Draw("HIST")

    h_ratio.GetXaxis().SetLabelSize(0.06)
    h_ratio.GetXaxis().SetTitleOffset(1.3)

    latex = ROOT.TLatex()
    latex.SetNDC()              # normalized (0–1) coords
    latex.SetTextSize(0.04)
    latex.SetTextFont(42)
    latex.DrawLatex(0.67, 0.75, f"M_{{A}} = {M} GeV")

    CMS_label(c)

    latex = ROOT.TLatex()
    latex.SetNDC()              # normalized (0–1) coords
    latex.SetTextSize(0.04)
    latex.SetTextFont(42)
    latex.DrawLatex(0.40, 0.85, "After preselection (at least 1 b-Jet)")

    c.SaveAs(Plot_dir+f"pt_ratio_{M}.png")


    ROOT.gStyle.SetOptStat(0)

    # import array

    # stops = array.array('d', [0.0, 1.0])

    # red   = array.array('d', [0.0, 1.0])
    # green = array.array('d', [0.0, 1.0])
    # blue  = array.array('d', [0.5, 0.0])

    # ROOT.TColor.CreateGradientColorTable(
    #     2, stops, red, green, blue, 255
    # )

    # ROOT.gStyle.SetNumberContours(255)

    h2 = ROOT.TH2F("h2",
                ";pT ratio (subleading p_{T}/leading_p_{T});dR",
                100, 0, 1,
                100, 0, 1)

    for l, s in zip(ratio_before_sel, dR_bb_all_before_sel):
        h2.Fill(l, s)

    c = ROOT.TCanvas("c2", "", 800, 600)

    ROOT.gStyle.SetPalette(ROOT.kRainBow) 

    h2.Draw("COLZ")

    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.04)
    latex.SetTextFont(42)
    latex.SetTextColor(ROOT.kRed)
    latex.DrawLatex(0.70, 0.12, f"M_{{A}} = {M} GeV")

    CMS_label(c)

    c.SaveAs(Plot_dir+f"pt_ratio_vs_dR_{M}.png")

    pairs = ak.cartesian(
        {"reco": top2_sorted_by_pt, "gen": two_bjet_bquark},
        axis=1
    )

    dR = pairs["reco"].deltaR(pairs["gen"])


    dR_matrix = ak.unflatten(dR, 2, axis=1)

    closest_gen_index = ak.argmin(dR_matrix, axis=2)
    min_dR = ak.min(dR_matrix, axis=2)

    matched = min_dR < 0.4

    unique_match = (
        (closest_gen_index[:,0] != closest_gen_index[:,1]) &
        matched[:,0] &
        matched[:,1]
    )

    same_b = (
        (closest_gen_index[:,0] == closest_gen_index[:,1]) &
        matched[:,0] &
        matched[:,1]
    )

    total = len(dR_matrix)

    print("Unique matching:", ak.sum(unique_match)/total)
    print("Same b double match:", ak.sum(same_b)/total)
    print("One matched:", ak.sum(ak.sum(matched,axis=1)==1)/total)
    print("None matched:", ak.sum(ak.sum(matched,axis=1)==0)/total)




if __name__ == "__main__":

    masses = [15, 20, 25, 30, 35, 40, 45, 50, 60]

    for M in masses:
        run_for_mass(M)



        