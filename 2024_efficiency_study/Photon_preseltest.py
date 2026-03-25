#!/usr/bin/env python3
import os
import awkward as ak
import ROOT
import numpy as np
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

ROOT.gStyle.SetOptStat(0)
ROOT.gROOT.SetBatch(True)

# ======================
# CONFIGURATION
# ======================

outdir = "/eos/user/b/bbapi/www/Analysis_plots/Overlay_plots/WH_2024"

Log = True


file = '/eos/user/b/bbapi/MC_contacts/2024_signal_samples_WH/CMSSW_15_0_15/src/WH_2024_M60.root'
factory = NanoEventsFactory.from_root(
    f"{file}:Events",
    schemaclass=NanoAODSchema,
)
events = factory.events()

events = ak.materialize(events)

# cuts
MUON_PT_CUT, ELE_PT_CUT = 24.0, 30.0
PHO1_PT_CUT, PHO2_PT_CUT = 20.0, 15.0
BJET_PT_CUT = 20.0

ELE_PHO_ETA_MAX = 2.5
MU_JET_ETA_MAX  = 2.4
ETA_GAP_LO, ETA_GAP_HI = 1.44, 1.57

BTAG_B_CUT    = 0.1272   # btagUParTAK4B [Medium WP 2024]
BTAG_PROBB_CUT = 0.38 # btagUParTAK4probbb [Taken from https://cms.cern.ch/iCMS/analysisadmin/cadilines?line=NPS-25-010&tp=an&id=3025&ancode=NPS-25-010]


# -------------------------
# Step 0: initial
events_0 = events

# -------------------------
# Step 1: exactly ONE lepton (e OR μ)
good_mu = (events_0.Muon.pt > MUON_PT_CUT)
good_el = (events_0.Electron.pt > ELE_PT_CUT)

mask_1lep = (ak.num(events_0.Muon[good_mu]) + ak.num(events_0.Electron[good_el])) == 1
events_1lep = events_0[mask_1lep]

# -----------------------------------------
# Exactly one lepton selection (SAFE)
# -----------------------------------------

# pT cuts

# # Count leptons (no pT cuts yet)
# n_ele = ak.num(events_0.Electron)
# n_mu  = ak.num(events_0.Muon)

# # Exactly one lepton in the event
# exactly_one_lepton = (n_ele + n_mu) == 1

# # Leading lepton pT (safe for jagged arrays)
# ele_pt0 = ak.firsts(events_0.Electron.pt)
# mu_pt0  = ak.firsts(events_0.Muon.pt)

# # Electron-only channel
# electron_channel = (
#     exactly_one_lepton &
#     (n_ele == 1) &
#     (ele_pt0 > ELE_PT_CUT)
# )

# # Muon-only channel
# muon_channel = (
#     exactly_one_lepton &
#     (n_mu == 1) &
#     (mu_pt0 > MUON_PT_CUT)
# )

# # Final 1-lepton selection
# lepton_channel_mask = electron_channel | muon_channel

# # Apply selection
# events_1lep = events_0[lepton_channel_mask]


# -------------------------
# Step 2: leading photon pT > 20
photons = events_1lep.Photon[ak.argsort(events_1lep.Photon.pt, axis=1, ascending=False)]
mask_pho1 = ak.num(photons) >= 1
events_pho1 = events_1lep[mask_pho1][photons[mask_pho1].pt[:, 0] > PHO1_PT_CUT]

# -------------------------
# Step 3: subleading photon pT > 10
photons = events_pho1.Photon[ak.argsort(events_pho1.Photon.pt, axis=1, ascending=False)]
mask_pho2 = ak.num(photons) >= 2
events_pho2 = events_pho1[mask_pho2][photons[mask_pho2].pt[:, 1] > PHO2_PT_CUT]

# -------------------------
# Step 4: ≥1 jet with pT > 20
mask_bjet = ak.num(events_pho2.Jet[events_pho2.Jet.pt > BJET_PT_CUT]) >= 1
events_bjet = events_pho2[mask_bjet]

# -------------------------
# Step 5: eta cuts on all objects

events_final = events_bjet[
    # electrons: |η|<2.5 and NOT in transition region
    ak.all(
        (np.abs(events_bjet.Electron.eta) < ELE_PHO_ETA_MAX) &
        ~((np.abs(events_bjet.Electron.eta) > ETA_GAP_LO) &
          (np.abs(events_bjet.Electron.eta) < ETA_GAP_HI)),
        axis=1
    )
    &
    # photons: |η|<2.5 and NOT in transition region
    ak.all(
        (np.abs(events_bjet.Photon.eta) < ELE_PHO_ETA_MAX) &
        ~((np.abs(events_bjet.Photon.eta) > ETA_GAP_LO) &
          (np.abs(events_bjet.Photon.eta) < ETA_GAP_HI)),
        axis=1
    )
    &
    # muons: |η|<2.4
    ak.all(np.abs(events_bjet.Muon.eta) < MU_JET_ETA_MAX, axis=1)
    &
    # jets: |η|<2.4
    ak.all(np.abs(events_bjet.Jet.eta) < MU_JET_ETA_MAX, axis=1)
]

jets = events_final.Jet

bjets_B = jets[jets.btagUParTAK4B > BTAG_B_CUT]
bjets_probb = jets[jets.btagUParTAK4probbb > BTAG_PROBB_CUT]

nbjets_B = ak.num(bjets_B)
nbjets_probb = ak.num(bjets_probb)

mask_cat1 = nbjets_B >= 2
events_cat1 = events_final[mask_cat1]

mask_cat2 = (nbjets_probb >= 1) & (~mask_cat1)
events_cat2 = events_final[mask_cat2]

mask_cat3 = (nbjets_B == 1) & (~mask_cat1) & (~mask_cat2)
events_cat3 = events_final[mask_cat3]

print("Total final events :", len(events_final))
print("Cat1 (>=2 bjets B) :", len(events_cat1))
print("Cat2 (>=1 probbb)  :", len(events_cat2))
print("Cat3 (==1 bjet B)  :", len(events_cat3))

print("Sum of cats        :", len(events_cat1) + len(events_cat2) + len(events_cat3))


# -------------------------

# =========================
# Cutflow summary
# =========================
print("Cutflow:")
print(f"Initial events              : {len(events_0)}")
print(f"Lepton pT cut               : {len(events_1lep)}")
print(f"Leading photon pT > 20      : {len(events_pho1)}")
print(f"Subleading photon pT > 15   : {len(events_pho2)}")
print(f">=1 b-jet                   : {len(events_bjet)}")
print(f"After eta cuts              : {len(events_final)}")


# stages = {
#     "No selection": events_0,
#     "lepton pT cut (> 30 GeV)": events_1lep,
#     "lead photon pT cut (> 20 GeV)": events_pho1,
#     "sublead photon pT cut (> 15 GeV)": events_pho2,
#     "b-jet pT cut (> 20 GeV)": events_bjet,
#     "eta cut": events_final,
# }

stages = {
    "No selection": events_0,
    "lep pT cut (e> 30 or mu> 24 GeV)": events_1lep,
    "lead photon pT cut (> 20 GeV)": events_pho1,
    "sublead photon pT cut (> 15 GeV)": events_pho2,
    "Jet pT cut (> 20 GeV)": events_bjet,
    "eta cut": events_final,
}

stages_cat1 = {
    **stages,
    ">= 2 Jets (UParTAK4B > 0.1272)": events_cat1,
}

stages_cat2 = {
    **stages,
    "1 Jet (UParTAK4probbb > 0.38)": events_cat2,
}

stages_cat3 = {
    **stages,
    "== 1 Jet (UParTAK4B > 0.1272)": events_cat3,
}



for k, ev in stages.items():
    print(
        k,
        ak.sum(ak.num(ev.Jet.pt) >= 2)
    )



def plot_overlay_from_arrays(
    arrays_dict, varname, title,
    nbins, xmin, xmax,
    outdir, logy=True
):
    # c = ROOT.TCanvas(f"c_{varname}", "", 800, 700)
    # c = ROOT.TCanvas(f"c_{varname}", "", 1000, 700)
    c = ROOT.TCanvas(f"c_{varname}", "", 1200, 700)
    c.SetRightMargin(0.40)
    c.SetLeftMargin(0.12)
    c.SetBottomMargin(0.12)

    c.SetRightMargin(0.35)
    # leg = ROOT.TLegend(0.6, 0.7, 0.88, 0.88)
    leg = ROOT.TLegend(0.62, 0.15, 0.95, 0.85)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.028)

    lat = ROOT.TLatex()
    lat.SetNDC()
    lat.SetTextFont(42)
    lat.SetTextSize(0.028)

    lat.DrawLatex(0.62, 0.88, "M = 60 GeV")


    colors = [
        ROOT.kRed, ROOT.kBlue, ROOT.kGreen+2,
        ROOT.kMagenta, ROOT.kOrange+1,
        ROOT.kCyan+1, ROOT.kGray+2
    ]

    hists = []
    ymax = 0

    for i, (label, values) in enumerate(arrays_dict.items()):
        if values is None or len(values) == 0:
            continue

        h = ROOT.TH1F(
            f"h_{varname}_{label}",
            "", nbins, xmin, xmax
        )
        # h.SetMinimum(0.1)
        if logy:
            h.SetMinimum(0.1)

        h.SetLineColor(colors[i % len(colors)])
        h.SetLineWidth(2)

        for v in ak.flatten(values, axis=None):
            h.Fill(v)

        ymax = max(ymax, h.GetMaximum())
        hists.append((h, label))

    if logy:
        c.SetLogy()

    first = True
    for h, label in hists:
        h.GetXaxis().SetTitle(title)
        h.GetYaxis().SetTitle("Events")
        h.SetMaximum(1.3 * ymax)

        if first:
            h.Draw("HIST")
            first = False
        else:
            h.Draw("HIST SAME")

        # leg.AddEntry(h, label, "l")
        entries = int(h.GetEntries())
        mean = h.GetMean()

        leg.AddEntry(
            h,
            f"{label}  (N={entries})",
            "l"
        )


    leg.Draw()

    c.SaveAs(f"{outdir}/{varname}.pdf")
    c.SaveAs(f"{outdir}/{varname}.png")
    c.Close()


photon_pt  = {k: v.Photon.pt  for k, v in stages.items()}
photon_eta = {k: v.Photon.eta for k, v in stages.items()}
photon_phi = {k: v.Photon.phi for k, v in stages.items()}

plot_overlay_from_arrays(photon_pt,  "photon_pt",  "Photon p_{T} [GeV]", 250, 0, 500, outdir)
plot_overlay_from_arrays(photon_eta, "photon_eta", "Photon #eta",       100, -3, 3, outdir)
plot_overlay_from_arrays(photon_phi, "photon_phi", "Photon #phi",        64, -3.2, 3.2, outdir)


electron_pt  = {k: v.Electron.pt  for k, v in stages.items()}
electron_eta = {k: v.Electron.eta for k, v in stages.items()}
electron_phi = {k: v.Electron.phi for k, v in stages.items()}

plot_overlay_from_arrays(electron_pt,  "electron_pt",  "Electron p_{T} [GeV]", 250, 0, 500, outdir)
plot_overlay_from_arrays(electron_eta, "electron_eta", "Electron #eta",       100, -3, 3, outdir)
plot_overlay_from_arrays(electron_phi, "electron_phi", "Electron #phi",        64, -3.2, 3.2, outdir)

muon_pt  = {k: v.Muon.pt  for k, v in stages.items()}
muon_eta = {k: v.Muon.eta for k, v in stages.items()}
muon_phi = {k: v.Muon.phi for k, v in stages.items()}

plot_overlay_from_arrays(muon_pt,  "muon_pt",  "Muon p_{T} [GeV]", 250, 0, 500, outdir)
plot_overlay_from_arrays(muon_eta, "muon_eta", "Muon #eta",       100, -3, 3, outdir)
plot_overlay_from_arrays(muon_phi, "muon_phi", "Muon #phi",        64, -3.2, 3.2, outdir)


jet_pt  = {k: v.Jet.pt  for k, v in stages.items()}
jet_eta = {k: v.Jet.eta for k, v in stages.items()}
jet_phi = {k: v.Jet.phi for k, v in stages.items()}

plot_overlay_from_arrays(jet_pt,  "jet_pt",  "Jet p_{T} [GeV]", 250, 0, 500, outdir)
plot_overlay_from_arrays(jet_eta, "jet_eta", "Jet #eta",       100, -4, 4, outdir)
plot_overlay_from_arrays(jet_phi, "jet_phi", "Jet #phi",        64, -3.2, 3.2, outdir)


photon_lead_pt  = {}
photon_sub_pt   = {}
photon_lead_eta = {}
photon_sub_eta  = {}
photon_lead_phi = {}
photon_sub_phi  = {}

for k, ev in stages.items():

    # -------- leading photon (>=1 photon)
    mask1 = ak.num(ev.Photon.pt) >= 1
    if ak.any(mask1):
        pho1 = ev.Photon[mask1]
        pho1 = pho1[ak.argsort(pho1.pt, axis=1, ascending=False)]

        photon_lead_pt[k]  = pho1[:, 0].pt
        photon_lead_eta[k] = pho1[:, 0].eta
        photon_lead_phi[k] = pho1[:, 0].phi
    else:
        photon_lead_pt[k]  = None
        photon_lead_eta[k] = None
        photon_lead_phi[k] = None

    # -------- subleading photon (>=2 photons)
    mask2 = ak.num(ev.Photon.pt) >= 2   
    if ak.any(mask2):
        pho2 = ev.Photon[mask2]
        pho2 = pho2[ak.argsort(pho2.pt, axis=1, ascending=False)]

        photon_sub_pt[k]  = pho2[:, 1].pt
        photon_sub_eta[k] = pho2[:, 1].eta
        photon_sub_phi[k] = pho2[:, 1].phi
    else:
        photon_sub_pt[k]  = None
        photon_sub_eta[k] = None
        photon_sub_phi[k] = None


electron_lead_pt  = {}
electron_lead_eta = {}
electron_lead_phi = {}

for k, ev in stages.items():
    mask = ak.num(ev.Electron.pt) >= 1
    if not ak.any(mask):
        electron_lead_pt[k] = None
        electron_lead_eta[k] = None
        electron_lead_phi[k] = None
        continue

    ele = ev.Electron[mask]
    ele = ele[ak.argsort(ele.pt, axis=1, ascending=False)]

    electron_lead_pt[k]  = ele[:, 0].pt
    electron_lead_eta[k] = ele[:, 0].eta
    electron_lead_phi[k] = ele[:, 0].phi


muon_lead_pt  = {}
muon_lead_eta = {}
muon_lead_phi = {}

for k, ev in stages.items():
    mask = ak.num(ev.Muon.pt) >= 1
    if not ak.any(mask):
        muon_lead_pt[k] = None
        muon_lead_eta[k] = None
        muon_lead_phi[k] = None
        continue

    mu = ev.Muon[mask]
    mu = mu[ak.argsort(mu.pt, axis=1, ascending=False)]

    muon_lead_pt[k]  = mu[:, 0].pt
    muon_lead_eta[k] = mu[:, 0].eta
    muon_lead_phi[k] = mu[:, 0].phi


jet_lead_pt  = {}
jet_sub_pt   = {}
jet_lead_eta = {}
jet_sub_eta  = {}
jet_lead_phi = {}
jet_sub_phi  = {}

for k, ev in stages.items():

    # -------- leading jet (>=1 jet)
    mask1 = ak.num(ev.Jet.pt) >= 1
    if ak.any(mask1):
        jets1 = ev.Jet[mask1]
        jets1 = jets1[ak.argsort(jets1.pt, axis=1, ascending=False)]

        jet_lead_pt[k]  = jets1[:, 0].pt
        jet_lead_eta[k] = jets1[:, 0].eta
        jet_lead_phi[k] = jets1[:, 0].phi
    else:
        jet_lead_pt[k]  = None
        jet_lead_eta[k] = None
        jet_lead_phi[k] = None

    # -------- subleading jet (>=2 jets)
    mask2 = ak.num(ev.Jet.pt) >= 2
    if ak.any(mask2):
        jets2 = ev.Jet[mask2]
        jets2 = jets2[ak.argsort(jets2.pt, axis=1, ascending=False)]

        jet_sub_pt[k]  = jets2[:, 1].pt
        jet_sub_eta[k] = jets2[:, 1].eta
        jet_sub_phi[k] = jets2[:, 1].phi
    else:
        jet_sub_pt[k]  = None
        jet_sub_eta[k] = None
        jet_sub_phi[k] = None



plot_overlay_from_arrays(photon_lead_pt, "photon_lead_pt",
                          "Leading photon p_{T} [GeV]", 250, 0, 500, outdir)
plot_overlay_from_arrays(photon_sub_pt,  "photon_sub_pt",
                          "Subleading photon p_{T} [GeV]", 250, 0, 500, outdir)


plot_overlay_from_arrays(jet_lead_pt, "jet_lead_pt",
                          "Leading jet p_{T} [GeV]", 250, 0, 500, outdir)

plot_overlay_from_arrays(jet_sub_pt, "jet_sub_pt",
                          "Subleading jet p_{T} [GeV]", 250, 0, 500, outdir)

plot_overlay_from_arrays(
    photon_lead_eta,
    "photon_lead_eta",
    "Leading photon #eta",
    100, -3, 3,
    outdir
)

plot_overlay_from_arrays(
    photon_sub_eta,
    "photon_sub_eta",
    "Subleading photon #eta",
    100, -3, 3,
    outdir
)

plot_overlay_from_arrays(
    photon_lead_phi,
    "photon_lead_phi",
    "Leading photon #phi",
    64, -3.2, 3.2,
    outdir
)

plot_overlay_from_arrays(
    photon_sub_phi,
    "photon_sub_phi",
    "Subleading photon #phi",
    64, -3.2, 3.2,
    outdir
)

plot_overlay_from_arrays(
    jet_lead_eta,
    "jet_lead_eta",
    "Leading jet #eta",
    100, -3, 3,
    outdir
)

plot_overlay_from_arrays(
    jet_sub_eta,
    "jet_sub_eta",
    "Subleading jet #eta",
    100, -3, 3,
    outdir
)

plot_overlay_from_arrays(
    jet_lead_phi,
    "jet_lead_phi",
    "Leading jet #phi",
    64, -3.2, 3.2,
    outdir
)

plot_overlay_from_arrays(
    jet_sub_phi,
    "jet_sub_phi",
    "Subleading jet #phi",
    64, -3.2, 3.2,
    outdir
)



plot_overlay_from_arrays(
    electron_lead_pt,
    "electron_lead_pt",
    "Leading electron p_{T} [GeV]",
    250, 0, 500,
    outdir
)

plot_overlay_from_arrays(
    electron_lead_eta,
    "electron_lead_eta",
    "Leading electron #eta",
    100, -3, 3,
    outdir
)

plot_overlay_from_arrays(
    electron_lead_phi,
    "electron_lead_phi",
    "Leading electron #phi",
    64, -3.2, 3.2,
    outdir
)

plot_overlay_from_arrays(
    muon_lead_pt,
    "muon_lead_pt",
    "Leading muon p_{T} [GeV]",
    250, 0, 500,
    outdir
)

plot_overlay_from_arrays(
    muon_lead_eta,
    "muon_lead_eta",
    "Leading muon #eta",
    100, -3, 3,
    outdir
)

plot_overlay_from_arrays(
    muon_lead_phi,
    "muon_lead_phi",
    "Leading muon #phi",
    64, -3.2, 3.2,
    outdir
)


import vector
vector.register_awkward()

dipho_mass = {}

for k, ev in stages.items():
    if not hasattr(ev, "Photon"):
        dipho_mass[k] = None
        continue

    mask = ak.num(ev.Photon.pt) >= 2
    if not ak.any(mask):
        dipho_mass[k] = None
        continue

    pho = ev.Photon[mask]

    # sort by pT
    pho = pho[ak.argsort(pho.pt, axis=1, ascending=False)]

    # build massless 4-vectors
    pho_vec = ak.zip(
        {
            "pt": pho.pt,
            "eta": pho.eta,
            "phi": pho.phi,
            "mass": ak.zeros_like(pho.pt),
        },
        with_name="Momentum4D",
    )

    dipho = pho_vec[:, 0] + pho_vec[:, 1]
    dipho_mass[k] = dipho.mass


import vector
vector.register_awkward()

dijet_mass = {}

for k, ev in stages.items():
    # correct collection name
    if not hasattr(ev, "Jet"):
        dijet_mass[k] = None
        continue

    mask = ak.num(ev.Jet.pt) >= 2
    if not ak.any(mask):
        dijet_mass[k] = None
        continue

    jets = ev.Jet[mask]

    # sort by pT
    jets = jets[ak.argsort(jets.pt, axis=1, ascending=False)]

    # build proper jet 4-vectors (USE jet.mass!)
    jet_vec = ak.zip(
        {
            "pt": jets.pt,
            "eta": jets.eta,
            "phi": jets.phi,
            "mass": jets.mass,
        },
        with_name="Momentum4D",
    )

    dijet = jet_vec[:, 0] + jet_vec[:, 1]
    dijet_mass[k] = dijet.mass


plot_overlay_from_arrays(
    dipho_mass,
    "diphoton_mass",
    "m_{#gamma#gamma} [GeV]",
    200, 0, 200,
    outdir
)

plot_overlay_from_arrays(
    dijet_mass,
    "dijet_mass",
    "m_{bb} [GeV]",
    200, 0, 200,
    outdir
)


def plot_all_objects_for_category_stages(
    stages,
    category_name,
    outdir,
    btag_mode,   # "cat1", "cat2", "cat3"
):
    """
    stages: dict(stage_name -> events)
    btag_mode:
      - "cat1": >=2 bjets, take top-2 by score then sort by pT
      - "cat2": >=1 bjet, take highest probbb
      - "cat3": ==1 bjet, take highest B
    """

    import os
    os.makedirs(outdir, exist_ok=True)

    # ------------------------
    # Inclusive object plots
    # ------------------------
    for obj, label, eta_range in [
        ("Photon",   "Photon",   (-3, 3)),
        ("Electron", "Electron", (-3, 3)),
        ("Muon",     "Muon",     (-3, 3)),
        ("Jet",      "Jet",      (-4, 4)),
    ]:
        plot_overlay_from_arrays(
            {k: getattr(v, obj).pt for k, v in stages.items()},
            f"{obj.lower()}_pt", f"{label} p_{{T}} [GeV]",
            250, 0, 500, outdir
        )
        plot_overlay_from_arrays(
            {k: getattr(v, obj).eta for k, v in stages.items()},
            f"{obj.lower()}_eta", f"{label} #eta",
            100, *eta_range, outdir
        )
        plot_overlay_from_arrays(
            {k: getattr(v, obj).phi for k, v in stages.items()},
            f"{obj.lower()}_phi", f"{label} #phi",
            64, -3.2, 3.2, outdir
        )

    # ------------------------
    # Helper for lead / sublead
    # ------------------------
    def lead_sublead(arr, idx):
        mask = ak.num(arr.pt) > idx
        if not ak.any(mask):
            return None
        a = arr[mask]
        a = a[ak.argsort(a.pt, axis=1, ascending=False)]
        return a[:, idx]

    # ------------------------
    # Diphoton invariant mass (ALL categories)
    # ------------------------
    dipho_mass = {}

    for k, ev in stages.items():

        photons = ev.Photon
        mask2 = ak.num(photons) >= 2

        if not ak.any(mask2):
            dipho_mass[k] = None
            continue

        pho = photons[mask2]
        pho = pho[ak.argsort(pho.pt, axis=1, ascending=False)]

        pho1 = pho[:, 0]
        pho2 = pho[:, 1]

        # build 4-vectors explicitly (photons are massless)
        v1 = ak.zip(
            {
                "pt": pho1.pt,
                "eta": pho1.eta,
                "phi": pho1.phi,
                "mass": ak.zeros_like(pho1.pt),
            },
            with_name="Momentum4D",
        )

        v2 = ak.zip(
            {
                "pt": pho2.pt,
                "eta": pho2.eta,
                "phi": pho2.phi,
                "mass": ak.zeros_like(pho2.pt),
            },
            with_name="Momentum4D",
        )

        dipho_mass[k] = (v1 + v2).mass

    plot_overlay_from_arrays(
        dipho_mass,
        "diphoton_mass",
        "m_{#gamma#gamma} [GeV]",
        100, 0, 200,
        outdir
    )



    # ------------------------
    # Photon lead / sublead
    # ------------------------
    pho_lead, pho_sub = {}, {}
    for k, ev in stages.items():
        pho_lead[k] = lead_sublead(ev.Photon, 0)
        pho_sub[k]  = lead_sublead(ev.Photon, 1)

    plot_overlay_from_arrays(
        {k: None if v is None else v.pt for k, v in pho_lead.items()},
        "photon_lead_pt", "Leading photon p_{T} [GeV]",
        250, 0, 500, outdir
    )
    plot_overlay_from_arrays(
        {k: None if v is None else v.pt for k, v in pho_sub.items()},
        "photon_sub_pt", "Subleading photon p_{T} [GeV]",
        250, 0, 500, outdir
    )

    # ------------------------
    # Leading leptons
    # ------------------------
    for obj, label in [("Electron", "electron"), ("Muon", "muon")]:
        lead = {}
        for k, ev in stages.items():
            lead[k] = lead_sublead(getattr(ev, obj), 0)

        plot_overlay_from_arrays(
            {k: None if v is None else v.pt for k, v in lead.items()},
            f"{label}_lead_pt", f"Leading {label} p_{{T}} [GeV]",
            250, 0, 500, outdir
        )
        plot_overlay_from_arrays(
            {k: None if v is None else v.eta for k, v in lead.items()},
            f"{label}_lead_eta", f"Leading {label} #eta",
            100, -3, 3, outdir
        )
        plot_overlay_from_arrays(
            {k: None if v is None else v.phi for k, v in lead.items()},
            f"{label}_lead_phi", f"Leading {label} #phi",
            64, -3.2, 3.2, outdir
        )

    # ------------------------
    # B-JET LOGIC (CATEGORY DEPENDENT)
    # ------------------------
    bjet_lead, bjet_sub = {}, {}

    for k, ev in stages.items():
        jets = ev.Jet

        if btag_mode == "cat1":

            # bjets = jets[jets.btagUParTAK4B > BTAG_B_CUT]

            mask2 = ak.num(jets) >= 2
            if not ak.any(mask2):
                bjet_lead[k] = None
                bjet_sub[k]  = None
                continue

            bjets = jets[mask2]

            # take top-2 by B score
            bjets = bjets[
                ak.argsort(bjets.btagUParTAK4B, axis=1, ascending=False)
            ][:, :2]

            # then order those two by pT
            bjets = bjets[
                ak.argsort(bjets.pt, axis=1, ascending=False)
            ]

            bjet_lead[k] = bjets[:, 0]
            bjet_sub[k]  = bjets[:, 1]

            # ------------------------
            # Dijet invariant mass (CAT1 ONLY)
            # ------------------------
            dijet_mass = {}

            for k, ev in stages.items():

                jets = ev.Jet
                # bjets = jets[jets.btagUParTAK4B > BTAG_B_CUT]

                mask2 = ak.num(jets) >= 2
                if not ak.any(mask2):
                    dijet_mass[k] = None
                    continue

                bjets = jets[mask2]

                # top-2 by btag score
                bjets = bjets[
                    ak.argsort(bjets.btagUParTAK4B, axis=1, ascending=False)
                ]

                # pad so slicing is safe
                bjets = ak.pad_none(bjets, 2)
                bjets2 = bjets[:, :2]

                # reorder those two by pT
                bjets2 = bjets2[
                    ak.argsort(bjets2.pt, axis=1, ascending=False)
                ]

                jet1 = ak.firsts(bjets2)
                jet2 = ak.firsts(bjets2[:, 1:])

                dijet_mass[k] = (jet1 + jet2).mass

            plot_overlay_from_arrays(
                dijet_mass,
                "dijet_mass",
                "m_{bb} [GeV]",
                100, 0, 200,
                outdir
            )




        elif btag_mode == "cat2":
            # bjets = jets[jets.btagUParTAK4probbb > BTAG_PROBB_CUT]

            mask1 = ak.num(jets) >= 1
            bjets = jets[mask1]

            if not ak.any(mask1):
                bjet_lead[k] = None
                bjet_sub[k]  = None
                continue

            bjets = bjets[
                ak.argsort(bjets.btagUParTAK4probbb, axis=1, ascending=False)
            ]

            bjets = ak.pad_none(bjets, 1)
            bjet_lead[k] = ak.firsts(bjets)
            bjet_sub[k]  = None


        elif btag_mode == "cat3":
            # bjets = jets[jets.btagUParTAK4B > BTAG_B_CUT]

            mask1 = ak.num(jets) >= 1
            bjets = jets[mask1]

            if not ak.any(mask1):
                bjet_lead[k] = None
                bjet_sub[k]  = None
                continue

            bjets = bjets[
                ak.argsort(bjets.btagUParTAK4B, axis=1, ascending=False)
            ]

            bjets = ak.pad_none(bjets, 1)
            bjet_lead[k] = ak.firsts(bjets)
            bjet_sub[k]  = None


    # ------------------------
    # B-jet plots
    # ------------------------
    plot_overlay_from_arrays(
        {k: None if v is None else v.pt for k, v in bjet_lead.items()},
        "bjet_lead_pt", "Leading b-jet p_{T} [GeV]",
        250, 0, 500, outdir
    )

    if btag_mode == "cat1":
        plot_overlay_from_arrays(
            {k: None if v is None else v.pt for k, v in bjet_sub.items()},
            "bjet_sub_pt", "Subleading b-jet p_{T} [GeV]",
            250, 0, 500, outdir
        )


plot_all_objects_for_category_stages(
    stages_cat1, "cat1", f"{outdir}/cat1", btag_mode="cat1"
)

plot_all_objects_for_category_stages(
    stages_cat2, "cat2", f"{outdir}/cat2", btag_mode="cat2"
)

plot_all_objects_for_category_stages(
    stages_cat3, "cat3", f"{outdir}/cat3", btag_mode="cat3"
)



