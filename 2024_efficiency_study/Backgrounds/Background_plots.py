import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import ROOT


file = '/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/skimmed_TTto2L2Nu/TTto2L2Nu_preselected.root'
# file = 'root://cms-xrd-global.cern.ch//store/mc/RunIII2024Summer24NanoAODv15/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/NANOAODSIM/150X_mcRun3_2024_realistic_v2-v3/2810000/a808fa28-4a7b-4807-aa90-9678f775f9b2.root'
factory = NanoEventsFactory.from_root(
    f"{file}:Events",
    schemaclass=NanoAODSchema,
)
events = factory.events()

pt_thr = 10

Process = "TTto2L2Nu"

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

# Plot_dir = "/eos/user/b/bbapi/www/Background_study_2024/" + Process + "/15_GeV/"
Plot_dir = "/eos/user/b/bbapi/www/Background_study_2024/" + Process + "/"

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

    barrel_cut = is_barrel & (photons.mvaID > 0.0439603)
    endcap_cut = is_endcap & (photons.mvaID > -0.249526)

    good_photons = (
        (photons.pt > pt_thr) &  # updated pT cut on photons
        valid_eta &
        (barrel_cut | endcap_cut) &
        (~photons.pixelSeed)
    )

    selected_photons = photons[good_photons]

    return selected_photons, selected_bjets, selected_leptons, selected_electrons, selected_muons

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


def make_hist1d(
    array,
    name="hist",
    title="Histogram",
    nbins=50,
    xmin=0,
    xmax=100,
    xlabel="x",
    ylabel="Events",
    outfile="hist.root",
    imgfile="hist.png",
    Process = Process,
    save_root=False,
    logy=False, 
    Normalized=False,
    stage = "After preselection"
):

    ROOT.gStyle.SetOptStat(1111111)

    # flatten awkward arrays if needed
    if isinstance(array, ak.Array):
        array = ak.to_numpy(ak.flatten(array))
    else:
        array = np.array(array)

    # histogram
    h = ROOT.TH1F(name, f";{xlabel};{ylabel}", nbins, xmin, xmax)

    for v in array:
        h.Fill(v)

    if Normalized:
        h.Scale(1.0 / h.Integral())

    # canvas
    c = ROOT.TCanvas("c","",800,600)
    if logy:
        c.SetLogy()

    h.SetLineWidth(2)
    h.Draw("hist")
    CMS_label(c)

    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.04)
    latex.SetTextFont(42)
    latex.DrawLatex(0.63, 0.8, Process)

    latex = ROOT.TLatex()
    latex.SetNDC()              # normalized (0–1) coords
    latex.SetTextSize(0.04)
    latex.SetTextFont(42)
    latex.DrawLatex(0.58, 0.70, "After preselection")

    c.Update()
    draw_side_statboxes(c, [h])
    c.Update()

    # save image
    c.SaveAs(Plot_dir+imgfile)

    if save_root:
        # save ROOT file
        f = ROOT.TFile(Plot_dir+outfile,"RECREATE")
        h.Write()
        c.Write()
        f.Close()

    return h


def plot_overlay(
        var1,
        var2,
        label1="Sample 1",
        label2="Sample 2",
        xlabel="Variable",
        nbins=100,
        xmin=0,
        xmax=100,
        logy=True,
        normalize=True,
        flatten=False,
        dataset_label=Process,
        extra_label="After preselection",
        output_name="plot",
        plot_dir=Plot_dir
    ):

    ROOT.gStyle.SetOptStat(1111111)



    # Histograms
    h1 = ROOT.TH1F("h1",";{};Events".format(xlabel),nbins,xmin,xmax)
    h2 = ROOT.TH1F("h2",";{};Events".format(xlabel),nbins,xmin,xmax)

    # Fill

    if flatten:
        for v in np.array(ak.flatten(var1)):
            h1.Fill(v)

        for v in np.array(ak.flatten(var2)):
            h2.Fill(v)
    else:
        for v in np.array(var1):
            h1.Fill(v)

        for v in np.array(var2):
            h2.Fill(v)


    # Normalize
    if normalize:
        if h1.Integral()>0:
            h1.Scale(1.0/h1.Integral())
        if h2.Integral()>0:
            h2.Scale(1.0/h2.Integral())

    # Style
    h1.SetLineColor(ROOT.kRed)
    h1.SetLineWidth(2)

    h2.SetLineColor(ROOT.kBlue)
    h2.SetLineWidth(2)

    h1.SetMaximum(1.2*max(h1.GetMaximum(),h2.GetMaximum()))

    # Canvas
    c = ROOT.TCanvas("c","",800,600)

    if logy:
        c.SetLogy()

    h1.Draw("hist")
    h2.Draw("hist same")

    # Legend
    leg = ROOT.TLegend(0.55,0.75,0.78,0.88)
    leg.AddEntry(h1,label1,"l")
    leg.AddEntry(h2,label2,"l")
    leg.Draw()

    # CMS label
    CMS_label(c)

    # Dataset text
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.04)
    latex.SetTextFont(42)

    if dataset_label:
        latex.DrawLatex(0.13,0.83,dataset_label)

    if extra_label:
        latex.DrawLatex(0.55,0.70,extra_label)

    c.Update()

    draw_side_statboxes(c,[h1,h2])

    c.Update()

    # Save
    c.SaveAs(plot_dir + output_name + ".png")

    f = ROOT.TFile(plot_dir + output_name + ".root","RECREATE")
    c.Write()
    f.Close()

gen = events[event_mask].GenPart

selected_photons = selected_photons[event_mask]

selected_photons = selected_photons[ak.argsort(selected_photons.pt, ascending=False)]

mother_pdg = gen[gen[selected_photons.genPartIdx].genPartIdxMother].pdgId

mother_ids = ak.flatten(mother_pdg)

from particle import Particle

# ---------------------------------
# Use your real jagged array here
# mother_pdgId = ...
# ---------------------------------

# Flatten jagged array
# flat_np = np.array(mother_ids)

p1 = mother_pdg[:,0]
p2   = mother_pdg[:,1]

from particle import Particle

# Convert to numpy
p1 = ak.to_numpy(p1)
p2 = ak.to_numpy(p2)

# --- determine particle contributions ---
all_pdgs = np.concatenate([p1, p2])

unique_pdgs, pdg_counts = np.unique(all_pdgs, return_counts=True)

fractions = pdg_counts / pdg_counts.sum()

threshold = 0.02

# Count unique PDG IDs
unique_ids, counts = np.unique(all_pdgs, return_counts=True)
total = np.sum(counts)

# Keep only major contributors (>1%)
threshold = 0.01
major_ids = []
major_counts = []
others_count = 0

for pid, cnt in zip(unique_ids, counts):
    if cnt / total > threshold:
        major_ids.append(pid)
        major_counts.append(cnt)
    else:
        others_count += cnt

if others_count > 0:
    major_ids.append(999999)
    major_counts.append(others_count)

# Create histogram
n_bins = len(major_ids)
h = ROOT.TH1F(
    "h",
    ";Mother Particle of Reco Photons;Counts",
    n_bins,
    0,
    n_bins
)
h.GetXaxis().SetTitleOffset(1.3)
h.SetLabelSize(0.04)
h.SetFillColor(ROOT.kBlue)

# Fill histogram with particle names
for i, (pid, cnt) in enumerate(zip(major_ids, major_counts), start=1):

    if pid == 999999:
        label = "Others"
    else:
        try:
            label = Particle.from_pdgid(int(pid)).name
        except:
            label = str(pid)

    h.GetXaxis().SetBinLabel(i, label)
    h.SetBinContent(i, cnt)

# Draw
ROOT.gStyle.SetOptStat(0)
c = ROOT.TCanvas("c", "", 1000, 600)
c.SetBottomMargin(0.14)
h.Draw("bar")
CMS_label(c)

latex = ROOT.TLatex()
latex.SetNDC()              # normalized (0–1) coords
latex.SetTextSize(0.04)
latex.SetTextFont(42)
latex.DrawLatex(0.13, 0.85, Process)

latex = ROOT.TLatex()
latex.SetNDC()              # normalized (0–1) coords
latex.SetTextSize(0.04)
latex.SetTextFont(42)
latex.DrawLatex(0.13, 0.75, "After preselection")

c.Update()

c.SaveAs(Plot_dir + "genPart_selected_phos_" + Process + "_lead_sublead.png")

flat_np = np.array(mother_ids)

unique_pdgs, pdg_counts = np.unique(flat_np, return_counts=True)

fractions = pdg_counts / pdg_counts.sum()

threshold = 0.02

# Count unique PDG IDs
unique_ids, counts = np.unique(all_pdgs, return_counts=True)
total = np.sum(counts)

# Keep only major contributors (>1%)
threshold = 0.01
major_ids = []
major_counts = []
others_count = 0

for pid, cnt in zip(unique_ids, counts):
    if cnt / total > threshold:
        major_ids.append(pid)
        major_counts.append(cnt)
    else:
        others_count += cnt

if others_count > 0:
    major_ids.append(999999)
    major_counts.append(others_count)

# Create histogram
n_bins = len(major_ids)
h = ROOT.TH1F(
    "h",
    ";Mother Particle of Reco Photons;Counts",
    n_bins,
    0,
    n_bins
)
h.GetXaxis().SetTitleOffset(1.3)
h.SetLabelSize(0.04)
h.SetFillColor(ROOT.kBlue)

# Fill histogram with particle names
for i, (pid, cnt) in enumerate(zip(major_ids, major_counts), start=1):

    if pid == 999999:
        label = "Others"
    else:
        try:
            label = Particle.from_pdgid(int(pid)).name
        except:
            label = str(pid)

    h.GetXaxis().SetBinLabel(i, label)
    h.SetBinContent(i, cnt)

# Draw
ROOT.gStyle.SetOptStat(0)
c = ROOT.TCanvas("c", "", 1000, 600)
c.SetBottomMargin(0.14)
h.Draw("bar")
CMS_label(c)

latex = ROOT.TLatex()
latex.SetNDC()              # normalized (0–1) coords
latex.SetTextSize(0.04)
latex.SetTextFont(42)
latex.DrawLatex(0.13, 0.85, Process)

latex = ROOT.TLatex()
latex.SetNDC()              # normalized (0–1) coords
latex.SetTextSize(0.04)
latex.SetTextFont(42)
latex.DrawLatex(0.13, 0.75, "After preselection")

c.Update()

c.SaveAs(Plot_dir + "genPart_selected_phos_" + Process + "_all.png")

p1 = mother_pdg[:,0]
p2   = mother_pdg[:,1]

from particle import Particle

# Convert to numpy
p1 = ak.to_numpy(p1)
p2 = ak.to_numpy(p2)

# --- determine particle contributions ---
all_pdgs = np.concatenate([p1, p2])

unique_pdgs, pdg_counts = np.unique(all_pdgs, return_counts=True)

fractions = pdg_counts / pdg_counts.sum()

threshold = 0.02

major_pdgs = unique_pdgs[fractions >= threshold]
major_counts = pdg_counts[fractions >= threshold]

# sort by contribution (largest first)
order = np.argsort(-major_counts)
major_pdgs = major_pdgs[order]

pdgs = list(major_pdgs)

# --- define histogram size ---
others_bin = len(pdgs) + 1
n = len(pdgs) + 1

# delete existing histogram if ROOT already has one
if ROOT.gROOT.FindObject("h2D"):
    ROOT.gROOT.FindObject("h2D").Delete()

h2D = ROOT.TH2F(
    "h2D",
    ";Mother #gamma_{1};Mother #gamma_{2}",
    n, 0, n,
    n, 0, n
)

# --- map pdg -> bin ---
pdg_to_bin = {pdg: i+1 for i, pdg in enumerate(pdgs)}

# --- fill histogram ---
for d, m in zip(p1, p2):

    x = pdg_to_bin.get(d, others_bin)
    y = pdg_to_bin.get(m, others_bin)

    h2D.Fill(x-0.5, y-0.5)

# --- pdg -> particle name ---
def name(pdg):
    try:
        return Particle.from_pdgid(int(pdg)).name
    except:
        return str(pdg)

# --- set axis labels ---
for i, pdg in enumerate(pdgs):

    label = name(pdg)

    h2D.GetXaxis().SetBinLabel(i+1, label)
    h2D.GetYaxis().SetBinLabel(i+1, label)

h2D.GetXaxis().SetBinLabel(n, "Others")
h2D.GetYaxis().SetBinLabel(n, "Others")

# --- plotting ---
c = ROOT.TCanvas("c","c",1800, 1600)

ROOT.gStyle.SetOptStat(0)

h2D.Draw("COLZ TEXT")
h2D.SetMarkerColor(ROOT.kWhite)
h2D.GetXaxis().SetTitleOffset(1.4)

c.SetRightMargin(0.15)
c.SetLeftMargin(0.16)
c.SetBottomMargin(0.10)

CMS_label(c)

latex = ROOT.TLatex()
latex.SetNDC()
latex.SetTextSize(0.04)
latex.SetTextFont(42)
latex.DrawLatex(0.13, 0.03, Process)

# latex.DrawLatex(0.13, 0.75, "After preselection")

c.SaveAs(Plot_dir+"p1_p2_2D_photons_" + Process +".png")
c.SaveAs(Plot_dir+"p1_p2_2D_photons_" + Process + ".pdf")


dR_ele_pho_lead = delta_r_manual(gen[(abs(gen.pdgId) == 11)&(gen.status == 1)&(gen.pt > 10)], selected_photons[:,0])
dR_ele_pho_sub = delta_r_manual(gen[(abs(gen.pdgId) == 11)&(gen.status == 1)&(gen.pt > 10)], selected_photons[:,1])
dR_ele_pho_lead_min = ak.min(dR_ele_pho_lead, axis=1)
dR_ele_pho_sub_min = ak.min(dR_ele_pho_sub, axis=1)

make_hist1d(
    dR_ele_pho_lead_min,
    name="dR_ele_pho_lead_min",
    title="Minimum Delta R between Electrons and Leading Photons",
    nbins=50,
    xmin=0,
    xmax=10.0,
    xlabel="min #DeltaR(e, #gamma_{lead})",
    ylabel="Events",
    imgfile="dR_ele_pho_lead_min.png",
    Process = Process,
    save_root=True,
    logy=True
)

make_hist1d(
    dR_ele_pho_sub_min,
    name="dR_ele_pho_sub_min",
    title="Minimum Delta R between Electrons and Sub-leading Photons",
    nbins=50,
    xmin=0,
    xmax=10.0,
    xlabel="min #DeltaR(e, #gamma_{sub-lead})",
    ylabel="Events",
    imgfile="dR_ele_pho_sub_min.png",
    Process = Process,
    save_root=True,
    logy=True
)

dR_mu_pho_lead = delta_r_manual(gen[(abs(gen.pdgId) == 13)&(gen.status == 1)&(gen.pt > 10)], selected_photons[:,0])
dR_mu_pho_sub = delta_r_manual(gen[(abs(gen.pdgId) == 13)&(gen.status == 1)&(gen.pt > 10)], selected_photons[:,1])
dR_mu_pho_lead_min = ak.min(dR_mu_pho_lead, axis=1)
dR_mu_pho_sub_min = ak.min(dR_mu_pho_sub, axis=1)

make_hist1d(
    dR_mu_pho_lead_min,
    name="dR_mu_pho_lead_min",
    title="Minimum Delta R between Muons and Leading Photons",
    nbins=50,
    xmin=0,
    xmax=10.0,
    xlabel="min #DeltaR(#mu, #gamma_{lead})",
    ylabel="Events",
    imgfile="dR_mu_pho_lead_min.png",
    Process = Process,
    save_root=True,
    logy=True
)

make_hist1d(
    dR_mu_pho_sub_min,
    name="dR_mu_pho_sub_min",
    title="Minimum Delta R between Muons and sub-Leading Photons",
    nbins=50,
    xmin=0,
    xmax=10.0,
    xlabel="min #DeltaR(#mu, #gamma_{sub-lead})",
    ylabel="Events",
    imgfile="dR_mu_pho_sub_min.png",
    Process = Process,
    save_root=True,
    logy=True
)

dR_ele_fake_ele_pho = delta_r_manual(gen[(abs(gen.pdgId) == 11)&(gen.status == 1)&(gen.pt > 10)], selected_photons[gen[selected_photons.genPartIdx].pdgId == 11])
dR_ele_fake_ele_pho_min = ak.min(dR_ele_fake_ele_pho, axis=1)

make_hist1d(
    dR_ele_fake_ele_pho_min,
    name="dR_ele_fake_ele_pho_min",
    title="Minimum Delta R between Electrons and Electrons faking as Photons",
    nbins=50,
    xmin=0,
    xmax=10.0,
    xlabel="min #DeltaR(e, #gamma_{fake-ele})",
    ylabel="Events",
    imgfile="dR_ele_fake_ele_pho_min.png",
    Process = Process,
    save_root=True,
    logy=True
)

flattened_phos = ak.flatten(selected_photons)
pi0_pho = flattened_phos[abs(ak.flatten(mother_pdg)) == 111]
jet_fake_pho = flattened_phos[(abs(ak.flatten(mother_pdg)) != 22) & (abs(ak.flatten(mother_pdg)) != 11) & (abs(ak.flatten(mother_pdg)) != 13) & (abs(ak.flatten(mother_pdg)) != 111)]
true_pho = flattened_phos[abs(ak.flatten(mother_pdg)) == 22]

# pi0_pho and true_pho should be your 1D arrays (numpy/awkward flattened)


# ROOT.gStyle.SetOptStat(1111111)

# plot_overlay(
#     var1=true_pho.hoe,
#     var2=pi0_pho.hoe,
#     label1="True photons",
#     label2="#pi^{0} photons",
#     xlabel="H/E",
#     nbins=50,
#     xmin=0,
#     xmax=0.6,
#     logy=True,
#     normalize=True,
#     output_name="hoe_overlay_normalized_pi0"
# )

# ROOT.gStyle.SetOptStat(1111111)

# plot_overlay(
#     var1=true_pho.hoe,
#     var2=pi0_pho.hoe,
#     label1="True photons",
#     label2="#pi^{0} photons",
#     xlabel="H/E",
#     nbins=50,
#     xmin=0,
#     xmax=0.6,
#     logy=True,
#     normalize=False,
#     output_name="hoe_overlay_pi0"
# )

# ROOT.gStyle.SetOptStat(1111111)

# plot_overlay(
#     var1=true_pho.r9,
#     var2=pi0_pho.r9,
#     label1="True photons",
#     label2="#pi^{0} photons",
#     xlabel="R_{9}",
#     nbins=50,
#     xmin=0,
#     xmax=2.0,
#     logy=True,
#     normalize=True,
#     output_name="R9_overlay_normalized_pi0"
# )

# ROOT.gStyle.SetOptStat(1111111)

# plot_overlay(
#     var1=true_pho.r9,
#     var2=pi0_pho.r9,
#     label1="True photons",
#     label2="#pi^{0} photons",
#     xlabel="R_{9}",
#     nbins=50,
#     xmin=0,
#     xmax=2.0,
#     logy=True,
#     normalize=False,
#     output_name="R9_overlay_pi0"
# )

# ROOT.gStyle.SetOptStat(1111111)

# plot_overlay(
#     var1=true_pho.pfPhoIso03,
#     var2=pi0_pho.pfPhoIso03,
#     label1="True photons",
#     label2="#pi^{0} photons",
#     xlabel="pfPhoIso03",
#     nbins=50,
#     xmin=0,
#     xmax=10.0,
#     logy=True,
#     normalize=True,
#     output_name="pfPhoIso03_overlay_normalized_pi0"
# )

# ROOT.gStyle.SetOptStat(1111111)

# plot_overlay(
#     var1=true_pho.pfPhoIso03,
#     var2=pi0_pho.pfPhoIso03,
#     label1="True photons",
#     label2="#pi^{0} photons",
#     xlabel="pfPhoIso03",
#     nbins=50,
#     xmin=0,
#     xmax=10.0,
#     logy=True,
#     normalize=False,
#     output_name="pfPhoIso03_overlay_pi0"
# )

# ROOT.gStyle.SetOptStat(1111111)

# plot_overlay(
#     var1=true_pho.hoe,
#     var2=jet_fake_pho.hoe,
#     label1="True photons",
#     label2="Jet_fake photons",
#     xlabel="H/E",
#     nbins=50,
#     xmin=0,
#     xmax=0.6,
#     logy=True,
#     normalize=True,
#     output_name="hoe_overlay_normalized_jet_fake"
# )

# ROOT.gStyle.SetOptStat(1111111)

# plot_overlay(
#     var1=true_pho.hoe,
#     var2=jet_fake_pho.hoe,
#     label1="True photons",
#     label2="Jet_fake photons",
#     xlabel="H/E",
#     nbins=50,
#     xmin=0,
#     xmax=0.6,
#     logy=True,
#     normalize=False,
#     output_name="hoe_overlay_jet_fake"
# )

# ROOT.gStyle.SetOptStat(1111111)

# plot_overlay(
#     var1=true_pho.r9,
#     var2=jet_fake_pho.r9,
#     label1="True photons",
#     label2="Jet_fake photons",
#     xlabel="R_{9}",
#     nbins=50,
#     xmin=0,
#     xmax=2.0,
#     logy=True,
#     normalize=True,
#     output_name="R9_overlay_normalized_jet_fake"
# )

# ROOT.gStyle.SetOptStat(1111111)

# plot_overlay(
#     var1=true_pho.r9,
#     var2=jet_fake_pho.r9,
#     label1="True photons",
#     label2="Jet_fake photons",
#     xlabel="R_{9}",
#     nbins=50,
#     xmin=0,
#     xmax=2.0,
#     logy=True,
#     normalize=False,
#     output_name="R9_overlay_jet_fake"
# )

# ROOT.gStyle.SetOptStat(1111111)

# plot_overlay(
#     var1=true_pho.pfPhoIso03,
#     var2=jet_fake_pho.pfPhoIso03,
#     label1="True photons",
#     label2="Jet_fake photons",
#     xlabel="pfPhoIso03",
#     nbins=50,
#     xmin=0,
#     xmax=10.0,
#     logy=True,
#     normalize=True,
#     output_name="pfPhoIso03_overlay_normalized_jet_fake"
# )

# ROOT.gStyle.SetOptStat(1111111)

# plot_overlay(
#     var1=true_pho.pfPhoIso03,
#     var2=jet_fake_pho.pfPhoIso03,
#     label1="True photons",
#     label2="Jet_fake photons",
#     xlabel="pfPhoIso03",
#     nbins=50,
#     xmin=0,
#     xmax=10.0,
#     logy=True,
#     normalize=False,
#     output_name="pfPhoIso03_overlay_jet_fake"
# )


ROOT.gStyle.SetOptStat(1111111)

plot_overlay(
    var1=pi0_pho.hoe,
    var2=jet_fake_pho.hoe,
    label1="#pi^{0} photons",
    label2="Jet fake photons",
    xlabel="H/E",
    nbins=50,
    xmin=0,
    xmax=0.6,
    logy=True,
    normalize=True,
    output_name="hoe_overlay_normalized_pi0_jetfake"
)

ROOT.gStyle.SetOptStat(1111111)

plot_overlay(
    var1=pi0_pho.hoe,
    var2=jet_fake_pho.hoe,
    label1="#pi^{0} photons",
    label2="Jet fake photons",
    xlabel="H/E",
    nbins=50,
    xmin=0,
    xmax=0.6,
    logy=True,
    normalize=False,
    output_name="hoe_overlay_pi0_jetfake"
)

ROOT.gStyle.SetOptStat(1111111)

plot_overlay(
    var1=pi0_pho.r9,
    var2=jet_fake_pho.r9,
    label1="#pi^{0} photons",
    label2="Jet fake photons",
    xlabel="R_{9}",
    nbins=50,
    xmin=0,
    xmax=2.0,
    logy=True,
    normalize=True,
    output_name="R9_overlay_normalized_pi0_jetfake"
)

ROOT.gStyle.SetOptStat(1111111)

plot_overlay(
    var1=pi0_pho.r9,
    var2=jet_fake_pho.r9,
    label1="#pi^{0} photons",
    label2="Jet fake photons",
    xlabel="R_{9}",
    nbins=50,
    xmin=0,
    xmax=2.0,
    logy=True,
    normalize=False,
    output_name="R9_overlay_pi0_jetfake"
)

ROOT.gStyle.SetOptStat(1111111)

plot_overlay(
    var1=pi0_pho.pfPhoIso03,
    var2=jet_fake_pho.pfPhoIso03,
    label1="#pi^{0} photons",
    label2="Jet fake photons",
    xlabel="pfPhoIso03",
    nbins=50,
    xmin=0,
    xmax=10.0,
    logy=True,
    normalize=True,
    output_name="pfPhoIso03_overlay_normalized_pi0_jetfake"
)

ROOT.gStyle.SetOptStat(1111111)

plot_overlay(
    var1=pi0_pho.pfPhoIso03,
    var2=jet_fake_pho.pfPhoIso03,
    label1="#pi^{0} photons",
    label2="Jet fake photons",
    xlabel="pfPhoIso03",
    nbins=50,
    xmin=0,
    xmax=10.0,
    logy=True,
    normalize=False,
    output_name="pfPhoIso03_overlay_pi0_jetfake"
)

pho_gen = gen[gen[selected_photons.genPartIdx].genPartIdxMother]

ele_faking_pho = selected_photons[((abs(pho_gen.pdgId) == 11)) & (abs(gen[selected_photons.genPartIdx].pdgId) == 11)]
ele_faking_pho = ele_faking_pho[ak.num(ele_faking_pho.pt)>0]

brems_pho = selected_photons[((abs(pho_gen.pdgId) == 11)) & (abs(gen[selected_photons.genPartIdx].pdgId) == 22)]
brems_pho = brems_pho[ak.num(brems_pho.pt)>0]

ROOT.gStyle.SetOptStat(1111111)

plot_overlay(
    var1=brems_pho.sieie,
    var2=ele_faking_pho.sieie,
    label1="brem photon",
    label2="electron faking photon",
    xlabel="#sigma_{i#etai#eta}",
    nbins=200,
    xmin=0,
    xmax=0.1,
    logy=True,
    flatten = True,
    normalize=True,
    output_name="sieie_overlay_normalized"
)

ROOT.gStyle.SetOptStat(1111111)

ROOT.gStyle.SetOptStat(1111111)

plot_overlay(
    var1=brems_pho.sieie,
    var2=ele_faking_pho.sieie,
    label1="brem photon",
    label2="electron faking photon",
    xlabel="#sigma_{i#etai#eta}",
    nbins=200,
    xmin=0,
    xmax=0.1,
    logy=True,
    flatten = True,
    normalize=False,
    output_name="sieie_overlay"
)

plot_overlay(
    var1=brems_pho.r9,
    var2=ele_faking_pho.r9,
    label1="brem photon",
    label2="electron faking photon",
    xlabel="R9",
    nbins=200,
    xmin=0,
    xmax=2.0,
    logy=True,
    flatten = True,
    normalize=True,
    output_name="R9_overlay_normalized"
)

ROOT.gStyle.SetOptStat(1111111)

plot_overlay(
    var1=brems_pho.r9,
    var2=ele_faking_pho.r9,
    label1="brem photon",
    label2="electron faking photon",
    xlabel="R9",
    nbins=200,
    xmin=0,
    xmax=2.0,
    logy=True,
    flatten = True,
    normalize=False,
    output_name="R9_overlay"
)

ROOT.gStyle.SetOptStat(1111111)

plot_overlay(
    var1=brems_pho.sieip,
    var2=ele_faking_pho.sieip,
    label1="brem photon",
    label2="electron faking photon",
    xlabel="#sigma_{i#etaip}",
    nbins=200,
    xmin=0,
    xmax=0.001,
    logy=True,
    flatten = True,
    normalize=True,
    output_name="sieip_overlay_normalized"
)

ROOT.gStyle.SetOptStat(1111111)

plot_overlay(
    var1=brems_pho.sieip,
    var2=ele_faking_pho.sieip,
    label1="brem photon",
    label2="electron faking photon",
    xlabel="#sigma_{i#etaip}",
    nbins=200,
    xmin=0,
    xmax=0.001,
    logy=True,
    flatten = True,
    normalize=False,
    output_name="sieip_overlay"
)

ROOT.gStyle.SetOptStat(1111111)

plot_overlay(
    var1=brems_pho.pt,
    var2=ele_faking_pho.pt,
    label1="brem photon",
    label2="electron faking photon",
    xlabel="p_{T}",
    nbins=200,
    xmin=0,
    xmax=200,
    logy=True,
    flatten = True,
    normalize=True,
    output_name="pt_overlay_normalized"
)

ROOT.gStyle.SetOptStat(1111111)

plot_overlay(
    var1=brems_pho.pt,
    var2=ele_faking_pho.pt,
    label1="brem photon",
    label2="electron faking photon",
    xlabel="p_{T}",
    nbins=200,
    xmin=0,
    xmax=200,
    logy=True,
    flatten = True,
    normalize=False,
    output_name="pt_overlay"
)

brems_frac = len(ak.flatten(brems_pho))/(2*len(selected_photons))
fake_ele_frac = len(ak.flatten(ele_faking_pho))/(2*len(selected_photons))

true_brem_pho_frac = ak.sum(brems_pho.electronVeto)/len(ak.flatten(brems_pho.electronVeto))
true_fake_ele_pho_frac = ak.sum(ele_faking_pho.electronVeto)/len(ak.flatten(ele_faking_pho.electronVeto))

# Histogram with 2 bins
h = ROOT.TH1F("h_frac", ";; Electron Veto passing Fraction", 2, 0, 2)

h.SetBinContent(1, true_brem_pho_frac)
h.SetBinContent(2, true_fake_ele_pho_frac)

h.GetXaxis().SetBinLabel(1, "brem photon")
h.GetXaxis().SetBinLabel(2, "electron fake photon")

# Style
h.SetFillColor(ROOT.kAzure+1)
h.SetBarWidth(0.5)
h.SetBarOffset(0.25)
h.GetXaxis().SetLabelSize(0.06)
h.GetXaxis().SetTitleOffset(1.1)

# Canvas
c = ROOT.TCanvas("c","",700,600)

h.SetMaximum(1.0)
h.SetMinimum(0.0)
h.Draw("bar")

c.Update()

CMS_label(c)

latex = ROOT.TLatex()
latex.SetNDC()
latex.SetTextSize(0.04)
latex.SetTextFont(42)
latex.DrawLatex(0.55, 0.83, Process)

latex = ROOT.TLatex()
latex.SetNDC()              # normalized (0–1) coords
latex.SetTextSize(0.04)
latex.SetTextFont(42)
latex.DrawLatex(0.55, 0.70, "After preselection")

c.SaveAs(Plot_dir+"electronVeto_fraction_" + Process + ".png")

# Histogram with 2 bins
h = ROOT.TH1F("h_frac", ";; Brem vs ele-fake-pho percentage", 2, 0, 2)

h.SetBinContent(1, brems_frac*100)
h.SetBinContent(2, fake_ele_frac*100)

h.GetXaxis().SetBinLabel(1, "brem photon")
h.GetXaxis().SetBinLabel(2, "electron fake photon")

# Style
h.SetFillColor(ROOT.kAzure+1)
h.SetBarWidth(0.5)
h.SetBarOffset(0.25)
h.GetXaxis().SetLabelSize(0.06)
h.GetXaxis().SetTitleOffset(1.1)

# Canvas
c = ROOT.TCanvas("c","",700,600)

h.SetMaximum(20)
h.SetMinimum(0.0)
h.Draw("bar")

c.Update()

CMS_label(c)

latex = ROOT.TLatex()
latex.SetNDC()
latex.SetTextSize(0.04)
latex.SetTextFont(42)
latex.DrawLatex(0.55, 0.83, Process)

latex = ROOT.TLatex()
latex.SetNDC()              # normalized (0–1) coords
latex.SetTextSize(0.04)
latex.SetTextFont(42)
latex.DrawLatex(0.55, 0.70, "After preselection")

c.SaveAs(Plot_dir+"fake_pho_fraction_" + Process + ".png")

#pho_gen is the reco matched gen particle's mother particle
brems = selected_photons[(abs(pho_gen.pdgId) == 11) & (abs(gen[selected_photons.genPartIdx].pdgId) == 22)]

one_mask = ak.num(brems.pt)>0

ele_gen = gen[abs(gen.pdgId)==11]
ele_gen = ele_gen[one_mask]

reco_matched = gen[selected_photons.genPartIdx]

matched_reco_pho = reco_matched[reco_matched.pdgId == 22]
matched_reco_pho = matched_reco_pho[one_mask]
matched_reco_pho = ak.pad_none(matched_reco_pho, 2)
matched_reco_pho_lead = matched_reco_pho[:,0]
matched_reco_pho_sublead = matched_reco_pho[:,1]

bremmed_ele1 = ele_gen[ele_gen.genPartIdxMother == matched_reco_pho_lead.genPartIdxMother]
bremmed_ele2 = ele_gen[ele_gen.genPartIdxMother == matched_reco_pho_sublead.genPartIdxMother]

brem1_dr = delta_r_manual(bremmed_ele1, matched_reco_pho_lead)
brem2_dr = delta_r_manual(bremmed_ele2, matched_reco_pho_sublead)

all_brem_dr = ak.concatenate([ak.flatten(brem1_dr),ak.flatten(brem2_dr)])
all_brem_dr = all_brem_dr[ak.num(all_brem_dr)>0]

make_hist1d(
    all_brem_dr,
    name="dR_bremmed_ele_brem_pho",
    title="Delta R between bremmed electrons and brem photon (Gen)",
    nbins=50,
    xmin=0,
    xmax=10.0,
    xlabel="#Delta R between bremmed electron and brem photon (Gen)",
    ylabel="Events",
    imgfile="dR_bremmed_ele_brem_pho.png",
    Process = Process,
    save_root=True,
    logy=True
)

# your awkward array (example variable name)
pdg_array = pho_gen.pdgId   # replace with your array

# flatten the particle list
flat = ak.to_numpy(ak.flatten(pdg_array))

# define PDG groups
groups = {
    "W+/W-": [24, -24],
    "e+/e-": [11, -11],
    "pi0": [111],
    "mu+/mu-": [13, -13],
    "tau+/tau-": [15, -15],
    "t/anti-top": [6, -6]
}

counts = {}

# count particles in each group
for name, pdgs in groups.items():
    counts[name] = np.sum(np.isin(flat, pdgs))

# count others
all_pdgs = np.concatenate(list(groups.values()))
mask_all = np.isin(flat, all_pdgs)
counts["others"] = np.sum(~mask_all)

# convert to fractions
total = len(flat)
fractions = {k: v/total for k,v in counts.items()}

# ROOT histogram (bar plot)
labels = list(fractions.keys())
values = list(fractions.values())

h = ROOT.TH1F("h_fraction", ";Particle Type;Fraction",
              len(labels), 0, len(labels))

for i, (label, val) in enumerate(zip(labels, values), start=1):
    h.SetBinContent(i, val)
    h.GetXaxis().SetBinLabel(i, label)

h.SetFillColor(ROOT.kAzure+1)
h.SetBarWidth(0.8)
h.SetBarOffset(0.1)

c = ROOT.TCanvas("c", "c", 800, 600)
h.Draw("bar2")

CMS_label(c)

latex = ROOT.TLatex()
latex.SetNDC()
latex.SetTextSize(0.04)
latex.SetTextFont(42)
latex.DrawLatex(0.13, 0.83, Process)

latex = ROOT.TLatex()
latex.SetNDC()              # normalized (0–1) coords
latex.SetTextSize(0.04)
latex.SetTextFont(42)
latex.DrawLatex(0.55, 0.70, "After preselection")

c.SaveAs(Plot_dir+"particle_fraction_" + Process + "_all.png")


# -------------------------
# Extract PDG IDs
# -------------------------
pdg_array = pho_gen.pdgId

# ensure at least 2 photons per event
pdg_padded = ak.pad_none(pdg_array, 2)

lead_pdg = ak.to_numpy(pdg_padded[:, 0])
sublead_pdg = ak.to_numpy(pdg_padded[:, 1])

# remove NaN / None
lead_pdg = lead_pdg[~np.isnan(lead_pdg)]
sublead_pdg = sublead_pdg[~np.isnan(sublead_pdg)]

# -------------------------
# Define PDG groups
# -------------------------
groups = {
    # "gamma": [22],
    "W+/W-": [24, -24],
    "e+/e-": [11, -11],
    "pi0": [111],
    "mu+/mu-": [13, -13],
    "tau+/tau-": [15, -15],
    "t/anti-top": [6, -6],
    # "b_hadrons": [511, 521, 531, 541, 5122, 5132, 5232, 5332],
    # "c_hadrons": [411, 421, 431, 441, 4122, 4132, 4232, 4332]
}

# -------------------------
# Function-like inline logic
# -------------------------
def compute_fraction(arr):
    counts = {}
    for name, pdgs in groups.items():
        counts[name] = np.sum(np.isin(arr, pdgs))

    all_pdgs = np.concatenate(list(groups.values()))
    mask_all = np.isin(arr, all_pdgs)
    counts["others"] = np.sum(~mask_all)

    total = len(arr)
    return {k: v/total if total > 0 else 0 for k, v in counts.items()}

fractions_lead = compute_fraction(lead_pdg)
fractions_sublead = compute_fraction(sublead_pdg)

# -------------------------
# Plot: Leading photon
# -------------------------
labels = list(fractions_lead.keys())

h_lead = ROOT.TH1F("h_lead", ";Particle Type;Fraction",
                  len(labels), 0, len(labels))

for i, label in enumerate(labels, start=1):
    h_lead.SetBinContent(i, fractions_lead[label])
    h_lead.GetXaxis().SetBinLabel(i, label)

h_lead.SetFillColor(ROOT.kAzure+1)
h_lead.SetBarWidth(0.8)
h_lead.SetBarOffset(0.1)

c1 = ROOT.TCanvas("c1", "c1", 800, 600)
h_lead.Draw("bar2")

CMS_label(c1)

latex = ROOT.TLatex()
latex.SetNDC()
latex.SetTextSize(0.04)
latex.DrawLatex(0.13, 0.83, Process)
latex.DrawLatex(0.55, 0.70, "Leading photon")

c1.SaveAs(Plot_dir + "particle_fraction_lead.png")

# -------------------------
# Plot: Subleading photon
# -------------------------
h_sub = ROOT.TH1F("h_sub", ";Particle Type;Fraction",
                 len(labels), 0, len(labels))

for i, label in enumerate(labels, start=1):
    h_sub.SetBinContent(i, fractions_sublead[label])
    h_sub.GetXaxis().SetBinLabel(i, label)

h_sub.SetFillColor(ROOT.kAzure+1)
h_sub.SetBarWidth(0.8)
h_sub.SetBarOffset(0.1)

c2 = ROOT.TCanvas("c2", "c2", 800, 600)
h_sub.Draw("bar2")

CMS_label(c2)

latex = ROOT.TLatex()
latex.SetNDC()
latex.SetTextSize(0.04)
latex.DrawLatex(0.13, 0.83, Process)
latex.DrawLatex(0.55, 0.70, "Subleading photon")

c2.SaveAs(Plot_dir + "particle_fraction_sublead.png")

