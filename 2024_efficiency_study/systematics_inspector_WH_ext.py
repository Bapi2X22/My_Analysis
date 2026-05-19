import awkward as ak
import numpy as np
import ROOT
from collections import defaultdict
import os


out_dir = "/eos/user/b/bbapi/www/Systematics_study/WH2024/final_selections/"
os.makedirs(out_dir, exist_ok=True)

# ===============================
# Configuration
# ===============================
base_dir = "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study"
# nominal_dir = base_dir + "nominal/diphoton/"
nominal_photon_dir = base_dir + "nominal/diphoton/"
nominal_jet_dir    = base_dir + "nominal/diphoton/"

process = "WH-2024M30"

dir_list = [
    "ScaleEB2G_IJazZ_down", "ScaleEB2G_IJazZ_up",
    "ScaleEE2G_IJazZ_down", "ScaleEE2G_IJazZ_up",
    "Smearing2G_IJazZ_down", "Smearing2G_IJazZ_up",
    "FNUF_down", "FNUF_up",
    "Material_down", "Material_up"
]

jec_list = ['jec_syst_Total_down', 'jec_syst_Total_up']
# energy_list = ['energyErrShift_down', 'energyErrShift_up']
energy_list = []

all_systs = dir_list + jec_list + energy_list

NBINS = 50
NBINS_Scale = 20
PT_MIN, PT_MAX = 0, 200
EnergyErr_MIN, EnergyErr_MAX = 0, 20
RATIO_MIN_jec, RATIO_MAX_jec = 0.9, 1.1
RATIO_spec_MIN, RATIO_spec_MAX = 0.994, 1.006
SS_ratio_min, SS_ratio_max = 0.9, 1.1
Scale_Min, Scale_Max = 0.985, 1.015
RATIO_MIN, RATIO_MAX = 0.5, 1.5
DIST_MIN, DIST_MAX = 0.90, 1.1
MASS_MIN, MASS_MAX = 20.0, 40.0
MASS_NBINS = 80

mass_branch = "mass"

# ROOT.gStyle.SetOptStat(0)

ROOT.gStyle.SetStatFontSize(0.030)
ROOT.gStyle.SetOptStat(111111)

def get_diphoton_mass(pt1, eta1, phi1, pt2, eta2, phi2):
    deta = eta1 - eta2
    
    # Proper Δφ wrapping
    dphi = (phi1 - phi2 + np.pi) % (2*np.pi) - np.pi
    
    m2 = 2 * pt1 * pt2 * (np.cosh(deta) - np.cos(dphi))
    
    return np.sqrt(m2)

def get_stats(hist):
    obj = hist.GetListOfFunctions().FindObject("stats")
    if not obj:
        raise RuntimeError("Stats not created")
    return ROOT.BindObject(obj, ROOT.TPaveStats)


def sort_by_event(file):
    order = np.argsort(ak.to_numpy(file.event))
    return file[order]


# ===============================
# Helper: load and align events
# ===============================
def load_common_triplet(file_nom, up_dir, down_dir):
    file_nom  = ak.from_parquet(file_nom)
    file_up   = ak.from_parquet(up_dir)
    file_down = ak.from_parquet(down_dir)

    idx_nom  = ak.to_numpy(file_nom.event)
    idx_up   = ak.to_numpy(file_up.event)
    idx_down = ak.to_numpy(file_down.event)

    # intersection: Nom ∩ Up ∩ Down 
    common_idx = np.intersect1d(idx_nom, idx_up, assume_unique=False)
    common_idx = np.intersect1d(common_idx, idx_down, assume_unique=False)

    # masks
    mask_nom  = np.isin(idx_nom,  common_idx)
    mask_up   = np.isin(idx_up,   common_idx)
    mask_down = np.isin(idx_down, common_idx)

    file_nom  = sort_by_event(file_nom[mask_nom])
    file_up   = sort_by_event(file_up[mask_up])
    file_down = sort_by_event(file_down[mask_down])

    return (
        file_nom,
        file_up,
        file_down,
    )

syst_pairs = defaultdict(dict)

for d in all_systs:
    if d.endswith("_up"):
        syst_pairs[d[:-3]]["up"] = d      # remove "_up"
    elif d.endswith("_down"):
        syst_pairs[d[:-5]]["down"] = d    # remove "_down"


# file_nominal_photon = ak.from_parquet(nominal_photon_dir)
# file_nominal_jet    = ak.from_parquet(nominal_jet_dir)


photon_branches = {
    "lead":    "pholead_pt",
    "sublead": "phosublead_pt",
}

photon_energy_branches = {
    "lead":    "pholead_energyErr",
    "sublead": "phosublead_energyErr",
}

jet_jec_branches = {
    "lead": {
        "raw": "first_jet_pt_raw",
        "nom":  "first_jet_pt",
        "up":   "first_jet_pt",
        "down": "first_jet_pt",
    },  
    "sublead": {
        "raw": "second_jet_pt_raw",
        "nom":  "second_jet_pt",
        "up":   "second_jet_pt",
        "down": "second_jet_pt",
    },
}



jec_bases = set(
    s.replace("_up", "").replace("_down", "") for s in jec_list
)

energy_bases = set(
    s.replace("_up", "").replace("_down", "") for s in energy_list
)

DO_MASS = True

for syst, pair in syst_pairs.items():
    print(f"Processing {syst}")

    is_jec = "jec_syst_Total" in syst

    is_energy = syst in energy_bases

    is_Smearing = ("Smearing" in syst)

    is_scale = "Scale" in syst

    is_mat_fnuf = ("FNUF" in syst) | ("Material" in syst)

    # breakpoint()

    # if is_jec:
    #     # -------------------------------
    #     # JEC: ONLY nominal dijet file
    #     # -------------------------------
    #     file_nom = file_nominal_jet

    # else:
        # -------------------------------
        # Photon systematics
        # -------------------------------
    nom_dir  = f"{base_dir}/NTuples_WH_2024_single_correction_check_HDNA_{syst}/WH-2024M30/nominal/diphoton/*.parquet"
    up_dir   = f"{base_dir}/NTuples_WH_2024_single_correction_check_HDNA_{syst}/WH-2024M30/{pair['up']}/diphoton/*.parquet"
    down_dir = f"{base_dir}/NTuples_WH_2024_single_correction_check_HDNA_{syst}/WH-2024M30/{pair['down']}/diphoton/*.parquet"


    file_nom, file_up, file_down = load_common_triplet(
        nom_dir,
        up_dir,
        down_dir
    )

    # --------------------------------
    # Loop over lead / sublead
    # --------------------------------
    for tag in ["lead", "sublead"]:

        if is_jec:
            pt_raw = np.asarray(getattr(file_nom, jet_jec_branches[tag]["raw"]))
            pt_nom  = np.asarray(getattr(file_nom, jet_jec_branches[tag]["nom"]))
            pt_up   = np.asarray(getattr(file_up, jet_jec_branches[tag]["up"]))
            pt_down = np.asarray(getattr(file_down, jet_jec_branches[tag]["down"]))
            obj_label = "jet"
        
        elif is_energy:
            branch = photon_energy_branches[tag]
            pt_nom  = np.asarray(getattr(file_nom, branch))
            pt_up   = np.asarray(getattr(file_up, branch))
            pt_down = np.asarray(getattr(file_down, branch))
            obj_label = "photon"

        else:
            branch = photon_branches[tag]
            pt_raw = np.asarray(getattr(file_nom, branch+"_raw"))
            pt_nom  = np.asarray(getattr(file_nom, branch))
            pt_up   = np.asarray(getattr(file_up, branch))
            pt_down = np.asarray(getattr(file_down, branch))
            obj_label = "photon"

        if DO_MASS and (not is_jec):
            mass_raw = get_diphoton_mass(
                file_nom.pholead_pt_raw,
                file_nom.pholead_eta,
                file_nom.pholead_phi,
                file_nom.phosublead_pt_raw,
                file_nom.phosublead_eta,
                file_nom.phosublead_phi
            )
            m_nom  = np.asarray(getattr(file_nom,  mass_branch))
            m_up   = np.asarray(getattr(file_up,   mass_branch))
            m_down = np.asarray(getattr(file_down, mass_branch))
    

        # ===============================
        # Histograms
        # ===============================
        if is_energy:
            PT_MIN, PT_MAX = EnergyErr_MIN, EnergyErr_MAX
        if is_energy:
            h_nom  = ROOT.TH1F(f"h_nom_{syst}_{tag}",  f"{syst}: {tag}_EnergyErr", NBINS, PT_MIN, PT_MAX)
            h_up   = ROOT.TH1F(f"h_up_{syst}_{tag}",   f"{syst}: {tag}_EnergyErr", NBINS, PT_MIN, PT_MAX)
            h_down = ROOT.TH1F(f"h_down_{syst}_{tag}", f"{syst}: {tag}_EnergyErr", NBINS, PT_MIN, PT_MAX)

        else:
            h_raw  = ROOT.TH1F(f"h_raw_{syst}_{tag}",  f"{syst}: {tag}_pT", NBINS, PT_MIN, PT_MAX)
            h_nom  = ROOT.TH1F(f"h_nom_{syst}_{tag}",  f"{syst}: {tag}_pT", NBINS, PT_MIN, PT_MAX)
            h_up   = ROOT.TH1F(f"h_up_{syst}_{tag}",   f"{syst}: {tag}_pT", NBINS, PT_MIN, PT_MAX)
            h_down = ROOT.TH1F(f"h_down_{syst}_{tag}", f"{syst}: {tag}_pT", NBINS, PT_MIN, PT_MAX)

        if DO_MASS and (not is_jec):

            h_m_raw  = ROOT.TH1F(
                f"h_m_raw_{syst}",
                f"{syst}: m_{{#gamma#gamma}}; m_{{#gamma#gamma}} [GeV]; Events",
                MASS_NBINS, MASS_MIN, MASS_MAX
            )

            h_m_nom  = ROOT.TH1F(
                f"h_m_nom_{syst}",
                f"{syst}: m_{{#gamma#gamma}}; m_{{#gamma#gamma}} [GeV]; Events",
                MASS_NBINS, MASS_MIN, MASS_MAX
            )

            h_m_up = ROOT.TH1F(
                f"h_m_up_{syst}",
                f"{syst}: m_{{#gamma#gamma}}; m_{{#gamma#gamma}} [GeV]; Events",
                MASS_NBINS, MASS_MIN, MASS_MAX
            )

            h_m_down = ROOT.TH1F(
                f"h_m_down_{syst}",
                f"{syst}: m_{{#gamma#gamma}}; m_{{#gamma#gamma}} [GeV]; Events",
                MASS_NBINS, MASS_MIN, MASS_MAX
            )
        
        if DO_MASS and (not is_jec):
            for r, n, u, d in zip(mass_raw, m_nom, m_up, m_down):
                h_m_raw.Fill(r)
                h_m_nom.Fill(n)
                h_m_up.Fill(u)
                h_m_down.Fill(d)

        for r, n, u, d in zip(pt_raw, pt_nom, pt_up, pt_down):
            h_raw.Fill(r)
            h_nom.Fill(n)
            h_up.Fill(u)
            h_down.Fill(d)


        if DO_MASS and (not is_jec):
            h_m_raw.SetLineColor(ROOT.kBlack)

            h_m_nom.SetLineColor(ROOT.kGreen)
            h_m_nom.SetLineWidth(2)

            h_m_up.SetLineColor(ROOT.kRed)
            h_m_up.SetLineWidth(2)

            h_m_down.SetLineColor(ROOT.kBlue)
            h_m_down.SetLineWidth(2)

    
        h_raw.SetLineColor(ROOT.kBlack)
        h_raw.SetLineWidth(2)

        h_nom.SetLineColor(ROOT.kGreen)
        h_nom.SetLineWidth(2)

        h_up.SetLineColor(ROOT.kRed)
        h_up.SetLineWidth(2)

        h_down.SetLineColor(ROOT.kBlue)
        h_down.SetLineWidth(2)

        # ===============================
        # Canvas with ratio pad
        # ===============================
        c = ROOT.TCanvas(f"c_{syst}_{tag}", syst, 800, 800)

        pad1 = ROOT.TPad("pad1", "", 0, 0.30, 1, 1.00)
        pad2 = ROOT.TPad("pad2", "", 0, 0.00, 1, 0.30)

        pad1.SetRightMargin(0.22)
        pad2.SetRightMargin(0.22)

        pad1.SetBottomMargin(0.02)
        pad2.SetTopMargin(0.05)
        pad2.SetBottomMargin(0.30)

        pad1.Draw()
        pad2.Draw()

        pad1.cd()

        ROOT.gStyle.SetOptStat(111111)


        h_raw.SetStats(1)
        h_nom.SetStats(1)
        h_up.SetStats(1)
        h_down.SetStats(1)

        h_raw.Draw("HIST")
        h_nom.Draw("HIST SAMES")
        h_up.Draw("HIST SAMES")
        h_down.Draw("HIST SAMES")

        pad1.Update()   # MUST come after all Draw calls


        st_raw  = get_stats(h_raw)
        st_nom  = get_stats(h_nom)
        st_up   = get_stats(h_up)
        st_down = get_stats(h_down)

        # x1, x2 = 0.78, 0.98   # narrower box
        h  = 0.14        # taller boxes

        x1, x2 = 0.80, 0.98   # ← move right
        y_top  = 0.92
        gap = 0.02
        # y_top = 0.95

        stats = [st_raw, st_nom, st_up, st_down]
        hists = [h_raw, h_nom, h_up, h_down]

        for i, (st, hst) in enumerate(zip(stats, hists)):
            y2 = y_top - i * (h + gap)
            y1 = y2 - h

            st.SetX1NDC(x1)
            st.SetX2NDC(x2)
            st.SetY1NDC(y1)
            st.SetY2NDC(y2)

            st.SetTextColor(hst.GetLineColor())
            st.SetFillStyle(0)
            st.SetBorderSize(1)


        pad1.Modified()
        pad1.Update()


        h_raw.GetYaxis().SetTitle("Entries")
        h_raw.GetXaxis().SetLabelSize(0)

        leg = ROOT.TLegend(0.78, 0.10, 0.99, 0.25)
        leg.SetBorderSize(0)
        leg.SetFillStyle(0)
        leg.SetTextSize(0.025)

        leg.AddEntry(h_raw,  "Raw", "l")
        leg.AddEntry(h_nom,  "Nominal", "l")
        leg.AddEntry(h_up,   pair['up'], "l")
        leg.AddEntry(h_down, pair['down'], "l")

        leg.Draw()

        latex = ROOT.TLatex()
        latex.SetNDC(True)          # normalized coordinates (0–1)
        latex.SetTextFont(42)
        latex.SetTextSize(0.045)

        latex.DrawLatex(0.58, 0.85, process)


        # ---- Ratio pad
        pad2.cd()


        h_ratio_nom = h_nom.Clone(f"h_ratio_nom_{syst}")
        h_ratio_nom.Divide(h_raw)
        h_ratio_nom.SetTitle("")

        h_ratio_up = h_up.Clone(f"h_ratio_up_{syst}")
        h_ratio_up.Divide(h_nom)
        h_ratio_up.SetTitle("")

        h_ratio_down = h_down.Clone(f"h_ratio_down_{syst}")
        h_ratio_down.Divide(h_nom)
        h_ratio_down.SetTitle("")

        h_ratio_up.SetStats(0)
        h_ratio_down.SetStats(0)
        h_ratio_nom.SetStats(0)

        h_ratio_nom.SetMinimum(RATIO_MIN)
        h_ratio_nom.SetMaximum(RATIO_MAX)

        h_ratio_nom.SetMarkerStyle(21)  # full square
        h_ratio_nom.SetMarkerSize(1.0)
        h_ratio_nom.SetLineColor(ROOT.kGreen)
        h_ratio_nom.SetMarkerColor(ROOT.kGreen)

        h_ratio_up.SetMarkerStyle(20)    # full circle
        h_ratio_up.SetMarkerSize(1.0)
        h_ratio_up.SetLineColor(ROOT.kRed)
        h_ratio_up.SetMarkerColor(ROOT.kRed)

        h_ratio_down.SetMarkerStyle(24)  # open circle
        h_ratio_down.SetMarkerSize(1.0)
        h_ratio_down.SetLineColor(ROOT.kBlue)
        h_ratio_down.SetMarkerColor(ROOT.kBlue)

        h_ratio_nom.GetYaxis().SetTitle("Syst / nom or nom / raw")
        h_ratio_nom.GetYaxis().SetTitleSize(0.10)
        h_ratio_nom.GetYaxis().SetLabelSize(0.08)

        h_ratio_nom.GetXaxis().SetTitle("p_{T} [GeV]")
        h_ratio_nom.GetXaxis().SetTitleSize(0.12)
        h_ratio_nom.GetXaxis().SetLabelSize(0.10)


        h_ratio_nom.Draw("P")
        h_ratio_up.Draw("P SAME")
        h_ratio_down.Draw("P SAME")

        leg_ratio = ROOT.TLegend(0.80, 0.70, 1.0, 0.90)
        leg_ratio.SetBorderSize(0)
        leg_ratio.SetFillStyle(0)
        leg_ratio.SetTextSize(0.06)

        leg_ratio.AddEntry(h_ratio_nom,  "Nom / Raw", "p")
        leg_ratio.AddEntry(h_ratio_up,   "Up / Nominal", "p")
        leg_ratio.AddEntry(h_ratio_down, "Down / Nominal", "p")
        leg_ratio.Draw()


        line = ROOT.TLine(PT_MIN, 1.0, PT_MAX, 1.0)
        line.SetLineStyle(2)
        line.Draw()

        # c.SaveAs(f"{out_dir}/photon_{tag}_pt_{syst}_up_down_vs_nominal.png")
        if is_energy:
            c.SaveAs(f"{out_dir}/{obj_label}_{tag}_EnergyErr_{syst}_up_down_vs_nominal.png")
        else:
            c.SaveAs(
            f"{out_dir}/{obj_label}_{tag}_pt_{syst}_up_down_vs_nominal.png"
        )

        if DO_MASS and (not is_jec):

            c_m = ROOT.TCanvas(f"c_mass_{syst}_{tag}", "", 800, 800)

            pad1m = ROOT.TPad("pad1m", "", 0, 0.30, 1, 1.00)
            pad2m = ROOT.TPad("pad2m", "", 0, 0.00, 1, 0.30)

            pad1m.SetRightMargin(0.22)
            pad2m.SetRightMargin(0.22)

            pad1m.SetBottomMargin(0.02)
            pad2m.SetTopMargin(0.05)
            pad2m.SetBottomMargin(0.30)

            pad1m.Draw()
            pad2m.Draw()

            pad1m.cd()

            ROOT.gStyle.SetOptStat(111111)

            max_val = max(
                h_m_raw.GetMaximum(),
                h_m_nom.GetMaximum(),
                h_m_up.GetMaximum(),
                h_m_down.GetMaximum()
            )

            ymax = 1.2 * max_val

            h_m_raw.SetMaximum(ymax)


            h_m_raw.Draw("HIST")
            h_m_nom.Draw("HIST SAMES")
            h_m_up.Draw("HIST SAMES")
            h_m_down.Draw("HIST SAMES")

            pad1m.Update()

            st_m_raw  = get_stats(h_m_raw)
            st_m_nom  = get_stats(h_m_nom)
            st_m_up   = get_stats(h_m_up)
            st_m_down = get_stats(h_m_down)

            x1, x2 = 0.80, 0.98
            y_top  = 0.92
            h      = 0.13
            gap    = 0.01

            stats = [st_m_raw, st_m_nom, st_m_up, st_m_down]
            hists = [h_m_raw, h_m_nom, h_m_up, h_m_down]

            for i, (st, hst) in enumerate(zip(stats, hists)):
                y2 = y_top - i * (h + gap)
                y1 = y2 - h

                st.SetX1NDC(x1)
                st.SetX2NDC(x2)
                st.SetY1NDC(y1)
                st.SetY2NDC(y2)

                st.SetTextColor(hst.GetLineColor())
                st.SetFillStyle(0)
                st.SetBorderSize(1)

            pad1m.Modified()
            pad1m.Update()

            h_m_raw.GetXaxis().SetLabelSize(0)

            leg_m = ROOT.TLegend(0.77, 0.10, 1.0, 0.25)
            leg_m.SetBorderSize(0)
            leg_m.SetFillStyle(0)
            leg_m.SetTextSize(0.020)

            leg_m.AddEntry(h_m_nom,  "Nominal", "l")
            leg_m.AddEntry(h_m_raw,  "Raw", "l")
            leg_m.AddEntry(h_m_up,   f"{syst} Up", "l")
            leg_m.AddEntry(h_m_down, f"{syst} Down", "l")

            leg_m.Draw()

            latex_m = ROOT.TLatex()
            latex_m.SetNDC(True)          # normalized coordinates (0–1)
            latex_m.SetTextFont(42)
            latex_m.SetTextSize(0.045)

            latex_m.DrawLatex(0.58, 0.85, process)

            pad2m.cd()

            # pad2m.SetLogy()

            h_m_ratio_nom = h_m_nom.Clone(f"h_m_ratio_nom_{syst}")
            h_m_ratio_nom.Divide(h_m_raw)
            h_m_ratio_nom.SetTitle("")
            h_m_ratio_up   = h_m_up.Clone(f"h_m_ratio_up_{syst}")
            h_m_ratio_up.Divide(h_m_nom)
            h_m_ratio_up.SetTitle("")
            h_m_ratio_down = h_m_down.Clone(f"h_m_ratio_down_{syst}")
            h_m_ratio_down.Divide(h_m_nom)
            h_m_ratio_down.SetTitle("")

            h_m_ratio_nom.SetStats(0)
            h_m_ratio_up.SetStats(0)
            h_m_ratio_down.SetStats(0)

            h_m_ratio_nom.SetMinimum(RATIO_MIN)
            h_m_ratio_nom.SetMaximum(RATIO_MAX)

            h_m_ratio_nom.SetMarkerStyle(21)  # full square
            h_m_ratio_nom.SetMarkerSize(1.0)
            h_m_ratio_nom.SetLineColor(ROOT.kGreen)
            h_m_ratio_nom.SetMarkerColor(ROOT.kGreen)

            h_m_ratio_up.SetMarkerStyle(20)    # full circle
            h_m_ratio_up.SetMarkerSize(1.0)
            h_m_ratio_up.SetLineColor(ROOT.kRed)
            h_m_ratio_up.SetMarkerColor(ROOT.kRed)

            h_m_ratio_down.SetMarkerStyle(24)  # open circle
            h_m_ratio_down.SetMarkerSize(1.0)
            h_m_ratio_down.SetLineColor(ROOT.kBlue)
            h_m_ratio_down.SetMarkerColor(ROOT.kBlue)

            h_m_ratio_nom.GetYaxis().SetTitle("")
            h_m_ratio_nom.Draw("P")
            h_m_ratio_up.Draw("P SAME")
            h_m_ratio_down.Draw("P SAME")

                
            h_m_ratio_nom.GetXaxis().SetTitle(f"m_{{#gamma#gamma}} [GeV]")
            h_m_ratio_nom.GetXaxis().SetTitleSize(0.12)
            h_m_ratio_nom.GetXaxis().SetLabelSize(0.10)
            h_m_ratio_nom.GetYaxis().SetLabelSize(0.07)

            leg_m_ratio = ROOT.TLegend(0.80, 0.50, 0.98, 0.65)
            leg_m_ratio.SetBorderSize(0)
            leg_m_ratio.SetFillStyle(0)
            leg_m_ratio.SetTextSize(0.07)

            leg_m_ratio.AddEntry(h_m_ratio_nom,   "Nom / Raw", "l")
            leg_m_ratio.AddEntry(h_m_ratio_up,   "Up / Nom", "l")
            leg_m_ratio.AddEntry(h_m_ratio_down, "Down / Nom", "l")

            leg_m_ratio.Draw()

            # save
            outname = f"{out_dir}/mass_{tag}_{syst}_up_down_vs_nominal.png"
            c_m.SaveAs(outname)

            c_m.Close()
            del c_m

        # ===============================
        # Ratio distribution (Up & Down)
        # ===============================

        if is_jec:
            DIST_MIN, DIST_MAX = RATIO_MIN_jec, RATIO_MAX_jec
        elif is_mat_fnuf:
            DIST_MIN, DIST_MAX = RATIO_spec_MIN, RATIO_spec_MAX
        elif is_Smearing:
            DIST_MIN, DIST_MAX = SS_ratio_min, SS_ratio_max
        elif is_scale:
            DIST_MIN, DIST_MAX = Scale_Min, Scale_Max
        else:
            DIST_MIN, DIST_MAX = RATIO_MIN, RATIO_MAX


        if is_energy:
            h_dist_up = ROOT.TH1F(
            f"h_dist_up_{syst}_{tag}",
            f"{syst}: {tag}_EnergyErr ratio; EnergyErr_{{syst}} / EnergyErr_{{nom}}; Events",
            200, DIST_MIN, DIST_MAX
        )

            h_dist_down = ROOT.TH1F(
                f"h_dist_down_{syst}_{tag}",
                f"{syst}: {tag}_EnergyErr ratio; EnergyErr_{{syst}} / EnergyErr_{{nom}}; Events",
                200, DIST_MIN, DIST_MAX
            )
        else:
            h_dist_nom = ROOT.TH1F(
                f"h_dist_nom_{syst}_{tag}",
                f"{syst}: {tag}_pT ratio; pT_{{nom}} / pT_{{raw}} (pT_{{syst}} / pT_{{nom}}); Events",
                200, DIST_MIN, DIST_MAX
            )

            h_dist_up = ROOT.TH1F(
                f"h_dist_up_{syst}_{tag}",
                f"{syst}: {tag}_pT ratio; pT_{{syst}} / pT_{{nom}}; Events",
                200, DIST_MIN, DIST_MAX
            )

            h_dist_down = ROOT.TH1F(
                f"h_dist_down_{syst}_{tag}",
                "",
                200, DIST_MIN, DIST_MAX
            )

        for r, n, u, d in zip(pt_raw, pt_nom, pt_up, pt_down):
            if r > 0:
                h_dist_nom.Fill(n / r)
            if n > 0:
                h_dist_up.Fill(u / n)
                h_dist_down.Fill(d / n)

        max_val = max(
            h_dist_nom.GetMaximum(),
            h_dist_up.GetMaximum(),
            h_dist_down.GetMaximum()
        )

        ymax = 1.2 * max_val

        h_dist_nom.SetMaximum(ymax)

        
        h_dist_nom.SetLineColor(ROOT.kGreen)
        h_dist_nom.SetLineWidth(2)
        h_dist_nom.SetLineStyle(0)
        h_dist_up.SetLineColor(ROOT.kRed)
        h_dist_up.SetLineStyle(1)
        h_dist_down.SetLineColor(ROOT.kBlue)
        h_dist_down.SetLineStyle(2)
        h_dist_up.SetFillStyle(0)
        h_dist_up.SetLineWidth(2)
        h_dist_down.SetLineWidth(2)
        h_dist_down.SetFillStyle(0)
    

        c_dist = ROOT.TCanvas(f"c_dist_{syst}_{tag}", "", 800, 600)
        c_dist.SetRightMargin(0.22)

        ROOT.gStyle.SetOptStat(111111)

        h_dist_nom.SetStats(1)
        h_dist_up.SetStats(1)
        h_dist_down.SetStats(1)

        h_dist_nom.Draw("HIST")
        h_dist_up.Draw("HIST SAMES")
        h_dist_down.Draw("HIST SAMES")   # SAMES is important

        c_dist.Update()

        st_nom  = get_stats(h_dist_nom)
        st_up_syst   = get_stats(h_dist_up)
        st_down_syst = get_stats(h_dist_down)


        # x1, x2 = 0.78, 0.98   # narrower box
        h  = 0.14        # taller boxes

        x1, x2 = 0.80, 0.98   # ← move right
        y_top  = 0.92
        gap = 0.02
        # y_top = 0.95

        stats = [st_nom, st_up_syst, st_down_syst]
        hists = [h_dist_nom, h_dist_up, h_dist_down]

        for i, (st, hst) in enumerate(zip(stats, hists)):
            y2 = y_top - i * (h + gap)
            y1 = y2 - h

            st.SetX1NDC(x1)
            st.SetX2NDC(x2)
            st.SetY1NDC(y1)
            st.SetY2NDC(y2)

            st.SetTextColor(hst.GetLineColor())
            st.SetFillStyle(0)
            st.SetBorderSize(1)


        c_dist.Modified()
        c_dist.Update()

        leg2 = ROOT.TLegend(0.78, 0.30, 0.99, 0.44)
        leg2.SetBorderSize(0)
        leg2.SetTextSize(0.016)
        leg2.SetFillStyle(0)   
        leg2.AddEntry(h_dist_nom,  "Nominal/Raw", "l") 
        leg2.AddEntry(h_dist_up,   f"{syst} Up/Nom", "l")
        leg2.AddEntry(h_dist_down, f"{syst} Down/Nom", "l")
        leg2.Draw()

        latex_r = ROOT.TLatex()
        latex_r.SetNDC(True)          # normalized coordinates (0–1)
        latex_r.SetTextFont(42)
        latex_r.SetTextSize(0.045)

        latex_r.DrawLatex(0.58, 0.85, process)

        # c_dist.SaveAs(f"{out_dir}/{tag}_pt_ratio_distribution_{syst}.png")
        if is_energy:
            c_dist.SaveAs(
            f"{out_dir}/{obj_label}_{tag}_EnergyErr_ratio_distribution_{syst}.png"
        )
        else:
            c_dist.SaveAs(
            f"{out_dir}/{obj_label}_{tag}_pt_ratio_distribution_{syst}.png"
        )

        root_out = ROOT.TFile(f"{out_dir}/{obj_label}_{tag}_histograms_{syst}.root", "RECREATE")
        h_nom.Write()
        h_up.Write()
        h_down.Write()
        h_ratio_up.Write()
        h_ratio_down.Write()
        h_dist_up.Write()
        h_dist_down.Write()
        root_out.Close()

print("All systematics processed successfully")

