import glob
import pyarrow.parquet as pq
import awkward as ak
import ROOT

# -------------------------------
# Inputs
# -------------------------------
base_dirs = {
    "Tight jetID + No dR cut between lep and Jet + 10 geV pho cut": "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/NTuples_WH_2024_official_tight_JetId_no_dR_jme_10GeV_pho/",
    "Tight jetID + No dR cut between lep and Jet": "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/NTuples_WH_2024_official_tight_JetId_no_dR_jmue/",
    "Tight jetID instead of TightLepveto": "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/NTuples_WH_2024_official_tight_JetId/",
    "Old cut": "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/NTuples_WH_2024_official_oldcut/",
    "Default HiggsDNA preselection": "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/NTuples_WH_2024_HDNA_presel_official_tightercut_new/"
}

Mass_points = ["12", "15", "20", "25", "30", "40", "45", "50", "55", "60"]
mass_vals = [int(m) for m in Mass_points]

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

# -------------------------------
# ROOT Style
# -------------------------------
ROOT.gStyle.SetOptStat(0)

canvas = ROOT.TCanvas("c", "Efficiency vs Mass", 800, 600)
legend = ROOT.TLegend(0.3, 0.6, 0.88, 0.88)

colors = [ROOT.kRed, ROOT.kBlue, ROOT.kGreen+2, ROOT.kBlack, ROOT.kMagenta]

graphs = []

# -------------------------------
# Loop over selections
# -------------------------------
for i, (label, base_dir) in enumerate(base_dirs.items()):

    efficiencies = []

    for mass in Mass_points:
        inside_dir = f"{base_dir}WH-2024M{mass}/nominal/diphoton/"
        files = glob.glob(f"{inside_dir}/*.parquet")

        sum_genw_beforesel = 0.0

        for f in files:
            pf = pq.ParquetFile(f)
            meta = pf.schema_arrow.metadata

            if meta and b"sum_genw_presel" in meta:
                val = meta[b"sum_genw_presel"].decode()
                if val != "Data":
                    sum_genw_beforesel += float(val)

        # Load events
        if len(files) > 0:
            array = ak.from_parquet(files, columns=["pholead_pt"])
            n_events = len(array.pholead_pt)
        else:
            n_events = 0

        eff = (n_events / sum_genw_beforesel) * 100 if sum_genw_beforesel > 0 else 0
        efficiencies.append(eff)

    # -------------------------------
    # Create TGraph
    # -------------------------------
    graph = ROOT.TGraph(len(mass_vals))

    for j, (m, e) in enumerate(zip(mass_vals, efficiencies)):
        graph.SetPoint(j, m, e)

    graph.SetLineColor(colors[i])
    graph.SetMarkerColor(colors[i])
    graph.SetMarkerStyle(20 + i)
    graph.SetLineWidth(2)

    graphs.append(graph)

    draw_opt = "APL" if i == 0 else "PL SAME"
    graph.Draw(draw_opt)

    # Axis labels only once
    if i == 0:
        graph.GetXaxis().SetTitle("Mass (GeV)")
        graph.GetYaxis().SetTitle("Efficiency (%)")
        graph.GetYaxis().SetRangeUser(0, 2)
        graph.SetTitle("")

    legend.AddEntry(graph, label, "lp")

# -------------------------------
# Final touches
# -------------------------------
legend.Draw()
canvas.SetGrid()

CMS_label(canvas)

canvas.SaveAs("efficiency_vs_mass.png")
canvas.Draw()
