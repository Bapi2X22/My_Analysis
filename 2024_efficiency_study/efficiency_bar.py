import glob
import pyarrow.parquet as pq
import awkward as ak
import ROOT

mass = "30"

# -------------------------------
# Ordered cumulative cuts
# -------------------------------
cuts = [
    ("Old cut", "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/NTuples_WH_2024_official_old_cut_using_HDNA/"),
    ("+ ele veto", "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/NTuples_WH_2024_official_HDNA_ele_veto/"),
    ("+ Muon Id", "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/NTuples_WH_2024_official_HDNA_muon_id/"),
    ("+ Muon Iso", "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/NTuples_WH_2024_official_HDNA_muon_isoId/"),
    ("+ Global Muon", "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/NTuples_WH_2024_official_HDNA_globalmuon/"),
    ("+ Tight JetID", "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/NTuples_WH_2024_official_HDNA_tight_jetId/"),
    ("+ dR( Jet,#gamma)>0.4", "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/NTuples_WH_2024_official_HDNA_dr_jetpho/"),
    ("+ dR( Jet,e)>0.4", "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/NTuples_WH_2024_official_HDNA_dr_jetele/"),
    ("+ dR( Jet,#mu)>0.4", "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/NTuples_WH_2024_official_HDNA_dr_jetmu/"),
    ("+ Extra pho cut", "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/NTuples_WH_2024_official_HDNA_all_pho_cut_end/")
]

# -------------------------------
# ROOT setup
# -------------------------------
ROOT.gStyle.SetOptStat(0)

canvas = ROOT.TCanvas("c", "Cut Flow Efficiency", 1000, 600)

hist = ROOT.TH1F("h", "", len(cuts), 0, len(cuts))

eff_list = []

# -------------------------------
# Loop
# -------------------------------
for i, (label, base_dir) in enumerate(cuts):

    inside_dir = f"{base_dir}WH-2024M{mass}/nominal/diphoton/"
    files = glob.glob(f"{inside_dir}/*.parquet")

    sum_genw = 0.0

    for f in files:
        pf = pq.ParquetFile(f)
        meta = pf.schema_arrow.metadata

        if meta and b"sum_genw_presel" in meta:
            val = meta[b"sum_genw_presel"].decode()
            if val != "Data":
                sum_genw += float(val)

    if len(files) > 0:
        arr = ak.from_parquet(files, columns=["pholead_pt"])
        n_events = len(arr.pholead_pt)
    else:
        n_events = 0

    eff = (n_events / sum_genw) * 100 if sum_genw > 0 else 0

    eff_list.append(eff)

    bin_idx = i + 1
    hist.SetBinContent(bin_idx, eff)
    hist.GetXaxis().SetBinLabel(bin_idx, label)

# -------------------------------
# Style
# -------------------------------
hist.SetFillColorAlpha(ROOT.kBlue, 0.7)
hist.SetLineColor(ROOT.kBlack)

hist.GetYaxis().SetTitle("Efficiency (%)")
hist.SetTitle("")

hist.SetMaximum(1)
hist.SetMinimum(0)

# Rotate labels
hist.LabelsOption("v", "X")
hist.SetLabelSize(0.04, "X")

# -------------------------------
# Draw
# -------------------------------
hist.Draw("BAR")

latex2 = ROOT.TLatex()
latex2.SetTextSize(0.035)
latex2.SetTextAlign(22)  # center

for i in range(len(eff_list)):
    if i == 0:
        continue  # no previous cut

    prev = eff_list[i-1]
    curr = eff_list[i]

    if prev > 0:
        drop = (curr - prev) / prev * 100.0
    else:
        drop = 0

    # Format as -x%
    text = f"{drop:.1f}%"

    # Position: center of bin, slightly above bar
    x = hist.GetBinCenter(i+1)
    y = curr + 0.03  # adjust depending on your y-range

    latex2.DrawLatex(x, y, text)

canvas.SetGrid()
canvas.SetBottomMargin(0.25)

# CMS label
latex = ROOT.TLatex()
latex.SetNDC()
latex.SetTextFont(61)
latex.SetTextSize(0.06)
latex.DrawLatex(0.12, 0.92, "CMS")

latex.SetTextFont(52)
latex.SetTextSize(0.045)
latex.DrawLatex(0.23, 0.92, "Simulation Preliminary")

latex.SetTextFont(42)
latex.SetTextAlign(31)
latex.DrawLatex(0.88, 0.92, "(2024, 109 fb^{-1})")

latex.SetTextFont(42)
latex.SetTextAlign(31)
latex.DrawLatex(0.78, 0.82, "M_{A} = 30 GeV")

canvas.SaveAs("cutflow_efficiency_30GeV.png")
canvas.Draw()
