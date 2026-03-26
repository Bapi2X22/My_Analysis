import json
import os
import ROOT 

ROOT.gROOT.SetBatch(True)  # prevents GUI memory usage

json_file = "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/jet_jerc.json"
corr_name = "Summer24Prompt24_V2_MC_L2Relative_AK4PFPuppi"

with open(json_file) as f:
    jerc = json.load(f)

# ---- Find correction ----
correction = None
for corr in jerc["corrections"]:
    if corr["name"] == corr_name:
        correction = corr["data"]
        break

if correction is None:
    raise ValueError(f"{corr_name} not found")

outdir = "/eos/user/b/bbapi/www/Systematics_study/JSON_plots/JERC/"
os.makedirs(outdir, exist_ok=True)

eta_edges = correction["edges"]
eta_contents = correction["content"]
phi_edges = eta_contents[0]["edges"]

pt_min, pt_max = 1, 1000

print("eta_edges: ", eta_edges)
# ---- Loop over eta bins ----
for i_eta, eta_bin in enumerate(eta_contents):

    eta_low  = eta_edges[i_eta]
    eta_high = eta_edges[i_eta + 1]

    phi_edges = eta_contents[i_eta]["edges"]

    print("eta_low: ", eta_low)
    print("eta_high: ", eta_high)

    print("phi_edges: ", phi_edges)

    canvas = ROOT.TCanvas(f"c_eta_{i_eta}", "", 800, 600)
    canvas.SetLogx()

    legend = ROOT.TLegend(0.55, 0.65, 0.88, 0.88)
    legend.SetBorderSize(0)

    phi_bins = eta_bin["content"]

    first = True
    funcs = []
    phi_labels = []

    ymin, ymax = 1e9, -1e9

    colors = [ROOT.kRed, ROOT.kBlue, ROOT.kGreen+2]

    for i_phi, phi_bin in enumerate(phi_bins):

        formula = phi_bin
        phi_low, phi_high = phi_edges[0+i_phi], phi_edges[1+i_phi]

        expr = formula["expression"]
        params = formula["parameters"]

        f = ROOT.TF1(f"f_eta{i_eta}_phi{i_phi}", expr, pt_min, pt_max)

        for ip, val in enumerate(params):
            f.SetParameter(ip, val)

        # f.SetLineColor(ROOT.kBlue + i_phi)
        f.SetLineColor(colors[i_phi])
        f.SetLineStyle(1 + i_phi) 
        f.SetLineWidth(2)

        # ---- Estimate y-range ----
        for pt in [1, 10, 50, 100, 500, 1000]:
            val = f.Eval(pt)
            ymin = min(ymin, val)
            ymax = max(ymax, val)

        funcs.append(f)

        phi_labels.append(f"{phi_low:.2f} <= #phi <= {phi_high:.2f}")

    # ---- Set proper margins ----
    margin = 0.02 * (ymax - ymin)
    ymin -= margin
    ymax += margin

    # ---- Draw ----
    for i, f in enumerate(funcs):

        if i == 0:
            f.SetMinimum(ymin)
            f.SetMaximum(ymax)
            f.GetXaxis().SetTitle("Jet p_{T} (GeV)")
            f.GetYaxis().SetTitle("Correction")
            f.SetTitle("")   # remove title
            f.Draw()
        else:
            f.Draw("SAME")

    # ---- Legend (smaller) ----
    legend = ROOT.TLegend(0.60, 0.70, 0.85, 0.88)
    legend.SetBorderSize(0)
    legend.SetTextSize(0.025)

    for i, f in enumerate(funcs):
        legend.AddEntry(f, phi_labels[i], "l")

    legend.Draw()

    # ---- Eta label inside canvas ----
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.035)

    latex.DrawLatex(0.15, 0.92, f"{eta_low:.2f} <= #eta <= {eta_high:.2f}")
    # latex.DrawLatex(0.15, 0.86, corr_name)

    # ---- Save ----
    fname = f"{outdir}/eta_{i_eta}_{eta_low:.2f}_{eta_high:.2f}"
    canvas.SaveAs(fname + ".png")
    canvas.SaveAs(fname + ".pdf")

    # MEMORY CLEANUP
    for f in funcs:
        f.Delete()
    del funcs

    canvas.Close()
    del canvas
    del legend
