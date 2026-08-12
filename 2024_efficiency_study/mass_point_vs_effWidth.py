import ROOT
from array import array

# Data
mass = array('d', [12, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
width = array('d', [0.35, 0.37, 0.43, 0.58, 0.66, 0.78, 0.93, 1.05, 1.14, 1.25, 1.30])

# Canvas
c = ROOT.TCanvas("c", "Effective Width", 700, 600)
c.SetMargin(0.13, 0.05, 0.12, 0.05)

# Graph
gr = ROOT.TGraph(len(mass), mass, width)
gr.SetTitle(";M_{A} (GeV);Effective width (GeV)")

gr.SetMarkerStyle(20)
gr.SetMarkerSize(1.3)
gr.SetMarkerColor(ROOT.kBlue + 1)

gr.SetLineColor(ROOT.kBlue + 1)
gr.SetLineWidth(3)

gr.GetXaxis().CenterTitle()
gr.GetYaxis().CenterTitle()
gr.GetXaxis().SetTitleSize(0.05)
gr.GetYaxis().SetTitleSize(0.05)
gr.GetXaxis().SetLabelSize(0.045)
gr.GetYaxis().SetLabelSize(0.045)

gr.Draw("ALP")   # A: axes, L: line, P: markers

# Optional CMS-style text
latex = ROOT.TLatex()
latex.SetNDC()
latex.SetTextFont(42)
latex.SetTextSize(0.04)
latex.DrawLatex(0.15, 0.92, "#bf{CMS} #it{Simulation}")

c.SaveAs("effective_width_vs_mass.pdf")
c.SaveAs("effective_width_vs_mass.png")
