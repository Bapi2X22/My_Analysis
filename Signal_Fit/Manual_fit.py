import ROOT
from collections import OrderedDict as od
import scipy
import numpy as np
from array import array
import ctypes
import os


import argparse

parser = argparse.ArgumentParser(
    description="N-Gaussian RooFit signal model builder / fitter"
)

# --- Model structure ---
parser.add_argument("--n-gauss", type=int, default=3,
                    help="Number of Gaussian components in PDF")

parser.add_argument("--mh", type=float, default=60.0,
                    help="Central MH value used in dMH = MH - mh")

parser.add_argument("--mh-poly-order", type=int, default=0,
                    help="Polynomial order for dm/sigma vs MH")

# --- Fit range ---
parser.add_argument("--fit-lo", type=float, default=53.0,
                    help="Fit lower edge")

parser.add_argument("--fit-hi", type=float, default=67.0,
                    help="Fit upper edge")

# --- Workspace + dataset selection ---
parser.add_argument("--proc", type=str, default="WHM60Y2018",
                    help="Process name used in PDF name prefix")

parser.add_argument("--cat", type=str, default="DiPho_pt",
                    help="Category name used in PDF name suffix")

parser.add_argument(
    "--xbins",
    type=int,
    default=80,
    help="Number of bins for xvar (CMS_hgg_mass) in RooDataHist / frame"
)

args = parser.parse_args()

def find_free_legend_position_roohist(
    roo_hist,
    xmin=None,
    xmax=None
):
    n = roo_hist.GetN()
    xs = roo_hist.GetX()
    ys = roo_hist.GetY()

    # X range
    if xmin is None:
        xmin = min(xs[i] for i in range(n))
    if xmax is None:
        xmax = max(xs[i] for i in range(n))

    # Y range
    ymax = max(ys[i] for i in range(n))
    ymid = 0.35 * ymax      # <-- important: keep legend below peak
    xmid = 0.5 * (xmin + xmax)

    regions = {
        "TL": [],
        "TR": [],
        "BL": [],
        "BR": [],
    }

    for i in range(n):
        x, y = xs[i], ys[i]
        if y <= 0:
            continue

        if x < xmid and y < ymid:
            regions["BL"].append(y)
        elif x < xmid and y >= ymid:
            regions["TL"].append(y)
        elif x >= xmid and y < ymid:
            regions["BR"].append(y)
        else:
            regions["TR"].append(y)

    scores = {k: sum(v) if v else 0.0 for k, v in regions.items()}
    best = min(scores, key=scores.get)

    # Tuned NDC positions (rectangular canvas friendly)
    if best == "BL": return (0.18, 0.30, 0.42, 0.52)
    if best == "BR": return (0.58, 0.30, 0.82, 0.52)
    if best == "TL": return (0.18, 0.62, 0.42, 0.84)
    if best == "TR": return (0.58, 0.62, 0.82, 0.84)


def poisson_interval(x,eSumW2,level=0.68):
  print("Poisson interval: ", "x: ", x, "eSumW2: ", eSumW2)
  neff = x**2/(eSumW2**2)
  scale = abs(x)/neff
  print("neff: ", neff, "scale: ", scale)
  l = scipy.stats.gamma.interval(level, neff, scale=scale,)[0]
  u = scipy.stats.gamma.interval(level, neff+1, scale=scale,)[1]
  print("l_full: ", scipy.stats.gamma.interval(level, neff, scale=scale,))
  print("u_full: ", scipy.stats.gamma.interval(level, neff+1, scale=scale,))
  print("l: ", l, "u: ", u)
  # protect against no effective entries
  l[neff==0] = 0.
  # protect against no variance
  l[eSumW2==0.] = 0.
  u[eSumW2==0.] = np.inf
  print("After protection l: ", l, "u: ", u)
  # convert to upper and lower errors
  eLo, eHi = abs(l-x),abs(u-x)
  print("eLo: ", eLo, "eHi: ", eHi)
  return eLo, eHi


def calcChi2(x,pdf,d,errorType="Poisson",_verbose=False,fitRange=[50,70], storeErrors=False):

  k = 0. # number of non empty bins (for calc degrees of freedom)
  normFactor = d.sumEntries()
  print("normFactor =",normFactor)
  
  # Using numpy and poisson error
  bins, nPdf, nData, eDataSumW2 = [], [],[],[]
  print("Number of entries in data hist =",d.numEntries())
  for i in range(d.numEntries()):
    p = d.get(i)
    print("p: ", p.getRealValue(x.GetName()))
    x.setVal(p.getRealValue(x.GetName()))
    if( x.getVal() < fitRange[0] )|( x.getVal() > fitRange[1] ): continue
    ndata = d.weight()
    if ndata*ndata == 0: continue
    print("binVolume: ", d.binVolume())
    npdf = pdf.getVal(ROOT.RooArgSet(x))*normFactor*d.binVolume()
    eLo, eHi = ctypes.c_double(), ctypes.c_double()
    #eLo, eHi = ROOT.Double(), ROOT.Double()
    d.weightError(eLo,eHi,ROOT.RooAbsData.SumW2)
    bins.append(i)
    nPdf.append(npdf)
    nData.append(ndata)
    eDataSumW2.append(eHi) if npdf>ndata else eDataSumW2.append(eLo)
    k += 1

  # Convert to numpy array
  nPdf = np.asarray(nPdf)
  nData = np.asarray(nData)
  eDataSumW2 = np.asarray([e.value for e in eDataSumW2], dtype=float)
  print("nPdf: ", nPdf)
  print("nData: ", nData)
  print("eDataSumW2: ", eDataSumW2)
  #eDataSumW2 = np.asarray(eDataSumW2)

  if errorType == 'Poisson':
    # Change error to poisson intervals: take max interval as error
    eLo,eHi = poisson_interval(nData,eDataSumW2,level=0.68)
    eLoArr = np.asarray(eLo)
    eHiArr = np.asarray(eHi)
    #eDataPoisson = 0.5*(eHi+eLo)
    eDataPoisson = np.maximum(eHi,eLo) 
    #eDataPoisson = (nPdf>nData)*eHi + (nPdf<=nData)*eLo 
    print("Final eDataPoisson (Poisson Interval): ", eDataPoisson)
    e = eDataPoisson
    # Calculate chi2 terms
    terms = (nPdf-nData)**2/(eDataPoisson**2)
   
  # If verbose: print to screen
  if _verbose:
    for i in range(len(terms)):
      print(" --> [DEBUG] Bin %g : nPdf = %.6f, nData = %.6f, e(%s) = %.6f --> chi2 term = %.6f"%(bins[i],nPdf[i],nData[i],errorType,e[i],terms[i]))

  # Sum terms
  result = terms.sum()

  if storeErrors:
    return result, k, eLoArr, eHiArr

  return result,k





pLUT = od()
pLUT['DCB'] = od()
pLUT['DCB']['dm_p0'] = [0.1,-2.5,2.5]
pLUT['DCB']['dm_p1'] = [0.0,-0.1,0.1]
pLUT['DCB']['dm_p2'] = [0.0,-0.001,0.001]
pLUT['DCB']['sigma_p0'] = [2.,1.,20.]
pLUT['DCB']['sigma_p1'] = [0.0,-0.1,0.1]
pLUT['DCB']['sigma_p2'] = [0.0,-0.001,0.001]
pLUT['DCB']['n1_p0'] = [20.,1.00001,500]
pLUT['DCB']['n1_p1'] = [0.0,-0.1,0.1]
pLUT['DCB']['n1_p2'] = [0.0,-0.001,0.001]
pLUT['DCB']['n2_p0'] = [20.,1.00001,500]
pLUT['DCB']['n2_p1'] = [0.0,-0.1,0.1]
pLUT['DCB']['n2_p2'] = [0.0,-0.001,0.001]
pLUT['DCB']['a1_p0'] = [1.,1.,10.]
pLUT['DCB']['a1_p1'] = [0.0,-0.1,0.1]
pLUT['DCB']['a1_p2'] = [0.0,-0.001,0.001]
pLUT['DCB']['a2_p0'] = [1.,1.,20.]
pLUT['DCB']['a2_p1'] = [0.0,-0.1,0.1]
pLUT['DCB']['a2_p2'] = [0.0,-0.001,0.001]
pLUT['Gaussian_wdcb'] = od()
pLUT['Gaussian_wdcb']['dm_p0'] = [0.1,-1.5,1.5]
pLUT['Gaussian_wdcb']['dm_p1'] = [0.01,-0.01,0.01]
pLUT['Gaussian_wdcb']['dm_p2'] = [0.01,-0.01,0.01]
pLUT['Gaussian_wdcb']['sigma_p0'] = [1.5,1.0,4.]
pLUT['Gaussian_wdcb']['sigma_p1'] = [0.0,-0.1,0.1]
pLUT['Gaussian_wdcb']['sigma_p2'] = [0.0,-0.001,0.001]
pLUT['Frac'] = od()
pLUT['Frac']['p0'] = [0.25,0.01,0.99]
pLUT['Frac']['p1'] = [0.,-0.05,0.05]
pLUT['Frac']['p2'] = [0.,-0.0001,0.0001]
pLUT['Gaussian'] = od()
pLUT['Gaussian']['dm_p0'] = [0.1,-5.,5.]
pLUT['Gaussian']['dm_p1'] = [0.0,-0.01,0.01]
pLUT['Gaussian']['dm_p2'] = [0.0,-0.01,0.01]
pLUT['Gaussian']['sigma_p0'] = ['func',0.5,10.0]
pLUT['Gaussian']['sigma_p1'] = [0.0,-0.01,0.01]
pLUT['Gaussian']['sigma_p2'] = [0.0,-0.01,0.01]
pLUT['FracGaussian'] = od()
pLUT['FracGaussian']['p0'] = ['func',0.01,0.99]
pLUT['FracGaussian']['p1'] = [0.01,-0.005,0.005]
pLUT['FracGaussian']['p2'] = [0.00001,-0.00001,0.00001]  

dir = '/eos/user/b/bbapi/My_Analysis/NTuples_bbgg/dr_ele_pho/root/WHM60-RunIISummer20UL18NanoAODv2/ws_WHM60Y2018/output_WHM60Y2018_M60_13TeV_amcatnlo_pythia8_WHM60Y2018.root'

f = ROOT.TFile.Open(dir)

f.ls()

w = f.Get("tagsDumper")
w.ls()

ws = w.Get("cms_hgg_13TeV")

ws.Print('v')

dataset = ws.data("WHM60Y2018_60_13TeV_DiPho_pt")

dataset.Print("v")

dc = dataset.emptyClone()

Vars = od()
Varlists = od()
Polynomials = od()
Pdfs = od()
Coeffs = od()
Splines = od()
  
def buildNGaussians(nGaussians,_recursive=True):


  # Loop over NGaussians
  for g in range(0,nGaussians):
    # Define polynominal functions for mean and sigma (in MH)
    for f in ['dm','sigma']: 
      k = "%s_g%g"%(f,g)
      Varlists[k] = ROOT.RooArgList("%s_coeffs"%k)
      # Create coeff for polynominal of order MHPolyOrder: y = a+bx+cx^2+...
      for po in range(0,MHPolyOrder+1):
        # p0 value of sigma is function of g (creates gaussians of increasing width)
        if(f == "sigma")&(po==0): 
          Vars['%s_p%g'%(k,po)] = ROOT.RooRealVar("%s_p%g"%(k,po),"%s_p%g"%(k,po),(g+1)*1.0,pLUT['Gaussian']["%s_p%s"%(f,po)][1],pLUT['Gaussian']["%s_p%s"%(f,po)][2])
        else:
          Vars['%s_p%g'%(k,po)] = ROOT.RooRealVar("%s_p%g"%(k,po),"%s_p%g"%(k,po),pLUT['Gaussian']["%s_p%s"%(f,po)][0],pLUT['Gaussian']["%s_p%s"%(f,po)][1],pLUT['Gaussian']["%s_p%s"%(f,po)][2])
        Varlists[k].add( Vars['%s_p%g'%(k,po)] ) 
      # Define polynominal
      Polynomials[k] = ROOT.RooPolyVar(k,k,dMH,Varlists[k])
    # Mean function
    Polynomials['mean_g%g'%g] = ROOT.RooFormulaVar("mean_g%g"%g,"mean_g%g"%g,"(@0+@1)",ROOT.RooArgList(MH,Polynomials['dm_g%g'%g]))
    # Build Gaussian
    Pdfs['gaus_g%g'%g] = ROOT.RooGaussian("gaus_g%g"%g,"gaus_g%g"%g,xvar,Polynomials['mean_g%g'%g],Polynomials['sigma_g%g'%g])
    # Relative fractions: also polynomials of order MHPolyOrder (define up to n=nGaussians-1)
    if g < nGaussians-1:
      Varlists['frac_g%g'%g] = ROOT.RooArgList("frac_g%g_coeffs"%g)
      for po in range(0,MHPolyOrder+1):
        if po == 0:
          Vars['frac_g%g_p%g'%(g,po)] = ROOT.RooRealVar("frac_g%g_p%g"%(g,po),"frac_g%g_p%g"%(g,po),0.5-0.05*g,pLUT['FracGaussian']['p%g'%po][1],pLUT['FracGaussian']['p%g'%po][2])
        else:
          Vars['frac_g%g_p%g'%(g,po)] = ROOT.RooRealVar("frac_g%g_p%g"%(g,po),"frac_g%g_p%g"%(g,po),pLUT['FracGaussian']['p%g'%po][0],pLUT['FracGaussian']['p%g'%po][1],pLUT['FracGaussian']['p%g'%po][2])
        Varlists['frac_g%g'%g].add( Vars['frac_g%g_p%g'%(g,po)] )
      # Define Polynomial
      Polynomials['frac_g%g'%g] = ROOT.RooPolyVar("frac_g%g"%g,"frac_g%g"%g,dMH,Varlists['frac_g%g'%g])
      # Constrain fraction to not be above 1 or below 0
      Polynomials['frac_g%g_constrained'%g] = ROOT.RooFormulaVar('frac_g%g_constrained'%g,'frac_g%g_constrained'%g,"(@0>0)*(@0<1)*@0+ (@0>1.0)*0.9999",ROOT.RooArgList(Polynomials['frac_g%g'%g]))
      Coeffs['frac_g%g_constrained'%g] = Polynomials['frac_g%g_constrained'%g]
  # End of loop over n Gaussians
  
  # Define total PDF
  _pdfs, _coeffs = ROOT.RooArgList(), ROOT.RooArgList()
  for g in range(0,nGaussians): 
    _pdfs.add(Pdfs['gaus_g%g'%g])
    if g < nGaussians-1: _coeffs.add(Coeffs['frac_g%g_constrained'%g])
  Pdfs['final'] = ROOT.RooAddPdf("%s_%s"%(proc,cat),"%s_%s"%(proc,cat),_pdfs,_coeffs,_recursive)
  print("g0 =", Pdfs.get("gaus_g0", None))
  print("g1 =", Pdfs.get("gaus_g1", None))
  print("g2 =", Pdfs.get("gaus_g2", None))
  return Pdfs['final']

DataHists = od()

weight = ROOT.RooRealVar("weight","weight",-10000,10000)
mass = ws.var("CMS_hgg_mass")
xvar   = mass
xvar.SetTitle("CMS_Agg_mass")
xvar.setBins(args.xbins) 
# xvar.setBins(15)
sumw = dataset.sumEntries()
for i in range(0,dataset.numEntries()):
    xvar.setVal(dataset.get(i).getRealValue(xvar.GetName()))
    weight.setVal((1/sumw)*dataset.weight())
    dc.add(ROOT.RooArgSet(xvar,weight),weight.getVal())
    # Convert to RooDataHist
    DataHists = ROOT.RooDataHist("%s_hist"%dataset.GetName(),"%s_hist"%dataset.GetName(),ROOT.RooArgSet(xvar),dc)

DataHists.Print("v")

MH = ROOT.RooRealVar("MH","m_{H}", 56, 64)
MH.setUnit("GeV")
MH.setConstant(True)
# dMH         = ROOT.RooFormulaVar("dMH","dMH","@0-60.0",ROOT.RooArgList(MH))
MH_central = args.mh

dMH = ROOT.RooFormulaVar(
    "dMH","dMH",
    f"@0-{MH_central}",
    ROOT.RooArgList(MH)
)
MHPolyOrder = args.mh_poly_order
proc        = "WHM60Y2018"
cat         = "DiPho_pt"


N_GAUSS = args.n_gauss   # <-- change n here

PDF = buildNGaussians(N_GAUSS, _recursive=True)

PDF.Print('v')

frame = xvar.frame(ROOT.RooFit.Title(f"Sum of {N_GAUSS} Gaussians"))

mean = ROOT.RooRealVar("mean","mean",args.mh,args.mh-5.0,args.mh+5.0)

fit_lo = args.fit_lo
fit_hi = args.fit_hi

xvar.setRange("fit", fit_lo, fit_hi)

colors = [ROOT.kBlue, ROOT.kGreen+2, ROOT.kMagenta+1,
          ROOT.kOrange+7, ROOT.kCyan+2, ROOT.kViolet]



print(">>> Final global fit (all parameters floating)")
fit_result = PDF.fitTo(
    DataHists,
    ROOT.RooFit.Save(True),
    ROOT.RooFit.Range("fit"),
    ROOT.RooFit.SumW2Error(True),
    ROOT.RooFit.PrintLevel(-1)
)

#--------------------------------------------------------------------------------------------------------------------------------


nParams = fit_result.floatParsFinal().getSize()


fit_result.Print()

params_text = []
pars = fit_result.floatParsFinal()

for i in range(pars.getSize()):
    p = pars.at(i)
    params_text.append(
        f"{p.GetName()} = {p.getVal():.4g}#pm {p.getError():.4g}"
    )


#----------------------------------
# Draw data
#----------------------------------
DataHists.plotOn(frame)

PDF.plotOn(
    frame,
    ROOT.RooFit.LineColor(ROOT.kRed),
    ROOT.RooFit.Name("model_total"),
    ROOT.RooFit.Range("fit")     # <-- THIS
)

for i in range(N_GAUSS):
    PDF.plotOn(
        frame,
        ROOT.RooFit.Components(f"gaus_g{i}"),
        ROOT.RooFit.LineStyle(ROOT.kDashed),
        ROOT.RooFit.LineColor(colors[i % len(colors)]),
        ROOT.RooFit.Range("fit"),
        ROOT.RooFit.Name(f"gaus_g{i}_curve")
    )



#----------------------------------
# Now create canvas + draw frame
#----------------------------------
# c = ROOT.TCanvas("c", f"{N_GAUSS}G Fit", 900, 700)
c = ROOT.TCanvas("c", f"{N_GAUSS}G Fit", 1100, 700)
c.SetLeftMargin(0.12)
c.SetRightMargin(0.35)   # <-- reserve space for info box
c.SetBottomMargin(0.12)
c.SetTopMargin(0.10)

frame.Draw()

data_hist_name = None
curve_name = "model_total"

n_items = int(frame.numItems())   # <-- force integer

for i in range(n_items):
    obj = frame.getObject(int(i))   # also force int here
    name = obj.GetName()

    # first RooHist encountered = data
    if obj.InheritsFrom("RooHist") and data_hist_name is None:
        data_hist_name = name

print("Data hist:", data_hist_name)
print("Model curve:", curve_name)

nParams = fit_result.floatParsFinal().getSize()

# chi2 = frame.chiSquare(
#     curve_name,
#     data_hist_name,
#     nParams
# )

chi2_val, nUsedBins = calcChi2(
    xvar,
    PDF,
    DataHists,
    errorType="Poisson",
    _verbose=True,
    fitRange=[fit_lo, fit_hi]
)

ndof = nUsedBins - fit_result.floatParsFinal().getSize()
chi2ndf = chi2_val / ndof

print(f"Chi2 = {chi2_val:.3f}")
print(f"NDOF = {ndof}")
print(f"Chi2/NDOF = {chi2ndf:.3f}")


# # ndf = int(frame.GetNbinsX()) - nParams

roo_hist = frame.findObject(data_hist_name)

# fit_min, fit_max = fit_lo, fit_hi

# fit_min, fit_max = fit_lo, fit_hi

# N_used = 0

# for i in range(roo_hist.GetN()):
#     x = roo_hist.GetX()[i]
#     if x >= fit_min and x <= fit_max:
#         N_used += 1

# ndf = N_used - nParams

# print(f"chi^2       = {chi2 * ndf}")
# print(f"NDF      = {ndf}")
# print(f"chi^2 / NDF = {chi2}")


# legend placement
legend_coords = find_free_legend_position_roohist(
    roo_hist,
    xmin=fit_lo,
    xmax=fit_hi
)
legend = ROOT.TLegend(*legend_coords)
# legend.Draw()


legend.SetBorderSize(0)
legend.SetFillStyle(0)
legend.SetTextSize(0.030)

legend.AddEntry(frame.findObject("model_total"),
                "Total model", "l")

for i in range(N_GAUSS):
    legend.AddEntry(
        frame.findObject(f"gaus_g{i}_curve"),
        f"Gaussian {i+1}",
        "l"
    )


legend.Draw()

# info = ROOT.TPaveText(ix1, iy1, ix2, iy2, "NDC")

info = ROOT.TPaveText(0.68, 0.15, 0.98, 0.88, "NDC")
info.SetFillStyle(0)
info.SetBorderSize(0)
info.SetTextAlign(12)
info.SetTextSize(0.028)

info.AddText(f"#chi^{{2}} / NDF = {chi2ndf:.3f}")
info.AddText(f"NDF = {ndof}")
info.AddText("")
info.AddText("Fit parameters:")

for t in params_text:
    info.AddText(t)

info.Draw()

c.Update()

c.SaveAs(f"Manual_GaussFit_{N_GAUSS}G.png")


# ============================================================
# Make a ZOOMED copy (50–70 GeV) — does NOT touch the fit
# ============================================================

plot_lo = args.mh - 10
plot_hi = args.mh + 10

# Clone the RooPlot (contains data + PDFs already)
frame_zoom = frame.Clone(f"{frame.GetName()}_zoom")
frame_zoom.SetTitle(f"{frame.GetTitle()}  (Zoomed version)")


# Zoom the x-axis only
frame_zoom.GetXaxis().SetLimits(plot_lo, plot_hi)

# Create second canvas
c_zoom = ROOT.TCanvas("c_zoom", f"{N_GAUSS}G Fit (50–70 GeV)", 1100, 700)
c_zoom.SetLeftMargin(0.12)
c_zoom.SetRightMargin(0.35)
c_zoom.SetBottomMargin(0.12)
c_zoom.SetTopMargin(0.10)

frame_zoom.Draw()

# Re-draw legend (reuse same one)
legend.Draw()

# Re-draw info box
info.Draw()

c_zoom.Update()
c_zoom.SaveAs(f"Manual_GaussFit_{N_GAUSS}G_ZOOM.png")

