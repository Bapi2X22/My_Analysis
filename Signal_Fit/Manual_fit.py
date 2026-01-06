import ROOT
from collections import OrderedDict as od


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

PDF = buildNGaussians(3, _recursive=True)

PDF.Print('v')


N_GAUSS = args.n_gauss   # <-- change n here

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
c = ROOT.TCanvas("c", f"{N_GAUSS}G Fit", 900, 700)
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

chi2 = frame.chiSquare(
    curve_name,
    data_hist_name,
    nParams
)

# # ndf = int(frame.GetNbinsX()) - nParams

roo_hist = frame.findObject(data_hist_name)

fit_min, fit_max = fit_lo, fit_hi

fit_min, fit_max = fit_lo, fit_hi

N_used = 0

for i in range(roo_hist.GetN()):
    x = roo_hist.GetX()[i]
    if x >= fit_min and x <= fit_max:
        N_used += 1

ndf = N_used - nParams

print(f"chi^2       = {chi2 * ndf}")
print(f"NDF      = {ndf}")
print(f"chi^2 / NDF = {chi2}")



#----------------------------------
# Build legend from drawn objects
#----------------------------------
legend = ROOT.TLegend(0.60, 0.60, 0.88, 0.88)
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

info = ROOT.TPaveText(0.60, 0.25, 0.88, 0.58, "NDC")
info.SetFillStyle(0)
info.SetBorderSize(0)
info.SetTextSize(0.028)

info.AddText(f"#chi^2 / NDF = {chi2:.3f}")
info.AddText(f"NDF = {ndf}")

info.AddText("")
info.AddText("Fit parameters:")

for t in params_text:
    info.AddText(t)

info.Draw()

c.Update()

c.SaveAs(f"Manual_GaussFit_{N_GAUSS}G.png")
