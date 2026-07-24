#!/usr/bin/env python3

import ROOT
import glob
import os
import re

ROOT.gROOT.SetBatch(True)

base_dir = "/eos/user/b/bbapi/flashggNew/CMSSW_14_1_0_pre4/src/flashggFinalFit/Signal"

coverage = 0.683
nscan = 10000


def smallest_interval(pdf, x, coverage=0.683, nscan=10000):

    xmin = x.getMin()
    xmax = x.getMax()

    cdf = pdf.createCdf(ROOT.RooArgSet(x))

    xs = [xmin + (xmax - xmin) * i / (nscan - 1) for i in range(nscan)]

    cdf_vals = []
    for xx in xs:
        x.setVal(xx)
        cdf_vals.append(cdf.getVal())

    best_width = 1e30
    best_low = None
    best_high = None

    j = 0

    for i in range(nscan):

        while j < nscan and (cdf_vals[j] - cdf_vals[i]) < coverage:
            j += 1

        if j == nscan:
            break

        width = xs[j] - xs[i]

        if width < best_width:
            best_width = width
            best_low = xs[i]
            best_high = xs[j]

    return best_low, best_high, best_width / 2.


print(f"{'Mass':>5} {'Low':>10} {'High':>10} {'SigmaEff':>10}")
print("-"*42)

dirs = sorted(glob.glob(os.path.join(base_dir,
                "outdir_packaged_2024_M*_BDT_15GeV0p7_forSigEff")))

for d in dirs:

    m = re.search(r"_M(\d+)_", d)
    if not m:
        continue

    mass = int(m.group(1))

    files = glob.glob(os.path.join(d, "*.root"))
    if len(files) != 1:
        print(f"Skipping {d}")
        continue

    f = ROOT.TFile.Open(files[0])

    w = f.Get("wsig_13TeV")
    if not w:
        print(f"No workspace in {files[0]}")
        continue

    # Find signal pdf automatically
    pdf = None
    it = w.allPdfs().createIterator()

    obj = it.Next()
    while obj:
        if obj.GetName().startswith("hggpdfsmrel_"):
            pdf = obj
            break
        obj = it.Next()

    if pdf is None:
        print(f"No signal pdf in {files[0]}")
        continue

    x = w.var("CMS_hgg_mass")

    low, high, sigma = smallest_interval(pdf, x,
                                         coverage=coverage,
                                         nscan=nscan)

    print(f"{mass:5d} {low:10.4f} {high:10.4f} {sigma:10.4f}")

    f.Close()
