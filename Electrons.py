#!/usr/bin/env python3
import os
import argparse
import awkward as ak
import ROOT
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

ROOT.gStyle.SetOptStat(0)
ROOT.gROOT.SetBatch(True)


def plot_histogram(varname, values, bins, title, legend, outdir, outroot, log_scale = False):
    nbins, xmin, xmax = bins
    hname = f"h_{varname}"
    hist = ROOT.TH1F(hname, title, nbins, xmin, xmax)
    hist.GetXaxis().SetTitle(title)
    hist.GetYaxis().SetTitle("Events")

    # Flatten awkward array to 1D
    values_flat = ak.flatten(values, axis=None)
    for val in values_flat:
        hist.Fill(val)

    # Canvas
    c = ROOT.TCanvas(f"c_{varname}", "", 800, 700)
    hist.SetLineWidth(2)
    hist.SetLineColor(ROOT.kBlue + 1)
    hist.Draw("HIST")
    if log_scale:
        c.SetLogy()

    legend_box = ROOT.TLegend(0.65, 0.75, 0.88, 0.88)
    legend_box.AddEntry(hist, legend, "l")
    legend_box.Draw()

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    pdf_path = os.path.join(outdir, f"{varname}.pdf")
    png_path = os.path.join(outdir, f"{varname}.png")
    c.SaveAs(pdf_path)
    c.SaveAs(png_path)

    outroot.cd()
    hist.Write()
    c.Close()


def main():
    parser = argparse.ArgumentParser(description="Plot electron fields from Coffea NanoEvents")
    parser.add_argument("--input", required=True, help="Input ROOT file (NanoAOD)")
    parser.add_argument("--outdir", required=True, help="Output directory for plots")
    parser.add_argument("--listfile", help="Optional Python file defining plot_list[]")
    args = parser.parse_args()

    # Default list
    plot_list = [
        {"var": "pt", "bins": (100, 0, 200), "title": "Electron p_{T} [GeV]", "legend": "Electron pT"},
        {"var": "eta", "bins": (60, -3, 3), "title": "Electron #eta", "legend": "Electron eta"},
        {"var": "phi", "bins": (64, -3.2, 3.2), "title": "Electron #phi", "legend": "Electron phi"},
    ]

    if args.listfile:
        import importlib.util
        spec = importlib.util.spec_from_file_location("plotlist", args.listfile)
        plotlist_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(plotlist_module)
        plot_list = plotlist_module.plot_list

    # Load NanoEvents
    print(f"Loading {args.input}")
    factory = NanoEventsFactory.from_root(
        f"{args.input}:Events",
        schemaclass=NanoAODSchema,
    )

    events = factory.events()
    events = ak.materialize(events)

    # Extract electrons
    electrons = events.Electron
    print(f"Loaded {len(events)} events, {ak.num(electrons)} electrons/event (avg {ak.mean(ak.num(electrons)):.2f})")

    # Prepare output ROOT
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    outroot_path = os.path.join(args.outdir, "electron_plots.root")
    outroot = ROOT.TFile(outroot_path, "RECREATE")

    n_electrons = ak.num(electrons)
    print("Plotting number of electrons per event (default) ...")
    plot_histogram(
        "nElectrons",
        n_electrons,
        (10, 0, 10),
        "Number of electrons per event",
        "Electron multiplicity",
        args.outdir,
        outroot,
    )

    # Loop over plots
    for cfg in plot_list:
        varname = cfg["var"]
        print(f"Plotting {varname} ...")
        try:
            values = getattr(electrons, varname)
        except AttributeError:
            print(f"Warning: Variable '{varname}' not found in Electron branch. Skipping.")
            continue
        plot_histogram(varname, values, cfg["bins"], cfg["title"], cfg["legend"], args.outdir, outroot, log_scale=cfg.get("log_scale", False))

    outroot.Close()
    print(f"All plots saved to:")
    print(f"   → ROOT file: {outroot_path}")
    print(f"   → PDF/PNG:   {args.outdir}/")


if __name__ == "__main__":
    main()