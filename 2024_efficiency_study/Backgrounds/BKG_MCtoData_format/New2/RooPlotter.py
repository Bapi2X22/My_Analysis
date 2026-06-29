import ROOT

# -------------------------------------------------
# Open file
# -------------------------------------------------

f = ROOT.TFile.Open(
    "output_AllData_13p6TeV_powheg_AllData.root"
)

ws = f.Get("tagsDumper/cms_hgg_13TeV")

# -------------------------------------------------
# Observable
# -------------------------------------------------

mass = ws.var("CMS_hgg_mass")

# Set plotting range if desired
mass.setRange(10, 70)

# -------------------------------------------------
# Loop over categories
# -------------------------------------------------

for cat in ["CAT1", "CAT2", "CAT3"]:

    ds_name = f"AllData_125_13TeV_{cat}"

    ds = ws.data(ds_name)

    if not ds:
        print(f"{ds_name} not found")
        continue

    # -------------------------------------------------
    # Create frame
    # -------------------------------------------------

    frame = mass.frame(
        ROOT.RooFit.Title(ds_name)
    )

    # -------------------------------------------------
    # Plot dataset
    # -------------------------------------------------

    ds.plotOn(
        frame,
        ROOT.RooFit.Binning(60)
    )

    # -------------------------------------------------
    # Canvas
    # -------------------------------------------------

    c = ROOT.TCanvas(
        f"c_{cat}",
        "",
        800,
        600
    )

    # c.SetLogy()

    frame.Draw()

    c.SaveAs(f"{ds_name}_all.png")

print("Done!")