import ROOT
import os

files = [
    ("output_TTG1Jets_13p6TeV_amcatnlo_pythia8_TTG1Jets.root", 4.634),
    ("output_TTtoLNu2Q_13p6TeV_powheg_TTtoLNu2Q.root", 405.87),
    ("output_TTto2L2Nu_13p6TeV_powheg_TTto2L2Nu.root", 98.04),
    ("output_WGtoLNuG_13p6TeV_amcatnlo_pythia8_WGtoLNuG.root", 671.5)
]

lumi = 109.0 * 1000.0   # pb^-1

# ---------------------------------------------------------
# Output workspace
# ---------------------------------------------------------

out_ws = ROOT.RooWorkspace("cms_hgg_13TeV")

# ---------------------------------------------------------
# Loop over categories
# ---------------------------------------------------------

for cat in ["CAT1", "CAT2"]:

    print(f"\n================ {cat} ================\n")

    combined_ds = None

    # -----------------------------------------------------
    # Loop over files
    # -----------------------------------------------------

    for filename, xsec in files:

        print(f"Processing {filename}")

        f = ROOT.TFile.Open(filename)

        ws = f.Get("tagsDumper/cms_hgg_13TeV")

        # -------------------------------------------------
        # Extract process name
        # -------------------------------------------------

        base = os.path.basename(filename)

        process = base.replace("output_", "").replace(".root", "")

        process = process.split("_")[-1]

        dataset_name = f"{process}_125_13TeV_{cat}"

        print(f"Dataset = {dataset_name}")

        ds = ws.data(dataset_name)

        if not ds:
            print(f"Dataset {dataset_name} not found")
            f.Close()
            continue

        # -------------------------------------------------
        # Cross section normalization
        # -------------------------------------------------

        scale = xsec * lumi

        print(f"scale = {scale}")

        # -------------------------------------------------
        # Weight variable
        # -------------------------------------------------

        weight_name = ds.weightVar().GetName()

        vars = ds.get()

        scaled_ds = ROOT.RooDataSet(
            f"scaled_{dataset_name}",
            "",
            vars,
            ROOT.RooFit.WeightVar(weight_name)
        )

        # -------------------------------------------------
        # Fill scaled dataset
        # -------------------------------------------------

        for i in range(ds.numEntries()):

            row = ds.get(i)

            w = ds.weight()

            scaled_ds.add(row, w * scale)

        # -------------------------------------------------
        # Merge backgrounds
        # -------------------------------------------------

        if combined_ds is None:

            combined_ds = scaled_ds.Clone(
                f"AllData_125_13TeV_{cat}"
            )

        else:

            combined_ds.append(scaled_ds)

        f.Close()
    # -----------------------------------------------------
    # Import combined category dataset
    # -----------------------------------------------------

    if combined_ds:

        getattr(out_ws, "import")(combined_ds)

# ---------------------------------------------------------
# Save workspace inside tagsDumper
# ---------------------------------------------------------

out_file = ROOT.TFile(
    "output_AllData_13p6TeV_powheg_AllData.root",
    "RECREATE"
)

# Create tagsDumper directory
out_dir = out_file.mkdir("tagsDumper")

# Move into directory
out_dir.cd()

# Write workspace
out_ws.Write()

out_file.Close()

print("\nDone!")
