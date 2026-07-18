import ROOT
import os

files = [
    ("output_TTG1Jets_13p6TeV_amcatnlo_pythia8_TTG1Jets.root", 4.634),
    ("output_TTtoLNu2Q_13p6TeV_powheg_TTtoLNu2Q.root", 405.87),
    ("output_TTto2L2Nu_13p6TeV_powheg_TTto2L2Nu.root", 98.04),
    ("output_WGtoLNuG_13p6TeV_amcatnlo_pythia8_WGtoLNuG.root", 671.5),
    ("output_DYto2Mu50_13p6TeV_amcatnlo_pythia8_DYto2Mu50.root", 2124.08),
    ("output_DYto2E50_13p6TeV_amcatnlo_pythia8_DYto2E50.root", 2124.08)
]

lumi = 109.0 * 1000.0  # pb^-1

f = ROOT.TFile.Open("output_TTG1Jets_13p6TeV_amcatnlo_pythia8_TTG1Jets.root")
ws = f.Get("tagsDumper/cms_hgg_13TeV")

categories = set()

for obj in ws.allData():
    name = obj.GetName()
    if "_125_13TeV_" in name:
        categories.add(name.split("_125_13TeV_")[-1])

categories = sorted(categories)
print(categories)

f.Close()


# ---------------------------------------------------------
# Output workspace
# ---------------------------------------------------------

out_ws = ROOT.RooWorkspace("cms_hgg_13TeV")

# ---------------------------------------------------------
# Category
# ---------------------------------------------------------

# cat = "CAT1"

for cat in categories:

    # ---------------------------------------------------------
    # Pass 1: determine DY and non-DY yields
    # ---------------------------------------------------------

    yield_nonDY = 0.0
    yield_DY = 0.0

    for filename, xsec in files:

        f = ROOT.TFile.Open(filename)

        ws = f.Get("tagsDumper/cms_hgg_13TeV")

        if not ws:
            print(f"Workspace not found in {filename}")
            f.Close()
            continue

        base = os.path.basename(filename)

        process = base.replace("output_", "").replace(".root", "")
        process = process.split("_")[-1]

        dataset_name = f"{process}_125_13TeV_{cat}"

        ds = ws.data(dataset_name)

        if not ds:
            print(f"Dataset {dataset_name} not found")
            f.Close()
            continue

        scale = xsec * lumi

        sumW_original = 0.0

        for i in range(ds.numEntries()):

            ds.get(i)

            w = ds.weight() * scale

            sumW_original += w

        if "DY" in filename:
            yield_DY += sumW_original
        else:
            yield_nonDY += sumW_original

        f.Close()

    print("\n================ Yield Summary ================")
    print("non-DY =", yield_nonDY)
    print("DY     =", yield_DY)

    global_scale = (yield_nonDY + yield_DY) / yield_nonDY

    print("Global scale =", global_scale)

    # ---------------------------------------------------------
    # Pass 2: build merged dataset without DY
    # ---------------------------------------------------------

    combined_ds = None

    for filename, xsec in files:

        if "DY" in filename:
            print(f"Skipping DY sample {filename}")
            continue

        print(f"Processing {filename}")

        f = ROOT.TFile.Open(filename)

        ws = f.Get("tagsDumper/cms_hgg_13TeV")

        if not ws:
            print(f"Workspace not found in {filename}")
            f.Close()
            continue

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

        scale = xsec * lumi

        weight_name = ds.weightVar().GetName()

        vars = ds.get()

        sumW_original = 0.0
        sumW_abs = 0.0

        for i in range(ds.numEntries()):

            ds.get(i)

            w = ds.weight() * scale

            sumW_original += w
            sumW_abs += abs(w)

        norm = sumW_original / sumW_abs

        print("Original yield =", sumW_original)
        print("Abs yield      =", sumW_abs)
        print("Renorm factor  =", norm)

        scaled_ds = ROOT.RooDataSet(
            f"scaled_{dataset_name}",
            "",
            vars,
            ROOT.RooFit.WeightVar(weight_name)
        )

        for i in range(ds.numEntries()):

            row = ds.get(i)

            w = ds.weight() * scale

            scaled_ds.add(
                row,
                abs(w) * norm * global_scale
            )

        if combined_ds is None:

            combined_ds = scaled_ds.Clone(
                f"AllData_125_13TeV_{cat}"
            )

        else:

            combined_ds.append(scaled_ds)

        f.Close()

    # ---------------------------------------------------------
    # Check final normalization
    # ---------------------------------------------------------

    if combined_ds:

        print("\n================ Final Check ================")
        print("Combined yield =", combined_ds.sumEntries())
        print("Expected yield =", yield_nonDY + yield_DY)

        getattr(out_ws, "import")(combined_ds)

# ---------------------------------------------------------
# Save workspace
# ---------------------------------------------------------

out_file = ROOT.TFile(
    "output_AllData_13p6TeV_powheg_AllData_DY_norm.root",
    "RECREATE"
)

out_dir = out_file.mkdir("tagsDumper")
out_dir.cd()

out_ws.Write()

out_file.Close()

print("\nDone!")
