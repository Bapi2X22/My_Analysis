import ROOT

input_file = "root://xrootd-cms.infn.it///store/mc/RunIII2024Summer24NanoAODv15/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/NANOAODSIM/150X_mcRun3_2024_realistic_v2-v3/2810000/889b2f80-6b56-4188-8865-2074c23099c3.root"

f = ROOT.TFile.Open(input_file)
t = f.Get("Events")

branches = []

total_zip = 0
total_raw = 0

for b in t.GetListOfBranches():
    zip_bytes = b.GetZipBytes("*")
    raw_bytes = b.GetTotBytes("*")

    total_zip += zip_bytes
    total_raw += raw_bytes

    branches.append({
        "name": b.GetName(),
        "zip_mb": zip_bytes / 1024**2,
        "raw_mb": raw_bytes / 1024**2,
        "compression": raw_bytes / zip_bytes if zip_bytes else 0,
    })


def write_report(filename, branches):
    with open(filename, "w") as out:
        out.write(
            f"{'Branch':<70}"
            f"{'Disk (MiB)':>15}"
            f"{'Raw (MiB)':>15}"
            f"{'Comp.':>12}\n"
        )
        out.write("=" * 112 + "\n")

        for br in branches:
            out.write(
                f"{br['name']:<70}"
                f"{br['zip_mb']:15.2f}"
                f"{br['raw_mb']:15.2f}"
                f"{br['compression']:12.2f}\n"
            )

        out.write("=" * 112 + "\n")
        out.write(
            f"{'TOTAL':<70}"
            f"{total_zip/1024**2:15.2f}"
            f"{total_raw/1024**2:15.2f}"
            f"{total_raw/total_zip if total_zip else 0:12.2f}\n"
        )


# Report 1: sorted by disk size
write_report(
    "branch_storage_by_size_TTto2L2Nu.txt",
    sorted(branches, key=lambda x: x["zip_mb"], reverse=True)
)

# Report 2: sorted alphabetically
write_report(
    "branch_storage_alphabetical_TTto2L2Nu.txt",
    sorted(branches, key=lambda x: x["name"].lower())
)

print("Written:")
print("  branch_storage_by_size__TTto2L2Nu.txt")
print("  branch_storage_alphabetical_TTto2L2Nu.txt")
