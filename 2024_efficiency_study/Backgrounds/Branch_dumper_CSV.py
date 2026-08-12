import ROOT
import csv
import argparse

# -------------------------
# Command-line arguments
# -------------------------
parser = argparse.ArgumentParser(
    description="Dump ROOT branch sizes to text and CSV reports."
)
parser.add_argument(
    "input_file",
    help="Input ROOT file (local path or xrootd URL)"
)
parser.add_argument(
    "-o", "--output-prefix",
    default="branch_storage",
    help="Prefix for output files (default: branch_storage)"
)

args = parser.parse_args()

input_file = args.input_file
prefix = args.output_prefix

# -------------------------
# Open ROOT file
# -------------------------
f = ROOT.TFile.Open(input_file)
if not f or f.IsZombie():
    raise RuntimeError(f"Cannot open {input_file}")

t = f.Get("Events")
if not t:
    raise RuntimeError("Could not find TTree 'Events'.")

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


def write_csv(filename, branches):
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(["Branch", "Disk (MiB)", "Raw (MiB)", "Compression"])

        for br in branches:
            writer.writerow([
                br["name"],
                f"{br['zip_mb']:.2f}",
                f"{br['raw_mb']:.2f}",
                f"{br['compression']:.2f}",
            ])

        writer.writerow([
            "TOTAL",
            f"{total_zip/1024**2:.2f}",
            f"{total_raw/1024**2:.2f}",
            f"{total_raw/total_zip:.2f}" if total_zip else "0.00",
        ])


# Sort once
by_size = sorted(branches, key=lambda x: x["zip_mb"], reverse=True)
alphabetical = sorted(branches, key=lambda x: x["name"].lower())

# Write reports
write_report(f"{prefix}_by_size.txt", by_size)
write_report(f"{prefix}_alphabetical.txt", alphabetical)

write_csv(f"{prefix}_by_size.csv", by_size)
write_csv(f"{prefix}_alphabetical.csv", alphabetical)

print("Written:")
print(f"  {prefix}_by_size.txt")
print(f"  {prefix}_alphabetical.txt")
print(f"  {prefix}_by_size.csv")
print(f"  {prefix}_alphabetical.csv")
