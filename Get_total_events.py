import ROOT
import os
import argparse
from tqdm import tqdm

def count_entries(directory):
    total = 0
    root_files = sorted(
        [f for f in os.listdir(directory) if f.endswith(".root")]
    )

    if not root_files:
        print(f"No ROOT files found in: {directory}")
        return

    for fname in tqdm(root_files, desc="Processing ROOT files", unit="file"):
        path = os.path.join(directory, fname)
        f = ROOT.TFile.Open(path)
        if not f or f.IsZombie():
            tqdm.write(f"[Error] Could not open: {fname}")
            continue

        t = f.Get("Events")
        if not t:
            tqdm.write(f"[Warning] No 'Events' tree in: {fname}")
            f.Close()
            continue

        n = t.GetEntries()
        total += n
        tqdm.write(f"{fname}: {n} entries")
        f.Close()

    print("\n==============================")
    print(f" Total entries across all files: {total}")
    print("==============================")

def main():
    parser = argparse.ArgumentParser(
        description="Count total entries in 'Events' trees across all ROOT files in a directory."
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Path to the directory containing ROOT files (default: current directory)."
    )
    args = parser.parse_args()
    count_entries(args.directory)

if __name__ == "__main__":
    main()

