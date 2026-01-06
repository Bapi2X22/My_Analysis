#!/usr/bin/env python3
import subprocess
import sys
import json

def das_query(query):
    """Run dasgoclient and return output as string."""
    try:
        out = subprocess.check_output(query, shell=True, text=True)
        return out.strip()
    except subprocess.CalledProcessError:
        return ""

def get_nevents(dataset):
    """Return total number of events for a dataset."""
    query = f"dasgoclient -query='summary dataset={dataset}' -json"
    out = das_query(query)
    if not out:
        return None
    try:
        js = json.loads(out)
        return js[0]["summary"][0]["nevents"]
    except:
        return None


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 count_nevents_print_and_save.py datasets.txt")
        sys.exit(1)

    # Load datasets
    with open(sys.argv[1]) as f:
        datasets = [x.strip() for x in f if x.strip()]

    # Output-only-numbers file
    out_file = "nevents.txt"
    fout = open(out_file, "w")

    print("\n=== Dataset → Nevents ===\n")

    for ds in datasets:
        nev = get_nevents(ds)
        if nev is None:
            print(f"{ds}: ERROR")
            fout.write("ERROR\n")
        else:
            print(f"{ds}  →  {nev}")
            fout.write(str(nev) + "\n")

    fout.close()
    print(f"\nSaved numbers only to: {out_file}\n")


if __name__ == "__main__":
    main()


