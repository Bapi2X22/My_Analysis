import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import ROOT
import importlib.util
import argparse

parser = argparse.ArgumentParser(description="Compute MC b-tagging efficiency from NanoAOD ROOT files.")

parser.add_argument("--json", default = "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/configs/sample_BKG_2024.json", help="JSON file containing the ROOT file lists.")
parser.add_argument("--sample",default = "TTG1Jets_24SummerRun3",required=True, help="Sample name in the JSON file.")
parser.add_argument("--output", required=True, help="Output correctionlib JSON file.")
parser.add_argument("--limit", type=int, default=None, help="Maximum number of ROOT files to process. Default: all files.")
parser.add_argument("--algo", nargs="+", default=["btagUParTAK4B"], help="B-tag discriminator branch name(s).")
parser.add_argument("--wp",  nargs="+", default=["L", "M"], help="Working point name.")
parser.add_argument("--threshold", nargs="+", type=float, default=[0.0246, 0.1272], help="B-tag discriminator threshold.")

args = parser.parse_args()

path = "BTV/lib/btag_sf.py"

spec = importlib.util.spec_from_file_location("btag_sf", path)

btag_sf = importlib.util.module_from_spec(spec)

spec.loader.exec_module(btag_sf)

compute_btagging_efficiency = btag_sf.compute_btagging_efficiency_multifile

# working_points = {args.wp: args.threshold}

if len(args.wp) != len(args.threshold):
    parser.error("--wp and --threshold must have the same number of values.")

working_points = dict(zip(args.wp, args.threshold))

cs = compute_btagging_efficiency(args.json, args.sample, args.algo, working_points, args.output, limit=args.limit, skipbadfiles=True)

print("Done.")