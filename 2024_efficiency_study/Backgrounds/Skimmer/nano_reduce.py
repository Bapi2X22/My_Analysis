import argparse

from core.reducer import NanoReducer
from core.writer import NanoWriter

parser = argparse.ArgumentParser()

parser.add_argument("--input", required=True)
parser.add_argument("--output", default="skim.root")

parser.add_argument(
    "--no-jet-selection",
    action="store_true",
    help="Disable jet selection.",
)

parser.add_argument(
    "--no-electron-selection",
    action="store_true",
    help="Disable electron selection.",
)

parser.add_argument(
    "--no-muon-selection",
    action="store_true",
    help="Disable muon selection.",
)

parser.add_argument(
    "--no-photon-selection",
    action="store_true",
    help="Disable photon selection.",
)

parser.add_argument(
    "--no-event-selection",
    action="store_true",
    help="Disable event selection.",
)

parser.add_argument(
    "--data",
    action="store_true",
    help="Process data instead of MC.",
)

parser.add_argument(
    "--apply_trigger",
    action="store_true",
    help="Apply the configured HLT trigger selection.",
)

args = parser.parse_args()

if args.data:
    from core import config_data as config
    data_kind = "data"
else:
    from core import config as config
    data_kind = "mc"

store = NanoReducer(
    args.input,
    config=config,
    jet_selection=not args.no_jet_selection,
    electron_selection=not args.no_electron_selection,
    muon_selection=not args.no_muon_selection,
    photon_selection=not args.no_photon_selection,
    event_selection=not args.no_event_selection,
    apply_trigger=args.apply_trigger
).run()

NanoWriter(args.output, config=config, data_kind=data_kind).write(store)