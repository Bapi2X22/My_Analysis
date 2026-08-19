# import uproot

# class NanoWriter:

#     def __init__(self, filename):
#         self.filename = filename

#     def write(self, store):

#         branches = {}

#         # Event-level branches
#         branches.update(store.scalars)
#         branches.update(store.weights)

#         # Object collections
#         for name, collection in store.collections.items():
#             branches[name] = collection

#         with uproot.recreate(self.filename, compression=uproot.ZSTD(9)) as fout:

#             tree = fout.mktree(
#                 "Events",
#                 {
#                     name: array.type
#                     for name, array in branches.items()
#                 },
#             )

#             tree.extend(branches)

#         print(f"Wrote {self.filename}")





# import uproot
# import numpy as np

# MAX_EVENTS = 500000

# class NanoWriter:

#     def __init__(self, filename):
#         self.filename = filename

#     def write(self, store):

#         n_events = len(store.scalars["event"])

#         branches = {}

#         # Event-level branches
#         branches.update(store.scalars)
#         branches.update(store.weights)

#         # Object collections
#         for name, collection in store.collections.items():
#             branches[name] = collection

#         with uproot.recreate(
#             self.filename,
#             compression=uproot.ZSTD(9),
#         ) as fout:

#             #
#             # Events tree
#             #
#             events = fout.mktree(
#                 "Events",
#                 {
#                     name: array.type
#                     for name, array in branches.items()
#                 },
#             )

#             events.extend(branches)

#             #
#             # Metadata tree (one entry)
#             #
#             if store.metadata:

#                 metadata = {}

#                 for key, value in store.metadata.items():

#                     metadata[key] = np.array([value])

#                 meta = fout.mktree(
#                     "Metadata",
#                     {
#                         key: value.dtype
#                         for key, value in metadata.items()
#                     },
#                 )

#                 meta.extend(metadata)

#         print(f"Wrote {self.filename}")



import os

import awkward as ak
import numpy as np
import uproot

# from core.config import MAX_EVENTS_PER_FILE


# class NanoWriter:

#     def __init__(self, filename):
#         self.filename = filename

#     def write(self, store):

#         # Original (unskimmed) genWeight
#         genWeight_original = store.temp["genWeight_original"]

#         # Original indices of surviving events
#         original_index = ak.to_numpy(store.scalars["__original_index__"])

#         n_skim = len(original_index)

#         #
#         # Determine skimmed split points
#         #
#         if n_skim <= MAX_EVENTS_PER_FILE:
#             split_points = []
#         else:
#             split_points = np.arange(
#                 MAX_EVENTS_PER_FILE,
#                 n_skim,
#                 MAX_EVENTS_PER_FILE,
#             )

#         starts = np.concatenate((np.array([0]), split_points))
#         stops = np.concatenate((split_points, np.array([n_skim])))

#         #
#         # Original NanoAOD boundaries
#         #
#         original_start = 0

#         for i, (start, stop) in enumerate(zip(starts, stops)):

#             #
#             # Last file
#             #
#             if stop == n_skim:
#                 original_stop = len(genWeight_original)
#             else:
#                 # Original event corresponding to last skimmed event
#                 original_stop = original_index[stop - 1] + 1

#             #
#             # Slice branches
#             #
#             branches = {}

#             for name, array in store.scalars.items():

#                 if name == "__original_index__":
#                     continue

#                 branches[name] = array[start:stop]

#             for name, array in store.weights.items():
#                 branches[name] = array[start:stop]

#             for name, array in store.collections.items():
#                 branches[name] = array[start:stop]

#             metadata = {
#                 "sum_genw_presel": np.array([
#                     float(
#                         ak.sum(
#                             genWeight_original[
#                                 original_start:original_stop
#                             ]
#                         )
#                     )
#                 ]),

#                 "n_events_presel": np.array([
#                     original_stop - original_start
#                 ]),
#             }

#             # Add cutflow only when event selection was performed
#             if "cutflow_lepton" in store.temp:

#                 cutflow_lepton = store.temp["cutflow_lepton"]
#                 cutflow_lepton_photon = store.temp["cutflow_lepton_photon"]
#                 cutflow_final = store.temp["cutflow_final"]

#                 metadata.update({
#                     "cutflow_lepton": np.array([
#                         int(ak.sum(
#                             cutflow_lepton[
#                                 original_start:original_stop
#                             ]
#                         ))
#                     ]),

#                     "cutflow_lepton_photon": np.array([
#                         int(ak.sum(
#                             cutflow_lepton_photon[
#                                 original_start:original_stop
#                             ]
#                         ))
#                     ]),

#                     "cutflow_final": np.array([
#                         int(ak.sum(
#                             cutflow_final[
#                                 original_start:original_stop
#                             ]
#                         ))
#                     ]),
#                 })

#             #
#             # Output filename
#             #
#             if len(starts) == 1:
#                 outfile = self.filename
#             else:
#                 base, ext = os.path.splitext(self.filename)
#                 outfile = f"{base}_{i:03d}{ext}"

#             #
#             # Write ROOT file
#             #
#             with uproot.recreate(
#                 outfile,
#                 compression=uproot.ZSTD(9),
#             ) as fout:

#                 events = fout.mktree(
#                     "Events",
#                     {
#                         name: array.type
#                         for name, array in branches.items()
#                     },
#                 )

#                 events.extend(branches)

#                 meta = fout.mktree(
#                     "Metadata",
#                     {
#                         key: value.dtype
#                         for key, value in metadata.items()
#                     },
#                 )

#                 meta.extend(metadata)

#             print(
#                 f"Wrote {outfile}"
#                 f" | skimmed events = {stop-start}"
#                 f" | original events = {original_stop-original_start}"
#             )

#             #
#             # Next chunk starts here
#             #
#             original_start = original_stop



class NanoWriter:

    def __init__(self, filename, config, data_kind="mc"):
        self.filename = filename
        self.data_kind = data_kind
        self.config = config

    def write(self, store):

        MAX_EVENTS_PER_FILE = self.config.MAX_EVENTS_PER_FILE

        n_events_original = store.temp["n_events_original"]

        original_index = ak.to_numpy(
            store.scalars["__original_index__"]
        )

        n_skim = len(original_index)

        if self.data_kind == "mc":
            genWeight_original = store.temp[
                "genWeight_original"
            ]

        # Determine skimmed split points

        if n_skim <= MAX_EVENTS_PER_FILE:

            split_points = []

        else:

            split_points = np.arange(
                MAX_EVENTS_PER_FILE,
                n_skim,
                MAX_EVENTS_PER_FILE,
            )

        starts = np.concatenate(
            (
                np.array([0]),
                split_points,
            )
        )

        stops = np.concatenate(
            (
                split_points,
                np.array([n_skim]),
            )
        )

        # Original NanoAOD boundary

        original_start = 0

        for i, (start, stop) in enumerate(
            zip(starts, stops)
        ):
            # Determine original event range

            if stop == n_skim:

                # Last output file include all remaining original events
                original_stop = n_events_original

            else:

                # Find the original event corresponding to the last skimmed event in this file.
                # +1 because Python slicing is [start:stop)

                original_stop = (
                    original_index[stop - 1] + 1
                )

            branches = {}

            for name, array in store.scalars.items():

                if name == "__original_index__":
                    continue

                branches[name] = array[start:stop]

            for name, array in store.weights.items():

                branches[name] = array[start:stop]

            for name, collection in store.collections.items():

                branches[name] = collection[start:stop]

            metadata = {
                "n_events_presel": np.array([
                    original_stop - original_start
                ])
            }

            if self.data_kind == "mc":

                metadata["sum_genw_presel"] = np.array([
                    float(
                        ak.sum(
                            genWeight_original[
                                original_start:original_stop
                            ]
                        )
                    )
                ])

            # Cutflow metadata

            if "cutflow_lepton" in store.temp:

                cutflow_lepton = store.temp[
                    "cutflow_lepton"
                ]

                cutflow_lepton_photon = store.temp[
                    "cutflow_lepton_photon"
                ]

                cutflow_final = store.temp[
                    "cutflow_final"
                ]

                metadata.update({

                    "cutflow_lepton": np.array([
                        int(
                            ak.sum(
                                cutflow_lepton[
                                    original_start:original_stop
                                ]
                            )
                        )
                    ]),

                    "cutflow_lepton_photon": np.array([
                        int(
                            ak.sum(
                                cutflow_lepton_photon[
                                    original_start:original_stop
                                ]
                            )
                        )
                    ]),

                    "cutflow_final": np.array([
                        int(
                            ak.sum(
                                cutflow_final[
                                    original_start:original_stop
                                ]
                            )
                        )
                    ]),
                })

            if len(starts) == 1:

                outfile = self.filename

            else:

                base, ext = os.path.splitext(
                    self.filename
                )

                outfile = f"{base}_{i:03d}{ext}"

            with uproot.recreate(
                outfile,
                compression=uproot.ZSTD(9),
            ) as fout:

                events = fout.mktree(
                    "Events",
                    {
                        name: array.type
                        for name, array in branches.items()
                    },
                )

                events.extend(branches)

                meta = fout.mktree(
                    "Metadata",
                    {
                        key: value.dtype
                        for key, value in metadata.items()
                    },
                )

                meta.extend(metadata)

            print(
                f"Wrote {outfile}"
                f" | skimmed events = {stop - start}"
                f" | original events = "
                f"{original_stop - original_start}"
            )

            # Next output file starts at the original event

            original_start = original_stop