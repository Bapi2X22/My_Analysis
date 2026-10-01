import os
import ROOT

from config import (
    BACKGROUND_DIR,
    ROOT_BACKGROUND_DIR,
    OUTPUT_DIR,
    LUMI_FB,
    PHOTON_PT_MIN,
    PHOTON_ETA_MAX,
    OTHER_PT_MIN,
    PROCESSES,
    ROOT_PROCESSES,
    PROCESS_GROUPS
)

from analysis import run_process_group, run_root_process_group
from plot_config import (
    PLOTS,
    SINGLE_PLOTS,
    OVERLAY_PLOTS,
    STACK_PLOTS,
    STACK_COMPARISONS
)

from plotting import (
    plot_single,
    plot_overlay,
    plot_stack,
    plot_stack_comparison
)

from physics import is_allowed_parent, is_allowed_parent_TT


def main():

    results = {}
    results_root = {}


    print("=" * 70)
    print("Running analysis")
    print("=" * 70)

    root_output = os.path.join(OUTPUT_DIR, "Overlap_removal.root")
    root_file = ROOT.TFile(root_output, "UPDATE")

    # for process, config in PROCESSES.items():
    #     filename = os.path.join(BACKGROUND_DIR, config["files"],)
    #     print(f"\nProcessing: {process}")
    #     print(f"Input:     {filename}")
    #     print(f"Xsec:      {config['xsec']} pb")
    #     result = run_process(filename=filename, xsec_pb=config["xsec"], lumi_fb=LUMI_FB, photon_pt_min=PHOTON_PT_MIN, photon_eta_max=PHOTON_ETA_MAX, other_pt_min=OTHER_PT_MIN, high_dr_cut=HIGH_DR_CUT)
    #     results[process] = result

    for process, config in PROCESSES.items():
        if process in {"TTto2L2Nu", "TTtoLNu2Q", "TTG1Jets"}:
            parentage_func = is_allowed_parent_TT
        else:
            parentage_func = is_allowed_parent
        print(f"\nProcessing: {process}")
        results[process] = run_process_group(files=config["files"], xsecs=config["xsec"], background_dir=BACKGROUND_DIR, lumi_fb=LUMI_FB, photon_pt_min=PHOTON_PT_MIN, photon_eta_max=PHOTON_ETA_MAX, other_pt_min=OTHER_PT_MIN, parentage_func=parentage_func)

    for process, config in ROOT_PROCESSES.items():
        if process in {"TTto2L2Nu", "TTtoLNu2Q", "TTG1Jets"}:
            parentage_func = is_allowed_parent_TT
        else:
            parentage_func = is_allowed_parent
        print(f"\nProcessing: {process}")
        results_root[process] = run_root_process_group(directories=config["directories"], n_files=config["n_files"], xsecs=config["xsec"], background_dir=ROOT_BACKGROUND_DIR, lumi_fb=LUMI_FB, photon_pt_min=PHOTON_PT_MIN, photon_eta_max=PHOTON_ETA_MAX, other_pt_min=OTHER_PT_MIN, parentage_func=parentage_func)

    print("\n" + "=" * 70)
    print("Creating Plots after selection")
    print("\n" + "=" * 70)

    print("\n" + "=" * 70)
    print("Creating single plots")
    print("=" * 70)

    for process, result in results.items():
        for plot_name in SINGLE_PLOTS:
            spec = PLOTS[plot_name]
            plot_single(result=result, process=process, plot_name=plot_name, spec=spec, root_file = root_file, output_dir=OUTPUT_DIR)

    print("\n" + "=" * 70)
    print("Creating overlay plots")
    print("=" * 70)

    for job in OVERLAY_PLOTS:
        plot_name = job["plot"]
        group = job["group"]
        processes = PROCESS_GROUPS[group]
        available_processes = [process for process in processes if process in results]
        plot_overlay(results=results, processes=available_processes, plot_name=plot_name, spec=PLOTS[plot_name], root_file = root_file, output_dir=OUTPUT_DIR, group_name = group)

    print("\n" + "=" * 70)
    print("Creating stack plots")
    print("=" * 70)

    for job in STACK_PLOTS:
        plot_name = job["plot"]
        group = job["group"]
        processes = PROCESS_GROUPS[group]
        available_processes = [process for process in processes if process in results]
        plot_stack(results=results, processes=available_processes, plot_name=plot_name, spec=PLOTS[plot_name], root_file = root_file, output_dir=OUTPUT_DIR, group_name = group)


    print("\n" + "=" * 70)
    print("Creating stack comparison plots")
    print("=" * 70)

    for job in STACK_COMPARISONS:
        plot_name = job["plot"]
        group = job["group"]
        processes = PROCESS_GROUPS[group]
        available_processes = [process for process in processes if process in results]
        plot_stack_comparison(results=results, processes=available_processes, plot_name=plot_name, spec=PLOTS[job["plot"]], high_spec=PLOTS[job["high"]], low_spec=PLOTS[job["low"]], root_file=root_file, output_dir=OUTPUT_DIR, group_name=group)


    print("\n" + "=" * 70)
    print("Creating Plots before selection")
    print("\n" + "=" * 70)
    print("Creating single plots")
    print("=" * 70)

    for process, result in results_root.items():
        for plot_name in SINGLE_PLOTS:
            spec = PLOTS[plot_name]
            plot_single(result=result, process=process, plot_name=plot_name, spec=spec, root_file = root_file, output_dir=OUTPUT_DIR, is_root_file = True)

    print("\n" + "=" * 70)
    print("Creating overlay plots")
    print("=" * 70)

    for job in OVERLAY_PLOTS:
        plot_name = job["plot"]
        group = job["group"]
        processes = PROCESS_GROUPS[group]
        available_processes = [process for process in processes if process in results_root]
        plot_overlay(results=results_root, processes=available_processes, plot_name=plot_name, spec=PLOTS[plot_name], root_file = root_file, output_dir=OUTPUT_DIR, group_name = group, is_root_file = True)

    print("\n" + "=" * 70)
    print("Creating stack plots")
    print("=" * 70)

    for job in STACK_PLOTS:
        plot_name = job["plot"]
        group = job["group"]
        processes = PROCESS_GROUPS[group]
        available_processes = [process for process in processes if process in results_root]
        plot_stack(results=results_root, processes=available_processes, plot_name=plot_name, spec=PLOTS[plot_name], root_file = root_file, output_dir=OUTPUT_DIR, group_name = group, is_root_file = True)

    print("\n" + "=" * 70)
    print("Creating stack comparison plots")
    print("=" * 70)

    for job in STACK_COMPARISONS:
        plot_name = job["plot"]
        group = job["group"]
        processes = PROCESS_GROUPS[group]
        available_processes = [process for process in processes if process in results_root]
        plot_stack_comparison(results=results_root, processes=available_processes, plot_name=plot_name, spec=PLOTS[job["plot"]], high_spec=PLOTS[job["high"]], low_spec=PLOTS[job["low"]], root_file=root_file, output_dir=OUTPUT_DIR, group_name=group, is_root_file = True)

    print("\n" + "=" * 70)
    print("Analysis completed")
    print("=" * 70)

if __name__ == "__main__":
    main()