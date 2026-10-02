import os
import numpy as np
import ROOT
from plot_config import PROCESS_COLORS 
from config import DR_CUT
from utils import darker_color, set_stack_comparison_ymax, set_stack_ymax_log, merge_overflow_into_last_bin, make_cumulative_histograms, make_cumulative_histograms_solid, set_ymax_1p4, set_overlay_ymax_1p4,set_overlay_ymax_1p4_log,set_stack_ymax_1p4, CMS_label, draw_statboxes_horizontal
ROOT.gROOT.ProcessLine(".L /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/Overlap_removal/live_stats.C")

HIGH_CUT_PROCESSES = {
    "TTG1Jets",
    "WGtoLNuG",
    "DYGto2LG",
}

# def attach_live_stats(canvas, cut_histograms):
#     canvas.cd()
#     keep = []
#     for h in cut_histograms:
#         c = h.Clone(h.GetName())          # same name, so the C++ side can find it
#         c.SetDirectory(0)
#         c.SetLineColor(0)
#         c.SetLineWidth(0)
#         c.SetFillStyle(0)
#         c.Draw("HIST SAME")
#         keep.append(c)
#     live = ROOT.TExec("live_stats", "updateStats();")
#     live.Draw()
#     keep.append(live)
#     return keep                           # keep the return value alive


def attach_live_stats(canvas, cut_histograms=None):
    canvas.cd()
    keep = []
    for h in (cut_histograms or []):
        c = h.Clone(h.GetName())
        c.SetDirectory(0)
        c.SetLineColor(0)
        c.SetLineWidth(0)
        c.SetFillStyle(0)
        c.Draw("HIST SAME")
        keep.append(c)
    live = ROOT.TExec("live_stats", "updateStats();")
    live.Draw()
    keep.append(live)
    return keep

def make_histogram(values, weights, spec, name):
    hist = ROOT.TH1D(name, "", spec.bins, spec.range[0], spec.range[1])
    values = np.asarray(values)
    weights = np.asarray(weights)
    for value, weight in zip(values, weights):
        hist.Fill(float(value), float(weight))
        merge_overflow_into_last_bin(hist)
    if spec.normalize:
        integral = hist.Integral()
        if integral > 0:
            hist.Scale(1.0 / integral)
    hist.SetDirectory(0)
    return hist

def save_canvas(canvas, root_file, output_dir, directory, name):
    png_dir = os.path.join(output_dir, "PNG")
    pdf_dir = os.path.join(output_dir, "PNG")
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)
    canvas.SaveAs(os.path.join(png_dir, f"{name}.png"))
    canvas.SaveAs(os.path.join(pdf_dir, f"{name}.pdf"))
    root_dir = root_file.GetDirectory(directory)
    if not root_dir:
        root_dir = root_file.mkdir(directory)
    root_dir.cd()
    canvas.Write(name)
    root_file.cd()

def plot_single(result, process, plot_name, spec, root_file, output_dir, is_root_file = False):
    values = getattr(result, spec.data)
    weights = getattr(result, spec.weight)
    if is_root_file:
        hist = make_histogram(values, weights, spec, f"{process}_{plot_name}_before_sel")
    else:
        hist = make_histogram(values, weights, spec, f"{process}_{plot_name}")
    canvas = ROOT.TCanvas(f"canvas_{process}_{plot_name}", "", 800, 600)
    hist.SetTitle(process)
    hist.GetXaxis().SetTitle(spec.xlabel)
    hist.GetYaxis().SetTitle(spec.ylabel)
    set_ymax_1p4(hist)
    hist.Draw("HIST")
    if is_root_file:
        save_canvas(canvas, root_file, output_dir, "BeforeSel/Single", f"{process}_{plot_name}_before_sel")
    else:
        save_canvas(canvas, root_file, output_dir, "Single", f"{process}_{plot_name}")
    canvas.Close()


def plot_overlay(results, processes, plot_name, spec, root_file, output_dir, group_name, is_root_file = False):
    canvas = ROOT.TCanvas(f"canvas_overlay_{group_name}_{plot_name}", "", 800, 600)
    histograms = []
    for process in processes:
        result = results[process]
        values = getattr(result, spec.data)
        weights = getattr(result, spec.weight)
        color = PROCESS_COLORS.get(process, ROOT.kBlack)
        if is_root_file:
            hist = make_histogram(values, weights, spec, f"{process}_{plot_name}_overlay_{group_name}_before_sel")
        else:
            hist = make_histogram(values, weights, spec, f"{process}_{plot_name}_overlay_{group_name}")
        hist.SetLineColor(color)
        hist.SetStats(0)
        histograms.append(hist)
    if not histograms:
        canvas.Close()
        return
    set_overlay_ymax_1p4(histograms)
    histograms[0].GetXaxis().SetTitle(spec.xlabel)
    histograms[0].GetYaxis().SetTitle(spec.ylabel)
    histograms[0].Draw("HIST")
    for hist in histograms[1:]:
        hist.Draw("HIST SAME")
    # canvas.BuildLegend()
    # legend = ROOT.TLegend(0.13, 0.75, 0.30, 0.90)
    # legend.SetFillStyle(0)
    # legend.SetBorderSize(0)
    # for hist, process in zip(histograms, processes):
    #     if is_root_file:
    #         legend.AddEntry(hist, f"{process} Skimmed", "l")
    #     else:
    #         legend.AddEntry(hist, f"{process} selection", "l")
    # legend.Draw()

    legend_fill = ROOT.TH1F(f"legend_fill_{group_name}_{plot_name}","",1, 0, 1)
    legend_fill.SetDirectory(0)
    legend_fill.SetFillColor(ROOT.kGray + 1)
    legend_fill.SetFillStyle(1001)
    legend_fill.SetLineColor(ROOT.kGray + 1)
    legend_line = ROOT.TH1F(f"legend_line_{group_name}_{plot_name}","",1, 0, 1)
    legend_line.SetDirectory(0)
    legend_line.SetFillStyle(0)
    legend_line.SetLineColor(ROOT.kBlack)
    legend_line.SetLineStyle(ROOT.kSolid)
    legend_line.SetLineWidth(2)

    process_legend = ROOT.TLegend(0.13, 0.80, 0.40, 0.90)
    process_legend.SetFillStyle(0)
    process_legend.SetBorderSize(0)
    process_legend.SetTextSize(0.028)
    for hist, process in zip(histograms, processes):
        process_legend.AddEntry(hist, process, "f")
    process_legend.Draw()
    style_legend = ROOT.TLegend(0.13, 0.75, 0.40, 0.80)
    style_legend.SetFillStyle(0)
    style_legend.SetBorderSize(0)
    style_legend.SetTextSize(0.028)
    if is_root_file:
        style_legend.AddEntry(legend_fill,"Skimmed","l")
    else:
        style_legend.AddEntry(legend_fill,"Selection","l")
    style_legend.Draw()

    # CMS_label( canvas, lumi=109.0, year="2024", energy="13.6 TeV", status="Simulation Preliminary")
    if is_root_file:
        CMS_label(canvas,lumi=109.0,year="2024",energy="13.6 TeV",status="Simulation Preliminary", x=0.16)
    else:
        CMS_label(canvas,lumi=109.0,year="2024",energy="13.6 TeV",status="Simulation Preliminary")
    live_keep = attach_live_stats(canvas)
    # if is_root_file:
    #     statboxes = draw_statboxes_horizontal(histograms, processes, label_suffix1="Skimmed")
    # else:
    #     statboxes = draw_statboxes_horizontal(histograms, processes, label_suffix1 = "selection")
    statboxes = draw_statboxes_horizontal(histograms,processes,x1=0.30,x2=0.90,y1=0.80,y2=0.90, is_root_file=is_root_file)
    canvas.Modified()
    canvas.Update()
    if is_root_file:
        save_canvas(canvas, root_file, output_dir, "BeforeSel/Overlay", f"overlay_{group_name}_{plot_name}_before_sel")
    else:
        save_canvas(canvas, root_file, output_dir, "Overlay", f"overlay_{group_name}_{plot_name}")
    canvas.SetLogy()
    set_overlay_ymax_1p4_log(histograms)
    # Redraw stack
    histograms[0].GetXaxis().SetTitle(spec.xlabel)
    histograms[0].GetYaxis().SetTitle(spec.ylabel)
    histograms[0].Draw("HIST")
    for hist in histograms[1:]:
        hist.Draw("HIST SAME")
    if is_root_file:
        histograms[0].SetMinimum(1000)
    else:
        histograms[0].SetMinimum(0.1)
    # legend.Draw()
    process_legend.Draw()
    style_legend.Draw()
    live_keep2 = attach_live_stats(canvas)
    # Redraw stat boxes
    for statbox in statboxes:
        statbox.Draw()
    # CMS_label(canvas,lumi=109.0,year="2024",energy="13.6 TeV",status="Simulation Preliminary")
    if is_root_file:
        CMS_label(canvas,lumi=109.0,year="2024",energy="13.6 TeV",status="Simulation Preliminary", x=0.16)
    else:
        CMS_label(canvas,lumi=109.0,year="2024",energy="13.6 TeV",status="Simulation Preliminary")
    canvas.Modified()
    canvas.Update()
    if is_root_file:
        save_canvas(canvas,root_file,output_dir,"BeforeSel/Overlay",f"overlay_{group_name}_{plot_name}_log_before_sel")
    else:
        save_canvas(canvas,root_file,output_dir,"Overlay",f"overlay_{group_name}_{plot_name}_log")
    canvas.SetLogy(False)
    canvas.Close()


def plot_stack(results, processes, plot_name, spec, root_file, output_dir, group_name, is_root_file = False):
    canvas = ROOT.TCanvas(f"canvas_stack_{group_name}_{plot_name}", "", 800, 600)
    stack = ROOT.THStack(f"stack_{plot_name}", "")
    histograms = []
    for process in processes:
        result = results[process]
        values = getattr(result, spec.data)
        weights = getattr(result, spec.weight)
        color = PROCESS_COLORS.get(process, ROOT.kBlack)
        if is_root_file:
            hist = make_histogram(values, weights, spec, f"{process}_{plot_name}_stack_{group_name}_before_sel")
        else:
            hist = make_histogram(values, weights, spec, f"{process}_{plot_name}_stack_{group_name}")
        hist.SetFillColor(color)
        hist.SetLineColor(0)
        hist.SetLineWidth(0)
        histograms.append(hist)
        stack.Add(hist)
    set_stack_ymax_1p4(stack, histograms)
    stack.Draw("HIST")
    stack.GetXaxis().SetTitle(spec.xlabel)
    stack.GetYaxis().SetTitle(spec.ylabel)
    # canvas.BuildLegend()
    # legend = ROOT.TLegend(0.13, 0.75, 0.30, 0.90)
    # legend.SetFillStyle(0)
    # legend.SetBorderSize(0)
    # for hist, process in zip(histograms, processes):
    #     if is_root_file:
    #         legend.AddEntry(hist, f"{process} Skimmed", "f")
    #     else:
    #         legend.AddEntry(hist, f"{process} selection", "f")
    # legend.Draw()

    legend_fill = ROOT.TH1F(f"legend_fill_{group_name}_{plot_name}","",1, 0, 1)
    legend_fill.SetDirectory(0)
    legend_fill.SetFillColor(ROOT.kGray + 1)
    legend_fill.SetFillStyle(1001)
    legend_fill.SetLineColor(ROOT.kGray + 1)
    legend_line = ROOT.TH1F(f"legend_line_{group_name}_{plot_name}","",1, 0, 1)
    legend_line.SetDirectory(0)
    legend_line.SetFillStyle(0)
    legend_line.SetLineColor(ROOT.kBlack)
    legend_line.SetLineStyle(ROOT.kSolid)
    legend_line.SetLineWidth(2)
    process_legend = ROOT.TLegend(0.13, 0.80, 0.40, 0.90)
    process_legend.SetFillStyle(0)
    process_legend.SetBorderSize(0)
    process_legend.SetTextSize(0.028)
    for hist, process in zip(histograms, processes):
        process_legend.AddEntry(hist, process, "f")
    process_legend.Draw()
    style_legend = ROOT.TLegend(0.13, 0.75, 0.40, 0.80)
    style_legend.SetFillStyle(0)
    style_legend.SetBorderSize(0)
    style_legend.SetTextSize(0.028)
    if is_root_file:
        style_legend.AddEntry(legend_fill,"Skimmed","f")
    else:
        style_legend.AddEntry(legend_fill,"Selection","f")
    style_legend.Draw()

    # CMS_label( canvas, lumi=109.0, year="2024", energy="13.6 TeV", status="Simulation Preliminary")
    if is_root_file:
        CMS_label(canvas,lumi=109.0,year="2024",energy="13.6 TeV",status="Simulation Preliminary", x=0.16)
    else:
        CMS_label(canvas,lumi=109.0,year="2024",energy="13.6 TeV",status="Simulation Preliminary")
    live_keep = attach_live_stats(canvas)
    # if is_root_file:
    #     statboxes = draw_statboxes_horizontal(histograms, processes, label_suffix1="Skimmed")
    # else:
    #     statboxes = draw_statboxes_horizontal(histograms, processes, label_suffix1="selection")
    statboxes = draw_statboxes_horizontal(histograms,processes,x1=0.30,x2=0.90,y1=0.75,y2=0.90,show_selection=True, is_root_file=is_root_file)
    canvas.Modified()
    canvas.Update()
    if is_root_file:
        save_canvas(canvas, root_file, output_dir, "BeforeSel/Stack", f"stack_{group_name}_{plot_name}_linear_before_sel")
    else:
        save_canvas(canvas, root_file, output_dir, "Stack", f"stack_{group_name}_{plot_name}_linear")
    canvas.SetLogy()
    set_stack_ymax_log(stack, histograms)
    # Redraw stack
    stack.Draw("HIST")
    stack.GetXaxis().SetTitle(spec.xlabel)
    stack.GetYaxis().SetTitle(spec.ylabel)
    if is_root_file:
        stack.SetMinimum(1000)
    else:
        stack.SetMinimum(0.1)
    # legend.Draw()
    process_legend.Draw()
    style_legend.Draw()
    live_keep2 = attach_live_stats(canvas)
    # Redraw stat boxes
    for statbox in statboxes:
        statbox.Draw()
    # CMS_label(canvas,lumi=109.0,year="2024",energy="13.6 TeV",status="Simulation Preliminary")
    if is_root_file:
        CMS_label(canvas,lumi=109.0,year="2024",energy="13.6 TeV",status="Simulation Preliminary", x=0.16)
    else:
        CMS_label(canvas,lumi=109.0,year="2024",energy="13.6 TeV",status="Simulation Preliminary")
    canvas.Modified()
    canvas.Update()
    if is_root_file:
        save_canvas(canvas,root_file,output_dir,"BeforeSel/Stack",f"stack_{group_name}_{plot_name}_log_before_sel")
    else:
        save_canvas(canvas,root_file,output_dir,"Stack",f"stack_{group_name}_{plot_name}_log")
    canvas.SetLogy(False)
    canvas.Close()

def plot_stack_comparison( results, processes, plot_name, spec, high_spec, low_spec, root_file, output_dir, group_name, is_root_file = False):
    canvas = ROOT.TCanvas(f"canvas_stack_comparison_{group_name}_{plot_name}", "", 800, 600)
    stack = ROOT.THStack(f"stackcomp_{plot_name}_{group_name}", "")
    cut_histograms = []
    histograms = []
    # Normal stack
    for process in processes:
        result = results[process]
        values = getattr(result, spec.data)
        weights = getattr(result, spec.weight)
        color = PROCESS_COLORS.get(process, ROOT.kBlack)
        if is_root_file:
            hist = make_histogram( values, weights, spec, f"{process}_{plot_name}_stackcomp_{group_name}_before_sel")
        else:
            hist = make_histogram( values, weights, spec, f"{process}_{plot_name}_stackcomp_{group_name}")
        # hist.SetFillColor(color)
        # hist.SetFillColorAlpha(color, 0.55)
        hist.SetFillColor(color-9)
        hist.SetFillStyle(1001)
        hist.SetLineColor(0)
        hist.SetLineWidth(0)
        histograms.append(hist)
        stack.Add(hist)
    # Stack after additional cut
    for process in processes:
        result = results[process]
        if process in HIGH_CUT_PROCESSES:
            cut_spec = high_spec
        else:
            cut_spec = low_spec
        values = getattr(result, cut_spec.data)
        weights = getattr(result, cut_spec.weight)
        color = PROCESS_COLORS.get(process, ROOT.kBlack)
        if is_root_file:
            hist = make_histogram( values, weights, cut_spec, f"{process}_{plot_name}_cut_{group_name}_before_sel")
        else:
            hist = make_histogram( values, weights, cut_spec, f"{process}_{plot_name}_cut_{group_name}")
        hist.SetFillStyle(0)
        hist.SetLineColor(color+2)
        hist.SetLineStyle(ROOT.kDashed)
        hist.SetLineWidth(3)
        cut_histograms.append(hist)
    set_stack_ymax_1p4(stack, histograms)
    stack.Draw("HIST")
    stack.GetXaxis().SetTitle(spec.xlabel)
    stack.GetYaxis().SetTitle(spec.ylabel)
    cumulative_cut_histograms = make_cumulative_histograms(cut_histograms,processes,group_name,plot_name)
    for hist, process in zip(cumulative_cut_histograms,processes):
        hist.Draw("HIST SAME")
    # legend = ROOT.TLegend(0.13, 0.68, 0.38, 0.90)
    # legend.SetFillStyle(0)
    # legend.SetBorderSize(0)
    # # Process entries: filled
    # for hist, process in zip(histograms, processes):
    #     if is_root_file:
    #         legend.AddEntry(hist, f"{process} Skimmed", "f")
    #     else:
    #         legend.AddEntry(hist, f"{process} selection", "f")
    # # Additional cut entries
    # for hist, process in zip(cut_histograms, processes):
    #     if process in HIGH_CUT_PROCESSES:
    #         legend.AddEntry(hist,f"{process} dR > {DR_CUT}","l")
    #     else:
    #         legend.AddEntry(hist,f"{process} dR < {DR_CUT}","l")
    # legend.Draw()

    legend_fill = ROOT.TH1F(f"legend_fill_{group_name}_{plot_name}","",1, 0, 1)
    legend_fill.SetDirectory(0)
    legend_fill.SetFillColor(ROOT.kGray + 1)
    legend_fill.SetFillStyle(1001)
    legend_fill.SetLineColor(ROOT.kGray + 1)
    legend_line = ROOT.TH1F(f"legend_line_{group_name}_{plot_name}","",1, 0, 1)
    legend_line.SetDirectory(0)
    legend_line.SetFillStyle(0)
    legend_line.SetLineColor(ROOT.kBlack)
    legend_line.SetLineStyle(ROOT.kSolid)
    legend_line.SetLineWidth(2)
    process_legend = ROOT.TLegend(0.13, 0.78, 0.40, 0.90)
    process_legend.SetFillStyle(0)
    process_legend.SetBorderSize(0)
    process_legend.SetTextSize(0.028)
    for hist, process in zip(histograms, processes):
        process_legend.AddEntry(hist, process, "f")
    process_legend.Draw()
    style_legend = ROOT.TLegend(0.13, 0.68, 0.40, 0.78)
    style_legend.SetFillStyle(0)
    style_legend.SetBorderSize(0)
    style_legend.SetTextSize(0.028)
    style_legend.AddEntry(legend_line,"#Delta R cut","l")
    if is_root_file:
        style_legend.AddEntry(legend_fill,"Skimmed","f")
    else:
        style_legend.AddEntry(legend_fill,"Selection","f")
    style_legend.Draw()

    # CMS_label( canvas, lumi=109.0, year="2024", energy="13.6 TeV", status="Simulation Preliminary")
    if is_root_file:
        CMS_label(canvas,lumi=109.0,year="2024",energy="13.6 TeV",status="Simulation Preliminary", x=0.16)
    else:
        CMS_label(canvas,lumi=109.0,year="2024",energy="13.6 TeV",status="Simulation Preliminary")
    # if is_root_file:
    #     statboxes = draw_statboxes_horizontal( histograms, processes, x1=0.38, x2=0.88, y1 = 0.80, y2 = 0.90, show_mean=False, label_suffix1="Skimmed")
    # else:
    #     statboxes = draw_statboxes_horizontal( histograms, processes, x1=0.38, x2=0.88, y1 = 0.80, y2 = 0.90, show_mean=False, label_suffix1="selection")
    statboxes = draw_statboxes_horizontal(histograms,processes,x1=0.30,x2=0.90,y1=0.80,y2=0.90,show_mean=False,show_selection=True, is_root_file=is_root_file)
    # statboxes = draw_statboxes_horizontal( histograms, processes, x1=0.42, x2=0.88, y1 = 0.80, y2 = 0.90, show_mean=False)
    live_keep = attach_live_stats(canvas, cut_histograms)
    # cut_statboxes = draw_statboxes_horizontal(cut_histograms, processes, x1=0.42, x2=0.88, y1 = 0.70, y2 = 0.80, show_mean=False, label_suffix="after cut")
    # if is_root_file:
    #     cut_statboxes = draw_statboxes_horizontal(cut_histograms, processes, x1=0.38, x2=0.88, y1 = 0.70, y2 = 0.80, show_mean=False, label_suffix1="Skimmed", label_suffix2=True)
    # else:
    #     cut_statboxes = draw_statboxes_horizontal(cut_histograms, processes, x1=0.38, x2=0.88, y1 = 0.70, y2 = 0.80, show_mean=False, label_suffix1="selection", label_suffix2=True)
    cut_statboxes = draw_statboxes_horizontal(cut_histograms,processes,x1=0.30,x2=0.90,y1=0.70,y2=0.80,show_mean=False, show_dr = True, is_root_file=is_root_file)
    canvas.Modified()
    canvas.Update()
    if is_root_file:
        save_canvas( canvas, root_file, output_dir, "BeforeSel/StackComparison", f"stack_comparison_{group_name}_{plot_name}_linear_before_sel")
    else:
        save_canvas( canvas, root_file, output_dir, "StackComparison", f"stack_comparison_{group_name}_{plot_name}_linear")
    # Log scale
    canvas.SetLogy()
    set_stack_ymax_log(stack, histograms)
    # Redraw stack
    stack.Draw("HIST")
    stack.GetXaxis().SetTitle(spec.xlabel)
    stack.GetYaxis().SetTitle(spec.ylabel)
    if is_root_file:
        stack.SetMinimum(1000)
    else:
        stack.SetMinimum(0.1)
    # Redraw cumulative after-cut dashed stack
    for hist in cumulative_cut_histograms:
        hist.Draw("HIST SAME")
    # Redraw legend
    # legend.Draw()
    process_legend.Draw()
    style_legend.Draw()
    live_keep2 = attach_live_stats(canvas, cut_histograms)
    # Redraw stat boxes
    for statbox in statboxes:
        statbox.Draw()
    for statbox in cut_statboxes:
        statbox.Draw()
    if is_root_file:
        CMS_label(canvas,lumi=109.0,year="2024",energy="13.6 TeV",status="Simulation Preliminary", x=0.16)
    else:
        CMS_label(canvas,lumi=109.0,year="2024",energy="13.6 TeV",status="Simulation Preliminary")
    canvas.Modified()
    canvas.Update()
    if is_root_file:
        save_canvas(canvas,root_file,output_dir,"BeforeSel/StackComparison",f"stack_comparison_{group_name}_{plot_name}_log_before_sel")
    else:
        save_canvas(canvas,root_file,output_dir,"StackComparison",f"stack_comparison_{group_name}_{plot_name}_log")
    canvas.SetLogy(False)
    canvas.Close()

def plot_stack_comparison_swap(results,processes,plot_name,spec,high_spec,low_spec,root_file,output_dir,group_name,is_root_file=False):
    canvas = ROOT.TCanvas(f"canvas_stack_comparison_swap_{group_name}_{plot_name}","",800,600)
    normal_histograms = []
    cut_histograms = []
    for process in processes:
        result = results[process]
        values = getattr(result, spec.data)
        weights = getattr(result, spec.weight)
        if is_root_file:
            hist = make_histogram(values,weights,spec,f"{process}_{plot_name}_normal_{group_name}_before_sel_swap")
        else:
            hist = make_histogram(values,weights,spec,f"{process}_{plot_name}_normal_{group_name}_swap")
        color = PROCESS_COLORS.get(process, ROOT.kBlack)
        # Solid boundary
        hist.SetFillStyle(0)
        hist.SetLineColor(color)
        hist.SetLineStyle(ROOT.kSolid)
        hist.SetLineWidth(2)
        normal_histograms.append(hist)
    for process in processes:
        result = results[process]
        if process in HIGH_CUT_PROCESSES:
            cut_spec = high_spec
        else:
            cut_spec = low_spec
        values = getattr(result, cut_spec.data)
        weights = getattr(result, cut_spec.weight)
        if is_root_file:
            hist = make_histogram(values,weights,cut_spec,f"{process}_{plot_name}_cut_{group_name}_before_sel_swap")
        else:
            hist = make_histogram(values,weights,cut_spec,f"{process}_{plot_name}_cut_{group_name}_swap")
        color = PROCESS_COLORS.get(process, ROOT.kBlack)
        # Filled stack
        hist.SetFillColor(color)
        hist.SetFillStyle(1001)
        # Optional thin boundary
        hist.SetLineColor(color)
        hist.SetLineWidth(1)
        cut_histograms.append(hist)
    cumulative_normal_histograms = make_cumulative_histograms_solid(normal_histograms,processes,group_name,plot_name)
    # Make sure cumulative normal histograms are solid boundaries
    for hist, process in zip(cumulative_normal_histograms, processes):
        color = PROCESS_COLORS.get(process, ROOT.kBlack)
        hist.SetFillStyle(0)
        hist.SetLineColor(color)
        hist.SetLineStyle(ROOT.kSolid)
        hist.SetLineWidth(2)
    stack = ROOT.THStack(f"stackcomp_swap_{plot_name}_{group_name}","")
    for hist in cut_histograms:
        stack.Add(hist)
    # set_stack_ymax_1p4(stack, cut_histograms)
    # Linear
    set_stack_comparison_ymax(stack,cumulative_normal_histograms,linear=True)
    stack.Draw("HIST")
    stack.GetXaxis().SetTitle(spec.xlabel)
    stack.GetYaxis().SetTitle(spec.ylabel)
    # Draw normal cumulative boundaries
    for hist in cumulative_normal_histograms:
        hist.Draw("HIST SAME")
    # ============================================================
    # Legend
    # ============================================================
    # legend = ROOT.TLegend(0.13, 0.68, 0.40, 0.90)
    # legend.SetFillStyle(0)
    # legend.SetBorderSize(0)
    # # Cut stack entries
    # for hist, process in zip(cut_histograms, processes):
    #     if process in HIGH_CUT_PROCESSES:
    #         cut_label = f"{process} dR > {DR_CUT}"
    #     else:
    #         cut_label = f"{process} dR < {DR_CUT}"
    #     legend.AddEntry(hist, cut_label, "f")
    # # Normal cumulative entries
    # for hist, process in zip(cumulative_normal_histograms, processes):
    #     if is_root_file:
    #         label = f"{process} Skimmed"
    #     else:
    #         label = f"{process} selection"
    #     legend.AddEntry(hist, label, "l")
    # legend.Draw()

    legend_fill = ROOT.TH1F(f"legend_fill_{group_name}_{plot_name}","",1, 0, 1)
    legend_fill.SetDirectory(0)
    legend_fill.SetFillColor(ROOT.kGray + 1)
    legend_fill.SetFillStyle(1001)
    legend_fill.SetLineColor(ROOT.kGray + 1)
    legend_line = ROOT.TH1F(f"legend_line_{group_name}_{plot_name}","",1, 0, 1)
    legend_line.SetDirectory(0)
    legend_line.SetFillStyle(0)
    legend_line.SetLineColor(ROOT.kBlack)
    legend_line.SetLineStyle(ROOT.kSolid)
    legend_line.SetLineWidth(2)
    process_legend = ROOT.TLegend(0.13, 0.78, 0.40, 0.90)
    process_legend.SetFillStyle(0)
    process_legend.SetBorderSize(0)
    process_legend.SetTextSize(0.028)
    for hist, process in zip(cut_histograms, processes):
        process_legend.AddEntry(hist, process, "f")
    process_legend.Draw()
    style_legend = ROOT.TLegend(0.13, 0.68, 0.40, 0.78)
    style_legend.SetFillStyle(0)
    style_legend.SetBorderSize(0)
    style_legend.SetTextSize(0.028)
    style_legend.AddEntry(legend_fill,"#Delta R cut","f")
    if is_root_file:
        style_legend.AddEntry(legend_line,"Skimmed","l")
    else:
        style_legend.AddEntry(legend_line,"Selection","l")
    style_legend.Draw()
    if is_root_file:
        CMS_label(canvas,lumi=109.0,year="2024",energy="13.6 TeV",status="Simulation Preliminary",x=0.16)
    else:
        CMS_label(canvas,lumi=109.0,year="2024",energy="13.6 TeV",status="Simulation Preliminary")
    # if is_root_file:
    #     statboxes = draw_statboxes_horizontal(normal_histograms,processes,x1=0.38,x2=0.88,y1=0.80,y2=0.90,show_mean=False,label_suffix1="Skimmed")
    # else:
    #     statboxes = draw_statboxes_horizontal(normal_histograms,processes,x1=0.38,x2=0.88,y1=0.80,y2=0.90,show_mean=False,label_suffix1="selection")
    statboxes = draw_statboxes_horizontal(normal_histograms,processes,x1=0.30,x2=0.90,y1=0.80,y2=0.90,show_mean=False,show_selection=True, is_root_file=is_root_file)
    live_keep = attach_live_stats(canvas, cut_histograms)
    # if is_root_file:
    #     cut_statboxes = draw_statboxes_horizontal(cut_histograms,processes,x1=0.38,x2=0.88,y1=0.70,y2=0.80,show_mean=False,label_suffix1="Skimmed",label_suffix2=True)
    # else:
    #     cut_statboxes = draw_statboxes_horizontal(cut_histograms,processes,x1=0.38,x2=0.88,y1=0.70,y2=0.80,show_mean=False,label_suffix1="selection",label_suffix2=True)
    cut_statboxes = draw_statboxes_horizontal(cut_histograms,processes,x1=0.30,x2=0.90,y1=0.70,y2=0.80,show_mean=False, show_dr=True, is_root_file=is_root_file)
    canvas.Modified()
    canvas.Update()
    if is_root_file:
        save_canvas(canvas,root_file,output_dir,"BeforeSel/StackComparisonSwap",f"stack_comparison_swap_{group_name}_{plot_name}_linear_before_sel")
    else:
        save_canvas(canvas,root_file,output_dir,"StackComparisonSwap",f"stack_comparison_swap_{group_name}_{plot_name}_linear")
    canvas.SetLogy()
    # set_stack_ymax_log(stack, cut_histograms)
    # Linear
    set_stack_comparison_ymax(stack,cumulative_normal_histograms,linear=False)
    # Redraw filled CUT stack
    stack.Draw("HIST")
    stack.GetXaxis().SetTitle(spec.xlabel)
    stack.GetYaxis().SetTitle(spec.ylabel)
    if is_root_file:
        stack.SetMinimum(1000)
    else:
        stack.SetMinimum(0.1)
    # Redraw NORMAL cumulative solid boundaries
    for hist in cumulative_normal_histograms:
        hist.Draw("HIST SAME")
    # Redraw legend
    # legend.Draw()
    process_legend.Draw()
    style_legend.Draw()
    # Keep stats alive
    live_keep2 = attach_live_stats(canvas, cut_histograms)
    # Redraw stat boxes
    for statbox in statboxes:
        statbox.Draw()
    for statbox in cut_statboxes:
        statbox.Draw()
    # CMS
    if is_root_file:
        CMS_label(canvas,lumi=109.0,year="2024",energy="13.6 TeV",status="Simulation Preliminary",x=0.16)
    else:
        CMS_label(canvas,lumi=109.0,year="2024",energy="13.6 TeV",status="Simulation Preliminary")
    canvas.Modified()
    canvas.Update()
    if is_root_file:
        save_canvas(canvas,root_file,output_dir,"BeforeSel/StackComparisonSwap",f"stack_comparison_swap_{group_name}_{plot_name}_log_before_sel")
    else:
        save_canvas(canvas,root_file,output_dir,"StackComparisonSwap",f"stack_comparison_swap_{group_name}_{plot_name}_log")
    canvas.SetLogy(False)
    canvas.Close()
