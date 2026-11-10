import numpy as np
import pandas as pd
import xgboost as xgb
import awkward as ak
import pickle
import sys
import matplotlib.pyplot as plt
# import seaborn as sns
import uproot as uproot
import operator
import ROOT
from optparse import OptionParser
from xgboost import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, auc
from copy import copy
from datetime import datetime
from sklearn import metrics
import os
import argparse
import optuna
import json

# sys.path.append('tools/')
# from xgboost2tmva import *

output_dir = "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Process_BDT/Model_output/"

os.makedirs(output_dir, exist_ok=True)

#############################################################
def exportTree(final_dataset,treeNameToExport,export_filename): 
    export_file = uproot.recreate(export_filename)
    print("\n\t  file exported  : ",export_filename)
    allBranchesToWrite={ ky : final_dataset[ky].dtype for ky in final_dataset}
    export_file.mktree( treeNameToExport ,allBranchesToWrite)
    export_file[treeNameToExport].extend(final_dataset)
    export_file.close()

#############################################################

#########################################################
def getTH1FromNumpHist(nph):
    hist = ROOT.TH1D("","",len(nph[1])-1,np.array(nph[1],dtype=float))
    for i,freq in enumerate(nph[0]):
        hist.SetBinContent(i+1,freq)
    hist.ResetStats() 	
    return hist
#########################################################
mass_value = 15.0
def loadSignal(inputVars):

  vars = inputVars
  if 'evt_wgt' not in vars:
    vars.insert(0,'evt_wgt')

  # dataset_Sig = uproot.open("merged_signal.root")['DiphotonTree'].arrays(inputVars,library='pd')
#   dataset_Sig = uproot.open(f"Merged_Inputs/M{int(mass_value)}_merged.root")['DiphotonTree'].arrays(inputVars)
  dataset_Sig = uproot.open(f"Merged_Inputs/merged_signal_allCat.root")['DiphotonTree/Resolved'].arrays(inputVars)
  df_sig = ak.to_dataframe(dataset_Sig)
  # df_sig = dataset_Sig[vars]

  return df_sig

def loadMCBackground(inputVars, inputVarsMC):
  
  vars = inputVarsMC
  if 'evt_wgt' not in vars:
    vars.insert(0,'evt_wgt')
   
  # dataset_Bkg = uproot.open('merged_bkg_withMass.root')['DiphotonTree'].arrays(inputVars,library='pd')
  dataset_Bkg = uproot.open('Merged_Inputs/merged_bkg_allCat.root')['DiphotonTree/Resolved'].arrays(inputVars)

  df_bkg = ak.to_dataframe(dataset_Bkg)
  # df_bkg = dataset_Bkg

  renames = {}
  keys = inputVarsMC
  values = inputVars
  for i,key in enumerate(keys):
        renames[key] = values[i]
  print('Renames:',renames)   
      
  df_bkg.rename(columns = renames, inplace = True) 
  
  return df_bkg  

def drawScores(predY_train,predY_test,W_train,W_test,nBins,c_min,c_max,name):

  train_Sig = predY_train[Y_train==1]
  train_Bkg = predY_train[Y_train==0]
  test_Sig = predY_test[Y_test==1]
  test_Bkg = predY_test[Y_test==0]
  
  W_trainSig_sum = W_train[Y_train==1].sum(0) 
  W_trainBkg_sum = W_train[Y_train==0].sum(0)
  W_testSig_sum = W_test[Y_test==1].sum(0)
  W_testBkg_sum = W_test[Y_test==0].sum(0)
  
  #Get histograms of the classifiers
  Histo_train_Sig = np.histogram(train_Sig,bins=int(nBins),range=(c_min,c_max),weights=W_train[Y_train==1])
  Histo_train_Bkg = np.histogram(train_Bkg,bins=int(nBins),range=(c_min,c_max),weights=W_train[Y_train==0])
  Histo_test_Sig = np.histogram(test_Sig,bins=int(nBins),range=(c_min,c_max),weights=W_test[Y_test==1]*(W_trainSig_sum/W_testSig_sum))
  Histo_test_Bkg = np.histogram(test_Bkg,bins=int(nBins),range=(c_min,c_max),weights=W_test[Y_test==0]*(W_trainBkg_sum/W_testBkg_sum))
 
  #########################################################

  filename = os.path.join(output_dir, name + '_hists.root')
  fOut=ROOT.TFile(filename,'RECREATE')
  fOut.cd()
  th1f_Histo_train_Sig = getTH1FromNumpHist(Histo_train_Sig) ;  th1f_Histo_train_Sig.SetName(name+"_train_sig_mvaHist") ; th1f_Histo_train_Sig.Write()
  th1f_Histo_train_Bkg = getTH1FromNumpHist(Histo_train_Bkg) ;  th1f_Histo_train_Bkg.SetName(name+"_train_bkg_mvaHist") ; th1f_Histo_train_Bkg.Write()  
  th1f_Histo_test_Sig  = getTH1FromNumpHist(Histo_test_Sig ) ;  th1f_Histo_test_Sig.SetName(name+"_test_sig_mvaHist")   ; th1f_Histo_test_Sig.Write() 
  th1f_Histo_test_Bkg  = getTH1FromNumpHist(Histo_test_Bkg ) ;  th1f_Histo_test_Bkg.SetName(name+"_test_bkg_mvaHist")   ; th1f_Histo_test_Bkg.Write()  
  fOut.Close()
  
  final_dataset={
            'BDT' : train_Bkg
          }
  # filename=name+'_trees_bkg_train.root'
  filename = os.path.join(output_dir, name + '_trees_bkg_train.root')
  treeNameToExport='Tree'
  exportTree(final_dataset,treeNameToExport,export_filename=filename)

  final_dataset={
            'BDT' : test_Bkg
          }
  # filename=name+'_trees_bkg_test.root'
  filename = os.path.join(output_dir, name + '_trees_bkg_test.root')
  treeNameToExport='Tree'
  exportTree(final_dataset,treeNameToExport,export_filename=filename)

  final_dataset={
            'BDT' : train_Sig
          }
  # filename=name+'_trees_sig_train.root'
  filename = os.path.join(output_dir, name + '_trees_sig_train.root')
  treeNameToExport='Tree'
  exportTree(final_dataset,treeNameToExport,export_filename=filename)

  final_dataset={
            'BDT' : test_Sig
          }
  # filename=name+'_trees_sig_test.root'
  filename = os.path.join(output_dir, name + '_trees_sig_test.root')
  treeNameToExport='Tree'
  exportTree(final_dataset,treeNameToExport,export_filename=filename)

  #########################################################
  #Lets get the min/max of the Histograms
  AllHistos= [Histo_train_Sig,Histo_train_Bkg,Histo_test_Sig,Histo_test_Bkg]
  h_max = max([histo[0].max() for histo in AllHistos])*1.2
  h_min = max([histo[0].min() for histo in AllHistos])
 
  #Get the histogram properties (binning, widths, centers)
  bin_edges = Histo_train_Sig[1]
  bin_centers = ( bin_edges[:-1] + bin_edges[1:]  ) /2.
  bin_widths = (bin_edges[1:] - bin_edges[:-1])
 
  #To make error bar plots for the data, take the Poisson uncertainty sqrt(N)
  ErrorBar_test_Sig = np.sqrt(Histo_test_Sig[0])
  ErrorBar_test_Bkg = np.sqrt(Histo_test_Bkg[0])  
  
  #Plot  
  ax1 = plt.subplot(111)
  ax1.bar(bin_centers,Histo_train_Sig[0],facecolor='blue',linewidth=0,width=bin_widths,label='Sig (Train)',alpha=0.5)
  ax1.bar(bin_centers,Histo_train_Bkg[0],facecolor='red',linewidth=0,width=bin_widths,label='Bkg (Train)',alpha=0.5)
  ax1.errorbar(bin_centers, Histo_test_Sig[0], yerr=ErrorBar_test_Sig, xerr=None, ecolor='blue',c='blue',fmt='o',label='Sig (Test)')
  ax1.errorbar(bin_centers, Histo_test_Bkg[0], yerr=ErrorBar_test_Bkg, xerr=None, ecolor='red',c='red',fmt='o',label='Bkg (Test)')
  ax1.axvspan(0.0, c_max, color='blue',alpha=0.08)
  ax1.axvspan(c_min,0.0, color='red',alpha=0.08)
  ax1.axis([c_min, c_max, h_min, h_max])
  plt.title("BDT Signal--Bkg separation")
  plt.xlabel("BDT output")
  plt.ylabel("frequency")
  legend = ax1.legend(loc='upper center', shadow=True,ncol=2)
  for alabel in legend.get_texts():
    alabel.set_fontsize('small')
  filename_png = os.path.join(output_dir, name + '.png')
  filename_pdf = os.path.join(output_dir, name + '.pdf')
  plt.savefig(filename_png,dpi=300)
  plt.savefig(filename_pdf,dpi=300)  


def drawROC(Y_train,Y_test,predY_train,predY_test,W_train,W_test,name):

  f,ax = plt.subplots(figsize=(8,8))

  fpr, tpr, threshold = metrics.roc_curve(Y_train,  predY_train, sample_weight=W_train)
  train_auc = metrics.auc(fpr, tpr)
#  ax.plot(fpr, tpr,label=f'Train:  AUC = {round(train_auc,3)}')
  ax.plot(tpr, fpr,label=f'Train:  AUC = {round(train_auc,3)}')

  fpr, tpr, threshold = metrics.roc_curve(Y_test,  predY_test, sample_weight=W_test)
  test_auc = metrics.auc(fpr, tpr)
#  ax.plot(fpr, tpr,label=f'Test:  AUC = {round(test_auc,3)}')
  ax.plot(tpr, fpr,label=f'Test:  AUC = {round(test_auc,3)}')

  plt.legend(loc='lower right')
  plt.title("ROC")
  plt.xlabel('Signal Efficiency')
  plt.ylabel('1 - Background Efficiency')
#  plt.ylabel('True Positive Rate')
#  plt.xlabel('False Positive Rate')
  filename_png = os.path.join(output_dir, name + '.png')
  filename_pdf = os.path.join(output_dir, name + '.pdf')
  plt.savefig(filename_png,dpi=300)
  plt.savefig(filename_pdf,dpi=300)  
  
  print('AUC for training = %1.3f'%(train_auc) )
  print('AUC for test     = %1.3f'%(test_auc) )  



# def drawROC_mass_point(
#     Y_train,
#     Y_test,
#     predY_train,
#     predY_test,
#     W_train,
#     W_test,
#     mass_train,
#     mass_test,
#     name,
#     model_name
# ):

#     mass_points = [12, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]

#     mass_tolerance = 1.0

#     auc_results = []

#     for mass in mass_points:

#         # ------------------------------------------------
#         # Select signal at target mass +/- 1 GeV
#         # Keep all background events
#         # ------------------------------------------------

#         mask_train = (
#             (Y_train == 0) |
#             (
#                 (Y_train == 1) &
#                 (np.abs(mass_train - mass) <= mass_tolerance)
#             )
#         )

#         mask_test = (
#             (Y_test == 0) |
#             (
#                 (Y_test == 1) &
#                 (np.abs(mass_test - mass) <= mass_tolerance)
#             )
#         )

#         # Check event counts
#         n_sig_train = np.sum(
#             (Y_train == 1) & mask_train
#         )

#         n_sig_test = np.sum(
#             (Y_test == 1) & mask_test
#         )

#         if n_sig_train == 0 or n_sig_test == 0:
#             print(f"Skipping mass {mass}: no signal in train/test")
#             continue

#         # ------------------------------------------------
#         # Apply masks
#         # ------------------------------------------------

#         y_train = Y_train[mask_train]
#         y_test = Y_test[mask_test]

#         pred_train = predY_train[mask_train]
#         pred_test = predY_test[mask_test]

#         w_train = W_train[mask_train]
#         w_test = W_test[mask_test]

#         # ------------------------------------------------
#         # ROC
#         # ------------------------------------------------

#         f, ax = plt.subplots(figsize=(8, 8))

#         fpr_train, tpr_train, _ = metrics.roc_curve(
#             y_train,
#             pred_train,
#             sample_weight=w_train
#         )

#         train_auc = metrics.auc(
#             fpr_train,
#             tpr_train
#         )

#         ax.plot(
#             tpr_train,
#             1 - fpr_train,
#             label=f"Train: AUC = {train_auc:.3f}"
#         )

#         fpr_test, tpr_test, _ = metrics.roc_curve(
#             y_test,
#             pred_test,
#             sample_weight=w_test
#         )

#         test_auc = metrics.auc(
#             fpr_test,
#             tpr_test
#         )

#         ax.plot(
#             tpr_test,
#             1 - fpr_test,
#             label=f"Test: AUC = {test_auc:.3f}"
#         )

#         # ------------------------------------------------
#         # Formatting
#         # ------------------------------------------------

#         ax.plot(
#             [0, 1],
#             [1, 0],
#             linestyle="--",
#             color="gray"
#         )

#         ax.legend(loc="lower left")

#         ax.set_title(
#             f"ROC - Mass = {mass} GeV"
#         )

#         ax.set_xlabel("Signal Efficiency")
#         ax.set_ylabel("Background Rejection")

#         ax.set_xlim(0, 1)
#         ax.set_ylim(0, 1)

#         plt.tight_layout()

#         # ------------------------------------------------
#         # Save
#         # ------------------------------------------------

#         # Create model-specific ROC directory
#         roc_dir = os.path.join(output_dir, f"{model_name}_ROC")
#         os.makedirs(roc_dir, exist_ok=True)

#         # Save ROC plots
#         filename = f"ROC_M{mass}"

#         plt.savefig(
#             os.path.join(roc_dir, filename + ".png"),
#             dpi=300
#         )

#         plt.savefig(
#             os.path.join(roc_dir, filename + ".pdf"),
#             dpi=300
#         )
#         plt.close()

#         # ------------------------------------------------
#         # Print results
#         # ------------------------------------------------

#         print(
#             f"Mass = {mass:2d} GeV | "
#             f"Signal Train = {n_sig_train:7d} | "
#             f"Signal Test = {n_sig_test:7d} | "
#             f"Train AUC = {train_auc:.3f} | "
#             f"Test AUC = {test_auc:.3f}"
#         )

#         auc_results.append({
#             "Mass": mass,
#             "Signal_Train": n_sig_train,
#             "Signal_Test": n_sig_test,
#             "Train_AUC": train_auc,
#             "Test_AUC": test_auc
#         })

#     # ----------------------------------------------------
#     # Save AUC summary
#     # ----------------------------------------------------

#     auc_df = pd.DataFrame(auc_results)

#     auc_df.to_csv(
#         os.path.join(roc_dir, "AUC_summary.csv"),
#         index=False
#     )

#     return auc_df


def drawROC_mass_point(
    Y_train,
    Y_test,
    predY_train,
    predY_test,
    W_train,
    W_test,
    name,
    model_name,
    mass_train=None,
    mass_test=None,
    single_mass=None
):

    mass_points = [12, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    mass_tolerance = 1.0

    auc_results = []

    # ----------------------------------------------------
    # Create model-specific ROC directory
    # ----------------------------------------------------

    roc_dir = os.path.join(output_dir, f"{model_name}_ROC")
    os.makedirs(roc_dir, exist_ok=True)

    # ----------------------------------------------------
    # Determine whether mass-wise selection is required
    # ----------------------------------------------------

    if mass_train is None or mass_test is None:

        if single_mass is None:
            raise ValueError(
                "Provide single_mass when mass_train/mass_test "
                "are not available."
            )

        selected_mass_points = [single_mass]
        use_mass_selection = False

    else:

        selected_mass_points = mass_points
        use_mass_selection = True

    # ----------------------------------------------------
    # Loop over mass points
    # ----------------------------------------------------

    for mass in selected_mass_points:

        if use_mass_selection:

            # Select signal at target mass +/- tolerance
            # Keep all background events

            mask_train = (
                (Y_train == 0) |
                (
                    (Y_train == 1) &
                    (np.abs(mass_train - mass) <= mass_tolerance)
                )
            )

            mask_test = (
                (Y_test == 0) |
                (
                    (Y_test == 1) &
                    (np.abs(mass_test - mass) <= mass_tolerance)
                )
            )

        else:

            # Single mass sample: use all events

            mask_train = np.ones(len(Y_train), dtype=bool)
            mask_test = np.ones(len(Y_test), dtype=bool)

        # ------------------------------------------------
        # Check event counts
        # ------------------------------------------------

        n_sig_train = np.sum((Y_train == 1) & mask_train)
        n_sig_test = np.sum((Y_test == 1) & mask_test)

        n_bkg_train = np.sum((Y_train == 0) & mask_train)
        n_bkg_test = np.sum((Y_test == 0) & mask_test)

        if min(n_sig_train, n_sig_test, n_bkg_train, n_bkg_test) == 0:
            print(f"Skipping mass {mass}: missing signal/background")
            continue

        # ------------------------------------------------
        # Apply masks
        # ------------------------------------------------

        y_train = Y_train[mask_train]
        y_test = Y_test[mask_test]

        pred_train = predY_train[mask_train]
        pred_test = predY_test[mask_test]

        w_train = W_train[mask_train]
        w_test = W_test[mask_test]

        # ------------------------------------------------
        # Training ROC
        # ------------------------------------------------

        f, ax = plt.subplots(figsize=(8, 8))

        fpr_train, tpr_train, _ = metrics.roc_curve(
            y_train,
            pred_train,
            sample_weight=w_train
        )

        train_auc = metrics.auc(fpr_train, tpr_train)

        ax.plot(
            tpr_train,
            1 - fpr_train,
            label=f"Train: AUC = {train_auc:.3f}"
        )

        # ------------------------------------------------
        # Testing ROC
        # ------------------------------------------------

        fpr_test, tpr_test, _ = metrics.roc_curve(
            y_test,
            pred_test,
            sample_weight=w_test
        )

        test_auc = metrics.auc(fpr_test, tpr_test)

        ax.plot(
            tpr_test,
            1 - fpr_test,
            label=f"Test: AUC = {test_auc:.3f}"
        )

        # ------------------------------------------------
        # Formatting
        # ------------------------------------------------

        ax.plot(
            [0, 1],
            [1, 0],
            linestyle="--",
            color="gray"
        )

        ax.legend(loc="lower left")

        ax.set_title(f"ROC - Mass = {mass} GeV")

        ax.set_xlabel("Signal Efficiency")
        ax.set_ylabel("Background Rejection")

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        plt.tight_layout()

        # ------------------------------------------------
        # Save ROC plots
        # ------------------------------------------------

        filename = f"{name}_M{mass}"

        plt.savefig(
            os.path.join(roc_dir, filename + ".png"),
            dpi=300
        )

        plt.savefig(
            os.path.join(roc_dir, filename + ".pdf"),
            dpi=300
        )

        plt.close()

        # ------------------------------------------------
        # Print results
        # ------------------------------------------------

        print(
            f"Mass = {mass:5.1f} GeV | "
            f"Signal Train = {n_sig_train:7d} | "
            f"Signal Test = {n_sig_test:7d} | "
            f"Background Train = {n_bkg_train:7d} | "
            f"Background Test = {n_bkg_test:7d} | "
            f"Train AUC = {train_auc:.3f} | "
            f"Test AUC = {test_auc:.3f}"
        )

        auc_results.append({
            "Mass": mass,
            "Signal_Train": n_sig_train,
            "Signal_Test": n_sig_test,
            "Background_Train": n_bkg_train,
            "Background_Test": n_bkg_test,
            "Train_AUC": train_auc,
            "Test_AUC": test_auc
        })

    # ----------------------------------------------------
    # Save AUC summary
    # ----------------------------------------------------

    auc_df = pd.DataFrame(auc_results)

    auc_df.to_csv(
        os.path.join(roc_dir, "AUC_summary.csv"),
        index=False
    )

    return auc_df



if __name__ == '__main__':

  parser = argparse.ArgumentParser(
      description="Train or load an XGBoost BDT model"
  )

  parser.add_argument(
      "--useMC",
      help="Use MC for training",
      action="store_true",
      dest="useMC",
  )

  parser.add_argument(
      "--loadModel",
      help="Load a previously trained model (provide the model file path)",
      default=None,
  )

  parser.add_argument(
      "-m", "--model_path",
      required=True,
      help="Output path for the model to export",
  )

  parser.add_argument(
      "--optimize",
      action="store_true",
      help="Optimise BDT hyperparameters using Optuna",
  )

  parser.add_argument(
      "--trials",
      type=int,
      default=100,
      help="Number of Optuna trials (default: 100)",
  )

  options = parser.parse_args()


  optimize = options.optimize
  n_trials = options.trials
  useMC = options.useMC
  loadModel = options.loadModel
  model_path = options.model_path
  export = False
  if model_path:
    export = True
  model_name = os.path.splitext(os.path.basename(model_path))[0]

  # inputVars =   ["leppt","lepeta","b1eta","b2eta","pT1_by_mbb","pT2_by_mbb","Njets","pho1eta","pho2eta","pT1_by_mgg","pT2_by_mgg","leadbdeepjet","subleadbdeepjet","leadgMVA","subleadgMVA","delphi_gg","delphi_bb","delphi_bbgg","delphi_ggMET","true_mass"]
  # inputVarsMC = ["leppt","lepeta","b1eta","b2eta","pT1_by_mbb","pT2_by_mbb","Njets","pho1eta","pho2eta","pT1_by_mgg","pT2_by_mgg","leadbdeepjet","subleadbdeepjet","leadgMVA","subleadgMVA","delphi_gg","delphi_bb","delphi_bbgg","delphi_ggMET","true_mass"]
  
#   inputVars =   ["leppt","lepeta", "first_jet_eta","second_jet_eta", "pT1_by_mbb","pT2_by_mbb", "Njets","pholead_eta","phosublead_eta","first_jet_B","second_jet_B","pholead_mvaID","phosublead_mvaID", "pT1_by_mgg","pT2_by_mgg", "delphi_gg","delphi_bb","delphi_bbgg", "delphi_ggMET_Puppi", "pt_ratio", "mass_point", "evt_wgt"]
#   inputVarsMC = ["leppt","lepeta", "first_jet_eta","second_jet_eta", "pT1_by_mbb","pT2_by_mbb", "Njets","pholead_eta","phosublead_eta","first_jet_B","second_jet_B","pholead_mvaID","phosublead_mvaID", "pT1_by_mgg","pT2_by_mgg", "delphi_gg","delphi_bb","delphi_bbgg", "delphi_ggMET_Puppi", "pt_ratio", "mass_point", "evt_wgt"]

  inputVars =   ["leppt","lepeta", "first_jet_eta","second_jet_eta", "pT1_by_mbb","pT2_by_mbb", "Njets","pholead_eta","phosublead_eta","first_jet_B","second_jet_B","pholead_mvaID","phosublead_mvaID", "pT1_by_mgg","pT2_by_mgg", "delphi_gg","delphi_bb","delphi_bbgg", "delphi_ggMET_Puppi", "mass_point", "evt_wgt"]
  inputVarsMC = ["leppt","lepeta", "first_jet_eta","second_jet_eta", "pT1_by_mbb","pT2_by_mbb", "Njets","pholead_eta","phosublead_eta","first_jet_B","second_jet_B","pholead_mvaID","phosublead_mvaID", "pT1_by_mgg","pT2_by_mgg", "delphi_gg","delphi_bb","delphi_bbgg", "delphi_ggMET_Puppi", "mass_point", "evt_wgt"]

#   inputVars =   ["leppt","lepeta", "first_jet_eta","second_jet_eta", "pT1_by_mbb","pT2_by_mbb", "Njets","pholead_eta","phosublead_eta","first_jet_B","second_jet_B","pholead_mvaID","phosublead_mvaID", "pT1_by_mgg","pT2_by_mgg", "delphi_gg","delphi_bb","delphi_bbgg", "delphi_ggMET_Puppi", "energy_resol_lead", "energy_resol_sublead", "mass_point", "evt_wgt"]
#   inputVarsMC = ["leppt","lepeta", "first_jet_eta","second_jet_eta", "pT1_by_mbb","pT2_by_mbb", "Njets","pholead_eta","phosublead_eta","first_jet_B","second_jet_B","pholead_mvaID","phosublead_mvaID", "pT1_by_mgg","pT2_by_mgg", "delphi_gg","delphi_bb","delphi_bbgg", "delphi_ggMET_Puppi", "energy_resol_lead", "energy_resol_sublead", "mass_point", "evt_wgt"]

#   inputVars =   ["leppt","lepeta", "first_jet_eta","second_jet_eta", "pT1_by_mbb","pT2_by_mbb", "Njets","pholead_eta","phosublead_eta","first_jet_B","second_jet_B","pholead_mvaID","phosublead_mvaID", "pT1_by_mgg","pT2_by_mgg", "delphi_gg","delphi_bb","delphi_bbgg", "delphi_ggMET_Puppi", "evt_wgt"]
#   inputVarsMC = ["leppt","lepeta", "first_jet_eta","second_jet_eta", "pT1_by_mbb","pT2_by_mbb", "Njets","pholead_eta","phosublead_eta","first_jet_B","second_jet_B","pholead_mvaID","phosublead_mvaID", "pT1_by_mgg","pT2_by_mgg", "delphi_gg","delphi_bb","delphi_bbgg", "delphi_ggMET_Puppi", "evt_wgt"]

  # inputVars =   ["leppt","lepeta", "first_jet_eta","second_jet_eta", "pT1_by_mbb","pT2_by_mbb", "n_bJets","pholead_eta","phosublead_eta","first_jet_B","second_jet_B","pholead_mvaID","phosublead_mvaID", "delphi_gg","delphi_bb","delphi_bbgg", "delphi_ggMET_Puppi", "pt_ratio", "mass_point", "evt_wgt"]
  # inputVarsMC = ["leppt","lepeta", "first_jet_eta","second_jet_eta", "pT1_by_mbb","pT2_by_mbb", "n_bJets","pholead_eta","phosublead_eta","first_jet_B","second_jet_B","pholead_mvaID","phosublead_mvaID", "delphi_gg","delphi_bb","delphi_bbgg", "delphi_ggMET_Puppi", "pt_ratio", "mass_point", "evt_wgt"]


  print("Loading Signal...")
  df_sig = loadSignal(inputVars)

  print("Loading MC Bkgs...")
  if useMC:
    df_bkg = loadMCBackground(inputVars,inputVarsMC)
    # df_bkg = df_bkg.sample(frac=1/5.0, random_state=12345)

  #Add target for signal (1)
  df_sig['target'] = np.ones(len(df_sig.index))
  df_sig['target'] = df_sig['target'].astype('int')

  #Add target for backgrounds (0)
  df_bkg['target'] = np.zeros(len(df_bkg.index))
  df_bkg['target'] = df_bkg['target'].astype('int')


#   #Deal with negative weights (set to 1 and rescale accordingly)
  sig_sum = df_sig.sum(0)['evt_wgt']
  #df_sig['weight'] = df_sig['weight'].abs()
  df_sig['evt_wgt'] = df_sig['evt_wgt'].apply(lambda x: x if x>=0. else np.abs(x))
  sig_sum_abs = df_sig.sum(0)['evt_wgt']
  df_sig['evt_wgt'] = df_sig['evt_wgt'].apply(lambda x: x*(sig_sum/sig_sum_abs))
  sig_sum = df_sig.sum(0)['evt_wgt']
  print(sig_sum)

  #Deal with negative weights (set to 1 and rescale accordingly)
  bkg_sum = df_bkg.sum(0)['evt_wgt']
  #df_bkg['weight'] = df_bkg['weight'].abs()
  df_bkg['evt_wgt'] = df_bkg['evt_wgt'].apply(lambda x: x if x>=0. else np.abs(x))
  bkg_sum_abs = df_bkg.sum(0)['evt_wgt']
  df_bkg['evt_wgt'] = df_bkg['evt_wgt'].apply(lambda x: x*(bkg_sum/bkg_sum_abs))
  bkg_sum = df_bkg.sum(0)['evt_wgt']
#  df_bkg['evt_wgt'] = df_bkg['evt_wgt'].apply(lambda x: x*(sig_sum/bkg_sum))
  df_sig['evt_wgt'] = df_sig['evt_wgt'].apply(lambda x: x*(bkg_sum/sig_sum))
  print(df_bkg['evt_wgt'])
  print(df_sig.sum(0)['evt_wgt'])
  print(bkg_sum)

  #Deal with negative weights (set to 1 and rescale accordingly)
#   sig_sum = df_sig.sum(0)['evt_wgt']
#   #df_sig['weight'] = df_sig['weight'].abs()
#   df_sig['evt_wgt'] = df_sig['evt_wgt'].apply(lambda x: x if x>=0. else 1)
#   sig_sum_abs = df_sig.sum(0)['evt_wgt']
#   df_sig['evt_wgt'] = df_sig['evt_wgt'].apply(lambda x: x*(sig_sum/sig_sum_abs))
#   sig_sum = df_sig.sum(0)['evt_wgt']
#   print(sig_sum)

#   #Deal with negative weights (set to 1 and rescale accordingly)
#   bkg_sum = df_bkg.sum(0)['evt_wgt']
#   #df_bkg['weight'] = df_bkg['weight'].abs()
#   df_bkg['evt_wgt'] = df_bkg['evt_wgt'].apply(lambda x: x if x>=0. else 1)
#   bkg_sum_abs = df_bkg.sum(0)['evt_wgt']
#   df_bkg['evt_wgt'] = df_bkg['evt_wgt'].apply(lambda x: x*(bkg_sum/bkg_sum_abs))
#   bkg_sum = df_bkg.sum(0)['evt_wgt']
# #  df_bkg['evt_wgt'] = df_bkg['evt_wgt'].apply(lambda x: x*(sig_sum/bkg_sum))
#   df_sig['evt_wgt'] = df_sig['evt_wgt'].apply(lambda x: x*(bkg_sum/sig_sum))
#   print(df_bkg['evt_wgt'])
#   print(bkg_sum)

  print("Loading done!...")
  
  if 'evt_wgt' in inputVars:
    inputVars.remove('evt_wgt')
  if 'Norm_SFs' in inputVars:
    inputVars.remove('Norm_SFs')
         
  seed = int(datetime.now().timestamp())
  seed = 12345
  np.random.seed(seed)
  print("Seed:",seed)

  # X = pd.concat((df_sig.iloc[:,0:21] ,df_bkg.iloc[:,0:21] ))
  features = ['evt_wgt'] + inputVars

  X = pd.concat([
      df_sig[features],
      df_bkg[features]
  ])
  Y = pd.concat((df_sig['target'] ,df_bkg['target'] ))

#   test_size = 0.2
# #  test_size = 0.001
#   X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed, shuffle=True)
  
#   W_train = X_train['evt_wgt'].values *(10**6)
#   X_train = X_train.drop('evt_wgt',axis=1).values
#   Y_train = Y_train.values
#   W_test  = X_test['evt_wgt'].values *(10**6)
#   X_test = X_test.drop('evt_wgt',axis=1).values
#   Y_test = Y_test.values



  test_size = 0.2
  validation_size = 0.2

  # Reserve the test sample before hyperparameter optimisation
  X_train_val, X_test, Y_train_val, Y_test = train_test_split(
      X,
      Y,
      test_size=test_size,
      random_state=seed,
      stratify=Y,
  )

  # Split the remaining data into training and validation samples
  X_train, X_val, Y_train, Y_val = train_test_split(
      X_train_val,
      Y_train_val,
      test_size=validation_size,
      random_state=seed,
      stratify=Y_train_val,
  )

  # Extract event weights
  W_train = X_train["evt_wgt"].to_numpy() * (10**6)
  W_val = X_val["evt_wgt"].to_numpy() * (10**6)
  W_test = X_test["evt_wgt"].to_numpy() * (10**6)

  print("W_train: ", W_train)

  # Remove weights from input features
  X_train_features = X_train.drop("evt_wgt", axis=1).to_numpy()
  X_val_features = X_val.drop("evt_wgt", axis=1).to_numpy()
  X_test_features = X_test.drop("evt_wgt", axis=1).to_numpy()

  Y_train = Y_train.to_numpy()
  Y_val = Y_val.to_numpy()
  Y_test = Y_test.to_numpy()

  # Create DMatrix objects
  training = xgb.DMatrix(
      X_train_features,
      label=Y_train,
      weight=W_train,
      feature_names=inputVars,
  )

  validation = xgb.DMatrix(
      X_val_features,
      label=Y_val,
      weight=W_val,
      feature_names=inputVars,
  )

  testing = xgb.DMatrix(
      X_test_features,
      label=Y_test,
      weight=W_test,
      feature_names=inputVars,
  )

  # Combined training + validation data, used for final model fitting
  X_final = np.concatenate([X_train_features, X_val_features])
  Y_final = np.concatenate([Y_train, Y_val])
  W_final = np.concatenate([W_train, W_val])

  training_final = xgb.DMatrix(
      X_final,
      label=Y_final,
      weight=W_final,
      feature_names=inputVars,
  )

  print("Training events:", len(Y_train))
  print("Validation events:", len(Y_val))
  print("Test events:", len(Y_test))


  print("W_train :",W_train)
  print("W_test :",W_test)

#   #Ref: Anirban Bala
#   params = {
#     'booster': 'gbtree',
#     'eval_metric': 'auc',
#     'learning_rate': 0.5,
#     'max_depth': 3,
#     'gamma': 0.1,
#     'objective': 'binary:logistic',
#   }

#   #Testing
# #   params = {
# #     'booster': 'gbtree',
# #     'eval_metric': 'auc',
# #     'learning_rate': 0.3,
# #     'max_depth': 9,
# #     'gamma': 0.1,
# #     'objective': 'binary:logistic',
# # }

  
#   training = xgb.DMatrix(X_train,label=Y_train, weight=W_train,feature_names=inputVars)
#   testing  = xgb.DMatrix(X_test, label=Y_test , weight=W_test, feature_names=inputVars) 

#   # training = xgb.DMatrix(X_train,label=Y_train,feature_names=inputVars)
#   # testing  = xgb.DMatrix(X_test, label=Y_test, feature_names=inputVars) 
  

#   # if loadModel:
#   #   with open(loadModel,'rb') as f:
#   #       modelDict=pickle.load(f)
#   #       model=modelDict['xgbModel']
#   #       modelName=modelDict['modelName']
#   #       print("Loaded model ",modelName)
#   # else:
#   #   print("Training model! ")
#   #   model = xgb.train(params,training) 
#   # if export:
#   #   print("Model exported to : ",model_path)
#   #   modelDict={
#   #       'xgbModel' : model,
#   #       'modelName' : model_name,
#   #   }
#   #   with open(model_path,'wb') as f:
#   #       pickle.dump(modelDict,f)

#   if loadModel:
#     with open(loadModel, "rb") as f:
#         modelDict = pickle.load(f)

#     model = modelDict["xgbModel"]
#     modelName = modelDict["modelName"]

#     print("Loaded model:", modelName)

#   else:
#       print("Training model!")

#       # Ensure params and training are defined before this point.
#       model = xgb.train(params, training)

#   if export:
#       modelDict = {
#           "xgbModel": model,
#           "modelName": model_name,
#       }

#       with open(model_path, "wb") as f:
#           pickle.dump(modelDict, f)

#       print("Model exported to:", model_path)




  # --------------------------------------------------
  # Default XGBoost parameters
  # --------------------------------------------------
  params = {
      "booster": "gbtree",
      "objective": "binary:logistic",
      "eval_metric": "auc",
      "tree_method": "hist",
      "learning_rate": 0.5,
      "max_depth": 3,
      "gamma": 0.1,
      "seed": seed,
  }

  # --------------------------------------------------
  # Load an existing model, optimise, or train normally
  # --------------------------------------------------
  if loadModel:

      with open(loadModel, "rb") as f:
          modelDict = pickle.load(f)

      model = modelDict["xgbModel"]
      modelName = modelDict["modelName"]

      print("Loaded model:", modelName)

  elif optimize:

      def objective(trial):

          trial_params = {
              **params,
              "learning_rate": trial.suggest_float(
                  "learning_rate", 0.01, 0.5, log=True
              ),
              "max_depth": trial.suggest_int(
                  "max_depth", 2, 12
              ),
              "gamma": trial.suggest_float(
                  "gamma", 0.0, 5.0
              )
          }

          trial_model = xgb.train(
              params=trial_params,
              dtrain=training,
              num_boost_round=1000,
              evals=[(validation, "validation")],
              early_stopping_rounds=50,
              verbose_eval=False,
          )

          best_iteration = trial_model.best_iteration
          best_auc = float(trial_model.best_score)

          trial.set_user_attr(
              "best_iteration", best_iteration
          )

          print(
              f"Trial {trial.number}: "
              f"AUC={best_auc:.6f}, "
              f"best_iteration={best_iteration}"
          )

          return best_auc

      study = optuna.create_study(
          direction="maximize",
          study_name=f"{model_name}_optimization",
          sampler=optuna.samplers.TPESampler(seed=seed),
      )

      study.optimize(objective, n_trials=n_trials)

      print("\nBest validation AUC:", study.best_value)
      print("Best hyperparameters:", study.best_params)

      # Apply the best hyperparameters
      params.update(study.best_params)

      # Save optimisation results
      best_params_path = os.path.join(
          output_dir, model_name + "_best_params.json"
      )

      with open(best_params_path, "w") as f:
          json.dump(
              {
                  "best_validation_auc": study.best_value,
                  "best_iteration": study.best_trial.user_attrs[
                      "best_iteration"
                  ],
                  "best_params": study.best_params,
              },
              f,
              indent=4,
          )

      print("Best parameters saved to:", best_params_path)

      # Retrain using training + validation data
      best_num_rounds = (
          study.best_trial.user_attrs["best_iteration"] + 1
      )

      print(
          f"\nTraining final model with {best_num_rounds} rounds"
      )

      model = xgb.train(
          params=params,
          dtrain=training_final,
          num_boost_round=best_num_rounds,
          verbose_eval=False,
      )

  else:

      print("Training model with default parameters!")

      model = xgb.train(
          params=params,
          dtrain=training,
      )


  # --------------------------------------------------
  # Export model using your existing pickle structure
  # --------------------------------------------------
  modelDict = {
      "xgbModel": model,
      "modelName": model_name,
  }

  with open(model_path, "wb") as f:
      pickle.dump(modelDict, f)

  print("Model exported to:", model_path)


  predY_train = model.predict(training)
  predY_test  = model.predict(testing)
  
  label = '2024'
  if useMC:
    label = label + 'MC' 
  else:
    label = label + 'dataDriven' 
          
  drawScores(predY_train,predY_test,W_train,W_test,25,0.,1.,'BDT_'+model_name)
  drawROC(Y_train,Y_test,predY_train,predY_test,W_train,W_test,'ROC_'+model_name)
  mass_train = X_train["mass_point"].to_numpy()
  mass_test  = X_test["mass_point"].to_numpy()
  drawROC_mass_point(
        Y_train,
        Y_test,
        predY_train,
        predY_test,
        W_train,
        W_test,
        name="BDT_ROC",
        model_name=model_name,
        mass_train=mass_train,
        mass_test=mass_test
    )

#   drawROC_mass_point(
#     Y_train, Y_test,
#     predY_train, predY_test,
#     W_train, W_test,
#     name="BDT",
#     model_name=model_name,
#     single_mass=mass_value
# )
          
  mdl = model.get_dump()
  input_vars=[]
  for i,key in enumerate(inputVars):
    input_vars.append((key,'F'))
  # convert_model(mdl,input_variables=input_vars,output_xml='H2AA2bbggMVA_xgboost_model'+label+'.xml')  
