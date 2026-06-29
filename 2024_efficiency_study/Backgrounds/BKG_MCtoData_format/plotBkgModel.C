// plotBkgModel.C

#include "TFile.h"
#include "TCanvas.h"
#include "RooWorkspace.h"
#include "RooRealVar.h"
#include "RooPlot.h"
#include "RooAbsPdf.h"
#include "RooAbsData.h"
#include "RooAbsReal.h"
#include "RooFit.h"

using namespace RooFit;

// void plotBkgModel(const char* fname="/eos/user/b/bbapi/flashggNew/CMSSW_14_1_0_pre4/src/flashggFinalFit/Datacard/DataCards/higgsCombine.bdt_50_2024.AsymptoticLimits.mH50.root")
// {
//     TFile *f = TFile::Open(fname);

//     RooWorkspace *w = (RooWorkspace*)f->Get("w");
//     if(!w){
//         std::cout << "Workspace not found!" << std::endl;
//         return;
//     }

//     RooRealVar *mgg = w->var("CMS_hgg_mass");

//     RooAbsPdf *bkg =
//         w->pdf("shapeBkg_bkg_mass_CAT1");

//     RooAbsData *data =
//         w->data("data_obs");

//     RooAbsReal *Nbkg =
//         w->function("n_exp_final_binCAT1_proc_bkg_mass");

//     std::cout << "Background yield = "
//               << Nbkg->getVal()
//               << std::endl;

//     mgg->setRange("fullRange",10,70);

//     RooPlot *frame =
//         mgg->frame(
//             Range("fullRange"),
//             Bins(120)
//         );

//     data->plotOn(frame);

//     bkg->plotOn(
//         frame,
//         Normalization(
//             Nbkg->getVal(),
//             RooAbsReal::NumEvent
//         ),
//         LineColor(kRed),
//         LineWidth(2)
//     );

//     TCanvas *c = new TCanvas("c","c",800,600);

//     frame->SetTitle("CAT1 Background Model");
//     frame->GetXaxis()->SetTitle("m_{#gamma#gamma} (GeV)");
//     frame->GetYaxis()->SetTitle("Events");

//     frame->Draw();

//     c->SaveAs("bkg_model_CAT1.pdf");
//     c->SaveAs("bkg_model_CAT1.png");
// }


// void plotBkgModel()
void plotBkgModel(const char* fname="/eos/user/b/bbapi/flashggNew/CMSSW_14_1_0_pre4/src/flashggFinalFit/Datacard/DataCards/higgsCombine.bdt_50_2024.AsymptoticLimits.mH50.root")
{
    TFile *f = TFile::Open(fname);

    RooWorkspace *w = (RooWorkspace*)f->Get("w");
    if(!w){
        std::cout << "Workspace not found!" << std::endl;
        return;
    }
    auto mgg = w->var("CMS_hgg_mass");
    auto bkg = w->pdf("shapeBkg_bkg_mass_CAT1");

    auto idx = w->cat("pdfindex_AllData_125_13TeV_CAT1_2024_13TeV");
    // idx->setIndex(1);

    TGraph *g = new TGraph();

    int ip=0;
    for(int m=10; m<=70; ++m){
        mgg->setVal(m);
        g->SetPoint(ip++, m, bkg->getVal());
    }

    for(int m=10; m<=70; ++m){
    mgg->setVal(m);

    cout << m
         << "  index=" << idx->getIndex()
         << endl;
}

    TCanvas *c = new TCanvas("c","c",800,600);

    g->SetTitle("Background PDF;CMS_hgg_mass (GeV);PDF value");
    g->SetLineWidth(2);
    g->Draw("AL");

    c->SaveAs("bkgShape.pdf");
}
