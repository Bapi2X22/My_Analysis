// plot_templates.C

#include "TCanvas.h"
#include "TFile.h"
#include "TH1.h"
#include "TLegend.h"
#include "RooWorkspace.h"
#include "RooDataHist.h"
#include "RooRealVar.h"

void plot_templates()
{
    TFile *f = TFile::Open(
        "NTuples_WH_2024_HDNA_presel_latest_with_BDT_score/root/WH-2024M50/ws_WHM50Y2024/output_WHM50Y2024_M50_13p6TeV_amcatnlo_pythia8_WHM50Y2024.root");

    auto *dir = (TDirectory*)f->Get("tagsDumper");
    auto *ws  = (RooWorkspace*)dir->Get("cms_hgg_13TeV");

    RooRealVar *mass = ws->var("CMS_hgg_mass");

    std::vector<std::string> systs = {
        "ScaleEBZmmg",
        "ScaleEEZmmg",
        "Smearing",
        "energyErrShift",
        "FNUF",
        "Material",
        "jecsystTotal"
    };

    for (auto &syst : systs)
    {
        std::string upname =
            "WHM50Y2024_50_13TeV_CAT1_" + syst + "Up01sigma";

        std::string downname =
            "WHM50Y2024_50_13TeV_CAT1_" + syst + "Down01sigma";

        auto *up =
            (RooDataHist*)ws->data(upname.c_str());

        auto *down =
            (RooDataHist*)ws->data(downname.c_str());

        if (!up || !down)
        {
            std::cout << "Missing " << syst << std::endl;
            continue;
        }

        TH1 *hUp =
            up->createHistogram(
                Form("hUp_%s", syst.c_str()),
                *mass);

        TH1 *hDown =
            down->createHistogram(
                Form("hDown_%s", syst.c_str()),
                *mass);

        hUp->SetLineColor(kRed);
        hUp->SetLineWidth(2);

        hDown->SetLineColor(kBlue);
        hDown->SetLineWidth(2);

        TCanvas *c =
            new TCanvas(
                Form("c_%s", syst.c_str()),
                syst.c_str(),
                800,
                600);

        hUp->SetTitle(
            Form("%s ;m_{#gamma#gamma} [GeV];Bin Content",
                 syst.c_str()));

        double maxy =
            std::max(hUp->GetMaximum(),
                     hDown->GetMaximum());

        hUp->SetMaximum(1.2 * maxy);

        hUp->Draw("hist");
        hDown->Draw("hist same");

        TLegend *leg =
            new TLegend(0.20,0.75,0.38,0.88);

        leg->AddEntry(hUp,"Up","l");
        leg->AddEntry(hDown,"Down","l");
        leg->Draw();

        c->SaveAs(Form("%s.pdf",syst.c_str()));
        c->SaveAs(Form("%s.png",syst.c_str()));
    }
}
