#include "TFile.h"
#include "RooWorkspace.h"
#include "RooDataSet.h"
#include "RooArgSet.h"
#include "RooRealVar.h"

void printWeights() {

    TFile *f = TFile::Open(
        "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/BKG_MCtoData_format/New/output_AllData_13p6TeV_powheg_AllData_without_DY_new.root"
    );

    RooWorkspace *w = (RooWorkspace*)f->Get("tagsDumper/cms_hgg_13TeV");

    if (!w) {
        std::cout << "Workspace not found!" << std::endl;
        return;
    }

    RooDataSet *dset =
        (RooDataSet*)w->data("AllData_125_13TeV_CAT1");

    if (!dset) {
        std::cout << "Dataset not found!" << std::endl;
        // w->allData().Print("V");
        for (auto data : w->allData()) {
            data->Print("V");
        }
        return;
    }

    for (int i = 0; i < 10; ++i) {
        dset->get(i);  // load entry i

        std::cout << "Entry " << i
                << " weight = "
                << std::setprecision(15)
                << dset->weight()
                << std::endl;
    }

    std::cout << "Dataset has "
              << dset->numEntries()
              << " entries" << std::endl;

    for (int i = 0; i < std::min(10, (int)dset->numEntries()); ++i) {

        const RooArgSet *row = dset->get(i);

        double w_nominal =
            ((RooRealVar*)row->find("weight_nominal"))->getVal();

        double w_central =
            ((RooRealVar*)row->find("weight_central"))->getVal();

        double w_pu_up =
            ((RooRealVar*)row->find("weight_PileupUp"))->getVal();

        double w_pu_down =
            ((RooRealVar*)row->find("weight_PileupDown"))->getVal();

        std::cout << "Entry " << i
                  << "  nominal=" << std::setprecision(15) << w_nominal
                  << "  central=" << w_central
                  << "  PUUp=" << w_pu_up
                  << "  PUDown=" << w_pu_down
                  << std::endl;
    }

        RooRealVar *mass =
            (RooRealVar*)dset->get()->find("CMS_hgg_mass");

        std::cout << "Mass variable range = ["
                << mass->getMin() << ", "
                << mass->getMax() << "] GeV"
                << std::endl;
}
