void ws_checker() {

    // -------------------------------
    // 1. Open file and workspace
    // -------------------------------
    TFile *f = TFile::Open("/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/NTuples_BKG_2024_condor_15GeV/root/TTG1Jets-24SummerRun3/diphoton/ws_TTG1Jets/Data_13TeV_DiPho_pt.root");

    if (!f || f->IsZombie()) {
        std::cout << "Error opening file!" << std::endl;
        return;
    }

    RooWorkspace *w =
        (RooWorkspace*)f->Get("tagsDumper/cms_hgg_13TeV");

    if (!w) {
        std::cout << "Workspace not found!" << std::endl;
        return;
    }

    // -------------------------------
    // 2. Get variable and dataset
    // -------------------------------
    RooRealVar* mass = w->var("CMS_hgg_mass");

    RooDataSet* Data =
        (RooDataSet*)w->data("TTG1Jets_125_13TeV_DiPho_pt");

    if (!mass || !Data) {
        std::cout << "Mass or dataset missing!" << std::endl;
        return;
    }

    std::cout << "\n=== Dataset Info ===" << std::endl;
    Data->Print("v");

    // -------------------------------
    // 3. Check weights (first 10 events)
    // -------------------------------
    std::cout << "\n=== First 10 events ===" << std::endl;

    for (int i = 0; i < 10; i++) {
        const RooArgSet* row = Data->get(i);

        double m =
            ((RooRealVar*)row->find("CMS_hgg_mass"))->getVal();

        double w_internal = Data->weight();

        double w_central =
            ((RooRealVar*)row->find("weight_central"))->getVal();

        double w_nominal =
            ((RooRealVar*)row->find("weight_nominal"))->getVal();

        std::cout << "Event " << i
                  << "  mass = " << m
                  << "  weight = " << w_internal
                  << "  central = " << w_central
                  << "  nominal = " << w_nominal
                  << std::endl;
    }

    // -------------------------------
    // 4. Sum of weights
    // -------------------------------
    std::cout << "\nSum of weights (unbinned): "
              << Data->sumEntries() << std::endl;

    double lumi = 109 * 1000.0; // pb^-1
    double xsec = 4.634;  // pb

    double scale = lumi * xsec;

    // RooRealVar* w = (RooRealVar*)cms_hgg_13TeV->var("weight");

    // // define scaled weight
    // RooFormulaVar w_scaled("w_scaled", "@0 * @1", RooArgList(*w, RooConst(scale)));

    // RooDataSet* data = (RooDataSet*)cms_hgg_13TeV->data("data_obs");

    // RooDataSet data_scaled(
    //     "data_scaled",
    //     "data_scaled",
    //     data,
    //     *data->get(),
    //     "",
    //     "w_scaled"
    // );

// -------------------------------------------------------------------------------
    // RooRealVar scaleVar("scaleVar","scaleVar", scale);

    // // original weight variable
    // RooRealVar* wgt = w->var("weight");

    // // define scaled weight formula
    // RooFormulaVar w_scaled(
    //     "w_scaled",
    //     "@0 * @1",
    //     RooArgList(*wgt, scaleVar)
    // );

    // // ---- ADD as new column ----
    // RooDataSet* data_with_w = (RooDataSet*)Data->addColumn(w_scaled);

    // // ---- build new dataset using that column ----
    // RooDataSet data_scaled(
    //     "data_scaled",
    //     "data_scaled",
    //     *Data->get(),
    //     RooFit::Import(*Data),
    //     RooFit::WeightVar("w_scaled")
    // );
//---------------------------------------------------------------------------------------

    // -------------------------------
    // 1. Define range (remove edge bins)
    // -------------------------------
    int mgg_low =12;
    int mgg_high =78;
    int nBinsForMass = 4*(mgg_high-mgg_low);
    double binWidth = (mgg_high - mgg_low) / nBinsForMass;

    double new_low  = mgg_low  + binWidth;
    double new_high = mgg_high - binWidth;

    // -------------------------------
    // 2. Reduce dataset
    // -------------------------------
    RooDataSet* dataReduced = (RooDataSet*) Data->reduce(
        Form("CMS_hgg_mass > %f && CMS_hgg_mass < %f", new_low, new_high)
    );

    // -------------------------------
    // 3. Scale factor
    // -------------------------------
    // double lumi  = 109 * 1000.0; // pb^-1
    // double xsec  = 4.634;        // pb
    // double scale = lumi * xsec;

    // -------------------------------
    // 4. Create scaled dataset
    // -------------------------------
    RooArgSet* obs = (RooArgSet*) dataReduced->get()->snapshot();

    RooDataSet* dataScaled = new RooDataSet(
        "dataScaled",
        "dataScaled",
        *obs,
        RooFit::WeightVar("weight")   // label only
    );

    // -------------------------------
    // 5. Fill with scaled weights
    // -------------------------------
    for (int i = 0; i < dataReduced->numEntries(); i++) {

        const RooArgSet* row = dataReduced->get(i);

        // choose correct weight
        double w = dataReduced->weight();  
        // OR explicitly:
        // double w = ((RooRealVar*)row->find("weight_central"))->getVal();

        double w_new = w * scale;

        dataScaled->add(*row, w_new);
    }

    // -------------------------------
    // 6. Convert to binned dataset
    // -------------------------------
    RooDataHist data_binned(
        "data_binned",
        "data_binned",
        *mass,
        *dataScaled   // IMPORTANT
    );

    // -------------------------------
    // 7. Sanity checks
    // -------------------------------
    std::cout << "Reduced sum = "
            << dataReduced->sumEntries() << std::endl;

    std::cout << "Scaled sum  = "
            << dataScaled->sumEntries() << std::endl;

    std::cout << "Binned sum  = "
            << data_binned.sumEntries() << std::endl;

    // -------------------------------
    // 8. Plot
    // -------------------------------
    RooPlot* frame = mass->frame(RooFit::Range(new_low, new_high));

    dataScaled->plotOn(frame);
    // OR
    // data_binned.plotOn(frame);



    // -------------------------------
    // 5. Convert to RooDataHist
    // -------------------------------
    // RooDataHist data_binned("data_binned","data_binned",
    //                         *mass,
    //                         *Data);

    // std::cout << "Sum of weights (binned): "
    //           << data_binned.sumEntries() << std::endl;

    // // -------------------------------
    // // 6. Plot
    // // -------------------------------
    // // RooPlot* frame = mass->frame();

    // // Data->plotOn(frame);            // unbinned
    // // data_binned.plotOn(frame);      // binned

    // RooPlot* frame = mass->frame(RooFit::Range(11,79));

    // data_binned.plotOn(frame);
    // data_scaled.plotOn(frame);

    TCanvas *c = new TCanvas("c","c",800,600);
    frame->Draw();


    frame->SetMinimum(0);
    // frame->SetMaximum(1);

    // -------------------------------
    // 7. Save output
    // -------------------------------
    c->SaveAs("check_dataset.png");

    std::cout << "\nPlot saved as check_dataset.png" << std::endl;
}

// void ws_checker() {

//     // -------------------------------
//     // 1. Load workspace
//     // -------------------------------
//     TFile *f = TFile::Open("/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/NTuples_BKG_2024_condor_15GeV/root/TTG1Jets-24SummerRun3/diphoton/ws_TTG1Jets/Data_13TeV_DiPho_pt.root");

//     RooWorkspace *w =
//         (RooWorkspace*)f->Get("tagsDumper/cms_hgg_13TeV");

//     if (!w) {
//         std::cout << "Workspace not found!" << std::endl;
//         return;
//     }

//     // -------------------------------
//     // 2. Get variable & dataset
//     // -------------------------------
//     RooRealVar* mass = w->var("CMS_hgg_mass");

//     RooDataSet* Data =
//         (RooDataSet*)w->data("TTG1Jets_125_13TeV_DiPho_pt");

//     // -------------------------------
//     // 3. Create unit-weight dataset
//     // -------------------------------
//     RooDataSet* Data_unit =
//         new RooDataSet("Data_unit","Data_unit",
//                        *Data->get());

//     for (int i = 0; i < Data->numEntries(); i++) {
//         const RooArgSet* row = Data->get(i);
//         Data_unit->add(*row, 1.0);
//     }

//     // -------------------------------
//     // 4. Convert to RooDataHist
//     // -------------------------------
//     RooDataHist data_weighted("data_weighted","data_weighted",
//                               *mass,
//                               *Data);

//     RooDataHist data_unit("data_unit","data_unit",
//                           *mass,
//                           *Data_unit);

//     // -------------------------------
//     // 5. Weighted plot
//     // -------------------------------
//     RooPlot* frame_w = mass->frame(RooFit::Range(11,79));


//     data_weighted.plotOn(frame_w);

//     frame_w->SetMinimum(0);
//     frame_w->SetMaximum(frame_w->GetMaximum()*1.5);

//     TCanvas *c1 = new TCanvas("c1","Weighted",800,600);
//     frame_w->Draw();

//     c1->SaveAs("weighted.png");


//     // -------------------------------
//     // 6. Unit-weight plot
//     // -------------------------------
//     RooPlot* frame_u = mass->frame(RooFit::Range(11,79));

//     data_unit.plotOn(frame_u);

//     frame_u->SetMinimum(0);
//     frame_u->SetMaximum(200.0);

//     TCanvas *c2 = new TCanvas("c2","Unit weight",800,600);
//     frame_u->Draw();

//     c2->SaveAs("unit_weight.png");


//     // -------------------------------
//     // 7. Print totals
//     // -------------------------------
//     std::cout << "Weighted sum = "
//               << data_weighted.sumEntries() << std::endl;

//     std::cout << "Unit sum = "
//               << data_unit.sumEntries() << std::endl;
// }