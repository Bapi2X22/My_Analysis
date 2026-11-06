#include "TPad.h"
#include "TVirtualPad.h"
#include "TH1.h"
#include "THStack.h"
#include "TList.h"
#include "TAxis.h"
#include "TPaveText.h"
#include "TText.h"
#include "TString.h"

static TH1* findHist(TVirtualPad* pad, const TString& name) {
  TIter next(pad->GetListOfPrimitives());
  while (TObject* o = next()) {
    if (o->InheritsFrom(TH1::Class()) && name == o->GetName()) return (TH1*)o;
    if (o->InheritsFrom(THStack::Class())) {
      TIter hn(((THStack*)o)->GetHists());
      while (TObject* h = hn())
        if (name == h->GetName()) return (TH1*)h;
    }
  }
  return nullptr;
}

static void updateBoxesIn(TVirtualPad* boxPad, TVirtualPad* histPad, double xlo, double xhi) {
  bool touched = false;
  TIter next(boxPad->GetListOfPrimitives());
  while (TObject* o = next()) {
    if (!o->InheritsFrom(TPaveText::Class())) continue;
    TPaveText* pt = (TPaveText*)o;
    TString nm = pt->GetName();
    if (!nm.BeginsWith("stat_")) continue;
    TH1* h = findHist(histPad, nm(5, nm.Length()));
    if (!h) continue;
    int nbins = h->GetNbinsX();

    // Include all bins: underflow, regular bins and overflow.
    double integral = h->Integral(0, nbins + 1);
    TIter ln(pt->GetListOfLines());
    while (TObject* l = ln()) {
      TText* t = (TText*)l;
      if (TString(t->GetTitle()).BeginsWith("Integral")) {
        t->SetTitle(Form("Integral = %.2f", integral));
        touched = true;
      }
    }
  }
  if (touched) boxPad->Modified();
}

void updateStats() {
  TVirtualPad* pad = gPad;
  if (!pad) return;
  double xlo = pad->GetUxmin(), xhi = pad->GetUxmax();
  updateBoxesIn(pad, pad, xlo, xhi);
  TVirtualPad* mother = pad->GetMother();
  if (mother && mother != pad) {
    TIter nx(mother->GetListOfPrimitives());
    while (TObject* o = nx())
      if (o != pad && o->InheritsFrom(TVirtualPad::Class()))
        updateBoxesIn((TVirtualPad*)o, pad, xlo, xhi);
  }
}