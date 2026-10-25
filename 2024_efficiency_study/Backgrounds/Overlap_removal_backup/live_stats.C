#include "TPad.h"
#include "TH1.h"
#include "THStack.h"
#include "TList.h"
#include "TPaveText.h"
#include "TText.h"

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

// void updateStats() {
//   TVirtualPad* pad = gPad;
//   if (!pad) return;
//   double xlo = pad->GetUxmin(), xhi = pad->GetUxmax();
//   TIter next(pad->GetListOfPrimitives());
//   while (TObject* o = next()) {
//     if (!o->InheritsFrom(TPaveText::Class())) continue;
//     TPaveText* pt = (TPaveText*)o;
//     TString nm = pt->GetName();
//     if (!nm.BeginsWith("stat_")) continue;
//     TH1* h = findHist(pad, nm(5, nm.Length()));
//     if (!h) continue;
//     TAxis* ax = h->GetXaxis();
//     int b1 = ax->FindBin(xlo + 1e-9), b2 = ax->FindBin(xhi - 1e-9);
//     int nb = h->GetNbinsX();
//     double integral = h->Integral(b1, b2);
//     double overflow = (b2 < nb) ? h->Integral(b2 + 1, nb + 1) : h->GetBinContent(nb + 1);
//     TIter ln(pt->GetListOfLines());
//     while (TObject* l = ln()) {
//       TText* t = (TText*)l;
//       TString s = t->GetTitle();
//       if (s.BeginsWith("Integral")) t->SetTitle(Form("Integral = %.2f", integral));
//       else if (s.BeginsWith("Overflow")) t->SetTitle(Form("Overflow = %.2f", overflow));
//       else if (s.BeginsWith("Mean")) {
//         ax->SetRange(b1, b2);
//         t->SetTitle(Form("Mean = %.2f", h->GetMean()));
//         ax->SetRange(0, 0);
//       }
//     }
//   }
// }

void updateStats() {
  TVirtualPad* pad = gPad;
  if (!pad) return;
  double xlo = pad->GetUxmin(), xhi = pad->GetUxmax();
  TIter next(pad->GetListOfPrimitives());
  while (TObject* o = next()) {
    if (!o->InheritsFrom(TPaveText::Class())) continue;
    TPaveText* pt = (TPaveText*)o;
    TString nm = pt->GetName();
    if (!nm.BeginsWith("stat_")) continue;
    TH1* h = findHist(pad, nm(5, nm.Length()));
    if (!h) continue;
    TAxis* ax = h->GetXaxis();
    int b1 = ax->FindBin(xlo + 1e-9), b2 = ax->FindBin(xhi - 1e-9);
    int nb = h->GetNbinsX();
    double integral = h->Integral(b1, b2);
    double overflow = (b2 < nb) ? h->Integral(b2 + 1, nb + 1) : h->GetBinContent(nb + 1);
    TIter ln(pt->GetListOfLines());
    while (TObject* l = ln()) {
      TText* t = (TText*)l;
      TString s = t->GetTitle();
      if (s.BeginsWith("Integral")) {
        const char* fmt = (std::abs(integral) > 1e5) ? "Integral = %.2e" : "Integral = %.2f";
        t->SetTitle(Form(fmt, integral));
      }
      else if (s.BeginsWith("Overflow")) {
        const char* fmt = (std::abs(overflow) > 1e5) ? "Overflow = %.2e" : "Overflow = %.2f";
        t->SetTitle(Form(fmt, overflow));
      }
      else if (s.BeginsWith("Mean")) {
        ax->SetRange(b1, b2);
        t->SetTitle(Form("Mean = %.2f", h->GetMean()));
        ax->SetRange(0, 0);
      }
    }
  }
}
