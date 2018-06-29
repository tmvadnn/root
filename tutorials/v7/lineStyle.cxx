/// \file
/// \ingroup tutorial_v7
///
/// \macro_code
///
/// \date 2018-03-18
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!
/// \author Iliana Betsou

#include "ROOT/RCanvas.hxx"
#include "ROOT/RColor.hxx"
#include "ROOT/RText.hxx"
#include "ROOT/RLine.hxx"
#include "ROOT/RPadPos.hxx"

void lineStyle() {
   using namespace ROOT;
   using namespace ROOT::Experimental;

   auto canvas = Experimental::RCanvas::Create("Canvas Title");
   double num = 0.3;

   for (int i=10; i>0; i--){
      num = num + 0.05;

      RPadPos pt(.3_normal, RPadLength::Normal(num));
      auto optts = canvas->Draw(Experimental::RText(pt, Form("%d", i)));
      optts->SetTextSize(13);
      optts->SetTextAlign(32);
      optts->SetTextFont(52);

      RPadPos pl1(.32_normal, RPadLength::Normal(num));
      RPadPos pl2(.8_normal , RPadLength::Normal(num));
      auto optls = canvas->Draw(Experimental::RLine(pl1, pl2));
      optls->SetLineStyle(i);
   }

   canvas->Show();
}
