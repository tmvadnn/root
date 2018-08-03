/// \file
/// \ingroup tutorial_v7
///
/// \macro_code
///
/// \date 2017-10-17
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!
/// \author Olivier couet <Olivier.Couet@cern.ch>

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RCanvas.hxx"
#include "ROOT/RColor.hxx"
#include "ROOT/RText.hxx"
#include "ROOT/RPadPos.hxx"
#include "ROOT/TDirectory.hxx"

void text()
{
   using namespace ROOT;
   using namespace ROOT::Experimental;

   // Create a canvas to be displayed.
   auto canvas = Experimental::RCanvas::Create("Canvas Title");

   for (int i=0; i<=360; i+=10) {
      auto opts = canvas->Draw(Experimental::RText({0.5_normal, 0.6_normal}, "____  Hello World"));

      Experimental::RColor col(0.0015*i, 0.0025*i ,0.003*i);
      opts->SetTextColor(col);
      opts->SetTextSize(10+i/10);
      opts->SetTextAngle(i);
      opts->SetTextAlign(13);
      opts->SetTextFont(42);
   }

   canvas->Show();
}
