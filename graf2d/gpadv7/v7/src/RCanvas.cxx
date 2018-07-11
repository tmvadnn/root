/// \file RCanvas.cxx
/// \ingroup Gpad ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2015-07-10
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RCanvas.hxx"

#include "ROOT/TLogger.hxx"

#include <algorithm>
#include <memory>
#include <stdio.h>
#include <string.h>

#include "TROOT.h"

namespace {
static std::vector<std::shared_ptr<ROOT::Experimental::RCanvas>> &GetHeldCanvases()
{
   static std::vector<std::shared_ptr<ROOT::Experimental::RCanvas>> sCanvases;
   return sCanvases;
}
} // namespace

const std::vector<std::shared_ptr<ROOT::Experimental::RCanvas>> &ROOT::Experimental::RCanvas::GetCanvases()
{
   return GetHeldCanvases();
}

// void ROOT::Experimental::RCanvas::Paint() {
//  for (auto&& drw: fPrimitives) {
//    drw->Paint(*this);
//  }
// }

///////////////////////////////////////////////////////////////////////////////////////
/// Generates unique ID inside the canvas

std::string ROOT::Experimental::RCanvas::GenerateUniqueId()
{
   return std::to_string(fIdCounter++);
}

///////////////////////////////////////////////////////////////////////////////////////
/// Returns true is canvas was modified since last painting

bool ROOT::Experimental::RCanvas::IsModified() const
{
   return fPainter ? fPainter->IsCanvasModified(fModified) : fModified;
}

void ROOT::Experimental::RCanvas::Update(bool async, CanvasCallback_t callback)
{
   if (fPainter)
      fPainter->CanvasUpdated(fModified, async, callback);

   // SnapshotList_t lst;
   // for (auto&& drw: fPrimitives) {
   //   TSnapshot *snap = drw->CreateSnapshot(*this);
   //   lst.push_back(std::unique_ptr<TSnapshot>(snap));
   // }
}

std::shared_ptr<ROOT::Experimental::RCanvas> ROOT::Experimental::RCanvas::Create(const std::string &title)
{
   auto pCanvas = std::make_shared<RCanvas>();
   pCanvas->SetTitle(title);
   GetHeldCanvases().emplace_back(pCanvas);
   return pCanvas;
}

//////////////////////////////////////////////////////////////////////////
/// Create new display for the canvas
/// Parameter \par where specifies which program could be used for display creation
/// Possible values:
///
///      cef - Chromium Embeded Framework, local display, local communication
///      qt5 - Qt5 WebEngine (when running via rootqt5), local display, local communication
///  browser - default system web-browser, communication via random http port from range 8800 - 9800
///  <prog> - any program name which will be started instead of default browser, like firefox or /usr/bin/opera
///           one could also specify $url in program name, which will be replaced with canvas URL
///  native - either any available local display or default browser
///
///  Canvas can be displayed in several different places

void ROOT::Experimental::RCanvas::Show(const std::string &where)
{
   if (fPainter) {
      bool isany = (fPainter->NumDisplays() > 0);

      if (!where.empty())
         fPainter->NewDisplay(where);

      if (isany) return;
   }

   if (!fModified)
      fModified = 1; // 0 is special value, means no changes and no drawings

   if (!fPainter)
      fPainter = Internal::RVirtualCanvasPainter::Create(*this);

   if (fPainter) {
      fPainter->NewDisplay(where);
      fPainter->CanvasUpdated(fModified, true, nullptr); // trigger async display
   }
}

//////////////////////////////////////////////////////////////////////////
/// Close all canvas displays

void ROOT::Experimental::RCanvas::Hide()
{
   if (fPainter)
      delete fPainter.release();
}

//////////////////////////////////////////////////////////////////////////
/// Create image file for the canvas
/// Supported SVG (extension .svg), JPEG (extension .jpg or .jpeg) and PNG (extension .png)
/// \param async specifies if file can be created asynchronous to the caller thread
/// When operation completed, callback function is called

void ROOT::Experimental::RCanvas::SaveAs(const std::string &filename, bool async, CanvasCallback_t callback)
{

   if (filename.find(".json") != std::string::npos) {
      if (!fPainter) fPainter = Internal::RVirtualCanvasPainter::Create(*this);
      fPainter->DoWhenReady("JSON", filename, async, callback);
      return;
   }

   if (!fPainter || (fPainter->NumDisplays()==0))
      Show("batch_canvas");

   if (filename.find(".svg") != std::string::npos)
      fPainter->DoWhenReady("SVG", filename, async, callback);
   else if (filename.find(".png") != std::string::npos)
      fPainter->DoWhenReady("PNG", filename, async, callback);
   else if ((filename.find(".jpg") != std::string::npos) || (filename.find(".jpeg") != std::string::npos))
      fPainter->DoWhenReady("JPEG", filename, async, callback);
}

// TODO: removal from GetHeldCanvases().
