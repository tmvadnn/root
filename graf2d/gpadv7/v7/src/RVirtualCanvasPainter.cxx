/// \file RVirtualCanvasPainter.cxx
/// \ingroup Gpad ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2017-05-31
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!


/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RVirtualCanvasPainter.hxx>

#include <ROOT/TLogger.hxx>
#include <TSystem.h> // TSystem::Load

#include <exception>

namespace {
static void LoadCanvasPainterLibrary() {
  if (gSystem->Load("libROOTCanvasPainter") != 0)
    R__ERROR_HERE("Gpad") << "Loading of libROOTCanvasPainter failed!";
}
} // unnamed namespace


/// The implementation is here to pin the vtable.
ROOT::Experimental::Internal::RVirtualCanvasPainter::~RVirtualCanvasPainter() = default;

std::unique_ptr<ROOT::Experimental::Internal::RVirtualCanvasPainter::Generator>
   &ROOT::Experimental::Internal::RVirtualCanvasPainter::GetGenerator()
{
   /// The generator for implementations.
   static std::unique_ptr<Generator> generator;
   return generator;
}

std::unique_ptr<ROOT::Experimental::Internal::RVirtualCanvasPainter> ROOT::Experimental::Internal::
   RVirtualCanvasPainter::Create(const RCanvas &canv)
{
   if (!GetGenerator()) {
      LoadCanvasPainterLibrary();
      if (!GetGenerator()) {
         R__ERROR_HERE("Gpad") << "RVirtualCanvasPainter::Generator failed to register!";
         throw std::runtime_error("RVirtualCanvasPainter::Generator failed to initialize");
      }
   }
   return GetGenerator()->Create(canv);
}

/// The implementation is here to pin the vtable.
ROOT::Experimental::Internal::RVirtualCanvasPainter::Generator::~Generator() = default;

