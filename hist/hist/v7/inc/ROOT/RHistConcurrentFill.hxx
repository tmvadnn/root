/// \file ROOT/RHistConcurrentFill.h
/// \ingroup Hist ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2015-07-03
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RHistConcurrentFill
#define ROOT7_RHistConcurrentFill

#include "ROOT/RSpan.hxx"
#include "ROOT/RHistBufferedFill.hxx"

#include <mutex>

namespace ROOT {
namespace Experimental {

template <class HIST, int SIZE>
class RHistConcurrentFillManager;

/**
 \class RHistConcurrentFiller
 Buffers a thread's Fill calls and submits them to the
 RHistConcurrentFillManager. Enables multi-threaded filling.
 **/

template <class HIST, int SIZE>
class RHistConcurrentFiller: public Internal::RHistBufferedFillBase<RHistConcurrentFiller<HIST, SIZE>, HIST, SIZE> {
   RHistConcurrentFillManager<HIST, SIZE> &fManager;

public:
   using CoordArray_t = typename HIST::CoordArray_t;
   using Weight_t = typename HIST::Weight_t;

   RHistConcurrentFiller(RHistConcurrentFillManager<HIST, SIZE> &manager): fManager(manager) {}

   /// Thread-specific HIST::Fill().
   using Internal::RHistBufferedFillBase<RHistConcurrentFiller<HIST, SIZE>, HIST, SIZE>::Fill;

   /// Thread-specific HIST::FillN().
   void FillN(const std::span<CoordArray_t> xN, const std::span<Weight_t> weightN)
   {
      fManager.FillN(xN, weightN);
   }

   /// Thread-specific HIST::FillN().
   void FillN(const std::span<CoordArray_t> xN) { fManager.FillN(xN); }

   /// The buffer is full, flush it out.
   void Flush() { fManager.FillN(this->GetCoords(), this->GetWeights()); }

   HIST &GetHist() { return fManager->GetHist(); }
   operator HIST &() { return GetHist(); }

   static constexpr int GetNDim() { return HIST::GetNDim(); }
};

/**
 \class RHistConcurrentFillManager
 Manages the synchronization of calls to FillN().

 The HIST template can be a RHist instance. This class hands out
 RHistConcurrentFiller objects that can concurrently fill the histogram. They
 buffer calls to Fill() until the buffer is full, and then swap the buffer
 with that of the RHistConcurrentFillManager. The manager than fills the
 histogram.
 **/

template <class HIST, int SIZE = 1024>
class RHistConcurrentFillManager {
   friend class RHistConcurrentFiller<HIST, SIZE>;

public:
   using Hist_t = HIST;
   using CoordArray_t = typename HIST::CoordArray_t;
   using Weight_t = typename HIST::Weight_t;

private:
   HIST &fHist;
   std::mutex fFillMutex; // should become a spin lock

public:
   RHistConcurrentFillManager(HIST &hist): fHist(hist) {}

   RHistConcurrentFiller<HIST, SIZE> MakeFiller() { return RHistConcurrentFiller<HIST, SIZE>{*this}; }

   /// Thread-specific HIST::FillN().
   void FillN(const std::span<CoordArray_t> xN, const std::span<Weight_t> weightN)
   {
      std::lock_guard<std::mutex> lockGuard(fFillMutex);
      fHist.FillN(xN, weightN);
   }

   /// Thread-specific HIST::FillN().
   void FillN(const std::span<CoordArray_t> xN)
   {
      std::lock_guard<std::mutex> lockGuard(fFillMutex);
      fHist.FillN(xN);
   }
};

} // namespace Experimental
} // namespace ROOT

#endif
