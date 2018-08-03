/* @(#)root/hist:$Id$ */

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

// need to correctly generate dictionary for display items, used in v7 histpainter,
// but currently histpainter does not creates dictionary at all
#pragma extra_include "ROOT/RDisplayItem.hxx";

#pragma link C++ class ROOT::Experimental::RHistDrawingOpts<1>+;
#pragma link C++ class ROOT::Experimental::RHistDrawingOpts<2>+;
#pragma link C++ class ROOT::Experimental::RHistDrawingOpts<3>+;
#pragma link C++ class ROOT::Experimental::RHistDrawable<1>+;
#pragma link C++ class ROOT::Experimental::RHistDrawable<2>+;
#pragma link C++ class ROOT::Experimental::RHistDrawable<3>+;
#pragma link C++ class ROOT::Experimental::RDrawableBase<ROOT::Experimental::RHistDrawable<1>>+;
#pragma link C++ class ROOT::Experimental::RDrawableBase<ROOT::Experimental::RHistDrawable<2>>+;
#pragma link C++ class ROOT::Experimental::RDrawableBase<ROOT::Experimental::RHistDrawable<3>>+;
#pragma link C++ class ROOT::Experimental::RHistDrawableBase<ROOT::Experimental::RHistDrawable<1>>+;
#pragma link C++ class ROOT::Experimental::RHistDrawableBase<ROOT::Experimental::RHistDrawable<2>>+;
#pragma link C++ class ROOT::Experimental::RHistDrawableBase<ROOT::Experimental::RHistDrawable<3>>+;
#pragma link C++ class ROOT::Experimental::ROrdinaryDisplayItem<ROOT::Experimental::RHistDrawable<1>>+;
#pragma link C++ class ROOT::Experimental::ROrdinaryDisplayItem<ROOT::Experimental::RHistDrawable<2>>+;
#pragma link C++ class ROOT::Experimental::ROrdinaryDisplayItem<ROOT::Experimental::RHistDrawable<3>>+;
#pragma link C++ class ROOT::Experimental::Internal::TUniWeakPtr<ROOT::Experimental::Detail::RHistImplPrecisionAgnosticBase<1> >+;
#pragma link C++ class ROOT::Experimental::Internal::TUniWeakPtr<ROOT::Experimental::Detail::RHistImplPrecisionAgnosticBase<2>>+;
#pragma link C++ class ROOT::Experimental::Internal::TUniWeakPtr<ROOT::Experimental::Detail::RHistImplPrecisionAgnosticBase<3>>+;
#pragma link C++ class ROOT::Experimental::RStringEnumAttr<ROOT::Experimental::RHistDrawingOpts<1>::EStyle>+;
#pragma link C++ class ROOT::Experimental::RStringEnumAttr<ROOT::Experimental::RHistDrawingOpts<2>::EStyle>+;
#pragma link C++ class ROOT::Experimental::RStringEnumAttr<ROOT::Experimental::RHistDrawingOpts<3>::EStyle>+;
#pragma link C++ class ROOT::Experimental::RDrawingAttr<ROOT::Experimental::RStringEnumAttr<ROOT::Experimental::RHistDrawingOpts<1>::EStyle>>+;
#pragma link C++ class ROOT::Experimental::RDrawingAttr<ROOT::Experimental::RStringEnumAttr<ROOT::Experimental::RHistDrawingOpts<2>::EStyle>>+;
#pragma link C++ class ROOT::Experimental::RDrawingAttr<ROOT::Experimental::RStringEnumAttr<ROOT::Experimental::RHistDrawingOpts<3>::EStyle>>+;


#endif
