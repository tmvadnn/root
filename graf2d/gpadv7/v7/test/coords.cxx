// @(#)root/graf2d:$Id$
// Author: Axel Naumann <axel@cern.ch>, 2017-08-26

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

 #include "gtest/gtest.h"

#include "ROOT/RPadLength.hxx"

// Test addition / subtraction of coords
TEST(PadCoord, AddSubtract) {
   using namespace ROOT::Experimental;

   RPadLength cn{0.3_normal};
   EXPECT_DOUBLE_EQ(0.3, cn.fNormal.fVal);
   EXPECT_DOUBLE_EQ(0., cn.fPixel.fVal);
   EXPECT_DOUBLE_EQ(0., cn.fUser.fVal);

   RPadLength cn1{0.4_normal};
   cn += cn1;
   EXPECT_DOUBLE_EQ(0.7, cn.fNormal.fVal);
   EXPECT_DOUBLE_EQ(0., cn.fPixel.fVal);
   EXPECT_DOUBLE_EQ(0., cn.fUser.fVal);

   RPadLength cp{120_px};
   EXPECT_DOUBLE_EQ(120., cp.fPixel.fVal);
   EXPECT_DOUBLE_EQ(0., cp.fNormal.fVal);
   EXPECT_DOUBLE_EQ(0., cp.fUser.fVal);

   RPadLength sum = cn + cp;
   EXPECT_DOUBLE_EQ(0.7, sum.fNormal.fVal);
   EXPECT_DOUBLE_EQ(120., sum.fPixel.fVal);
   EXPECT_DOUBLE_EQ(0., sum.fUser.fVal);

   sum -= RPadLength(0.2_user);
   EXPECT_DOUBLE_EQ(0.7, sum.fNormal.fVal);
   EXPECT_DOUBLE_EQ(120., sum.fPixel.fVal);
   EXPECT_DOUBLE_EQ(-0.2, sum.fUser.fVal);

   sum *= 0.1;
   EXPECT_DOUBLE_EQ(0.07, sum.fNormal.fVal);
   EXPECT_DOUBLE_EQ(12., sum.fPixel.fVal);
   EXPECT_DOUBLE_EQ(-0.02, sum.fUser.fVal);

   RPadLength subtr(0.07_normal, 12_px, -0.02_user);
   EXPECT_DOUBLE_EQ(0.07, subtr.fNormal.fVal);
   EXPECT_DOUBLE_EQ(12., subtr.fPixel.fVal);
   EXPECT_DOUBLE_EQ(-0.02, subtr.fUser.fVal);

   sum -= subtr;
   EXPECT_NEAR(0., sum.fNormal.fVal, 1e-10);
   EXPECT_NEAR(0., sum.fPixel.fVal, 1e-10);
   EXPECT_NEAR(0., sum.fUser.fVal, 1e-10);
}
