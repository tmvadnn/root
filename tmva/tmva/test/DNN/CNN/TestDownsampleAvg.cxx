// @(#)root/tmva/tmva/cnn:$Id$
// Author: Vladimir Ilievski

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  :                                                                       *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Testing Downsample method                                                 *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Vladimir Ilievski      <ilievski.vladimir@live.com>  - CERN, Switzerland  *
 *                                                                                *
 * Copyright (c) 2005-2015:                                                       *
 *      CERN, Switzerland                                                         *
 *      U. of Victoria, Canada                                                    *
 *      MPI-K Heidelberg, Germany                                                 *
 *      U. of Bonn, Germany                                                       *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/



////////////////////////////////////////////////////////////////////
// Testing the DownsampleAvg function                                //
////////////////////////////////////////////////////////////////////

#include <iostream>
#include <cmath>

#include "TMVA/DNN/Architectures/Reference.h"
#include "TestConvNet.h"

using namespace TMVA::DNN;
using namespace TMVA::DNN::CNN;
using Matrix_t = typename TReference<double>::Matrix_t;


inline bool isInteger(double x) {return x == floor(x);}

size_t calculateDimension(size_t imgDim,
                          size_t fltDim,
                          size_t padding,
                          size_t stride)
{
   double dimension = ((imgDim - fltDim + 2 * padding) / stride) + 1;
   if(!isInteger(dimension)) {
      std::cout << "Not compatible hyper parameters" << std::endl;
      std::exit(EXIT_FAILURE);
   }
    
   return (size_t) dimension;
}



/*************************************************************************
 * Test 1:
 *  depth = 2, image height = 4, image width = 5,
 *  frame depth = 2, filter height = 2, filter width = 2,
 *  stride rows = 2, stride cols = 1,
 *  zero-padding height = 0, zero-padding width = 0,
 *************************************************************************/

void test1()
{
    
   double imgTest1[][20] =
      {
        {166,  212,  213,  150,  114,
         119,  109,  115,   88,  144,
         227,  208,  208,  235,   57,
          57,  165,  250,  139,   76},
        
        { 57,  255,  184,  162,  204,
         220,   11,  192,  183,  174,
           2,  153,  183,  175,   10,
          55,  123,  246,  138,   80}
      };
    
    
   double answerTest1[][8] =
      {
        {151.5,  162.25,  141.5,  124,
         164.25,  207.75,  208,  126.75},
        
        {135.75,  160.5,  180.25,  180.75,
         83.25,  176.25,  185.5,  100.75}
      };

    
   size_t imgDepthTest1 = 2;
   size_t imgHeightTest1 = 4;
   size_t imgWidthTest1 = 5;
   size_t fltHeightTest1 = 2;
   size_t fltWidthTest1 = 2;
   size_t strideRowsTest1 = 2;
   size_t strideColsTest1 = 1;
    
    
   Matrix_t A(imgDepthTest1, imgHeightTest1 * imgWidthTest1);
    
   for(size_t i = 0; i < (size_t) A.GetNrows(); i++){
      for(size_t j = 0; j < (size_t) A.GetNcols(); j++){
         A(i, j) = imgTest1[i][j];
      }
   }
    
    
   size_t height = calculateDimension(imgHeightTest1, fltHeightTest1,
                                      0, strideRowsTest1);
    
   size_t width = calculateDimension(imgWidthTest1, fltWidthTest1,
                                     0, strideColsTest1);

    
   Matrix_t idx(imgDepthTest1,  height * width);
   Matrix_t B(imgDepthTest1, height * width);
    
   for(size_t i = 0; i < (size_t) B.GetNrows(); i++){
      for(size_t j = 0; j < (size_t) B.GetNcols(); j++){
         B(i, j) = answerTest1[i][j];
      }
   }
    
    
    
   bool status = testDownsampleAvg<TReference<double>>(A, B,
                                                    imgHeightTest1, imgWidthTest1,
                                                    fltHeightTest1, fltWidthTest1,
                                                    strideRowsTest1, strideColsTest1);

   if(status)
      std::cout << "Test passed!" << std::endl;
   else
      std::cout << "Test not passed!" << std::endl;
}

/*************************************************************************
 * Test 1:
 *  depth = 1, image height = 6, image width = 6,
 *  frame depth = 1, filter height = 2, filter width = 3,
 *  stride rows = 1, stride cols = 3,
 *  zero-padding height = 0, zero-padding width = 0,
 *************************************************************************/

void test2()
{
    
   double imgTest2[][36] =
      {
        {200,  79,  69,  58,  98, 168,
          49, 230,  21, 141, 218,  38,
          72, 224,  14,  65, 147, 105,
          38,  27, 111, 160, 200,  48,
         109, 104, 153, 149, 233,  11,
          16,  91, 236, 183, 166, 155}
      };
    
    
   double answerTest2[][10] =
      {
        {108, 120.167,
         101.667, 119,
         81, 120.833,
         90.333, 133.5,
         118.167, 149.5}
      };
    
   size_t imgDepthTest2 = 1;
   size_t imgHeightTest2 = 6;
   size_t imgWidthTest2 = 6;
   size_t fltHeightTest2 = 2;
   size_t fltWidthTest2 = 3;
   size_t strideRowsTest2 = 1;
   size_t strideColsTest2 = 3;
    
    
   Matrix_t A(imgDepthTest2, imgHeightTest2 * imgWidthTest2);
    
   for(size_t i = 0; i < (size_t) A.GetNrows(); i++){
      for(size_t j = 0; j < (size_t) A.GetNcols(); j++){
         A(i, j) = imgTest2[i][j];
      }
   }
    
    
   size_t height = calculateDimension(imgHeightTest2, fltHeightTest2,
                                      0, strideRowsTest2);
    
   size_t width = calculateDimension(imgWidthTest2, fltWidthTest2,
                                     0, strideColsTest2);
    
    
   Matrix_t idx(imgDepthTest2,  height * width);
   Matrix_t B(imgDepthTest2, height * width);
    
   for(size_t i = 0; i < (size_t) B.GetNrows(); i++){
      for(size_t j = 0; j < (size_t) B.GetNcols(); j++){
         B(i, j) = answerTest2[i][j];
      }
   }
    
    
    
   bool status = testDownsampleAvg<TReference<double>>(A, B,
                                                    imgHeightTest2, imgWidthTest2,
                                                    fltHeightTest2, fltWidthTest2,
                                                    strideRowsTest2, strideColsTest2);
    
   if(status)
      std::cout << "Test passed!" << std::endl;
   else
      std::cout << "Test not passed!" << std::endl;
}


int main(){
   std::cout << "Testing Downsample function:" << std::endl;
    
   std::cout << "Test 1: " << std::endl;
   test1();
    
   std::cout << "Test 2: " << std::endl;
   test2();
}
