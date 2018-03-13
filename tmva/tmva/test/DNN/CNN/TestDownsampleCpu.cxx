// @(#)root/tmva/tmva/cnn:$Id$
// Author: Vladimir Ilievski

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  :                                                                       *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Testing Downsample method on a CPU architecture                           *
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
// Testing the Downsample function                                //
////////////////////////////////////////////////////////////////////

#include <iostream>
#include <cmath>

#include "TMVA/DNN/Architectures/Cpu.h"
#include "TestConvNet.h"

using namespace TMVA::DNN;
using namespace TMVA::DNN::CNN;
using Matrix_t = typename TCpu<double>::Matrix_t;

inline bool isInteger(double x)
{
   return x == floor(x);
}

size_t calculateDimension(size_t imgDim, size_t fltDim, size_t padding, size_t stride)
{
   double dimension = ((imgDim - fltDim + 2 * padding) / stride) + 1;
   if (!isInteger(dimension)) {
      std::cout << "Not compatible hyper parameters" << std::endl;
      std::exit(EXIT_FAILURE);
   }

   return (size_t)dimension;
}

/*************************************************************************
 * Test 1:
 *  depth = 2, image height = 4, image width = 5,
 *  frame depth = 2, filter height = 2, filter width = 2,
 *  stride rows = 2, stride cols = 1,
 *  zero-padding height = 0, zero-padding width = 0,
 *************************************************************************/

void testMaxPooling1()
{

   double imgTest1[][20] = {
      {166, 212, 213, 150, 114, 119, 109, 115, 88, 144, 227, 208, 208, 235, 57, 57, 165, 250, 139, 76},

      {57, 255, 184, 162, 204, 220, 11, 192, 183, 174, 2, 153, 183, 175, 10, 55, 123, 246, 138, 80}};

   double answerTest1[][8] = {{212, 213, 213, 150, 227, 250, 250, 235},

                              {255, 255, 192, 204, 153, 246, 246, 175}};

   double answerIdxTest1[][8] = {{1, 2, 2, 3, 10, 17, 17, 13},

                                 {1, 1, 7, 4, 11, 17, 17, 13}};

   size_t imgDepthTest1 = 2;
   size_t imgHeightTest1 = 4;
   size_t imgWidthTest1 = 5;
   size_t fltHeightTest1 = 2;
   size_t fltWidthTest1 = 2;
   size_t strideRowsTest1 = 2;
   size_t strideColsTest1 = 1;

   Matrix_t A(imgDepthTest1, imgHeightTest1 * imgWidthTest1);

   for (size_t i = 0; i < (size_t)A.GetNrows(); i++) {
      for (size_t j = 0; j < (size_t)A.GetNcols(); j++) {
         A(i, j) = imgTest1[i][j];
      }
   }

   size_t height = calculateDimension(imgHeightTest1, fltHeightTest1, 0, strideRowsTest1);

   size_t width = calculateDimension(imgWidthTest1, fltWidthTest1, 0, strideColsTest1);

   CNN::TPoolLayer<TCpu<double>> layer = CNN::TPoolLayer<TCpu<double>>(1, imgDepthTest1, imgHeightTest1, imgWidthTest1,
                                                                       height, width, 1, imgDepthTest1, height * width,
                                                                       fltHeightTest1, fltWidthTest1, strideRowsTest1,
                                                                       strideColsTest1, 1.0, "max");

   Matrix_t idx(imgDepthTest1, height * width);
   Matrix_t B(imgDepthTest1, height * width);

   for (size_t i = 0; i < (size_t)B.GetNrows(); i++) {
      for (size_t j = 0; j < (size_t)B.GetNcols(); j++) {
         idx(i, j) = answerIdxTest1[i][j];
         B(i, j) = answerTest1[i][j];
      }
   }

   bool outputStatus = testDownsampleOutput<TCpu<double>>(A, B, layer);
   bool indexStatus = testDownsampleIndex<TCpu<double>>(A, idx, layer);

   if(outputStatus && indexStatus)
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

void testMaxPooling2()
{

   double imgTest2[][36] = {{200, 79, 69,  58,  98,  168, 49,  230, 21,  141, 218, 38, 72, 224, 14,  65,  147, 105,
                             38,  27, 111, 160, 200, 48,  109, 104, 153, 149, 233, 11, 16, 91,  236, 183, 166, 155}};

   double answerTest2[][10] = {{230, 218, 230, 218, 224, 200, 153, 233, 236, 233}};

   double answerIdxTest2[][10] = {{7, 10, 7, 10, 13, 22, 26, 28, 32, 28}};

   size_t imgDepthTest2 = 1;
   size_t imgHeightTest2 = 6;
   size_t imgWidthTest2 = 6;
   size_t fltHeightTest2 = 2;
   size_t fltWidthTest2 = 3;
   size_t strideRowsTest2 = 1;
   size_t strideColsTest2 = 3;

   Matrix_t A(imgDepthTest2, imgHeightTest2 * imgWidthTest2);

   for (size_t i = 0; i < (size_t)A.GetNrows(); i++) {
      for (size_t j = 0; j < (size_t)A.GetNcols(); j++) {
         A(i, j) = imgTest2[i][j];
      }
   }

   size_t height = calculateDimension(imgHeightTest2, fltHeightTest2, 0, strideRowsTest2);

   size_t width = calculateDimension(imgWidthTest2, fltWidthTest2, 0, strideColsTest2);

   CNN::TPoolLayer<TCpu<double>> layer = CNN::TPoolLayer<TCpu<double>>(1, imgDepthTest2, imgHeightTest2, imgWidthTest2,
                              height, width, 1, imgDepthTest2,
                              height * width, fltHeightTest2,
                              fltWidthTest2, strideRowsTest2,
                              strideColsTest2, 1.0, "max");

   Matrix_t idx(imgDepthTest2, height * width);
   Matrix_t B(imgDepthTest2, height * width);

   for (size_t i = 0; i < (size_t)B.GetNrows(); i++) {
      for (size_t j = 0; j < (size_t)B.GetNcols(); j++) {
         idx(i, j) = answerIdxTest2[i][j];
         B(i, j) = answerTest2[i][j];
      }
   }

   bool outputStatus = testDownsampleOutput<TCpu<double>>(A, B, layer);
   bool indexStatus = testDownsampleIndex<TCpu<double>>(A, idx, layer);

   if(outputStatus && indexStatus)
      std::cout << "Test passed!" << std::endl;
   else
      std::cout << "Test not passed!" << std::endl;
}

void testAveragePooling1()
{

   double input[][36] =
      {
         {200,  79,  69,  58,  98, 168,
           49, 230,  21, 141, 218,  38,
           72, 224,  14,  65, 147, 105,
           38,  27, 111, 160, 200,  48,
          109, 104, 153, 149, 233,  11,
           16,  91, 236, 183, 166, 155}
      };


   double output[][10] =
      {
         {108.000, 120.167,
          101.667, 119.000,
           81.000, 120.833,
           90.333, 133.500,
          118.167, 149.500}
      };

   size_t depth = 1;
   size_t inputHeight = 6;
   size_t inputWidth = 6;
   size_t frameHeight = 2;
   size_t frameWidth = 3;
   size_t strideRows = 1;
   size_t strideCols = 3;


   Matrix_t A(depth, inputHeight * inputWidth);

   for(size_t i = 0; i < (size_t) A.GetNrows(); i++){
      for(size_t j = 0; j < (size_t) A.GetNcols(); j++){
         A(i, j) = input[i][j];
      }
   }

   size_t height = calculateDimension(inputHeight, frameHeight, 0, strideRows);

   size_t width = calculateDimension(inputWidth, frameWidth, 0, strideCols);


   CNN::TPoolLayer<TCpu<double>> layer = CNN::TPoolLayer<TCpu<double>>(1, depth, inputHeight,
         inputWidth, height, width,
         1, depth, height * width,
         frameHeight, frameWidth,
         strideRows, strideCols,
         1.0, "avg");


   Matrix_t B(depth, height * width);

   for(size_t i = 0; i < (size_t)B.GetNrows(); i++){
      for(size_t j = 0; j < (size_t)B.GetNcols(); j++){
         B(i, j) = output[i][j];
      }
   }

   bool status = testDownsampleOutput<TCpu<double>>(A, B, layer);

   if(status)
      std::cout << "Test passed!" << std::endl;
   else
      std::cout << "Test not passed!" << std::endl;
}

void testPoolingBackward()
{

   /* Activations of the previous layer. These will be computed by the backward pass. */
   double expected[][36] =
      {
         {2.5,  2.5,  2.5,  -1.1,  -1.1, -1.1,
          3.5,  3.5,  3.5,  -0.6,  -0.6, -0.6,
          3.0,  3.0,  3.0,  -0.5,  -0.5, -0.5,
          5.5,  5.5,  5.5,  -2.5,  -2.5, -2.5,
          3.0,  3.0,  3.0,   2.0,   2.0,  2.0,
         -0.5, -0.5, -0.5,   3.5,   3.5,  3.5}
      };

   /* Activation gradients, coming from the next layer. These will be back-propagated. */
   double next[][10] =
      {
         { 15.0, -6.6,
            6.0,  3.0,
           12.0, -6.0,
           21.0, -9.0,
           -3.0, 21.0}
      };

   size_t depth = 1;
   size_t inHeight = 6;
   size_t inWidth = 6;
   size_t frameHeight = 2;
   size_t frameWidth = 3;
   size_t strideRows = 1;
   size_t strideCols = 3;

   std::cout << "Filling expected matrix A" << std::endl;
   Matrix_t A(depth, inHeight * inWidth);

   for(size_t i = 0; i < (size_t) A.GetNrows(); i++){
      for(size_t j = 0; j < (size_t) A.GetNcols(); j++){
         A(i, j) = expected[i][j];
      }
   }

   size_t height = calculateDimension(inHeight, frameHeight, 0, strideRows);

   size_t width = calculateDimension(inWidth, frameWidth, 0, strideCols);

   CNN::TPoolLayer<TCpu<double>> layer = CNN::TPoolLayer<TCpu<double>>(1, depth, inHeight, inWidth,
                                                                                   height, width, 1, depth,
                                                                                   height * width, frameHeight,
                                                                                   frameWidth, strideRows,
                                                                                   strideCols, 1.0, "avg");


   /* Fill the activation gradients */
   for(size_t d = 0; d < depth; d++) {
      for(size_t i = 0; i < height * width; i++) {
         layer.GetActivationGradients()[0](d, i) = next[d][i];
      }
   }

   bool status = testPoolingBackward<TCpu<double>>(A, layer);

   if(status)
      std::cout << "Test passed!" << std::endl;
   else
      std::cout << "Test not passed!" << std::endl;
}

int main()
{
   std::cout << "Testing Downsample function:" << std::endl;

   std::cout << "Test Max 1: " << std::endl;
   testMaxPooling1();

   std::cout << "Test Max 2: " << std::endl;
   testMaxPooling2();

   std::cout << "Test Average 1: " << std::endl;
   testAveragePooling1();

   std::cout << "Test Backward Pooling (average): " << std::endl;
   testPoolingBackward();
}
