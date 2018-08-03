// @(#)root/tmva/tmva/cnn:$Id$
// Author: Manos Stergiadis

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
 *      Manos Stergiadis       <em.stergiadis@gmail.com>  - CERN, Switzerland     *
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
// Testing the Convolutional Layer                                //
////////////////////////////////////////////////////////////////////

#include <iostream>
#include <cmath>

#include "TestConvNet.h"

using namespace TMVA::DNN;
using namespace TMVA::DNN::CNN;

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

template<typename AFloat>
bool almostEqual(AFloat expected, AFloat computed, double epsilon = 0.0001) {
    return abs(computed - expected) < epsilon;
}

/*************************************************************************
 * Test 1: Forward Propagation
 *  batch size = 1
 *  image depth = 2, image height = 4, image width = 4,
 *  num frames = 3, filter height = 2, filter width = 2,
 *  stride rows = 2, stride cols = 2,
 *  zero-padding height = 0, zero-padding width = 0,
 *************************************************************************/
template<typename Architecture>
bool testForward1()
{
   using Matrix_t = typename Architecture::Matrix_t;
   double img[][16] = {
           {166, 212, 213, 150,
            114, 119, 109, 115,
             88, 144, 227, 208,
            208, 235,  57,  58},

           { 57,  255, 184, 162,
            204,  220,  11, 192,
            183,  174,   2, 153,
            184,  175,  10,  55}
   };

   double weights[][8] = {
           {2.0,  3.0,  0.5, -1.5,
            1.0,  1.5, -2.0, -3.0},

           {-0.5,  1.0,  2.5, -1.0,
             2.0,  1.5, -0.5,  1.0},

           {-1.0, -2.0, 1.5, 0.5,
             2.0, -1.5, 0.5, 1.0}
   };

   double biases[][1] = {
           {45},

           {60},

           {12}
   };

   double expected[][9] = {

           {263.0, 1062.0,  632.0,
            104.0,  224.0,  245.5,
            -44.5,  843.0, 1111.0},

           { 969.5, 1042.5, 1058.5,
            1018.5,  614.0,  942.0,
            1155.0, 1019.0,  522.5},

           {-294.0, -38.0,   42.5,
             207.5, 517.0,    5.5,
             437.5, 237.5, -682.0}
    };



   size_t imgDepth = 2;
   size_t imgHeight = 4;
   size_t imgWidth = 4;
   size_t numberFilters = 3;
   size_t fltHeight = 2;
   size_t fltWidth = 2;
   size_t strideRows = 1;
   size_t strideCols = 1;
   size_t zeroPaddingHeight = 0;
   size_t zeroPaddingWidth = 0;

   Matrix_t inputEvent(imgDepth, imgHeight * imgWidth);

   for (size_t i = 0; i < imgDepth; i++) {
      for (size_t j = 0; j < imgHeight * imgWidth; j++) {
         inputEvent(i, j) = img[i][j];
      }
   }
   std::vector<Matrix_t> input;
   input.push_back(inputEvent);

   Matrix_t weightsMatrix(numberFilters, fltHeight * fltWidth * imgDepth);
   Matrix_t biasesMatrix(numberFilters, 1);
   for (size_t i = 0; i < numberFilters; i++) {
       for (size_t j = 0; j < fltHeight * fltWidth * imgDepth; j++){
           weightsMatrix(i, j) = weights[i][j];
       }
       biasesMatrix(i, 0) = biases[i][0];
   }

   size_t height = calculateDimension(imgHeight, fltHeight, zeroPaddingHeight, strideRows);
   size_t width = calculateDimension(imgWidth, fltWidth, zeroPaddingWidth, strideCols);

   Matrix_t outputEvent(numberFilters, height * width);

   for (size_t i = 0; i < numberFilters; i++) {
      for (size_t j = 0; j < height * width; j++) {
         outputEvent(i, j) = expected[i][j];
      }
   }
   std::vector<Matrix_t> expectedOutput;
   expectedOutput.push_back(outputEvent);

   bool status = testConvLayerForward<Architecture>(input, expectedOutput, weightsMatrix, biasesMatrix, imgHeight,
                                                    imgWidth, imgDepth, fltHeight, fltWidth, numberFilters, strideRows,
                                                    strideCols, zeroPaddingHeight, zeroPaddingWidth);

   return status;
}

/*************************************************************************
* Test 1: Backward Propagation
*  batch size = 1
*  image depth = 2, image height = 5, image width = 5,
*  num frames = 2, filter height = 3, filter width = 3,
*  stride rows = 1, stride cols = 1,
*  zero-padding height = 0, zero-padding width = 0,
*************************************************************************/
template<typename Architecture>
bool testBackward1()
{
    using Matrix_t = typename Architecture::Matrix_t;

    size_t imgDepth = 2;
    size_t imgHeight = 5;
    size_t imgWidth = 5;
    size_t numberFilters = 2;
    size_t fltHeight = 3;
    size_t fltWidth = 3;
    size_t strideRows = 1;
    size_t strideCols = 1;
    size_t zeroPaddingHeight = 0;
    size_t zeroPaddingWidth = 0;

    size_t height = calculateDimension(imgHeight, fltHeight, zeroPaddingHeight, strideRows);
    size_t width = calculateDimension(imgWidth, fltWidth, zeroPaddingWidth, strideCols);
    size_t nLocalViews =  height * width;
    size_t batchSize = 1;

    double grad[][9] = {
            {0, 1.37, 0, 0, 0, 0, 0, -0.90, 0},
            {0, -0.37, 0, 0, 0, -0.25, 0.26, 0, 0}
    };

    Matrix_t gradEvent(numberFilters, height * width);

    for (size_t i = 0; i < numberFilters; i++) {
        for (size_t j = 0; j < height * width; j++) {
            gradEvent(i, j) = grad[i][j];
        }
    }
    std::vector<Matrix_t> activationGradients;
    activationGradients.push_back(gradEvent);

    double derivatives[][9] = {
            {1, 1, 1, 1 , 1, 1, 1, 1, 1},
            {1, 1, 1, 1 , 1, 1, 1, 1, 1}
    };

    Matrix_t dfEvent(numberFilters, height * width);

    for (size_t i = 0; i < numberFilters; i++) {
        for (size_t j = 0; j < height * width; j++) {
            dfEvent(i, j) = derivatives[i][j];
        }
    }
    std::vector<Matrix_t> df;
    df.push_back(dfEvent);

    double W[][18] = {
            {1, 0.31, -0.35, -0.33, 0.40, 0.26, -0.30, 0.29, -0.31,
             0.21, 0.44, -0.36, 0.03, -0.27, -0.53, 0.24, 0.22, -0.35},

            {-0.06, -0.35, 0.10, -0.49, -0.88, 0.35, 0.03, 0, -0.19,
             -0.09, 0.29, 0.10, -0.15, -0.11, 0.02, 0.08, 0.17, 0.35}
    };

    Matrix_t weights(numberFilters, imgDepth * fltHeight * fltWidth);

    for (size_t i = 0; i < numberFilters; i++) {
        for (size_t j = 0; j < imgDepth * fltHeight * fltWidth; j++) {
            weights(i, j) = W[i][j];
        }
    }

    double activationsPreviousLayer[][25] = {
            {-0.33, -0.05, -0.34, -0.15, -0.13, 0.44, 0.51, 0.26, 1.21,
             -0.11, 0.86, 0, -0.56, -1.11, 1.22, -0.02, 0.29, 2.32,
             1.87, -0.31, -0.08, -0.20, -1.62, -1.40, -0.80},

            {0.42, -1.12, 1.10, -0.73, 0.38, 0.66, -2.36, 1.13, -1.19, 0.39,
             0.28, 0.04, 1.01, 0.31, -1.87, -0.09, 0.21, -2.51, 0.11, 0.84,
             -0.04, 0.56, 0.48, 0.09, -0.08}
    };

    Matrix_t activationsBackwardEvent(imgDepth, imgHeight * imgWidth);

    for (size_t i = 0; i < imgDepth; i++) {
        for (size_t j = 0; j < imgHeight * imgWidth; j++) {
            activationsBackwardEvent(i, j) = activationsPreviousLayer[i][j];
        }
    }
    std::vector<Matrix_t> activationsBackward;
    activationsBackward.push_back(activationsBackwardEvent);

    /////////////////////// Fill the expected output //////////////////////////
    double expectedActivationGradsBackward[][25] = {
            {0, 1.39, 0.55, -0.51, 0, 0, -0.27, 0.88,  0.31,  -0.02, -0.01, -1.41, 0.26,  0.18,  -0.08, -0.12, 0.06,
             -0.27, -0.23, 0.04,  0,    0.27,  -0.31, 0.27, 0},

            {0, 0.32, 0.49, -0.53, 0, 0, 0.09,  -0.30, -0.80, -0.02, -0.02, 0.18,  -0.09, -0.25, 0,     -0.03, -0.05,
             0.22,  0.43,  -0.08, 0.02, -0.17, -0.10, 0.31, 0}
    };

    Matrix_t expectedActivationGradientsBackwardEvent(imgDepth, imgHeight * imgWidth);

    for (size_t i = 0; i < imgDepth; i++) {
        for (size_t j = 0; j < imgHeight * imgWidth; j++) {
            expectedActivationGradientsBackwardEvent(i, j) = expectedActivationGradsBackward[i][j];
        }
    }

    std::vector<Matrix_t> computedActivationGradientsBackward;

    /////////////////////// Fill the expected weights gradients //////////////////////////
    double expectedWeightGrads[][18] = {
            {-0.06, 0.03, 0.79, 0.43, -1.73, -0.02, 0.18, 0.69, -0.26, -1.57, 0.59, -1.27, -3.42, 3.80,
             -1.72, -0.44, 0.95, 0.34},

            {0.17, -0.17, -0.06, -0.05, 0.25, -0.14, -0.60, -0.31, 0.06, 0.20, -0.09, 0.43, 0.59, -0.44,
             0.25, 0.60, -0.25, -0.19}
    };

    Matrix_t expectedWeightGradients(numberFilters, imgDepth * fltHeight * fltWidth);
    for (size_t i = 0; i < numberFilters; i++) {
        for (size_t j = 0; j < imgDepth * fltHeight * fltWidth; j++) {
            expectedWeightGradients(i, j) = expectedWeightGrads[i][j];
        }
    }

    /////////////////////// Fill the expected bias gradients //////////////////////////
    double expectedBiasGrads[][1] = {
            {0.47},
            {-0.36}
    };

    Matrix_t expectedBiasGradients(imgDepth, 1);
    for (size_t i = 0; i < imgDepth; i++) {
        expectedBiasGradients(i, 0) = expectedBiasGrads[i][0];
    }


    // Init outputs - these should be filled by the computation.
    computedActivationGradientsBackward.emplace_back(imgDepth, imgHeight * imgWidth);
    Matrix_t computedWeightGradients(numberFilters, imgDepth * fltHeight * fltWidth);
    Matrix_t computedBiasGradients(numberFilters, 1);

    Architecture::ConvLayerBackward(computedActivationGradientsBackward, computedWeightGradients, computedBiasGradients,
                                    df, activationGradients, weights, activationsBackward,
                                    batchSize, imgHeight, imgWidth, numberFilters, height,
                                    width, imgDepth, fltHeight, fltWidth, nLocalViews);


    // Check correctness.
    bool status = true;
    status &= Architecture::AlmostEquals(expectedActivationGradientsBackwardEvent, computedActivationGradientsBackward[0]);
    status &= Architecture::AlmostEquals(expectedWeightGradients, computedWeightGradients);
    status &= Architecture::AlmostEquals(expectedWeightGradients, computedWeightGradients);
    return status;
};
