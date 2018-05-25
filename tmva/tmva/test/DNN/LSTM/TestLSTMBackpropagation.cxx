// @(#)root/tmva $Id$
// Author: Harshit Prasad 04/07/18

/*************************************************************************
 * Copyright (C) 2018, Harshit Prasad                                    *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////
// Testing LSTMLayer backpropagation                              //
////////////////////////////////////////////////////////////////////

#include <iostream>
#include "TMVA/DNN/Architectures/Reference.h"
#include "TestLSTMBackpropagation.h"

using namespace TMVA::DNN;
using namespace TMVA::DNN::LSTM;


int main() {
   std::cout << "Testing LSTM backward pass\n";

   /*! timesteps, batchsize, statesize, inputsize */
   testLSTMBackpropagation<TReference<double>>(1, 2, 1, 2, 1e-5);

   //testLSTMBackpropagation<TReference<double>>(2, 2, 1, 10, 1e-5);

   //testLSTMBackpropagation<TReference<double>>(1, 2, 2, 10, 1e-5);

   //testLSTMBackpropagation<TReference<double>>(2, 1, 2, 5, 1e-5);

   //testLSTMBackpropagation<TReference<double>>(4, 2, 3, 10, 1e-5);

   /*! using a fixed input */
   //testLSTMBackpropagation<TReference<double>>(3, 1, 4, 5, 1e-5, {true});

   /*! with a dense layer */
   //testLSTMBackpropagation<TReference<double>>(4, 32, 10, 5, 1e-5, {false, true});

   /*! with an additional LSTM layer */
   //testLSTMBackpropagation<TReference<double>>(4, 32, 10, 5, 1e-5, {false, true, true});

   return 0;
}
