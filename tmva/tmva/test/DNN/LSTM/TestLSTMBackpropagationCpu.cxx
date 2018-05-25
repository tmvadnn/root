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
#include "TMVA/DNN/Architectures/Cpu.h"
#include "TestLSTMBackpropagation.h"
#include "TROOT.h"

using namespace TMVA::DNN;
using namespace TMVA::DNN::LSTM;

int main() {
   std::cout << "Testing LSTM backward pass\n";
   
   using Scalar_t = Double_t;

   /*! timesteps, batchsize, statesize, inputsize */
   testLSTMBackpropagation<TCpu<Scalar_t>>(1, 2, 1, 2, 1e-5);

   /*testLSTMBackpropagation<TCpu<Scalar_t>>(1, 2, 3, 2, 1e-5); 

   testLSTMBackpropagation<TCpu<Scalar_t>>(2, 3, 4, 5, 1e-5);

   testLSTMBackpropagation<TCpu<Scalar_t>>(4, 2, 10, 5, 1e-5);

   testLSTMBackpropagation<TCpu<Scalar_t>>(5, 64, 10, 5, 1e-5);*/


   /* using a fixed input */
   //testLSTMBackpropagation<TCpu<Scalar_t>>(3, 1, 10, 5, 1e-5, {true});

   /* with a dense layer */
   //testLSTMBackpropagation<TCpu<Scalar_t>>(4, 32, 10, 20, 1e-5, {false, true});

   /* with an additional LSTM layer */
   //testLSTMBackpropagation<TCpu<Scalar_t>>(4, 32, 10, 5, 1e-5, {false, true, true}); 


   return 0;
}
