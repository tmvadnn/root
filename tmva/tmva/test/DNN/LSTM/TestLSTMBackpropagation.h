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
// Generic tests of the LSTM Layer Backward pass                  //
////////////////////////////////////////////////////////////////////

#ifndef TMVA_TEST_DNN_TEST_LSTM_TEST_BWDPASS_H
#define TMVA_TEST_DNN_TEST_LSTM_TEST_BWDPASS_H

#include <iostream>
#include <vector>

#include "../Utility.h"
#include "Math/Functor.h"
#include "Math/RichardsonDerivator.h"

#include "TMVA/DNN/Functions.h"
#include "TMVA/DNN/DeepNet.h"

using namespace TMVA::DNN;
using namespace TMVA::DNN::LSTM;

//______________________________________________________________________________
template <typename Architecture>
auto printTensor(const std::vector<typename Architecture::Matrix_t> &A, const std::string name = "matrix")
-> void
{
  std::cout << name << "\n";
  for (size_t l = 0; l < A.size(); ++l) {
     for (size_t i = 0; i < (size_t) A[l].GetNrows(); ++i) {
        for (size_t j = 0; j < (size_t) A[l].GetNcols(); ++j) {
            std::cout << A[l](i, j) << " ";
        }
        std::cout << "\n";
      }
      std::cout << "********\n";
  } 
}
//______________________________________________________________________________
template <typename Architecture>
auto printMatrix(const typename Architecture::Matrix_t &A, const std::string name = "matrix")
-> void
{
  std::cout << name << "\n";
  for (size_t i = 0; i < (size_t) A.GetNrows(); ++i) {
    for (size_t j = 0; j < (size_t) A.GetNcols(); ++j) {
        std::cout << A(i, j) << " ";
    }
    std::cout << "\n";
  }
  std::cout << "********\n";
}

/*! Compute the loss of the net as a function of the weight at index (i,j) in
 *  layer l. dx is added as an offset to the current value of the weight. */
//______________________________________________________________________________
template <typename Architecture>
auto evaluate_net_weight(TDeepNet<Architecture> &net, std::vector<typename Architecture::Matrix_t> & X,
                         const typename Architecture::Matrix_t &Y, const typename Architecture::Matrix_t &W, size_t l,
                         size_t k, size_t i, size_t j, typename Architecture::Scalar_t xvalue) ->
   typename Architecture::Scalar_t
{
    using Scalar_t = typename Architecture::Scalar_t;

    Scalar_t prev_value = net.GetLayerAt(l)->GetWeightsAt(k).operator()(i,j);
    net.GetLayerAt(l)->GetWeightsAt(k).operator()(i,j) = xvalue;
    Scalar_t res = net.Loss(X, Y, W, false, false);
    net.GetLayerAt(l)->GetWeightsAt(k).operator()(i,j) = prev_value;
    //std::cout << "compute loss for weight  " << xvalue << "  " << prev_value << " result " << res << std::endl;
    return res;
}

/*! Compute the loss of the net as a function of the weight at index i in
 *  layer l. dx is added as an offset to the current value of the weight. */
//______________________________________________________________________________
template <typename Architecture>
auto evaluate_net_bias(TDeepNet<Architecture> &net, std::vector<typename Architecture::Matrix_t> & X,
                       const typename Architecture::Matrix_t &Y, const typename Architecture::Matrix_t &W, size_t l,
                       size_t k, size_t i, typename Architecture::Scalar_t xvalue) -> typename Architecture::Scalar_t
{
    using Scalar_t = typename Architecture::Scalar_t;
 
    Scalar_t prev_value = net.GetLayerAt(l)->GetBiasesAt(k).operator()(i,0);
    net.GetLayerAt(l)->GetBiasesAt(k).operator()(i,0) = xvalue;
    Scalar_t res = net.Loss(X, Y, W, false, false);
    net.GetLayerAt(l)->GetBiasesAt(k).operator()(i,0) = prev_value; 
    return res;
}

/* Generate a DeepNet, test backward pass. */
//______________________________________________________________________________
template <typename Architecture>
bool testLSTMBackpropagation(size_t timeSteps, size_t batchSize, size_t stateSize, 
                             size_t inputSize, typename Architecture::Scalar_t /*dx = 1.E-5*/,
                             std::vector<bool> options = {}, bool debug = false)
{
    bool failed = false;
    if (options.size() == 0) options = std::vector<bool>(4);
    bool randomInput = !options[0];
    bool addDenseLayer = options[1];
    bool addExtraLSTM = options[2];

    using Matrix_t = typename Architecture::Matrix_t;
    using Tensor_t = std::vector<Matrix_t>;
    using LSTMLayer_t = TBasicLSTMLayer<Architecture>;
    using DenseLayer_t = TDenseLayer<Architecture>;
    using Net_t = TDeepNet<Architecture>;
    using Scalar_t = typename Architecture::Scalar_t;

    // Defining inputs at each timestep.
    Tensor_t XArch;
    for (size_t i = 0; i < batchSize; ++i) {
        XArch.emplace_back(timeSteps, inputSize);
    }

    printTensor<Architecture>(XArch, "XArch Matrix");

    // Random inputs.
    if (randomInput) {
        for(size_t i = 0; i < batchSize; ++i) {
            for (size_t l = 0; l < (size_t) XArch[i].GetNrows(); ++l) {
                for (size_t m = 0; m < (size_t) XArch[i].GetNcols(); ++m) {
                    XArch[i](l, m) = gRandom->Uniform(-1,1);
                }
            }
        }
    } else {
        R__ASSERT(inputSize <= 6);
        R__ASSERT(timeSteps <= 3);
        R__ASSERT(batchSize <= 1);
        double xinput[] = { -1, 1, -2, 2, -3, 3,
                             -0.5, 0.5, -0.8, 0.9, -2, 1.5,
                             -0.2, 0.1, -0.5, 0.4, -1, 1 };
        TMatrixD Input(3, 6, xinput);
        for (size_t i = 0; i < batchSize; ++i) {
            auto &mat = XArch[i];
            // Timestep 0
            for (size_t l = 0; l < timeSteps; ++l) {
                for (size_t m = 0; m < timeSteps; ++m) {
                    mat(l,m) = Input(l,m);
                }
            }
        }
        gRandom->SetSeed(1); // For weights intialization.
    }

    if (debug) printTensor<Architecture>(XArch, "input");
    size_t outputSize = timeSteps * stateSize;
    
    if (addDenseLayer) outputSize = 1;

    Matrix_t Y(batchSize, outputSize), weights(batchSize, 1);
    // Initialize random matrix.
    for (size_t i = 0; i < (size_t) Y.GetNrows(); ++i) {
        for (size_t j = 0; j < (size_t) Y.GetNcols(); ++j) {
            Y(i,j) = gRandom->Integer(2);
        }
    }
    printMatrix<Architecture>(Y, "Matrix Y");
    // Initialize weight matrix
    fillMatrix(weights, 1.0);
    printMatrix<Architecture>(weights, "Weights Matrix");

    

    std::cout << "Testing weights backpropagation using LSTM with batchsize = " << batchSize << "input = " << inputSize << "state = " << stateSize << "time = " << timeSteps;
    if (randomInput) {
        std::cout << "\nusing a random input";
    } else {
        std::cout << "\nusing a fixed input";
    }
    if (addDenseLayer) {
        std::cout << "and a dense layer";
    }
    if (addExtraLSTM) {
        std::cout << "and a extra LSTM unit";
    }
    std::cout << std::endl;

    Net_t lstm(batchSize, batchSize, timeSteps, inputSize, 0, 0, 0, ELossFunction::kMeanSquaredError, EInitialization::kGauss);
    LSTMLayer_t* layer = lstm.AddBasicLSTMLayer(stateSize, inputSize, timeSteps);

    if (addExtraLSTM) lstm.AddBasicLSTMLayer(stateSize, stateSize, timeSteps);
    lstm.AddReshapeLayer(1, 1, timeSteps * stateSize, true); 

    DenseLayer_t *dlayer1 = nullptr;
    DenseLayer_t *dlayer2 = nullptr;
    if (addDenseLayer) {
        dlayer1 = lstm.AddDenseLayer(10, TMVA::DNN::EActivationFunction::kTanh);
        dlayer2 = lstm.AddDenseLayer(1, TMVA::DNN::EActivationFunction::kIdentity);
    }

    // Initialize layer.
    lstm.Initialize();

    /*! There will be 8 different weight matrices.
     *  1. 4 input weight matrices w.r.t each gate.
     *  2. 4 state weight matrices w.r.t each gate. */
    auto & weights_input = layer->GetWeightsInputGate();
    if (debug) printMatrix<Architecture>(weights_input, "Input Gate: input weights");
    #if 0
        for (int i = 0; i < stateSize; ++i) { 
            for (int j = 0; j < inputSize; ++j) { 
                weights_input(i,j) =  gRandom->Uniform(-1,1);
            }
            weights_input(i,i) = 1.0; 
        }
    #endif
    auto & weights_forget = layer->GetWeightsForgetGate();
    if (debug) printMatrix<Architecture>(weights_forget, "Forget Gate: input weights");
    #if 0
        for (int i = 0; i < stateSize; ++i) { 
            for (int j = 0; j < inputSize; ++j) { 
                weights_forget(i,j) =  gRandom->Uniform(-1,1);
            }
            weights_forget(i,i) = 1.0; 
        }
    #endif
    auto & weights_candidate = layer->GetWeightsCandidate();
    if (debug) printMatrix<Architecture>(weights_candidate, "Candidate Gate: input weights");
    #if 0
        for (int i = 0; i < stateSize; ++i) { 
            for (int j = 0; j < inputSize; ++j) { 
                weights_candidate(i,j) =  gRandom->Uniform(-1,1);
            }
            weights_candidate(i,i) = 1.0; 
        }
    #endif
    auto & weights_output = layer->GetWeightsOutputGate();
    if (debug) printMatrix<Architecture>(weights_output, "Output Gate: input weights");
    #if 0
        for (int i = 0; i < stateSize; ++i) { 
            for (int j = 0; j < inputSize; ++j) { 
                weights_output(i,j) =  gRandom->Uniform(-1,1);
            }
            weights_output(i,i) = 1.0; 
        }
    #endif
   
    auto & weights_input_state = layer->GetWeightsInputGateState();
    if (debug) printMatrix<Architecture>(weights_input_state, "Input Gate: state weights");
    #if 0
        for (int i = 0; i < stateSize; ++i) { 
            for (int j = 0; j < stateSize; ++j) { 
                weights_input_state(i,j) = gRandom->Uniform(-1,1);
            }
            weights_input_state(i,i) = 0.5; 
        }
    #endif
    auto & weights_forget_state = layer->GetWeightsForgetGateState();
    if (debug) printMatrix<Architecture>(weights_forget_state, "Forget Gate: state weights");
    #if 0
        for (int i = 0; i < stateSize; ++i) { 
            for (int j = 0; j < stateSize; ++j) { 
                weights_forget_state(i,j) = gRandom->Uniform(-1,1);
            }
            weights_forget_state(i,i) = 0.5; 
        }
    #endif
    auto & weights_candidate_state = layer->GetWeightsCandidateState();
    if (debug) printMatrix<Architecture>(weights_candidate_state, "Candidate Gate: state weights");
    #if 0
        for (int i = 0; i < stateSize; ++i) { 
            for (int j = 0; j < stateSize; ++j) { 
                weights_candidate_state(i,j) = gRandom->Uniform(-1,1);
            }
            weights_candidate_state(i,i) = 0.5; 
        }
    #endif
    auto & weights_output_state = layer->GetWeightsOutputGateState();
    if (debug) printMatrix<Architecture>(weights_output_state, "Output Gate: state weights");
    #if 0
        for (int i = 0; i < stateSize; ++i) { 
            for (int j = 0; j < stateSize; ++j) { 
                weights_output_state(i,j) = gRandom->Uniform(-1,1);
            }
            weights_output_state(i,i) = 0.5; 
        }
    #endif
    auto & input_bias = layer->GetInputGateBias();
    if (debug) input_bias.Print();
    #if 0
    for (int i = 0; i < (size_t) input_gate.GetNrows(); ++i) { 
        for (int j = 0; j < (size_t) input_gate.GetNcols(); ++j) { 
            input_gate(i,j) = gRandom->Uniform(-0.5,0.5);
        }
    }
    #endif
    auto & forget_bias = layer->GetForgetGateBias();
    if (debug) forget_bias.Print();
    #if 0
    for (int i = 0; i < (size_t) forget_gate.GetNrows(); ++i) { 
        for (int j = 0; j < (size_t) forget_gate.GetNcols(); ++j) { 
            forget_gate(i,j) = gRandom->Uniform(-0.5,0.5);
        }
    }
    #endif
    auto & candidate_bias = layer->GetCandidateBias();
    if (debug) candidate_bias.Print();
    #if 0
    for (int i = 0; i < (size_t) candidate_gate.GetNrows(); ++i) { 
        for (int j = 0; j < (size_t) candidate_gate.GetNcols(); ++j) { 
            candidate_gate(i,j) = gRandom->Uniform(-0.5,0.5);
        }
    }
    #endif
    auto & output_bias = layer->GetOutputGateBias();
    if (debug) output_bias.Print();
    #if 0
    for (int i = 0; i < (size_t) output_gate.GetNrows(); ++i) { 
        for (int j = 0; j < (size_t) output_gate.GetNcols(); ++j) { 
            output_gate(i,j) = gRandom->Uniform(-0.5,0.5);
        }
    }
    #endif

    lstm.Forward(XArch);

    /* TODO: Error needs to be fixed arising from Backward() method.
     * ERROR: For 'testLSTMBackpropagation' tests, this runs into infinite loop with error:
     * Error in <operator()>: Request column(188704) outside matrix range of 0 - 1
     * For 'testLSTMBackPropagationCpu', it works. */
    lstm.Backward(XArch, Y, weights);

    if (debug)  {
        auto & out1 = layer->GetOutput();
        printTensor<Architecture>(out1, "output");
        if (dlayer1) {
            auto & out2 = dlayer1->GetOutput();
            printTensor<Architecture>(out2, "dense layer1 output");
            auto & out3 = dlayer2->GetOutput();
            printTensor<Architecture>(out3, "dense layer2 output");
        }
    }

    std::cout << "Verify....Done!" << "\n";
    Double_t maximum_error = 0.0;
    std::string maxErrorType; 

    // We'll take first partial derivative. 
    ROOT::Math::RichardsonDerivator deriv;

    // Testing input gate: input weights
    auto &Wi = layer->GetWeightsAt(0);
    auto &dWi = layer->GetWeightGradientsAt(0);
    for (size_t i = 0; i < (size_t) Wi.GetNrows(); ++i) {
        for (size_t j = 0; j < (size_t) Wi.GetNcols(); ++j) {
            auto f = [&lstm, &XArch, &Y, &weights_input, i, j](Scalar_t x) {
                return evaluate_net_weight(lstm, XArch, Y, weights_input, 0, 0, i, j, x);
            };
            ROOT::Math::Functor1D func(f);
            double dy = deriv.Derivative1(func, Wi(i,j), 1.E-5);
            Scalar_t dy_ref = dWi(i,j);

            printMatrix<Architecture>(Wi, "dy/dx reference");

            // Compute relative error if dy != 0
            Scalar_t error;
            std::string errorType;
            if (std::fabs(dy_ref) > 1e-15) {
                error = std::fabs((dy - dy_ref) / dy_ref);
                errorType = "relative";
            } else {
                error = std::fabs(dy - dy_ref);
                errorType = "absolute";
            }

            if (debug) std::cout << "Input Gate: input weight gradients (" << i << "," << j << ") : (comp, ref) " << dy << ", " << dy_ref << std::endl;

            if (error >= maximum_error) {
                maximum_error = error;
                maxErrorType = errorType;
            }
        }
    }
    std::cout << "\rInput Gate: testing input weight gradients: ";
    std::cout << "Maximum error (" << maxErrorType << ") : " << print_error(maximum_error) << std::endl;
    if (maximum_error > 1.E-2) {
        std::cerr << "\e[31m Error \e[39m in input weight gradients of input gate" << std::endl;
        failed = true;
    }

    // Testing forget gate: input weights
    maximum_error = 0.0;
    auto &Wf = layer->GetWeightsAt(0);
    auto &dWf = layer->GetWeightGradientsAt(0);

    for (size_t i = 0; i < (size_t) Wf.GetNrows(); ++i) {
        for (size_t j = 0; j < (size_t) Wf.GetNcols(); ++j) {
            auto f = [&lstm, &XArch, &Y, &weights_forget, i, j](Scalar_t x) {
                return evaluate_net_weight(lstm, XArch, Y, weights_forget, 0, 0, i, j, x);
            };
            ROOT::Math::Functor1D func(f);
            double dy = deriv.Derivative1(func, Wf(i,j), 1.E-5);
            Scalar_t dy_ref = dWf(i,j);

            printMatrix<Architecture>(Wf, "dy/dx ref: forget gate { input weights } ");

            // Compute relative error if dy != 0
            Scalar_t error;
            std::string errorType;
            if (std::fabs(dy_ref) > 1e-15) {
                error = std::fabs((dy - dy_ref) / dy_ref);
                errorType = "relative";
            } else {
                error = std::fabs(dy - dy_ref);
                errorType = "absolute";
            }

            if (debug) std::cout << "Forget Gate: input weight gradients (" << i << "," << j << ") : (comp, ref) " << dy << ", " << dy_ref << std::endl;

            if (error >= maximum_error) {
                maximum_error = error;
                maxErrorType = errorType;
            }
        }
    }
    std::cout << "\rForget Gate: testing input weight gradients: ";
    std::cout << "Maximum error (" << maxErrorType << ") : " << print_error(maximum_error) << std::endl;
    if (maximum_error > 1.E-2) {
        std::cerr << "\e[31m Error \e[39m in input weight gradients of forget gate" << std::endl;
        failed = true;
    }

    // Testing candidate gate: input weights
    maximum_error = 0.0;
    auto &Wc = layer->GetWeightsAt(0);
    auto &dWc = layer->GetWeightGradientsAt(0);

    for (size_t i = 0; i < (size_t) Wc.GetNrows(); ++i) {
        for (size_t j = 0; j < (size_t) Wc.GetNcols(); ++j) {
            auto f = [&lstm, &XArch, &Y, &weights_candidate, i, j](Scalar_t x) {
                return evaluate_net_weight(lstm, XArch, Y, weights_candidate, 0, 0, i, j, x);
            };
            ROOT::Math::Functor1D func(f);
            double dy = deriv.Derivative1(func, Wc(i,j), 1.E-5);
            Scalar_t dy_ref = dWc(i,j);

            // Compute relative error if dy != 0
            Scalar_t error;
            std::string errorType;
            if (std::fabs(dy_ref) > 1e-15) {
                error = std::fabs((dy - dy_ref) / dy_ref);
                errorType = "relative";
            } else {
                error = std::fabs(dy - dy_ref);
                errorType = "absolute";
            }

            if (debug) std::cout << "Candidate Gate: input weight gradients (" << i << "," << j << ") : (comp, ref) " << dy << ", " << dy_ref << std::endl;

            if (error >= maximum_error) {
                maximum_error = error;
                maxErrorType = errorType;
            }
        }
    }
    std::cout << "\rCandidate Gate: testing input weight gradients: ";
    std::cout << "Maximum error (" << maxErrorType << ") : " << print_error(maximum_error) << std::endl;
    if (maximum_error > 1.E-2) {
        std::cerr << "\e[31m Error \e[39m in input weight gradients of candidate gate" << std::endl;
        failed = true;
    }
    
    // Testing output gate: input weights
    maximum_error = 0.0;
    auto &Wo = layer->GetWeightsAt(0);
    auto &dWo = layer->GetWeightGradientsAt(0);

    for (size_t i = 0; i < (size_t) Wo.GetNrows(); ++i) {
        for (size_t j = 0; j < (size_t) Wo.GetNcols(); ++j) {
            auto f = [&lstm, &XArch, &Y, &weights_output, i, j](Scalar_t x) {
                return evaluate_net_weight(lstm, XArch, Y, weights_output, 0, 0, i, j, x);
            };
            ROOT::Math::Functor1D func(f);
            double dy = deriv.Derivative1(func, Wo(i,j), 1.E-5);
            Scalar_t dy_ref = dWo(i,j);

            // Compute relative error if dy != 0
            Scalar_t error;
            std::string errorType;
            if (std::fabs(dy_ref) > 1e-15) {
                error = std::fabs((dy - dy_ref) / dy_ref);
                errorType = "relative";
            } else {
                error = std::fabs(dy - dy_ref);
                errorType = "absolute";
            }

            if (debug) std::cout << "Output Gate: input weight gradients (" << i << "," << j << ") : (comp, ref) " << dy << ", " << dy_ref << std::endl;

            if (error >= maximum_error) {
                maximum_error = error;
                maxErrorType = errorType;
            }
        }
    }
    std::cout << "\rOutput Gate: testing input weight gradients: ";
    std::cout << "Maximum error (" << maxErrorType << ") : " << print_error(maximum_error) << std::endl;
    if (maximum_error > 1.E-2) {
        std::cerr << "\e[31m Error \e[39m in input weight gradients of output gate" << std::endl;
        failed = true;
    }

    // Testing input gate: state weights gradients
    maximum_error = 0.0;
    auto &Wis = layer->GetWeightsAt(0);
    auto &dWis = layer->GetWeightGradientsAt(0);

    for (size_t i = 0; i < (size_t) Wis.GetNrows(); ++i) {
        for (size_t j = 0; j < (size_t) Wis.GetNcols(); ++j) {
            auto f = [&lstm, &XArch, &Y, &weights_input_state, i, j](Scalar_t x) {
                return evaluate_net_weight(lstm, XArch, Y, weights_input_state, 0, 0, i, j, x);
            };
            ROOT::Math::Functor1D func(f);
            double dy = deriv.Derivative1(func, Wis(i,j), 1.E-5);
            Scalar_t dy_ref = dWis(i,j);

            // Compute relative error if dy != 0
            Scalar_t error;
            std::string errorType;
            if (std::fabs(dy_ref) > 1e-15) {
                error = std::fabs((dy - dy_ref) / dy_ref);
                errorType = "relative";
            } else {
                error = std::fabs(dy - dy_ref);
                errorType = "absolute";
            }

            if (debug) std::cout << "Input Gate: state weight gradients (" << i << "," << j << ") : (comp, ref) " << dy << ", " << dy_ref << std::endl;

            if (error >= maximum_error) {
                maximum_error = error;
                maxErrorType = errorType;
            }
        }
    }
    std::cout << "\rInput Gate: testing state weight gradients: ";
    std::cout << "Maximum error (" << maxErrorType << ") : " << print_error(maximum_error) << std::endl;
    if (maximum_error > 1.E-2) {
        std::cerr << "\e[31m Error \e[39m in state weight gradients of input gate" << std::endl;
        failed = true;
    }

    // Testing forget gate: state weights gradients
    maximum_error = 0.0;
    auto &Wfs = layer->GetWeightsAt(0);
    auto &dWfs = layer->GetWeightGradientsAt(0);

    for (size_t i = 0; i < (size_t) Wfs.GetNrows(); ++i) {
        for (size_t j = 0; j < (size_t) Wfs.GetNcols(); ++j) {
            auto f = [&lstm, &XArch, &Y, &weights_forget_state, i, j](Scalar_t x) {
                return evaluate_net_weight(lstm, XArch, Y, weights_forget_state, 0, 0, i, j, x);
            };
            ROOT::Math::Functor1D func(f);
            double dy = deriv.Derivative1(func, Wfs(i,j), 1.E-5);
            Scalar_t dy_ref = dWfs(i,j);

            // Compute relative error if dy != 0
            Scalar_t error;
            std::string errorType;
            if (std::fabs(dy_ref) > 1e-15) {
                error = std::fabs((dy - dy_ref) / dy_ref);
                errorType = "relative";
            } else {
                error = std::fabs(dy - dy_ref);
                errorType = "absolute";
            }

            if (debug) std::cout << "Forget Gate: state weight gradients (" << i << "," << j << ") : (comp, ref) " << dy << ", " << dy_ref << std::endl;

            if (error >= maximum_error) {
                maximum_error = error;
                maxErrorType = errorType;
            }
        }
    }
    std::cout << "\rForget Gate: testing state weight gradients: ";
    std::cout << "Maximum error (" << maxErrorType << ") : " << print_error(maximum_error) << std::endl;
    if (maximum_error > 1.E-2) {
        std::cerr << "\e[31m Error \e[39m in state weight gradients of forget gate" << std::endl;
        failed = true;
    }

    // Testing candidate gate: state weights gradients k = 5
    maximum_error = 0.0;
    auto &Wcs = layer->GetWeightsAt(0);
    auto &dWcs = layer->GetWeightGradientsAt(0);

    for (size_t i = 0; i < (size_t) Wcs.GetNrows(); ++i) {
        for (size_t j = 0; j < (size_t) Wcs.GetNcols(); ++j) {
            auto f = [&lstm, &XArch, &Y, &weights_candidate_state, i, j](Scalar_t x) {
                return evaluate_net_weight(lstm, XArch, Y, weights_candidate_state, 0, 0, i, j, x);
            };
            ROOT::Math::Functor1D func(f);
            double dy = deriv.Derivative1(func, Wcs(i,j), 1.E-5);
            Scalar_t dy_ref = dWcs(i,j);

            // Compute relative error if dy != 0
            Scalar_t error;
            std::string errorType;
            if (std::fabs(dy_ref) > 1e-15) {
                error = std::fabs((dy - dy_ref) / dy_ref);
                errorType = "relative";
            } else {
                error = std::fabs(dy - dy_ref);
                errorType = "absolute";
            }

            if (debug) std::cout << "Candidate Gate: state weight gradients (" << i << "," << j << ") : (comp, ref) " << dy << ", " << dy_ref << std::endl;

            if (error >= maximum_error) {
                maximum_error = error;
                maxErrorType = errorType;
            }
        }
    }
    std::cout << "\rCandidate Gate: testing state weight gradients: ";
    std::cout << "Maximum error (" << maxErrorType << ") : " << print_error(maximum_error) << std::endl;
    if (maximum_error > 1.E-2) {
        std::cerr << "\e[31m Error \e[39m in state weight gradients of candidate gate" << std::endl;
        failed = true;
    }

    // Testing output gate: state weights gradients k = 7
    maximum_error = 0.0;
    auto &Wos = layer->GetWeightsAt(0);
    auto &dWos = layer->GetWeightGradientsAt(0);

    for (size_t i = 0; i < (size_t) Wos.GetNrows(); ++i) {
        for (size_t j = 0; j < (size_t) Wos.GetNcols(); ++j) {
            auto f = [&lstm, &XArch, &Y, &weights_output_state, i, j](Scalar_t x) {
                return evaluate_net_weight(lstm, XArch, Y, weights_output_state, 0, 0, i, j, x);
            };
            ROOT::Math::Functor1D func(f);
            double dy = deriv.Derivative1(func, Wos(i,j), 1.E-5);
            Scalar_t dy_ref = dWos(i,j);

            // Compute relative error if dy != 0
            Scalar_t error;
            std::string errorType;
            if (std::fabs(dy_ref) > 1e-15) {
                error = std::fabs((dy - dy_ref) / dy_ref);
                errorType = "relative";
            } else {
                error = std::fabs(dy - dy_ref);
                errorType = "absolute";
            }

            if (debug) std::cout << "Output Gate: state weight gradients (" << i << "," << j << ") : (comp, ref) " << dy << ", " << dy_ref << std::endl;

            if (error >= maximum_error) {
                maximum_error = error;
                maxErrorType = errorType;
            }
        }
    }
    std::cout << "\rOutput Gate: testing state weight gradients: ";
    std::cout << "Maximum error (" << maxErrorType << ") : " << print_error(maximum_error) << std::endl;
    if (maximum_error > 1.E-2) {
        std::cerr << "\e[31m Error \e[39m in state weight gradients of output gate" << std::endl;
        failed = true;
    }

    // Testing input gate: bias gradients
    maximum_error = 0.0;
    auto &Bi = layer->GetBiasesAt(0);
    auto &dBi = layer->GetBiasGradientsAt(0);

    for (size_t i = 0; i < (size_t) Bi.GetNrows(); ++i) {
        auto f = [&lstm, &XArch, &Y, &input_bias, i](Scalar_t x) {
            return evaluate_net_bias(lstm, XArch, Y, input_bias, 0, 0, i, x);
        };
        ROOT::Math::Functor1D func(f);
        double dy = deriv.Derivative1(func, Bi(i,0), 1.E-5);
        Scalar_t dy_ref = dBi(i,0);

        // Compute relative error if dy != 0
        Scalar_t error;
        std::string errorType;
        if (std::fabs(dy_ref) > 1e-15) {
            error = std::fabs((dy - dy_ref) / dy_ref);
            errorType = "relative";
        } else {
            error = std::fabs(dy - dy_ref);
            errorType = "absolute";
        }

        if (debug) std::cout << "Input Gate: bias gradients (" << i << ") : (comp, ref) " << dy << ", " << dy_ref << std::endl;

        if (error >= maximum_error) {
            maximum_error = error;
            maxErrorType = errorType;
        }
    }
    std::cout << "\rInput Gate: testing bias gradients: ";
    std::cout << "Maximum error (" << maxErrorType << ") : " << print_error(maximum_error) << std::endl;
    if (maximum_error > 1.E-2) {
        std::cerr << "\e[31m Error \e[39m in bias gradients of input gate" << std::endl;
        failed = true;
    }

    // Testing forget gate: bias gradients
    maximum_error = 0.0;
    auto &Bf = layer->GetBiasesAt(0);
    auto &dBf = layer->GetBiasGradientsAt(0);

    for (size_t i = 0; i < (size_t) Bf.GetNrows(); ++i) {
        auto f = [&lstm, &XArch, &Y, &forget_bias, i](Scalar_t x) {
            return evaluate_net_bias(lstm, XArch, Y, forget_bias, 0, 0, i, x);
        };
        ROOT::Math::Functor1D func(f);
        double dy = deriv.Derivative1(func, Bf(i,0), 1.E-5);
        Scalar_t dy_ref = dBf(i,0);

        // Compute relative error if dy != 0
        Scalar_t error;
        std::string errorType;
        if (std::fabs(dy_ref) > 1e-15) {
            error = std::fabs((dy - dy_ref) / dy_ref);
            errorType = "relative";
        } else {
            error = std::fabs(dy - dy_ref);
            errorType = "absolute";
        }

        if (debug) std::cout << "Forget Gate: bias gradients (" << i << ") : (comp, ref) " << dy << ", " << dy_ref << std::endl;

        if (error >= maximum_error) {
            maximum_error = error;
            maxErrorType = errorType;
        }
    }
    std::cout << "\rForget Gate: testing bias gradients: ";
    std::cout << "Maximum error (" << maxErrorType << ") : " << print_error(maximum_error) << std::endl;
    if (maximum_error > 1.E-2) {
        std::cerr << "\e[31m Error \e[39m in bias gradients of forget gate" << std::endl;
        failed = true;
    }

    // Testing candidate gate: bias gradients
    maximum_error = 0.0;
    auto &Bc = layer->GetBiasesAt(0);
    auto &dBc = layer->GetBiasGradientsAt(0);

    for (size_t i = 0; i < (size_t) Bc.GetNrows(); ++i) {
        auto f = [&lstm, &XArch, &Y, &candidate_bias, i](Scalar_t x) {
            return evaluate_net_bias(lstm, XArch, Y, candidate_bias, 0, 0, i, x);
        };
        ROOT::Math::Functor1D func(f);
        double dy = deriv.Derivative1(func, Bc(i,0), 1.E-5);
        Scalar_t dy_ref = dBc(i,0);

        // Compute relative error if dy != 0
        Scalar_t error;
        std::string errorType;
        if (std::fabs(dy_ref) > 1e-15) {
            error = std::fabs((dy - dy_ref) / dy_ref);
            errorType = "relative";
        } else {
            error = std::fabs(dy - dy_ref);
            errorType = "absolute";
        }

        if (debug) std::cout << "Candidate Gate: bias gradients (" << i << ") : (comp, ref) " << dy << ", " << dy_ref << std::endl;

        if (error >= maximum_error) {
            maximum_error = error;
            maxErrorType = errorType;
        }
    }
    std::cout << "\rCandidate Gate: testing bias gradients: ";
    std::cout << "Maximum error (" << maxErrorType << ") : " << print_error(maximum_error) << std::endl;
    if (maximum_error > 1.E-2) {
        std::cerr << "\e[31m Error \e[39m in bias gradients of candidate gate" << std::endl;
        failed = true;
    }

    // Testing output gate: bias gradients
    maximum_error = 0.0;
    auto &Bo = layer->GetBiasesAt(0);
    auto &dBo = layer->GetBiasGradientsAt(0);

    for (size_t i = 0; i < (size_t) Bo.GetNrows(); ++i) {
        auto f = [&lstm, &XArch, &Y, &output_bias, i](Scalar_t x) {
            return evaluate_net_bias(lstm, XArch, Y, output_bias, 0, 0, i, x);
        };
        ROOT::Math::Functor1D func(f);
        double dy = deriv.Derivative1(func, Bo(i,0), 1.E-5);
        Scalar_t dy_ref = dBo(i,0);

        // Compute relative error if dy != 0
        Scalar_t error;
        std::string errorType;
        if (std::fabs(dy_ref) > 1e-15) {
            error = std::fabs((dy - dy_ref) / dy_ref);
            errorType = "relative";
        } else {
            error = std::fabs(dy - dy_ref);
            errorType = "absolute";
        }

        if (debug) std::cout << "Output Gate: bias gradients (" << i << ") : (comp, ref) " << dy << ", " << dy_ref << std::endl;

        if (error >= maximum_error) {
            maximum_error = error;
            maxErrorType = errorType;
        }
    }
    std::cout << "\rOutput Gate: testing bias gradients: ";
    std::cout << "Maximum error (" << maxErrorType << ") : " << print_error(maximum_error) << std::endl;
    if (maximum_error > 1.E-2) {
        std::cerr << "\e[31m Error \e[39m in bias gradients of output gate" << std::endl;
        failed = true;
    }

    return failed;

}

#endif
