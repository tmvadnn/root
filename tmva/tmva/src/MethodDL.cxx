 // @(#)root/tmva/tmva/cnn:$Id$Ndl
// Author: Vladimir Ilievski, Saurav Shekhar

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodDL                                                              *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Deep Neural Network Method                                                *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Vladimir Ilievski  <ilievski.vladimir@live.com> - CERN, Switzerland       *
 *      Saurav Shekhar     <sauravshekhar01@gmail.com> - ETH Zurich, Switzerland  *
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

#include "TFormula.h"
#include "TString.h"
#include "TMath.h"

#include "TMVA/Tools.h"
#include "TMVA/Configurable.h"
#include "TMVA/IMethod.h"
#include "TMVA/ClassifierFactory.h"
#include "TMVA/MethodDL.h"
#include "TMVA/Types.h"
#include "TMVA/DNN/TensorDataLoader.h"
#include "TMVA/DNN/Functions.h"
#include "TMVA/DNN/DLMinimizers.h"
#include "TStopwatch.h"

#include <chrono>

REGISTER_METHOD(DL)
ClassImp(TMVA::MethodDL);

using namespace TMVA::DNN::CNN;
using namespace TMVA::DNN;

using TMVA::DNN::EActivationFunction;
using TMVA::DNN::ELossFunction;
using TMVA::DNN::EInitialization;
using TMVA::DNN::EOutputFunction;

namespace TMVA {

////////////////////////////////////////////////////////////////////////////////
TString fetchValueTmp(const std::map<TString, TString> &keyValueMap, TString key)
{
   key.ToUpper();
   std::map<TString, TString>::const_iterator it = keyValueMap.find(key);
   if (it == keyValueMap.end()) {
      return TString("");
   }
   return it->second;
}

////////////////////////////////////////////////////////////////////////////////
template <typename T>
T fetchValueTmp(const std::map<TString, TString> &keyValueMap, TString key, T defaultValue);

////////////////////////////////////////////////////////////////////////////////
template <>
int fetchValueTmp(const std::map<TString, TString> &keyValueMap, TString key, int defaultValue)
{
   TString value(fetchValueTmp(keyValueMap, key));
   if (value == "") {
      return defaultValue;
   }
   return value.Atoi();
}

////////////////////////////////////////////////////////////////////////////////
template <>
double fetchValueTmp(const std::map<TString, TString> &keyValueMap, TString key, double defaultValue)
{
   TString value(fetchValueTmp(keyValueMap, key));
   if (value == "") {
      return defaultValue;
   }
   return value.Atof();
}

////////////////////////////////////////////////////////////////////////////////
template <>
TString fetchValueTmp(const std::map<TString, TString> &keyValueMap, TString key, TString defaultValue)
{
   TString value(fetchValueTmp(keyValueMap, key));
   if (value == "") {
      return defaultValue;
   }
   return value;
}

////////////////////////////////////////////////////////////////////////////////
template <>
bool fetchValueTmp(const std::map<TString, TString> &keyValueMap, TString key, bool defaultValue)
{
   TString value(fetchValueTmp(keyValueMap, key));
   if (value == "") {
      return defaultValue;
   }

   value.ToUpper();
   if (value == "TRUE" || value == "T" || value == "1") {
      return true;
   }

   return false;
}

////////////////////////////////////////////////////////////////////////////////
template <>
std::vector<double> fetchValueTmp(const std::map<TString, TString> &keyValueMap, TString key,
                                  std::vector<double> defaultValue)
{
   TString parseString(fetchValueTmp(keyValueMap, key));
   if (parseString == "") {
      return defaultValue;
   }

   parseString.ToUpper();
   std::vector<double> values;

   const TString tokenDelim("+");
   TObjArray *tokenStrings = parseString.Tokenize(tokenDelim);
   TIter nextToken(tokenStrings);
   TObjString *tokenString = (TObjString *)nextToken();
   for (; tokenString != NULL; tokenString = (TObjString *)nextToken()) {
      std::stringstream sstr;
      double currentValue;
      sstr << tokenString->GetString().Data();
      sstr >> currentValue;
      values.push_back(currentValue);
   }
   return values;
}

////////////////////////////////////////////////////////////////////////////////
void MethodDL::DeclareOptions()
{
   // Set default values for all option strings

   DeclareOptionRef(fInputLayoutString = "0|0|0", "InputLayout", "The Layout of the input");

   DeclareOptionRef(fBatchLayoutString = "0|0|0", "BatchLayout", "The Layout of the batch");

   DeclareOptionRef(fLayoutString = "DENSE|(N+100)*2|SOFTSIGN,DENSE|0|LINEAR", "Layout", "Layout of the network.");

   DeclareOptionRef(fErrorStrategy = "CROSSENTROPY", "ErrorStrategy", "Loss function: Mean squared error (regression)"
                                                                      " or cross entropy (binary classification).");
   AddPreDefVal(TString("CROSSENTROPY"));
   AddPreDefVal(TString("SUMOFSQUARES"));
   AddPreDefVal(TString("MUTUALEXCLUSIVE"));

   DeclareOptionRef(fWeightInitializationString = "XAVIER", "WeightInitialization", "Weight initialization strategy");
   AddPreDefVal(TString("XAVIER"));
   AddPreDefVal(TString("XAVIERUNIFORM"));

   DeclareOptionRef(fRandomSeed = 0, "RandomSeed", "Random seed used for weight initialization and batch shuffling");


   DeclareOptionRef(fArchitectureString = "CPU", "Architecture", "Which architecture to perform the training on.");
   AddPreDefVal(TString("STANDARD"));
   AddPreDefVal(TString("CPU"));
   AddPreDefVal(TString("GPU"));
   AddPreDefVal(TString("OPENCL"));

   DeclareOptionRef(fTrainingStrategyString = "LearningRate=1e-1,"
                                              "Momentum=0.3,"
                                              "Repetitions=3,"
                                              "ConvergenceSteps=50,"
                                              "BatchSize=30,"
                                              "TestRepetitions=7,"
                                              "WeightDecay=0.0,"
                                              "Renormalize=L2,"
                                              "DropConfig=0.0,"
                                              "DropRepetitions=5|LearningRate=1e-4,"
                                              "Momentum=0.3,"
                                              "Repetitions=3,"
                                              "ConvergenceSteps=50,"
                                              "MaxEpochs=2000,"
                                              "BatchSize=20,"
                                              "TestRepetitions=7,"
                                              "WeightDecay=0.001,"
                                              "Renormalize=L2,"
                                              "DropConfig=0.0+0.5+0.5,"
                                              "DropRepetitions=5,"
                                              "Multithreading=True",
                    "TrainingStrategy", "Defines the training strategies.");
}

////////////////////////////////////////////////////////////////////////////////
void MethodDL::ProcessOptions()
{

   if (IgnoreEventsWithNegWeightsInTraining()) {
      Log() << kINFO << "Will ignore negative events in training!" << Endl;
   }

   if (fArchitectureString == "STANDARD") {
      Log() << kINFO << "The STANDARD architecture has been deprecated. "
                         "Please use Architecture=CPU or Architecture=CPU."
                         "See the TMVA Users' Guide for instructions if you "
                         "encounter problems."
            << Endl;
   }
   if (fArchitectureString == "OPENCL") {
      Log() << kERROR << "The OPENCL architecture has not been implemented yet. "
                         "Please use Architecture=CPU or Architecture=CPU for the "
                         "time being. See the TMVA Users' Guide for instructions "
                         "if you encounter problems."
            << Endl;
   }
   
   // the architecture can now be set at runtime as an option


   if (fArchitectureString == "GPU") {
#ifndef R__HAS_TMVAGPU    // case TMVA does not support GPU
      Log() << kERROR << "CUDA backend not enabled. Please make sure "
         "you have CUDA installed and it was successfully "
         "detected by CMAKE by using -Dcuda=On "
            << Endl;
#ifdef R__HAS_TMVACPU
      fArchitectureString = "CPU";
      Log() << kINFO << "Will use now the CPU architecture !" << Endl;
#else 
      fArchitectureString = "Standard";
      Log() << kINFO << "Will use now the Standard architecture !" << Endl;
#endif
#else
      Log() << kINFO << "Will use now the GPU architecture !" << Endl;
#endif
   }

   else if (fArchitectureString == "CPU") {
#ifndef R__HAS_TMVACPU  // TMVA has no CPU support
      Log() << kERROR << "Multi-core CPU backend not enabled. Please make sure "
                          "you have a BLAS implementation and it was successfully "
                         "detected by CMake as well that the imt CMake flag is set."
            << Endl;
#ifdef R__HAS_TMVAGPU
      fArchitectureString = "GPU";
      Log() << kINFO << "Will use now the GPU architecture !" << Endl;
#else 
      fArchitectureString = "STANDARD";
      Log() << kINFO << "Will use now the Standard architecture !" << Endl;
#endif
#else
      Log() << kINFO << "Will use now the CPU architecture !" << Endl;
#endif
   }

   else { 
      Log() << kINFO << "Will use the deprecated STANDARD architecture !" << Endl;
      fArchitectureString = "STANDARD";
   }

   // Input Layout
   ParseInputLayout();
   ParseBatchLayout();

   // Loss function and output.
   fOutputFunction = EOutputFunction::kSigmoid;
   if (fAnalysisType == Types::kClassification) {
      if (fErrorStrategy == "SUMOFSQUARES") {
         fLossFunction = ELossFunction::kMeanSquaredError;
      }
      if (fErrorStrategy == "CROSSENTROPY") {
         fLossFunction = ELossFunction::kCrossEntropy;
      }
      fOutputFunction = EOutputFunction::kSigmoid;
   } else if (fAnalysisType == Types::kRegression) {
      if (fErrorStrategy != "SUMOFSQUARES") {
         Log() << kWARNING << "For regression only SUMOFSQUARES is a valid "
               << " neural net error function. Setting error function to "
               << " SUMOFSQUARES now." << Endl;
      }

      fLossFunction = ELossFunction::kMeanSquaredError;
      fOutputFunction = EOutputFunction::kIdentity;
   } else if (fAnalysisType == Types::kMulticlass) {
      if (fErrorStrategy == "SUMOFSQUARES") {
         fLossFunction = ELossFunction::kMeanSquaredError;
      }
      if (fErrorStrategy == "CROSSENTROPY") {
         fLossFunction = ELossFunction::kCrossEntropy;
      }
      if (fErrorStrategy == "MUTUALEXCLUSIVE") {
         fLossFunction = ELossFunction::kSoftmaxCrossEntropy;
      }
      fOutputFunction = EOutputFunction::kSoftmax;
   }

   // Initialization
   // the biases will be always initialized to zero
   if (fWeightInitializationString == "XAVIER") {
      fWeightInitialization = DNN::EInitialization::kGlorotNormal;
   } else if (fWeightInitializationString == "XAVIERUNIFORM") {
      fWeightInitialization = DNN::EInitialization::kGlorotUniform;
   } else if (fWeightInitializationString == "GAUSS") {
      fWeightInitialization = DNN::EInitialization::kGauss;
   } else if (fWeightInitializationString == "UNIFORM") {
      fWeightInitialization = DNN::EInitialization::kUniform;
   } else if (fWeightInitializationString == "ZERO") {
      fWeightInitialization = DNN::EInitialization::kZero;
   } else {
      fWeightInitialization = DNN::EInitialization::kGlorotUniform;
   }

   // Training settings.

   KeyValueVector_t strategyKeyValues = ParseKeyValueString(fTrainingStrategyString, TString("|"), TString(","));
   for (auto &block : strategyKeyValues) {
      TTrainingSettings settings;

      settings.convergenceSteps = fetchValueTmp(block, "ConvergenceSteps", 100);
      settings.batchSize = fetchValueTmp(block, "BatchSize", 30);
      settings.maxEpochs = fetchValueTmp(block, "MaxEpochs", 2000);
      settings.testInterval = fetchValueTmp(block, "TestRepetitions", 7);
      settings.weightDecay = fetchValueTmp(block, "WeightDecay", 0.0);
      settings.learningRate = fetchValueTmp(block, "LearningRate", 1e-5);
      settings.momentum = fetchValueTmp(block, "Momentum", 0.3);
      settings.dropoutProbabilities = fetchValueTmp(block, "DropConfig", std::vector<Double_t>());

      TString regularization = fetchValueTmp(block, "Regularization", TString("NONE"));
      if (regularization == "L1") {
         settings.regularization = DNN::ERegularization::kL1;
      } else if (regularization == "L2") {
         settings.regularization = DNN::ERegularization::kL2;
      }

      TString strMultithreading = fetchValueTmp(block, "Multithreading", TString("True"));

      if (strMultithreading.BeginsWith("T")) {
         settings.multithreading = true;
      } else {
         settings.multithreading = false;
      }

      fTrainingSettings.push_back(settings);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// default initializations
void MethodDL::Init()
{
   // Nothing to do here
}

////////////////////////////////////////////////////////////////////////////////
/// Parse the input layout
void MethodDL::ParseInputLayout()
{
   // Define the delimiter
   const TString delim("|");

   // Get the input layout string
   TString inputLayoutString = this->GetInputLayoutString();

   size_t depth = 0;
   size_t height = 0;
   size_t width = 0;

   // Split the input layout string
   TObjArray *inputDimStrings = inputLayoutString.Tokenize(delim);
   TIter nextInputDim(inputDimStrings);
   TObjString *inputDimString = (TObjString *)nextInputDim();
   int idxToken = 0;

   for (; inputDimString != nullptr; inputDimString = (TObjString *)nextInputDim()) {
      switch (idxToken) {
      case 0: // input depth
      {
         TString strDepth(inputDimString->GetString());
         depth = (size_t)strDepth.Atoi();
      } break;
      case 1: // input height
      {
         TString strHeight(inputDimString->GetString());
         height = (size_t)strHeight.Atoi();
      } break;
      case 2: // input width
      {
         TString strWidth(inputDimString->GetString());
         width = (size_t)strWidth.Atoi();
      } break;
      }
      ++idxToken;
   }

   this->SetInputDepth(depth);
   this->SetInputHeight(height);
   this->SetInputWidth(width);
}

////////////////////////////////////////////////////////////////////////////////
/// Parse the input layout
void MethodDL::ParseBatchLayout()
{
   // Define the delimiter
   const TString delim("|");

   // Get the input layout string
   TString batchLayoutString = this->GetBatchLayoutString();

   size_t batchDepth = 0;
   size_t batchHeight = 0;
   size_t batchWidth = 0;

   // Split the input layout string
   TObjArray *batchDimStrings = batchLayoutString.Tokenize(delim);
   TIter nextBatchDim(batchDimStrings);
   TObjString *batchDimString = (TObjString *)nextBatchDim();
   int idxToken = 0;

   for (; batchDimString != nullptr; batchDimString = (TObjString *)nextBatchDim()) {
      switch (idxToken) {
      case 0: // input depth
      {
         TString strDepth(batchDimString->GetString());
         batchDepth = (size_t)strDepth.Atoi();
      } break;
      case 1: // input height
      {
         TString strHeight(batchDimString->GetString());
         batchHeight = (size_t)strHeight.Atoi();
      } break;
      case 2: // input width
      {
         TString strWidth(batchDimString->GetString());
         batchWidth = (size_t)strWidth.Atoi();
      } break;
      }
      ++idxToken;
   }

   this->SetBatchDepth(batchDepth);
   this->SetBatchHeight(batchHeight);
   this->SetBatchWidth(batchWidth);
}

////////////////////////////////////////////////////////////////////////////////
/// Create a deep net based on the layout string
template <typename Architecture_t, typename Layer_t>
void MethodDL::CreateDeepNet(DNN::TDeepNet<Architecture_t, Layer_t> &deepNet,
                             std::vector<DNN::TDeepNet<Architecture_t, Layer_t>> &nets)
{
   // Layer specification, layer details
   const TString layerDelimiter(",");
   const TString subDelimiter("|");

   TString layoutString = this->GetLayoutString();

   //std::cout << "Create Deepnet - layout string " << layoutString << "\t layers : " << deepNet.GetLayers().size() << std::endl;

   // Split layers
   TObjArray *layerStrings = layoutString.Tokenize(layerDelimiter);
   TIter nextLayer(layerStrings);
   TObjString *layerString = (TObjString *)nextLayer();


   for (; layerString != nullptr; layerString = (TObjString *)nextLayer()) {
      // Split layer details
      TObjArray *subStrings = layerString->GetString().Tokenize(subDelimiter);
      TIter nextToken(subStrings);
      TObjString *token = (TObjString *)nextToken();

      // Determine the type of the layer
      TString strLayerType = token->GetString();


      if (strLayerType == "DENSE") {
         ParseDenseLayer(deepNet, nets, layerString->GetString(), subDelimiter);
      } else if (strLayerType == "CONV") {
         ParseConvLayer(deepNet, nets, layerString->GetString(), subDelimiter);
      } else if (strLayerType == "MAXPOOL") {
         ParseMaxPoolLayer(deepNet, nets, layerString->GetString(), subDelimiter);
      } else if (strLayerType == "RESHAPE") {
         ParseReshapeLayer(deepNet, nets, layerString->GetString(), subDelimiter);
      } else if (strLayerType == "RNN") {
         ParseRnnLayer(deepNet, nets, layerString->GetString(), subDelimiter);
      } else if (strLayerType == "LSTM") {
         Log() << kFATAL << "LSTM Layer is not yet fully implemented" << Endl;
         //ParseLstmLayer(deepNet, nets, layerString->GetString(), subDelimiter);
      } else if (strLayerType == "PADDING2D") {
         ParsePaddingLayer2D(deepNet, nets, layerString->GetString(), subDelimiter);
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Pases the layer string and creates the appropriate dense layer
template <typename Architecture_t, typename Layer_t>
void MethodDL::ParseDenseLayer(DNN::TDeepNet<Architecture_t, Layer_t> &deepNet,
                               std::vector<DNN::TDeepNet<Architecture_t, Layer_t>> & /*nets*/, TString layerString,
                               TString delim)
{
   int width = 0;
   EActivationFunction activationFunction = EActivationFunction::kTanh;

   // this return number of input variables for the method
   // it can be used to deduce width of dense layer if specified as N+10
   // where N is the number of input variables 
   const size_t inputSize = GetNvar();

   // Split layer details
   TObjArray *subStrings = layerString.Tokenize(delim);
   TIter nextToken(subStrings);
   TObjString *token = (TObjString *)nextToken();
   int idxToken = 0;

   // loop on the tokens
   // order of sepcifying width and activation function is not relevant
   // both  100|TANH and TANH|100 are valid cases
   for (; token != nullptr; token = (TObjString *)nextToken()) {
      idxToken++;
      // first token defines the layer type- skip it 
      if (idxToken == 1) continue;
      // try a match with the activation function 
      TString strActFnc(token->GetString());
      if (strActFnc == "RELU") {
         activationFunction = DNN::EActivationFunction::kRelu;
      } else if (strActFnc == "TANH") {
         activationFunction = DNN::EActivationFunction::kTanh;
      } else if (strActFnc == "SYMMRELU") {
         activationFunction = DNN::EActivationFunction::kSymmRelu;
      } else if (strActFnc == "SOFTSIGN") {
         activationFunction = DNN::EActivationFunction::kSoftSign;
      } else if (strActFnc == "SIGMOID") {
         activationFunction = DNN::EActivationFunction::kSigmoid;
      } else if (strActFnc == "LINEAR") {
         activationFunction = DNN::EActivationFunction::kIdentity;
      } else if (strActFnc == "GAUSS") {
         activationFunction = DNN::EActivationFunction::kGauss;
      } else if (width == 0) {
         // no match found try to parse as text showing the width
         // support for input a formula where the variable 'x' is 'N' in the string
         // use TFormula for the evaluation
         TString  strNumNodes = strActFnc;
         // number of nodes
         TString strN("x");
         strNumNodes.ReplaceAll("N", strN);
         strNumNodes.ReplaceAll("n", strN);
         TFormula fml("tmp", strNumNodes);
         width = fml.Eval(inputSize);
      }

   }

   // Add the dense layer, initialize the weights and biases and copy
   TDenseLayer<Architecture_t> *denseLayer = deepNet.AddDenseLayer(width, activationFunction);
   denseLayer->Initialize();

   // add same layer to fNet
   if (fBuildNet) fNet->AddDenseLayer(width, activationFunction);

   //TDenseLayer<Architecture_t> *copyDenseLayer = new TDenseLayer<Architecture_t>(*denseLayer);

   // add the copy to all slave nets
   //for (size_t i = 0; i < nets.size(); i++) {
   //   nets[i].AddDenseLayer(copyDenseLayer);
   //}

   // check compatibility of added layer
   // for a dense layer input should be 1 x 1 x DxHxW
}

////////////////////////////////////////////////////////////////////////////////
/// Pases the layer string and creates the appropriate convolutional layer
template <typename Architecture_t, typename Layer_t>
void MethodDL::ParseConvLayer(DNN::TDeepNet<Architecture_t, Layer_t> &deepNet,
                              std::vector<DNN::TDeepNet<Architecture_t, Layer_t>> & /*nets*/, TString layerString,
                              TString delim)
{
   int depth = 0;
   int fltHeight = 0;
   int fltWidth = 0;
   int strideRows = 0;
   int strideCols = 0;
   int zeroPadHeight = 0;
   int zeroPadWidth = 0;
   EActivationFunction activationFunction = EActivationFunction::kTanh;

   // Split layer details
   TObjArray *subStrings = layerString.Tokenize(delim);
   TIter nextToken(subStrings);
   TObjString *token = (TObjString *)nextToken();
   int idxToken = 0;

   for (; token != nullptr; token = (TObjString *)nextToken()) {
      switch (idxToken) {
      case 1: // depth
      {
         TString strDepth(token->GetString());
         depth = strDepth.Atoi();
      } break;
      case 2: // filter height
      {
         TString strFltHeight(token->GetString());
         fltHeight = strFltHeight.Atoi();
      } break;
      case 3: // filter width
      {
         TString strFltWidth(token->GetString());
         fltWidth = strFltWidth.Atoi();
      } break;
      case 4: // stride in rows
      {
         TString strStrideRows(token->GetString());
         strideRows = strStrideRows.Atoi();
      } break;
      case 5: // stride in cols
      {
         TString strStrideCols(token->GetString());
         strideCols = strStrideCols.Atoi();
      } break;
      case 6: // zero padding height
      {
         TString strZeroPadHeight(token->GetString());
         zeroPadHeight = strZeroPadHeight.Atoi();
      } break;
      case 7: // zero padding width
      {
         TString strZeroPadWidth(token->GetString());
         zeroPadWidth = strZeroPadWidth.Atoi();
      } break;
      case 8: // activation function
      {
         TString strActFnc(token->GetString());
         if (strActFnc == "RELU") {
            activationFunction = DNN::EActivationFunction::kRelu;
         } else if (strActFnc == "TANH") {
            activationFunction = DNN::EActivationFunction::kTanh;
         } else if (strActFnc == "SYMMRELU") {
            activationFunction = DNN::EActivationFunction::kSymmRelu;
         } else if (strActFnc == "SOFTSIGN") {
            activationFunction = DNN::EActivationFunction::kSoftSign;
         } else if (strActFnc == "SIGMOID") {
            activationFunction = DNN::EActivationFunction::kSigmoid;
         } else if (strActFnc == "LINEAR") {
            activationFunction = DNN::EActivationFunction::kIdentity;
         } else if (strActFnc == "GAUSS") {
            activationFunction = DNN::EActivationFunction::kGauss;
         }
      } break;
      }
      ++idxToken;
   }

   // Add the convolutional layer, initialize the weights and biases and copy
   TConvLayer<Architecture_t> *convLayer = deepNet.AddConvLayer(depth, fltHeight, fltWidth, strideRows, strideCols,
                                                                zeroPadHeight, zeroPadWidth, activationFunction);
   convLayer->Initialize();

   // Add same layer to fNet
   if (fBuildNet) fNet->AddConvLayer(depth, fltHeight, fltWidth, strideRows, strideCols,
                      zeroPadHeight, zeroPadWidth, activationFunction);

   //TConvLayer<Architecture_t> *copyConvLayer = new TConvLayer<Architecture_t>(*convLayer);

   //// add the copy to all slave nets
   //for (size_t i = 0; i < nets.size(); i++) {
   //   nets[i].AddConvLayer(copyConvLayer);
   //}
}

////////////////////////////////////////////////////////////////////////////////
/// Pases the layer string and creates the appropriate max pool layer
template <typename Architecture_t, typename Layer_t>
void MethodDL::ParseMaxPoolLayer(DNN::TDeepNet<Architecture_t, Layer_t> &deepNet,
                                 std::vector<DNN::TDeepNet<Architecture_t, Layer_t>> & /*nets*/, TString layerString,
                                 TString delim)
{

   int frameHeight = 0;
   int frameWidth = 0;
   int strideRows = 0;
   int strideCols = 0;

   // Split layer details
   TObjArray *subStrings = layerString.Tokenize(delim);
   TIter nextToken(subStrings);
   TObjString *token = (TObjString *)nextToken();
   int idxToken = 0;

   for (; token != nullptr; token = (TObjString *)nextToken()) {
      switch (idxToken) {
      case 1: // frame height
      {
         TString strFrmHeight(token->GetString());
         frameHeight = strFrmHeight.Atoi();
      } break;
      case 2: // frame width
      {
         TString strFrmWidth(token->GetString());
         frameWidth = strFrmWidth.Atoi();
      } break;
      case 3: // stride in rows
      {
         TString strStrideRows(token->GetString());
         strideRows = strStrideRows.Atoi();
      } break;
      case 4: // stride in cols
      {
         TString strStrideCols(token->GetString());
         strideCols = strStrideCols.Atoi();
      } break;
      }
      ++idxToken;
   }

   // Add the Max pooling layer
   // TMaxPoolLayer<Architecture_t> *maxPoolLayer =
   deepNet.AddMaxPoolLayer(frameHeight, frameWidth, strideRows, strideCols);

   // Add the same layer to fNet
   if (fBuildNet) fNet->AddMaxPoolLayer(frameHeight, frameWidth, strideRows, strideCols);

   //TMaxPoolLayer<Architecture_t> *copyMaxPoolLayer = new TMaxPoolLayer<Architecture_t>(*maxPoolLayer);

   //// add the copy to all slave nets
   //for (size_t i = 0; i < nets.size(); i++) {
   //   nets[i].AddMaxPoolLayer(copyMaxPoolLayer);
   //}
}

////////////////////////////////////////////////////////////////////////////////
/// Pases the layer string and creates the appropriate reshape layer
template <typename Architecture_t, typename Layer_t>
void MethodDL::ParseReshapeLayer(DNN::TDeepNet<Architecture_t, Layer_t> &deepNet,
                                 std::vector<DNN::TDeepNet<Architecture_t, Layer_t>> & /*nets*/, TString layerString,
                                 TString delim)
{
   int depth = 0;
   int height = 0;
   int width = 0;
   bool flattening = false;

   // Split layer details
   TObjArray *subStrings = layerString.Tokenize(delim);
   TIter nextToken(subStrings);
   TObjString *token = (TObjString *)nextToken();
   int idxToken = 0;

   for (; token != nullptr; token = (TObjString *)nextToken()) {
      if (token->GetString() == "FLAT") idxToken=4; 
      switch (idxToken) {
      case 1: {
         TString strDepth(token->GetString());
         depth = strDepth.Atoi();
      } break;
      case 2: // height
      {
         TString strHeight(token->GetString());
         height = strHeight.Atoi();
      } break;
      case 3: // width
      {
         TString strWidth(token->GetString());
         width = strWidth.Atoi();
      } break;
      case 4: // flattening
      {
         TString flat(token->GetString());
         if (flat == "FLAT") {
            flattening = true;
         }
      } break;
      }
      ++idxToken;
   }

   // Add the reshape layer
   // TReshapeLayer<Architecture_t> *reshapeLayer =
   deepNet.AddReshapeLayer(depth, height, width, flattening);

   // Add the same layer to fNet
   if (fBuildNet) fNet->AddReshapeLayer(depth, height, width, flattening);

   //TReshapeLayer<Architecture_t> *copyReshapeLayer = new TReshapeLayer<Architecture_t>(*reshapeLayer);

   //// add the copy to all slave nets
   //for (size_t i = 0; i < nets.size(); i++) {
   //   nets[i].AddReshapeLayer(copyReshapeLayer);
   //}
}

////////////////////////////////////////////////////////////////////////////////
/// Pases the layer string and creates the appropriate rnn layer
template <typename Architecture_t, typename Layer_t>
void MethodDL::ParseRnnLayer(DNN::TDeepNet<Architecture_t, Layer_t> & deepNet,
                             std::vector<DNN::TDeepNet<Architecture_t, Layer_t>> & /*nets */, TString layerString,
                             TString delim)
{
   //    int depth = 0;
   int stateSize = 0;
   int inputSize = 0;
   int timeSteps = 0;
   bool rememberState = false;

   // Split layer details
   TObjArray *subStrings = layerString.Tokenize(delim);
   TIter nextToken(subStrings);
   TObjString *token = (TObjString *)nextToken();
   int idxToken = 0;

   for (; token != nullptr; token = (TObjString *)nextToken()) {
      switch (idxToken) {
         case 1:  // state size 
         {
            TString strstateSize(token->GetString());
            stateSize = strstateSize.Atoi();
         } break;
         case 2:  // input size
         {
            TString strinputSize(token->GetString());
            inputSize = strinputSize.Atoi();
         } break;
         case 3:  // time steps
         {
            TString strtimeSteps(token->GetString());
            timeSteps = strtimeSteps.Atoi();
         }
         case 4: // remember state (1 or 0)
         {
            TString strrememberState(token->GetString());
            rememberState = (bool) strrememberState.Atoi();
         } break;
      }
      ++idxToken;
   }

   // Add the recurrent layer, initialize the weights and biases and copy
   TBasicRNNLayer<Architecture_t> *basicRNNLayer = deepNet.AddBasicRNNLayer(stateSize, inputSize,
                                                                        timeSteps, rememberState);
   basicRNNLayer->Initialize();
    
   // Add same layer to fNet
   if (fBuildNet) fNet->AddBasicRNNLayer(stateSize, inputSize, timeSteps, rememberState);

   //TBasicRNNLayer<Architecture_t> *copyRNNLayer = new TBasicRNNLayer<Architecture_t>(*basicRNNLayer);

   //// add the copy to all slave nets
   //for (size_t i = 0; i < nets.size(); i++) {
   //   nets[i].AddBasicRNNLayer(copyRNNLayer);
   //}
}

////////////////////////////////////////////////////////////////////////////////
/// Pases the layer string and creates the appropriate lstm layer
template <typename Architecture_t, typename Layer_t>
void MethodDL::ParseLstmLayer(DNN::TDeepNet<Architecture_t, Layer_t> & /*deepNet*/,
                              std::vector<DNN::TDeepNet<Architecture_t, Layer_t>> & /*nets*/, TString layerString,
                              TString delim)
{
   // Split layer details
   TObjArray *subStrings = layerString.Tokenize(delim);
   TIter nextToken(subStrings);
   TObjString *token = (TObjString *)nextToken();
   int idxToken = 0;

   for (; token != nullptr; token = (TObjString *)nextToken()) {
      switch (idxToken) {
      }
      ++idxToken;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Pases the layer string and creates the appropriate padding layer
template <typename Architecture_t, typename Layer_t>
void MethodDL::ParsePaddingLayer2D(DNN::TDeepNet<Architecture_t, Layer_t> &deepNet,
                                 std::vector<DNN::TDeepNet<Architecture_t, Layer_t>> & /*nets*/, TString layerString,
                                 TString delim)
{
   int topPad = 0;
   int bottomPad = 0;
   int leftPad = 0;
   int rightPad = 0;

   //layout expected: topPad|bottomPad|leftPad|rightPad

   // Split layer details
   TObjArray *subStrings = layerString.Tokenize(delim);
   TIter nextToken(subStrings);
   TObjString *token = (TObjString *)nextToken();
   int idxToken = 0;

   for (; token != nullptr; token = (TObjString *)nextToken()) {

      switch (idxToken) {
      case 1: // top padding
      {
         TString strTopPad(token->GetString());
         topPad = strTopPad.Atoi();
      } break;
      case 2: // bottom padding
      {
         TString strBottomPad(token->GetString());
         bottomPad = strBottomPad.Atoi();
      } break;
      case 3: // left padding
      {
         TString strLeftPad(token->GetString());
         leftPad = strLeftPad.Atoi();
      } break;
      case 4: // right padding
      {
         TString strRightPad(token->GetString());
         rightPad = strRightPad.Atoi();
      } break;
      }
      ++idxToken;
   }

   // Add the padding layer
   deepNet.AddPaddingLayer2D(topPad, bottomPad, leftPad, rightPad);

   // Add the same layer to fNet
   if (fBuildNet) fNet->AddPaddingLayer2D(topPad, bottomPad, leftPad, rightPad);
   
}

////////////////////////////////////////////////////////////////////////////////
/// Standard constructor.
MethodDL::MethodDL(const TString &jobName, const TString &methodTitle, DataSetInfo &theData, const TString &theOption)
   : MethodBase(jobName, Types::kDL, methodTitle, theData, theOption), fInputDepth(), fInputHeight(), fInputWidth(),
     fBatchDepth(), fBatchHeight(), fBatchWidth(), fRandomSeed(0), fWeightInitialization(), fOutputFunction(), fLossFunction(),
     fInputLayoutString(), fBatchLayoutString(), fLayoutString(), fErrorStrategy(), fTrainingStrategyString(),
     fWeightInitializationString(), fArchitectureString(), fResume(false), fBuildNet(true), fTrainingSettings()
{
   // Nothing to do here
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor from a weight file.
MethodDL::MethodDL(DataSetInfo &theData, const TString &theWeightFile)
   : MethodBase(Types::kDL, theData, theWeightFile), fInputDepth(), fInputHeight(), fInputWidth(), fBatchDepth(),
     fBatchHeight(), fBatchWidth(), fRandomSeed(0), fWeightInitialization(), fOutputFunction(), fLossFunction(), fInputLayoutString(),
     fBatchLayoutString(), fLayoutString(), fErrorStrategy(), fTrainingStrategyString(), fWeightInitializationString(),
     fArchitectureString(), fResume(false), fBuildNet(true), fTrainingSettings()
{
   // Nothing to do here
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor.
MethodDL::~MethodDL()
{
   // Nothing to do here
}

////////////////////////////////////////////////////////////////////////////////
/// Parse key value pairs in blocks -> return vector of blocks with map of key value pairs.
auto MethodDL::ParseKeyValueString(TString parseString, TString blockDelim, TString tokenDelim) -> KeyValueVector_t
{
   // remove empty spaces
   parseString.ReplaceAll(" ",""); 
   KeyValueVector_t blockKeyValues;
   const TString keyValueDelim("=");

   TObjArray *blockStrings = parseString.Tokenize(blockDelim);
   TIter nextBlock(blockStrings);
   TObjString *blockString = (TObjString *)nextBlock();

   for (; blockString != nullptr; blockString = (TObjString *)nextBlock()) {
      blockKeyValues.push_back(std::map<TString, TString>());
      std::map<TString, TString> &currentBlock = blockKeyValues.back();

      TObjArray *subStrings = blockString->GetString().Tokenize(tokenDelim);
      TIter nextToken(subStrings);
      TObjString *token = (TObjString *)nextToken();

      for (; token != nullptr; token = (TObjString *)nextToken()) {
         TString strKeyValue(token->GetString());
         int delimPos = strKeyValue.First(keyValueDelim.Data());
         if (delimPos <= 0) continue;

         TString strKey = TString(strKeyValue(0, delimPos));
         strKey.ToUpper();
         TString strValue = TString(strKeyValue(delimPos + 1, strKeyValue.Length()));

         strKey.Strip(TString::kBoth, ' ');
         strValue.Strip(TString::kBoth, ' ');

         currentBlock.insert(std::make_pair(strKey, strValue));
      }
   }
   return blockKeyValues;
}

////////////////////////////////////////////////////////////////////////////////
/// What kind of analysis type can handle the CNN
Bool_t MethodDL::HasAnalysisType(Types::EAnalysisType type, UInt_t numberClasses, UInt_t /*numberTargets*/)
{
   if (type == Types::kClassification && numberClasses == 2) return kTRUE;
   if (type == Types::kMulticlass) return kTRUE;
   if (type == Types::kRegression) return kTRUE;

   return kFALSE;
}


////////////////////////////////////////////////////////////////////////////////
///  Implementation of architecture specific train method
///
template <typename Architecture_t>
void MethodDL::TrainDeepNet()
{
   
   using Scalar_t = typename Architecture_t::Scalar_t;
   using DeepNet_t = TMVA::DNN::TDeepNet<Architecture_t>;
   using TensorDataLoader_t = TTensorDataLoader<TMVAInput_t, Architecture_t>;

   bool debug = Log().GetMinType() == kDEBUG;

   // Determine the number of training and testing examples
   size_t nTrainingSamples = GetEventCollection(Types::kTraining).size();
   size_t nTestSamples = GetEventCollection(Types::kTesting).size();

   // Determine the number of outputs
   // //    size_t outputSize = 1;
   // //    if (fAnalysisType == Types::kRegression && GetNTargets() != 0) {
   // //       outputSize = GetNTargets();
   // //    } else if (fAnalysisType == Types::kMulticlass && DataInfo().GetNClasses() >= 2) {
   // //       outputSize = DataInfo().GetNClasses();
   // //    }

   // set the random seed for weight initialization
   Architecture_t::SetRandomSeed(fRandomSeed); 

   size_t trainingPhase = 1;
   for (TTrainingSettings &settings : this->GetTrainingSettings()) {

      size_t nThreads = 1;       // FIXME threads are hard coded to 1, no use of slave threads or multi-threading


      // After the processing of the options, initialize the master deep net
      size_t batchSize = settings.batchSize;
      // Should be replaced by actual implementation. No support for this now.
      size_t inputDepth = this->GetInputDepth();
      size_t inputHeight = this->GetInputHeight();
      size_t inputWidth = this->GetInputWidth();
      size_t batchDepth = this->GetBatchDepth();
      size_t batchHeight = this->GetBatchHeight();
      size_t batchWidth = this->GetBatchWidth();
      ELossFunction J = this->GetLossFunction();
      EInitialization I = this->GetWeightInitialization();
      ERegularization R = settings.regularization;
      Scalar_t weightDecay = settings.weightDecay;

      //Batch size should be included in batch layout as well. There are two possibilities:
      //  1.  Batch depth = batch size   one will input tensorsa as (batch_size x d1 x d2)
      //       This is case for example if first layer is a conv layer and d1 = image depth, d2 = image width x image height
      //  2.  Batch depth = 1, batch height = batch size  batxch width = dim of input features
      //        This should be case if first layer is a Dense 1 and input tensor must be ( 1 x batch_size x input_features )

      if (batchDepth != batchSize && batchDepth > 1) {
         Error("Train","Given batch depth of %zu (specified in BatchLayout)  should be equal to given batch size %zu",batchDepth,batchSize);
         return;
      }
      if (batchDepth == 1 && batchSize > 1 && batchSize != batchHeight ) {
         Error("Train","Given batch height of %zu (specified in BatchLayout)  should be equal to given batch size %zu",batchHeight,batchSize);
         return;
      }


      //check also that input layout compatible with batch layout
      bool badLayout = false;
      // case batch depth == batch size
      if (batchDepth == batchSize)
         badLayout = ( inputDepth * inputHeight * inputWidth != batchHeight * batchWidth ) ;
      // case batch Height is batch size
      if (batchHeight == batchSize && batchDepth == 1) 
         badLayout |=  ( inputDepth * inputHeight * inputWidth !=  batchWidth);
      if (badLayout) {
         Error("Train","Given input layout %zu x %zu x %zu is not compatible with  batch layout %zu x %zu x  %zu ",
               inputDepth,inputHeight,inputWidth,batchDepth,batchHeight,batchWidth);
         return;
      }


      DeepNet_t deepNet(batchSize, inputDepth, inputHeight, inputWidth, batchDepth, batchHeight, batchWidth, J, I, R, weightDecay);

      // create a copy of DeepNet for evaluating but with batch size = 1
      // fNet is the saved network and will be with CPU or Referrence architecture
      if (trainingPhase == 1) {
         fNet = std::unique_ptr<DeepNetImpl_t>(new DeepNetImpl_t(1, inputDepth, inputHeight, inputWidth, batchDepth, 
                                                                 batchHeight, batchWidth, J, I, R, weightDecay));
         fBuildNet = true; 
      }
      else
         fBuildNet = false; 

      // Initialize the vector of slave nets
      std::vector<DeepNet_t> nets{};
      nets.reserve(nThreads);
      for (size_t i = 0; i < nThreads; i++) {
         // create a copies of the master deep net
         nets.push_back(deepNet);
      }

      // Add all appropriate layers to deepNet and (if fBuildNet is true) also to fNet
      CreateDeepNet(deepNet, nets);

      if (trainingPhase > 1) {
         // copy initial weights from fNet to deepnet
         for (size_t i = 0; i < deepNet.GetDepth(); ++i) {
            const auto & nLayer = fNet->GetLayerAt(i); 
            const auto & dLayer = deepNet.GetLayerAt(i);
            // could use a traits for detecting equal architectures
           // dLayer->CopyWeights(nLayer->GetWeights()); 
           //  dLayer->CopyBiases(nLayer->GetBiases());
            Architecture_t::CopyDiffArch(dLayer->GetWeights(), nLayer->GetWeights() );
            Architecture_t::CopyDiffArch(dLayer->GetBiases(), nLayer->GetBiases() );
         }
      }

      // print the created network
      if (fBuildNet) { 
         Log()  << "*****   Deep Learning Network *****" << Endl;
         if (Log().GetMinType() <= kINFO)
            deepNet.Print();
      }

      // Loading the training and testing datasets
      TMVAInput_t trainingTuple = std::tie(GetEventCollection(Types::kTraining), DataInfo());
      TensorDataLoader_t trainingData(trainingTuple, nTrainingSamples, deepNet.GetBatchSize(),
                                      deepNet.GetBatchDepth(), deepNet.GetBatchHeight(), deepNet.GetBatchWidth(),
                                      deepNet.GetOutputWidth(), nThreads);

      TMVAInput_t testTuple = std::tie(GetEventCollection(Types::kTesting), DataInfo());
      TensorDataLoader_t testingData(testTuple, nTestSamples, deepNet.GetBatchSize(),
                                     deepNet.GetBatchDepth(), deepNet.GetBatchHeight(), deepNet.GetBatchWidth(),
                                     deepNet.GetOutputWidth(), nThreads);

      // Initialize the minimizer
      DNN::TDLGradientDescent<Architecture_t> minimizer(settings.learningRate, settings.convergenceSteps,
                                                        settings.testInterval);

      // Initialize the vector of batches, one batch for one slave network
      std::vector<TTensorBatch<Architecture_t>> batches{};

      bool converged = false;
      // count the steps until the convergence
      size_t stepCount = 0;
      size_t batchesInEpoch = nTrainingSamples / deepNet.GetBatchSize();

      // start measuring
      std::chrono::time_point<std::chrono::system_clock> tstart, tend;
      tstart = std::chrono::system_clock::now();

      Log() << "Training phase " << trainingPhase << " of " << this->GetTrainingSettings().size() << ":" << Endl;
      if (!fInteractive) {
         std::string separator(62, '-');
         Log() << separator << Endl;
         Log() << std::setw(10) << "Epoch"
               << " | " << std::setw(12) << "Train Err." << std::setw(12) << "Test Err." 
               << std::setw(12) << "t(s)/epoch" << std::setw(12)  << "Eval t(s)"
               << std::setw(12) << "nEvents/s"
               << std::setw(12) << "Conv. Steps" << Endl;
         Log() << separator << Endl;
      }

      // set up generator for shuffling the batches 
      // if seed is zero we have always a different order in the batches 
      size_t shuffleSeed = 0;
      if (fRandomSeed != 0) shuffleSeed = fRandomSeed + trainingPhase; 
      RandomGenerator<TRandom3> rng(shuffleSeed);

      // print weights before
      if (fBuildNet && debug) {
         Log() << "Initial Deep Net Weights " << Endl;
         auto & weights_tensor = deepNet.GetLayerAt(0)->GetWeights();
         for (size_t l = 0; l < weights_tensor.size(); ++l) 
            weights_tensor[l].Print();
         auto & bias_tensor = deepNet.GetLayerAt(0)->GetBiases();
         bias_tensor[0].Print();
      }

      Double_t minTestError = 0;

      while (!converged) {
         stepCount++;
         trainingData.Shuffle(rng);

         // execute all epochs
         //for (size_t i = 0; i < batchesInEpoch; i += nThreads) {

         for (size_t i = 0; i < batchesInEpoch; ++i ) {
            // Clean and load new batches, one batch for one slave net
            //batches.clear();
            //batches.reserve(nThreads);
            //for (size_t j = 0; j < nThreads; j++) {
            //   batches.push_back(trainingData.GetTensorBatch());
            //}

            auto my_batch = trainingData.GetTensorBatch();



            
         // execute one minimization step
         // StepMomentum is currently not written for single thread, TODO write it
            if (settings.momentum > 0.0) {
               //minimizer.StepMomentum(deepNet, nets, batches, settings.momentum);
               minimizer.Step(deepNet, my_batch.GetInput(), my_batch.GetOutput(), my_batch.GetWeights());
            } else {
               //minimizer.Step(deepNet, nets, batches);
               minimizer.Step(deepNet, my_batch.GetInput(), my_batch.GetOutput(), my_batch.GetWeights());
            }


         }
         //}


         if ((stepCount % minimizer.GetTestInterval()) == 0) {

            std::chrono::time_point<std::chrono::system_clock> t1,t2; 

            t1 = std::chrono::system_clock::now();

            // Compute test error.
            Double_t testError = 0.0;
            for (auto batch : testingData) {
               auto inputTensor = batch.GetInput();
               auto outputMatrix = batch.GetOutput();
               auto weights = batch.GetWeights();
               testError += deepNet.Loss(inputTensor, outputMatrix, weights);
            }


            t2 = std::chrono::system_clock::now();
            testError /= (Double_t)(nTestSamples / settings.batchSize);
            // copy configuration when reached a minimum error
            if (testError < minTestError ) {
               // Copy weights from deepNet to fNet
               Log() << std::setw(10) << stepCount << " Minimun Test error found - save the configuration " << Endl;
               for (size_t i = 0; i < deepNet.GetDepth(); ++i) {
                  const auto & nLayer = fNet->GetLayerAt(i); 
                  const auto & dLayer = deepNet.GetLayerAt(i); 
                  //nLayer->CopyWeights(dLayer->GetWeights()); 
                  //nLayer->CopyBiases(dLayer->GetBiases());
                  ArchitectureImpl_t::CopyDiffArch(nLayer->GetWeights(), dLayer->GetWeights() );
                  ArchitectureImpl_t::CopyDiffArch(nLayer->GetBiases(), dLayer->GetBiases() );
                  // std::cout << "Weights for layer " << i << std::endl;
                  // for (size_t k = 0; k < dlayer->GetWeights().size(); ++k) 
                  //    dLayer->GetWeightsAt(k).Print(); 
               }
               minTestError = testError;
            }
            else if ( minTestError <= 0. )
               minTestError = testError; 


            Double_t trainingError = 0.0;
            // Compute training error.
            for (auto batch : trainingData) {
               auto inputTensor = batch.GetInput();
               auto outputMatrix = batch.GetOutput();
               auto weights = batch.GetWeights();

               trainingError += deepNet.Loss(inputTensor, outputMatrix, weights);
            }
            trainingError /= (Double_t)(nTrainingSamples / settings.batchSize);

            // stop measuring
            tend = std::chrono::system_clock::now();

            // Compute numerical throughput.
            std::chrono::duration<double> elapsed_seconds = tend - tstart;
            std::chrono::duration<double> elapsed1 = t1-tstart;
            // std::chrono::duration<double> elapsed2 = t2-tstart;
            // time to compute training and test errors
            std::chrono::duration<double> elapsed_testing = tend-t1;  


            double seconds = elapsed_seconds.count();
            // double nGFlops = (double)(settings.testInterval * batchesInEpoch * settings.batchSize)*1.E-9;
            // nGFlops *= deepnet.GetNFlops() * 1e-9;
            double eventTime = elapsed1.count()/( batchesInEpoch * settings.testInterval * settings.batchSize);

            converged = minimizer.HasConverged(testError) || stepCount >= settings.maxEpochs;

            Log() << std::setw(10) << stepCount << " | " << std::setw(12) << trainingError << std::setw(12) << testError
                  << std::setw(12) << seconds/settings.testInterval
                  << std::setw(12) << elapsed_testing.count() 
                  << std::setw(12) << 1./eventTime 
                  << std::setw(12) << minimizer.GetConvergenceCount()
                  << Endl;

            if (converged) {
               Log() << Endl;
            }
            tstart = std::chrono::system_clock::now();
         }

         //if (stepCount % 10 == 0 || converged) { 
         if (converged && debug) { 
            Log() << "Final Deep Net Weights for phase  " << trainingPhase << " epoch " << stepCount  << Endl;
            auto & weights_tensor = deepNet.GetLayerAt(0)->GetWeights();
            auto & bias_tensor = deepNet.GetLayerAt(0)->GetBiases();
            for (size_t l = 0; l < weights_tensor.size(); ++l) 
               weights_tensor[l].Print();
            bias_tensor[0].Print();
         }


      }

      trainingPhase++;
   }  // end loop on training Phase

}

////////////////////////////////////////////////////////////////////////////////
void MethodDL::Train()
{
   if (fInteractive) {
      Log() << kFATAL << "Not implemented yet" << Endl;
      return;
   }

   if (this->GetArchitectureString() == "GPU") {
#ifdef R__HAS_TMVAGPU
      Log() << kINFO << "Start of deep neural network training on GPU." << Endl << Endl;
      TrainDeepNet<DNN::TCuda<Double_t> >(); 
#else
      Log() << kFATAL << "CUDA backend not enabled. Please make sure "
         "you have CUDA installed and it was successfully "
         "detected by CMAKE."
             << Endl;
      return;
#endif
   } else if (this->GetArchitectureString() == "OPENCL") {
      Log() << kFATAL << "OPENCL backend not yet supported." << Endl;
      return;
   } else if (this->GetArchitectureString() == "CPU") {
#ifdef R__HAS_TMVACPU
      Log() << kINFO << "Start of deep neural network training on CPU." << Endl << Endl;
      TrainDeepNet<DNN::TCpu<Double_t> >(); 
#else
      Log() << kFATAL << "Multi-core CPU backend not enabled. Please make sure "
                      "you have a BLAS implementation and it was successfully "
                      "detected by CMake as well that the imt CMake flag is set."
            << Endl;
      return;
#endif
   } else if (this->GetArchitectureString() == "STANDARD") {
      Log() << kINFO << "Start of deep neural network training on the STANDARD architecture" << Endl << Endl;
      TrainDeepNet<DNN::TReference<Double_t> >(); 
   }
   else {
      Log() << kFATAL << this->GetArchitectureString() << 
                      " is not  a supported archiectire for TMVA::MethodDL"
            << Endl;
   }
   
// /// definitions for CUDA
// #ifdef R__HAS_TMVAGPU // Included only if DNNCUDA flag is set.
//    using Architecture_t = DNN::TCuda<Double_t>;
// #else
// #ifdef R__HAS_TMVACPU // Included only if DNNCPU flag is set.
//    using Architecture_t = DNN::TCpu<Double_t>;
// #else
//    using Architecture_t = DNN::TReference<Double_t>;
// #endif
// #endif
}


////////////////////////////////////////////////////////////////////////////////
Double_t MethodDL::GetMvaValue(Double_t * /*errLower*/, Double_t * /*errUpper*/)
{
   using Matrix_t = typename ArchitectureImpl_t::Matrix_t;

   int nVariables = GetEvent()->GetNVariables();
   int batchWidth = fNet->GetBatchWidth();
   int batchDepth = fNet->GetBatchDepth();
   int batchHeight = fNet->GetBatchHeight();
   int nb = fNet->GetBatchSize();
   int noutput = fNet->GetOutputWidth();

   // note that batch size whould be equal to 1
   R__ASSERT(nb == 1); 

   std::vector<Matrix_t> X{};
   Matrix_t YHat(nb, noutput);

   // get current event
   const std::vector<Float_t> &inputValues = GetEvent()->GetValues();

   //   for (int i = 0; i < batchDepth; ++i)

   // find dimension of matrices
   // Tensor outer size must be equal to 1
   // because nb ==1 by definition
   int n1 = batchHeight;
   int n2 = batchWidth;
   // treat case where batchHeight is batchSize in case of first Dense layers
   if (batchDepth == 1 && GetInputHeight() == 1 && GetInputDepth() == 1) n1 = 1;

   X.emplace_back(Matrix_t(n1, n2));

   if (n1 > 1) {
      if (n1*n2 != nVariables) {
         std::cout << n1 << "  " << batchDepth << "  " << GetInputHeight() << "  " << GetInputDepth()  << std::endl;
      }
      R__ASSERT( n1*n2 == nVariables);
      // for CNN or RNN evaluations
      for (int j = 0; j < n1; ++j) {
         for (int k = 0; k < n2; k++) {
            X[0](j, k) = inputValues[j*n1+k];
         }
      }
   }
   else {
      R__ASSERT( n2 == nVariables);
      for (int k = 0; k < n2; k++) {
         X[0](0, k) = inputValues[k];
      }
   }

   // perform the prediction
   fNet->Prediction(YHat, X, fOutputFunction);

   double mvaValue = YHat(0, 0);

   // for debugging
#ifdef DEBUG_MVAVALUE
   using Tensor_t = std::vector<Matrix_t>; 
    TMatrixF  xInput(n1,n2, inputValues.data() ); 
    std::cout << "Input data - class " << GetEvent()->GetClass() << std::endl;
    xInput.Print(); 
    std::cout << "Output of DeepNet " << mvaValue << std::endl;
    auto & deepnet = *fNet; 
    std::cout << "Loop on layers " << std::endl;
    for (int l = 0; l < deepnet.GetDepth(); ++l) { 
       std::cout << "Layer " << l;
       const auto *  layer = deepnet.GetLayerAt(l);
       const Tensor_t & layer_output = layer->GetOutput();
       layer->Print();
       std::cout << "DNN output " << layer_output.size() << std::endl;
       for (size_t i = 0; i < layer_output.size(); ++i) {
#ifdef R__HAS_TMVAGPU
          //TMatrixD m(layer_output[i].GetNrows(), layer_output[i].GetNcols() , layer_output[i].GetDataPointer()  );
          TMatrixD m = layer_output[i];
#else
          TMatrixD m(layer_output[i].GetNrows(), layer_output[i].GetNcols() , layer_output[i].GetRawDataPointer()  );
#endif
          m.Print();
       }
       const Tensor_t & layer_weights = layer->GetWeights();
       std::cout << "DNN weights " << layer_weights.size() << std::endl;
       if (layer_weights.size() > 0) { 
          int i = 0; 
#ifdef R__HAS_TMVAGPU
          TMatrixD m = layer_weights[i];
//          TMatrixD m(layer_weights[i].GetNrows(), layer_weights[i].GetNcols() , layer_weights[i].GetDataPointer()  );
#else
          TMatrixD m(layer_weights[i].GetNrows(), layer_weights[i].GetNcols() , layer_weights[i].GetRawDataPointer()  );
#endif
          m.Print();
       }
    }
#endif

 
   
   return (TMath::IsNaN(mvaValue)) ? -999. : mvaValue;

}

////////////////////////////////////////////////////////////////////////////////
void MethodDL::AddWeightsXMLTo(void * parent) const
{
      // Create the parrent XML node with name "Weights"
   auto & xmlEngine = gTools().xmlengine(); 
   void* nn = xmlEngine.NewChild(parent, 0, "Weights");
   
   /*! Get all necessary information, in order to be able to reconstruct the net 
    *  if we read the same XML file. */

   // Deep Net specific info
   Int_t depth = fNet->GetDepth();

   Int_t inputDepth = fNet->GetInputDepth();
   Int_t inputHeight = fNet->GetInputHeight();
   Int_t inputWidth = fNet->GetInputWidth();

   Int_t batchSize = fNet->GetBatchSize();

   Int_t batchDepth = fNet->GetBatchDepth();
   Int_t batchHeight = fNet->GetBatchHeight();
   Int_t batchWidth = fNet->GetBatchWidth();

   char lossFunction = static_cast<char>(fNet->GetLossFunction());
   char initialization = static_cast<char>(fNet->GetInitialization());
   char regularization = static_cast<char>(fNet->GetRegularization());

   Double_t weightDecay = fNet->GetWeightDecay();

   // Method specific info (not sure these are needed)
   char outputFunction = static_cast<char>(this->GetOutputFunction());
   //char lossFunction = static_cast<char>(this->GetLossFunction());

   // Add attributes to the parent node
   xmlEngine.NewAttr(nn, 0, "NetDepth", gTools().StringFromInt(depth));

   xmlEngine.NewAttr(nn, 0, "InputDepth", gTools().StringFromInt(inputDepth));
   xmlEngine.NewAttr(nn, 0, "InputHeight", gTools().StringFromInt(inputHeight));
   xmlEngine.NewAttr(nn, 0, "InputWidth", gTools().StringFromInt(inputWidth));

   xmlEngine.NewAttr(nn, 0, "BatchSize", gTools().StringFromInt(batchSize));
   xmlEngine.NewAttr(nn, 0, "BatchDepth", gTools().StringFromInt(batchDepth));
   xmlEngine.NewAttr(nn, 0, "BatchHeight", gTools().StringFromInt(batchHeight));
   xmlEngine.NewAttr(nn, 0, "BatchWidth", gTools().StringFromInt(batchWidth));

   xmlEngine.NewAttr(nn, 0, "LossFunction", TString(lossFunction));
   xmlEngine.NewAttr(nn, 0, "Initialization", TString(initialization));
   xmlEngine.NewAttr(nn, 0, "Regularization", TString(regularization));
   xmlEngine.NewAttr(nn, 0, "OutputFunction", TString(outputFunction));

   gTools().AddAttr(nn, "WeightDecay", weightDecay);


   for (Int_t i = 0; i < depth; i++)
   {
      fNet->GetLayerAt(i) -> AddWeightsXMLTo(nn);
   }


}

////////////////////////////////////////////////////////////////////////////////
void MethodDL::ReadWeightsFromXML(void * rootXML)
{
   
   auto netXML = gTools().GetChild(rootXML, "Weights");
   if (!netXML){
      netXML = rootXML;
   }

   size_t netDepth;
   gTools().ReadAttr(netXML, "NetDepth", netDepth);

   size_t inputDepth, inputHeight, inputWidth;
   gTools().ReadAttr(netXML, "InputDepth", inputDepth);
   gTools().ReadAttr(netXML, "InputHeight", inputHeight);
   gTools().ReadAttr(netXML, "InputWidth", inputWidth);

   size_t batchSize, batchDepth, batchHeight, batchWidth;
   gTools().ReadAttr(netXML, "BatchSize", batchSize);
   // use always batchsize = 1
   //batchSize = 1; 
   gTools().ReadAttr(netXML, "BatchDepth", batchDepth);
   gTools().ReadAttr(netXML, "BatchHeight", batchHeight);
   gTools().ReadAttr(netXML, "BatchWidth",  batchWidth);

   char lossFunctionChar;
   gTools().ReadAttr(netXML, "LossFunction", lossFunctionChar);
   char initializationChar;
   gTools().ReadAttr(netXML, "Initialization", initializationChar);
   char regularizationChar;
   gTools().ReadAttr(netXML, "Regularization", regularizationChar);
   char outputFunctionChar;
   gTools().ReadAttr(netXML, "OutputFunction", outputFunctionChar);
   double weightDecay;
   gTools().ReadAttr(netXML, "WeightDecay", weightDecay);

   // create the net

   // DeepNetCpu_t is defined in MethodDL.h
   this->SetInputDepth(inputDepth);
   this->SetInputHeight(inputHeight);
   this->SetInputWidth(inputWidth);
   this->SetBatchDepth(batchDepth);
   this->SetBatchHeight(batchHeight);
   this->SetBatchWidth(batchWidth);
   
   

   fNet = std::unique_ptr<DeepNetImpl_t>(new DeepNetImpl_t(batchSize, inputDepth, inputHeight, inputWidth, batchDepth,
                                                   batchHeight, batchWidth,
                                                   static_cast<ELossFunction>(lossFunctionChar),
                                                   static_cast<EInitialization>(initializationChar),
                                                   static_cast<ERegularization>(regularizationChar),
                                                    weightDecay));

   fOutputFunction = static_cast<EOutputFunction>(outputFunctionChar);


   //size_t previousWidth = inputWidth;
   auto layerXML = gTools().xmlengine().GetChild(netXML);

   // loop on the layer and add them to the network
   for (size_t i = 0; i < netDepth; i++) {

      TString layerName = gTools().xmlengine().GetNodeName(layerXML);

      // case of dense layer 
      if (layerName == "DenseLayer") {

         // read width and activation function and then we can create the layer
         size_t width = 0;
         gTools().ReadAttr(layerXML, "Width", width);

         // Read activation function.
         TString funcString; 
         gTools().ReadAttr(layerXML, "ActivationFunction", funcString);
         EActivationFunction func = static_cast<EActivationFunction>(funcString.Atoi());


         fNet->AddDenseLayer(width, func, 0.0); // no need to pass dropout probability

      }
      // Convolutional Layer
      else if (layerName == "ConvLayer") {

         // read width and activation function and then we can create the layer
         size_t depth = 0;
         gTools().ReadAttr(layerXML, "Depth", depth);
         size_t fltHeight, fltWidth = 0;
         size_t strideRows, strideCols = 0;
         size_t padHeight, padWidth = 0;
         gTools().ReadAttr(layerXML, "FilterHeight", fltHeight);
         gTools().ReadAttr(layerXML, "FilterWidth", fltWidth);
         gTools().ReadAttr(layerXML, "StrideRows", strideRows);
         gTools().ReadAttr(layerXML, "StrideCols", strideCols);
         gTools().ReadAttr(layerXML, "PaddingHeight", padHeight);
         gTools().ReadAttr(layerXML, "PaddingWidth", padWidth);

         // Read activation function.
         TString funcString; 
         gTools().ReadAttr(layerXML, "ActivationFunction", funcString);
         EActivationFunction actFunction = static_cast<EActivationFunction>(funcString.Atoi());


         fNet->AddConvLayer(depth, fltHeight, fltWidth, strideRows, strideCols,
                            padHeight, padWidth, actFunction);

      }

      // MaxPool Layer
      else if (layerName == "MaxPoolLayer") {

         // read maxpool layer info
         size_t frameHeight, frameWidth = 0;
         size_t strideRows, strideCols = 0;
         gTools().ReadAttr(layerXML, "FrameHeight", frameHeight);
         gTools().ReadAttr(layerXML, "FrameWidth", frameWidth);
         gTools().ReadAttr(layerXML, "StrideRows", strideRows);
         gTools().ReadAttr(layerXML, "StrideCols", strideCols);

         fNet->AddMaxPoolLayer(frameHeight, frameWidth, strideRows, strideCols);
      }
      else if (layerName == "ReshapeLayer") {

         // read reshape layer info
         size_t depth, height, width = 0; 
         gTools().ReadAttr(layerXML, "Depth", depth);
         gTools().ReadAttr(layerXML, "Height", height);
         gTools().ReadAttr(layerXML, "Width", width);
         int flattening = 0;
         gTools().ReadAttr(layerXML, "Flattening",flattening );

         fNet->AddReshapeLayer(depth, height, width, flattening);

      }
      else if (layerName == "RNNLayer") {

         // read RNN layer info
         size_t  stateSize,inputSize, timeSteps = 0;
         int rememberState= 0;   
         gTools().ReadAttr(layerXML, "StateSize", stateSize);
         gTools().ReadAttr(layerXML, "InputSize", inputSize);
         gTools().ReadAttr(layerXML, "TimeSteps", timeSteps);
         gTools().ReadAttr(layerXML, "RememberState", rememberState );
         
         fNet->AddBasicRNNLayer(stateSize, inputSize, timeSteps, rememberState);
         
      }
      else if (layerName == "PaddingLayer2D") {

         // read reshape layer info
         size_t leftPad, rightPad, topPad, bottomPad = 0; 
         gTools().ReadAttr(layerXML, "LeftPad", leftPad);
         gTools().ReadAttr(layerXML, "RightPad", rightPad);
         gTools().ReadAttr(layerXML, "TopPad", topPad);
         gTools().ReadAttr(layerXML, "BottomPad", bottomPad);

         fNet->AddPaddingLayer2D(topPad, bottomPad, leftPad, rightPad);

      }      


      // read eventually weights and biases
      fNet->GetLayers().back()->ReadWeightsFromXML(layerXML);

      // read next layer
      layerXML = gTools().GetNextChild(layerXML);
   }
}

////////////////////////////////////////////////////////////////////////////////
void MethodDL::ReadWeightsFromStream(std::istream & /*istr*/)
{
}

////////////////////////////////////////////////////////////////////////////////
const Ranking *TMVA::MethodDL::CreateRanking()
{
   // TODO
   return NULL;
}

////////////////////////////////////////////////////////////////////////////////
void MethodDL::GetHelpMessage() const
{
   // TODO
}

} // namespace TMVA
