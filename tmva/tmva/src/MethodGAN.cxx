// @(#)root/tmva/tmva/cnn:$Id$
// Author: Vladimir Ilievski, Saurav Shekhar

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodGAN                                                              *
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
#include "TMVA/MethodGAN.h"
#include "TMVA/Types.h"
#include "TMVA/DNN/TensorDataLoader.h"
#include "TMVA/DNN/Functions.h"
#include "TMVA/DNN/DLMinimizers.h"
#include "TStopwatch.h"
#include "TMVA/MethodDL.h"

#include <chrono>

REGISTER_METHOD(GAN)
ClassImp(TMVA::MethodGAN);

using namespace TMVA::DNN::CNN;
using namespace TMVA::DNN;

using TMVA::DNN::EActivationFunction;
using TMVA::DNN::ELossFunction;
using TMVA::DNN::EInitialization;
using TMVA::DNN::EOutputFunction;

namespace TMVA {

////////////////////////////////////////////////////////////////////////////////
TString getValueTmp(const std::map<TString, TString> &keyValueMap, TString key)
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
T getValueTmp(const std::map<TString, TString> &keyValueMap, TString key, T defaultValue);

////////////////////////////////////////////////////////////////////////////////
template <>
int getValueTmp(const std::map<TString, TString> &keyValueMap, TString key, int defaultValue)
{
   TString value(getValueTmp(keyValueMap, key));
   if (value == "") {
      return defaultValue;
   }
   return value.Atoi();
}

////////////////////////////////////////////////////////////////////////////////
template <>
double getValueTmp(const std::map<TString, TString> &keyValueMap, TString key, double defaultValue)
{
   TString value(getValueTmp(keyValueMap, key));
   if (value == "") {
      return defaultValue;
   }
   return value.Atof();
}

////////////////////////////////////////////////////////////////////////////////
template <>
TString getValueTmp(const std::map<TString, TString> &keyValueMap, TString key, TString defaultValue)
{
   TString value(getValueTmp(keyValueMap, key));
   if (value == "") {
      return defaultValue;
   }
   return value;
}

////////////////////////////////////////////////////////////////////////////////
template <>
bool getValueTmp(const std::map<TString, TString> &keyValueMap, TString key, bool defaultValue)
{
   TString value(getValueTmp(keyValueMap, key));
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
std::vector<double> getValueTmp(const std::map<TString, TString> &keyValueMap, TString key,
                                  std::vector<double> defaultValue)
{
   TString parseString(getValueTmp(keyValueMap, key));
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
void MethodGAN::DeclareOptions()
{
   // Set default values for all option strings
   DeclareOptionRef(fInputLayoutString = "0|0|0##0|0|0", "InputLayout", "The Layout of the input");

   DeclareOptionRef(fBatchLayoutString = "0|0|0##0|0|0", "BatchLayout", "The Layout of the batch");

   DeclareOptionRef(fLayoutString = "DENSE|(N+100)*2|SOFTSIGN,DENSE|0|LINEAR##DENSE|(N+100)*2|SOFTSIGN,DENSE|0|LINEAR", "Layout", "Layout of the network.");

   DeclareOptionRef(fErrorStrategy = "CROSSENTROPY", "ErrorStrategy", "Loss function: Mean squared error (regression)"
                                                                      " or cross entropy (binary classification).");
   AddPreDefVal(TString("CROSSENTROPY"));
   AddPreDefVal(TString("SUMOFSQUARES"));
   AddPreDefVal(TString("MUTUALEXCLUSIVE"));

   DeclareOptionRef(fWeightInitializationString = "XAVIER", "WeightInitialization", "Weight initialization strategy");
   AddPreDefVal(TString("XAVIER"));
   AddPreDefVal(TString("XAVIERUNIFORM"));

   DeclareOptionRef(fArchitectureString = "CPU", "Architecture", "Which architecture to perform the training on.");
   AddPreDefVal(TString("STANDARD"));
   AddPreDefVal(TString("CPU"));
   AddPreDefVal(TString("GPU"));
   AddPreDefVal(TString("OPENCL"));

   DeclareOptionRef(fTrainingStrategyString = "GeneratorLearningRate=1e-1,"
                                              "GeneratorMomentum=0.3,"
                                              "GeneratorRepetitions=3,"
                                              "GeneratorConvergenceSteps=50,"
                                              "GeneratorBatchSize=30,"
                                              "GeneratorTestRepetitions=7,"
                                              "GeneratorWeightDecay=0.0,"
                                              "GeneratorRenormalize=L2,"
                                              "GeneratorDropConfig=0.0,"
                                              "GeneratorDropRepetitions=5,"
                                              "DiscriminatorLearningRate=1e-1,"
                                              "DiscriminatorMomentum=0.3,"
                                              "DiscriminatorRepetitions=3,"
                                              "DiscriminatorConvergenceSteps=50,"
                                              "DiscriminatorBatchSize=30,"
                                              "DiscriminatorTestRepetitions=7,"
                                              "DiscriminatorWeightDecay=0.0,"
                                              "DiscriminatorRenormalize=L2,"
                                              "DiscriminatorDropConfig=0.0,"
                                              "DiscriminatorDropRepetitions=5|"
                                              "GeneratorLearningRate=1e-4,"
                                              "GeneratorMomentum=0.3,"
                                              "GeneratorRepetitions=3,"
                                              "GeneratorConvergenceSteps=50,"
                                              "GeneratorMaxEpochs=2000,"
                                              "GeneratorBatchSize=20,"
                                              "GeneratorTestRepetitions=7,"
                                              "GeneratorWeightDecay=0.001,"
                                              "GeneratorRenormalize=L2,"
                                              "GeneratorDropConfig=0.0+0.5+0.5,"
                                              "GeneratorDropRepetitions=5,"
                                              "GeneratorMultithreading=True,"
					      "DiscriminatorLearningRate=1e-4,"
                                              "DiscriminatorMomentum=0.3,"
                                              "DiscriminatorRepetitions=3,"
                                              "DiscriminatorConvergenceSteps=50,"
                                              "DiscriminatorMaxEpochs=2000,"
                                              "DiscriminatorBatchSize=20,"
                                              "DiscriminatorTestRepetitions=7,"
                                              "DiscriminatorWeightDecay=0.001,"
                                              "DiscriminatorRenormalize=L2,"
                                              "DiscriminatorDropConfig=0.0+0.5+0.5,"
                                              "DiscriminatorDropRepetitions=5,"
                                              "DiscriminatorMultithreading=True",
					       
                           "TrainingStrategy", "Defines the training strategies.");
}  

////////////////////////////////////////////////////////////////////////////////
void MethodGAN::ProcessOptions()
{

   if (IgnoreEventsWithNegWeightsInTraining()) {
      Log() << kINFO << "Will ignore negative events in training!" << Endl;
      }

   if (fArchitectureString == "STANDARD") {
      Log() << kERROR << "The STANDARD architecture has been deprecated. "
                         "Please use Architecture=CPU or Architecture=CPU."
                         "See the TMVA Users' Guide for instructions if you "
                         "encounter problems."
            << Endl;
      Log() << kFATAL << "The STANDARD architecture has been deprecated. "
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
      Log() << kFATAL << "The OPENCL architecture has not been implemented yet. "
                         "Please use Architecture=CPU or Architecture=CPU for the "
                         "time being. See the TMVA Users' Guide for instructions "
                         "if you encounter problems."
            << Endl;
   }

   if (fArchitectureString == "GPU") {
#ifndef R__HAS_TMVACUDA // Included only if DNNCUDA flag is _not_ set.
      Log() << kERROR << "CUDA backend not enabled. Please make sure "
                         "you have CUDA installed and it was successfully "
                         "detected by CMAKE."
            << Endl;
      Log() << kFATAL << "CUDA backend not enabled. Please make sure "
                         "you have CUDA installed and it was successfully "
                         "detected by CMAKE."
            << Endl;
#endif // DNNCUDA
   }
   
  if (fArchitectureString == "CPU") {
#ifndef R__HAS_TMVACPU // Included only if DNNCPU flag is _not_ set.
      Log() << kERROR << "Multi-core CPU backend not enabled. Please make sure "
                         "you have a BLAS implementation and it was successfully "
                         "detected by CMake as well that the imt CMake flag is set."
            << Endl;
      Log() << kFATAL << "Multi-core CPU backend not enabled. Please make sure "
                         "you have a BLAS implementation and it was successfully "
                         "detected by CMake as well that the imt CMake flag is set."
            << Endl;
 #endif // DNNCPU
   }

   // Input Layout
   ParseInputLayout();
   ParseBatchLayout();
   ParseNetworkLayout();

   // Loss function and output.
   fOutputFunction = EOutputFunction::kSigmoid;
   if (fAnalysisType == Types::kClassification) {
      if (fErrorStrategy == "SUMOFSQUARES") {
         std::cout << "Error Strategy String" << fErrorStrategy << std::endl;
         fLossFunction = ELossFunction::kMeanSquaredError;
         
      }
      if (fErrorStrategy == "CROSSENTROPY") {
         std::cout << "Error Strategy String" << fErrorStrategy << std::endl;
         fLossFunction = ELossFunction::kCrossEntropy;
      }
      fOutputFunction = EOutputFunction::kSigmoid;
   } 
      
   else if (fAnalysisType == Types::kRegression) {
      if (fErrorStrategy != "SUMOFSQUARES") {
         Log() << kWARNING << "For regression only SUMOFSQUARES is a valid "
               << " neural net error function. Setting error function to "
               << " SUMOFSQUARES now." << Endl;
      }

      fLossFunction = ELossFunction::kMeanSquaredError;
      fOutputFunction = EOutputFunction::kIdentity;
   } 
   
   else if (fAnalysisType == Types::kMulticlass) {
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
   if (fWeightInitializationString == "XAVIER") {
      fWeightInitialization = DNN::EInitialization::kGauss;
   } else if (fWeightInitializationString == "XAVIERUNIFORM") {
      fWeightInitialization = DNN::EInitialization::kUniform;
   } else {
      fWeightInitialization = DNN::EInitialization::kGauss;
   }

   // Training settings.

   KeyValueVector_t strategyKeyValues = ParseKeyValueString(fTrainingStrategyString, TString("|"), TString(","));
   for (auto &block : strategyKeyValues) {

      GANTTrainingSettings settings;

      settings.generatorConvergenceSteps = getValueTmp(block, "GeneratorConvergenceSteps", 100);
      settings.generatorBatchSize = getValueTmp(block, "GeneratorBatchSize", 30);
      settings.generatorMaxEpochs = getValueTmp(block, "GeneratorMaxEpochs", 2000);
      settings.generatorTestInterval = getValueTmp(block, "GeneratorTestRepetitions", 7);
      settings.generatorWeightDecay = getValueTmp(block, "GeneratorWeightDecay", 0.0);
      settings.generatorLearningRate = getValueTmp(block, "GeneratorLearningRate", 1e-5);
      settings.generatorMomentum = getValueTmp(block, "GeneratorMomentum", 0.3);
      settings.generatorDropoutProbabilities = getValueTmp(block, "GeneratorDropConfig", std::vector<Double_t>());

      TString generatorRegularization = getValueTmp(block, "GeneratorRegularization", TString("NONE"));
      if (generatorRegularization == "L1") {
         settings.generatorRegularization = DNN::ERegularization::kL1;
      } else if (generatorRegularization == "L2") {
         settings.generatorRegularization = DNN::ERegularization::kL2;
      }

      TString generatorStrMultithreading = getValueTmp(block, "GeneratorMultithreading", TString("True"));

      if (generatorStrMultithreading.BeginsWith("T")) {
         settings.generatorMultithreading = true;
      } else {
         settings.generatorMultithreading = false;
      }

      settings.discriminatorConvergenceSteps = getValueTmp(block, "DiscriminatorConvergenceSteps", 100);
      settings.discriminatorBatchSize = getValueTmp(block, "DiscriminatorBatchSize", 256);
      settings.discriminatorMaxEpochs = getValueTmp(block, "DiscriminatorMaxEpochs", 2000);
      settings.discriminatorTestInterval = getValueTmp(block, "DiscriminatorTestRepetitions", 7);
      settings.discriminatorWeightDecay = getValueTmp(block, "DiscriminatorWeightDecay", 0.0);
      settings.discriminatorLearningRate = getValueTmp(block, "DiscriminatorLearningRate", 1e-5);
      settings.discriminatorMomentum = getValueTmp(block, "DiscriminatorMomentum", 0.3);
      settings.discriminatorDropoutProbabilities = getValueTmp(block, "DiscriminatorDropConfig", std::vector<Double_t>());
      
      TString discriminatorRegularization = getValueTmp(block, "DiscriminatorRegularization", TString("NONE"));
      if (discriminatorRegularization == "L1") {
         settings.discriminatorRegularization = DNN::ERegularization::kL1;
      } else if (discriminatorRegularization == "L2") {
         settings.discriminatorRegularization = DNN::ERegularization::kL2;
      }

      TString discriminatorStrMultithreading = getValueTmp(block, "DiscriminatorMultithreading", TString("True"));

      if (discriminatorStrMultithreading.BeginsWith("T")) {
         settings.discriminatorMultithreading = true;
      } else {
         settings.discriminatorMultithreading = false;
      }


      fTrainingSettings.push_back(settings);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// default initializations
void MethodGAN::Init()
{
   // Nothing to do here
}

////////////////////////////////////////////////////////////////////////////////
/// Parse the model layout
void MethodGAN::ParseNetworkLayout()
{
   // Define the delimiter for separation of Generator and Discriminator Strings
   const TString delim_model("##");
   const TString delim("|");

   // Get the input layout string
   TString networkLayoutString = this->GetLayoutString();

   //Split string into Generator and Discriminator layout strings
   TObjArray *modelStrings = networkLayoutString.Tokenize(delim_model);
   TIter nextModelDim(modelStrings);
   TObjString *modelDimString = (TObjString *)nextModelDim();
   int idxTokenModel = 0;

   for(; modelDimString != nullptr; modelDimString = (TObjString *)nextModelDim())
   {
      TString strNetworkLayout(modelDimString->GetString());

      if(idxTokenModel == 0)
         this->SetGeneratorNetworkLayout(strNetworkLayout);
      else if(idxTokenModel == 1)
         this->SetDiscriminatorNetworkLayout(strNetworkLayout);

      ++idxTokenModel;
   }
}


////////////////////////////////////////////////////////////////////////////////
/// Parse the input layout
void MethodGAN::ParseInputLayout()
{
   // Define the delimiter for separation of Generator and Discriminator Strings
   const TString delim_model("##");
   const TString delim("|");

   // Get the input layout string
   TString inputLayoutString = this->GetInputLayoutString();

   size_t depth = 0;
   size_t height = 0;
   size_t width = 0;

   //Split string into Generator and Discriminator layout strings
   TObjArray *modelStrings = inputLayoutString.Tokenize(delim_model);
   TIter nextModelDim(modelStrings);
   TObjString *modelDimString = (TObjString *)nextModelDim();
   int idxTokenModel = 0;

   for(; modelDimString != nullptr; modelDimString = (TObjString *)nextModelDim())
   {
      // Split the input layout string
      TObjArray *inputDimStrings = modelDimString->GetString().Tokenize(delim);
      TIter nextInputDim(inputDimStrings);
      TObjString *inputDimString = (TObjString *)nextInputDim();
      int idxToken = 0;

      for (; inputDimString != nullptr; inputDimString = (TObjString *)nextInputDim()) {
         switch (idxToken) {
         case 0: // input depth
         {
            TString strDepth(inputDimString->GetString());
            depth = (size_t)strDepth.Atoi();

            if(idxTokenModel == 0)
               this->SetGeneratorInputDepth(depth);
            else if(idxTokenModel == 1)
               this->SetDiscriminatorInputDepth(depth);
	
	    break;
         } 
         case 1: // input height
         {
            TString strHeight(inputDimString->GetString());
            height = (size_t)strHeight.Atoi();

            if(idxTokenModel == 0)
               this->SetGeneratorInputHeight(height);
            else if(idxTokenModel == 1)
               this->SetDiscriminatorInputHeight(height);

            break;
         }
         case 2: // input width
         {
            TString strWidth(inputDimString->GetString());
            width = (size_t)strWidth.Atoi();

            if(idxTokenModel == 0)
               this->SetGeneratorInputWidth(width);
            else if(idxTokenModel == 1)
               this->SetDiscriminatorInputWidth(width);

            break;
         }
         }
         ++idxToken;
      }
   
   ++idxTokenModel;
  
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Parse the batch layout
void MethodGAN::ParseBatchLayout()
{
   // Define the delimiter
   const TString delim_model("##");
   const TString delim("|");

   // Get the input layout string
   TString batchLayoutString = this->GetBatchLayoutString();

   size_t depth = 0;
   size_t height = 0;
   size_t width = 0;

   // Split the input layout string into Generator and Discriminator Strings
   TObjArray *modelDimStrings = batchLayoutString.Tokenize(delim_model);
   TIter nextModelDim(modelDimStrings);
   TObjString *modelDimString = (TObjString *)nextModelDim();
   int idxTokenModel = 0; 

   for(; modelDimString != nullptr; modelDimString = (TObjString *)nextModelDim())
   {

      TObjArray *batchDimStrings = modelDimString->GetString().Tokenize(delim);
      TIter nextBatchDim(batchDimStrings);
      TObjString *batchDimString = (TObjString *)nextBatchDim();
      int idxToken = 0;

      for (; batchDimString != nullptr; batchDimString = (TObjString *)nextBatchDim()) {
         switch (idxToken) {
         case 0: // input depth
         {
            TString strDepth(batchDimString->GetString());
            depth = (size_t)strDepth.Atoi();
    
            if(idxTokenModel == 0)
               this->SetGeneratorBatchDepth(depth);
            else if(idxTokenModel == 1)
               this->SetDiscriminatorBatchDepth(depth);

            break;
         }
         case 1: // input height
         {
            TString strHeight(batchDimString->GetString());
            height = (size_t)strHeight.Atoi();

            if(idxTokenModel == 0)
               this->SetGeneratorBatchHeight(height);
            else if(idxTokenModel == 1)
               this->SetDiscriminatorBatchHeight(height);

            break;
         }
         case 2: // input width
         {
            TString strWidth(batchDimString->GetString());
            width = (size_t)strWidth.Atoi();

            if(idxTokenModel == 0)
               this->SetGeneratorBatchWidth(width);
            else if(idxTokenModel == 1)
               this->SetDiscriminatorBatchWidth(width);

            break;
         }
         }
      
         ++idxToken;
      }
      ++idxTokenModel;
   }
}


////////////////////////////////////////////////////////////////////////////////
/// Create a deep net based on the layout string
template <typename Architecture_t, typename Layer_t>
void MethodGAN::CreateDeepNet(DNN::TDeepNet<Architecture_t, Layer_t> &deepNet,
                             std::vector<DNN::TDeepNet<Architecture_t, Layer_t>> &nets, std::unique_ptr<DeepNetImpl_t> &fNet, TString layoutString)
{

   // Layer specification, layer details
   const TString layerDelimiter(",");
   const TString subDelimiter("|");

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
      
      const size_t inputSize = GetEvent()->GetNVariables();

      if (strLayerType == "DENSE") {
         MethodDL::ParseDenseLayer(inputSize, deepNet, nets, layerString->GetString(), subDelimiter, fNet);
      } else if (strLayerType == "CONV") {
         MethodDL::ParseConvLayer(deepNet, nets, layerString->GetString(), subDelimiter, fNet);
      } else if (strLayerType == "MAXPOOL") {
         MethodDL::ParseMaxPoolLayer(deepNet, nets, layerString->GetString(), subDelimiter, fNet);
      } else if (strLayerType == "RESHAPE") {
         MethodDL::ParseReshapeLayer(deepNet, nets, layerString->GetString(), subDelimiter, fNet);
      } else if (strLayerType == "RNN") {
         MethodDL::ParseRnnLayer(deepNet, nets, layerString->GetString(), subDelimiter, fNet);
      } else if (strLayerType == "LSTM") {
         Log() << kFATAL << "LSTM Layer is not yet fully implemented" << Endl;
         //MethodDL::ParseLstmLayer(deepNet, nets, layerString->GetString(), subDelimiter);
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Standard constructor.
MethodGAN::MethodGAN(const TString &jobName, const TString &methodTitle, DataSetInfo &theData, const TString &theOption)
   : MethodBase(jobName, Types::kGAN, methodTitle, theData, theOption)
{
   // Nothing to do here
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor from a weight file.
MethodGAN::MethodGAN(DataSetInfo &theData, const TString &theWeightFile)
   : MethodBase(Types::kGAN, theData, theWeightFile)
{
   // Nothing to do here
}
//
////////////////////////////////////////////////////////////////////////////////
/// Constructor from a weight file.

////////////////////////////////////////////////////////////////////////////////
/// Destructor.
MethodGAN::~MethodGAN()
{
   // Nothing to do here
}

////////////////////////////////////////////////////////////////////////////////
/// Parse key value pairs in blocks -> return vector of blocks with map of key value pairs.
auto MethodGAN::ParseKeyValueString(TString parseString, TString blockDelim, TString tokenDelim) -> KeyValueVector_t
{
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
Bool_t MethodGAN::HasAnalysisType(Types::EAnalysisType type, UInt_t numberClasses, UInt_t /*numberTargets*/)
{
   if (type == Types::kClassification && numberClasses == 2) return kTRUE;
   if (type == Types::kMulticlass) return kTRUE;
   if (type == Types::kRegression) return kTRUE;

   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
void MethodGAN::Train()
{
   if (fInteractive) {
      Log() << kFATAL << "Not implemented yet" << Endl;
      return;
   }

   if (this->GetArchitectureString() == "GPU") {
#ifdef R__HAS_TMVACUDA
      Log() << kINFO << "Start of deep neural network training on GPU." << Endl << Endl;
#else
      Log() << kFATAL << "CUDA backend not enabled. Please make sure "
         "you have CUDA installed and it was successfully "
         "detected by CMAKE."
             << Endl;
      return;
#endif
   } else if (this->GetArchitectureString() == "OpenCL") {
      Log() << kFATAL << "OpenCL backend not yet supported." << Endl;
      return;
   } else if (this->GetArchitectureString() == "CPU") {
#ifdef R__HAS_TMVACPU
      Log() << kINFO << "Start of deep neural network training on CPU." << Endl << Endl;
#else
      Log() << kFATAL << "Multi-core CPU backend not enabled. Please make sure "
                      "you have a BLAS implementation and it was successfully "
                      "detected by CMake as well that the imt CMake flag is set."
            << Endl;
      return;
#endif
   }

/// definitions for CUDA
#ifdef R__HAS_TMVACUDA // Included only if DNNCUDA flag is set.
   using Architecture_t = DNN::TCuda<Double_t>;
#else
#ifdef R__HAS_TMVACPU // Included only if DNNCPU flag is set.
   using Architecture_t = DNN::TCpu<Double_t>;
#else
   using Architecture_t = DNN::TReference<Double_t>;
#endif
#endif

   using Scalar_t = Architecture_t::Scalar_t;
   using DeepNet_t = TMVA::DNN::TDeepNet<Architecture_t>;
   using TensorDataLoader_t = TTensorDataLoader<TMVAInput_t, Architecture_t>;

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

   size_t trainingPhase = 1;
   for (GANTTrainingSettings &settings : this->GetTrainingSettings()) {

      size_t nThreads = 1;       // FIXME threads are hard coded to 1, no use of slave threads or multi-threading

      Log() << "Training phase " << trainingPhase << " of " << this->GetTrainingSettings().size() << ":" << Endl;
      trainingPhase++;

      // After the processing of the options, initialize the master deep net
      size_t generatorBatchSize = settings.discriminatorBatchSize;
      size_t discriminatorBatchSize = settings.discriminatorBatchSize;

      // Should be replaced by actual implementation. No support for this now.
      size_t generatorInputDepth = this->GetGeneratorInputDepth();
      size_t generatorInputHeight = this->GetGeneratorInputHeight();
      size_t generatorInputWidth = this->GetGeneratorInputWidth();
      size_t generatorBatchDepth = this->GetGeneratorBatchDepth();
      size_t generatorBatchHeight = this->GetGeneratorBatchHeight();
      size_t generatorBatchWidth = this->GetGeneratorBatchWidth();

      size_t discriminatorInputDepth = this->GetDiscriminatorInputDepth();
      size_t discriminatorInputHeight = this->GetDiscriminatorInputHeight();
      size_t discriminatorInputWidth = this->GetDiscriminatorInputWidth();
      size_t discriminatorBatchDepth = this->GetDiscriminatorBatchDepth();
      size_t discriminatorBatchHeight = this->GetDiscriminatorBatchHeight();
      size_t discriminatorBatchWidth = this->GetDiscriminatorBatchWidth();

      ELossFunction generatorJ = this->GetLossFunction();
      EInitialization generatorI = this->GetWeightInitialization();
      ERegularization generatorR = settings.generatorRegularization;
      Scalar_t generatorWeightDecay = settings.generatorWeightDecay;
       
      ELossFunction discriminatorJ = this->GetLossFunction();     
      EInitialization discriminatorI = this->GetWeightInitialization();
      ERegularization discriminatorR = settings.discriminatorRegularization;
      Scalar_t discriminatorWeightDecay = settings.discriminatorWeightDecay;



/////////////////////////////////////////////////////////////////////////////////////////////////////   
      //Settings for Generator Model

      //Batch size should be included in batch layout as well. There are two possibilities:
      //  1.  Batch depth = batch size   one will input tensors as (batch_size x d1 x d2)
      //       This is case for example if first layer is a conv layer and d1 = image depth, d2 = image width x image height
      //  2.  Batch depth = 1, batch height = batch size  batxch width = dim of input features
      //        This should be case if first layer is a Dense 1 and input tensor must be ( 1 x batch_size x input_features )

      if (generatorBatchDepth != settings.generatorBatchSize && generatorBatchDepth > 1) {
         Error("TrainCpu","Given batch depth of %zu (specified in BatchLayout)  should be equal to given batch size %zu",generatorBatchDepth,settings.generatorBatchSize);
         return;
      }
      if (generatorBatchDepth == 1 && settings.generatorBatchSize > 1 && settings.generatorBatchSize != generatorBatchHeight ) {
         Error("TrainCpu","Given batch height of %zu (specified in BatchLayout)  should be equal to given batch size %zu",generatorBatchHeight,settings.generatorBatchSize);
         return;
      }


      //check also that input layout compatible with batch layout
      bool genBadLayout = false;
      // case batch depth == batch size
      if (generatorBatchDepth == settings.generatorBatchSize)
         genBadLayout = ( generatorInputDepth * generatorInputHeight * generatorInputWidth != generatorBatchHeight * generatorBatchWidth ) ;
      // case batch Height is batch size
      if (generatorBatchHeight == generatorBatchSize && generatorBatchDepth == 1) 
         genBadLayout |=  ( generatorInputDepth * generatorInputHeight * generatorInputWidth !=  generatorBatchWidth);
      if (genBadLayout) {
         Error("TrainCpu","Given input layout %zu x %zu x %zu is not compatible with  batch layout %zu x %zu x  %zu ",
               generatorInputDepth,generatorInputHeight,generatorInputWidth,generatorBatchDepth,generatorBatchHeight,generatorBatchWidth);
         return;
      }

      DeepNet_t generatorDeepNet(generatorBatchSize, generatorInputDepth, generatorInputHeight, generatorInputWidth, generatorBatchDepth, generatorBatchHeight, generatorBatchWidth, generatorJ, generatorI, generatorR, generatorWeightDecay);

      // create a copy of DeepNet for evaluating but with batch size = 1
      // fNet is the saved network and will be with CPU or Referrence architecture
      generatorFNet = std::unique_ptr<DeepNetImpl_t>(new DeepNetImpl_t(1, generatorInputDepth, generatorInputHeight, generatorInputWidth, generatorBatchDepth, generatorBatchHeight, generatorBatchWidth, generatorJ, generatorI, generatorR, generatorWeightDecay));

      // Initialize the vector of slave nets
      std::vector<DeepNet_t> generatorNets{};
      generatorNets.reserve(nThreads);
      for (size_t i = 0; i < nThreads; i++) {
         // create a copies of the master deep net
         generatorNets.push_back(generatorDeepNet);
      }

      TMVAInput_t generatorTrainingTuple = std::tie(GetEventCollection(Types::kTraining), DataInfo());

      // Add all appropriate layers to deepNet and copies to fNet
      CreateDeepNet(generatorDeepNet, generatorNets, generatorFNet, this->GetGeneratorNetworkLayoutString());

      // print the created network
      std::cout << "***** Generator Deep Learning Network *****\n";
      generatorDeepNet.Print();

      // Loading the training and testing datasets
      //TMVAInput_t generatorTrainingTuple = std::tie(GetEventCollection(Types::kTraining), DataInfo());
      TensorDataLoader_t generatorTrainingData(generatorTrainingTuple, nTrainingSamples, generatorDeepNet.GetBatchSize(),
                                      generatorDeepNet.GetBatchDepth(), generatorDeepNet.GetBatchHeight(), generatorDeepNet.GetBatchWidth(),
                                      generatorDeepNet.GetOutputWidth(), nThreads);

      TMVAInput_t generatorTestTuple = std::tie(GetEventCollection(Types::kTesting), DataInfo());
      TensorDataLoader_t generatorTestingData(generatorTestTuple, nTestSamples, generatorDeepNet.GetBatchSize(),
                                     generatorDeepNet.GetBatchDepth(), generatorDeepNet.GetBatchHeight(), generatorDeepNet.GetBatchWidth(),
                                     generatorDeepNet.GetOutputWidth(), nThreads);

      // Initialize the minimizer
      DNN::TDLGradientDescent<Architecture_t> generatorMinimizer(settings.generatorLearningRate, settings.generatorConvergenceSteps,
                                                        settings.generatorTestInterval);

      // Initialize the vector of batches, one batch for one slave network
      std::vector<TTensorBatch<Architecture_t>> generatorBatches{};

      bool generatorConverged = false;
      // count the steps until the convergence
      size_t generatorStepCount = 0;
      size_t generatorBatchesInEpoch = nTrainingSamples / generatorDeepNet.GetBatchSize();

      // start measuring
      std::chrono::time_point<std::chrono::system_clock> gen_tstart, gen_tend;
      gen_tstart = std::chrono::system_clock::now();

      if (!fInteractive) {
         Log() << std::setw(10) << "Epoch"
               << " | " << std::setw(12) << "Train Err." << std::setw(12) << "Test  Err." << std::setw(12) << "GFLOP/s"
               << std::setw(16) << "time(s)/epoch" << std::setw(12) << "Conv. Steps" << Endl;
         std::string separator(62, '-');
         Log() << separator << Endl;
      }

      Double_t genMinTestError = 0;
      // use generator with 0 seed to get always different values 
      RandomGenerator<TRandom3> genRng(0);   
      while (!generatorConverged) {
         generatorStepCount++;
         generatorTrainingData.Shuffle(genRng);

         // execute all epochs
         //for (size_t i = 0; i < batchesInEpoch; i += nThreads) {
         //std::cout << "Loop on batches " <<  batchesInEpoch << std::endl;
         for (size_t i = 0; i < generatorBatchesInEpoch; ++i ) {
            // Clean and load new batches, one batch for one slave net
            //batches.clear();
            //batches.reserve(nThreads);
            //for (size_t j = 0; j < nThreads; j++) {
            //   batches.push_back(trainingData.GetTensorBatch());
            //}

            auto generatorMy_batch = generatorTrainingData.GetTensorBatch();

            //std::cout << "retrieve batch # " << i << " data " << my_batch.GetInput()[0](0,0) << std::endl;

            //std::cout << "input size " << my_batch.GetInput().size() << " matrix  " << my_batch.GetInput().front().GetNrows() << " x " << my_batch.GetInput().front().GetNcols()   << std::endl;

         // execute one minimization step
         // StepMomentum is currently not written for single thread, TODO write it
            if (settings.generatorMomentum > 0.0) {
               //minimizer.StepMomentum(generatorDeepNet, nets, batches, settings.momentum);
               generatorMinimizer.Step(generatorDeepNet, generatorMy_batch.GetInput(), generatorMy_batch.GetOutput(), generatorMy_batch.GetWeights());
            } else {
               //minimizer.Step(generatorDeepNet, nets, batches);
               generatorMinimizer.Step(generatorDeepNet, generatorMy_batch.GetInput(), generatorMy_batch.GetOutput(), generatorMy_batch.GetWeights());
            }
         }
         //}


         if ((generatorStepCount % generatorMinimizer.GetTestInterval()) == 0) {

            std::chrono::time_point<std::chrono::system_clock> gen_t1,gen_t2; 

            gen_t1 = std::chrono::system_clock::now();

            // Compute test error.
            Double_t generatorTestError = 0.0;
            for (auto batch : generatorTestingData) {
               auto inputTensor = batch.GetInput();
               auto outputMatrix = batch.GetOutput();
               auto weights = batch.GetWeights();
               generatorTestError += generatorDeepNet.Loss(inputTensor, outputMatrix, weights);
            }


            gen_t2 = std::chrono::system_clock::now();
            generatorTestError /= (Double_t)(nTestSamples / settings.generatorBatchSize);
            // copy configuration when reached a minimum error
            if (generatorTestError < genMinTestError ) {
               // Copy weights from deepNet to fNet
               Log() << std::setw(10) << generatorStepCount << " Minimum Test error found - save the configuration " << Endl;
               for (size_t i = 0; i < generatorDeepNet.GetDepth(); ++i) {
                  const auto & generatorNLayer = generatorFNet->GetLayerAt(i); 
                  const auto & generatorDLayer = generatorDeepNet.GetLayerAt(i); 
                  generatorNLayer->CopyWeights(generatorDLayer->GetWeights()); 
                  generatorNLayer->CopyBiases(generatorDLayer->GetBiases());
                  // std::cout << "Weights for layer " << i << std::endl;
                  // for (size_t k = 0; k < dlayer->GetWeights().size(); ++k) 
                  //    dLayer->GetWeightsAt(k).Print(); 
               }
               genMinTestError = generatorTestError;
            }
            else if ( genMinTestError <= 0. )
               genMinTestError = generatorTestError; 


            Double_t generatorTrainingError = 0.0;
            // Compute training error.
            for (auto batch : generatorTrainingData) {
               auto inputTensor = batch.GetInput();
               auto outputMatrix = batch.GetOutput();
               auto weights = batch.GetWeights();

               //std::cout << "After  size " << batch.GetInput().size() << " matrix  " << batch.GetInput().front().GetNrows() << " x " << batch.GetInput().front().GetNcols()   << std::endl;

               generatorTrainingError += generatorDeepNet.Loss(inputTensor, outputMatrix, weights);
            }
            generatorTrainingError /= (Double_t)(nTrainingSamples / settings.generatorBatchSize);

            // stop measuring
            gen_tend = std::chrono::system_clock::now();

            // Compute numerical throughput.
            std::chrono::duration<double> gen_elapsed_seconds = gen_tend - gen_tstart;
            std::chrono::duration<double> gen_elapsed1 = gen_t1 - gen_tstart;
            std::chrono::duration<double> gen_elapsed2 = gen_t2 - gen_tstart;

            double gen_seconds = gen_elapsed_seconds.count();
            double gen_nFlops = (double)(settings.generatorTestInterval * generatorBatchesInEpoch);
            // nFlops *= net.GetNFlops() * 1e-9;

            generatorConverged = generatorMinimizer.HasConverged(generatorTestError) || generatorStepCount >= settings.generatorMaxEpochs;

            Log() << std::setw(10) << generatorStepCount << " | " << std::setw(12) << generatorTrainingError << std::setw(12) << generatorTestError
                  << std::setw(12) << gen_nFlops / gen_seconds << std::setw(12)
                  << std::setw(12) << gen_seconds/settings.generatorTestInterval 
                  << std::setw(12) << generatorMinimizer.GetConvergenceCount()
                  <<  std::setw(12) << gen_elapsed1.count()
                  << std::setw(12) << gen_elapsed2.count() 
                  << std::setw(12) << gen_seconds 

                  << Endl;

            if (generatorConverged) {
               Log() << Endl;
            }
            gen_tstart = std::chrono::system_clock::now();
         }
      }

/////////////////////////////////////////////////////////////////////////////////////////////////////   
      //Settings for Discriminator Model

      //Batch size should be included in batch layout as well. There are two possibilities:
      //  1.  Batch depth = batch size   one will input tensors as (batch_size x d1 x d2)
      //       This is case for example if first layer is a conv layer and d1 = image depth, d2 = image width x image height
      //  2.  Batch depth = 1, batch height = batch size  batxch width = dim of input features
      //        This should be case if first layer is a Dense 1 and input tensor must be ( 1 x batch_size x input_features )

      if (discriminatorBatchDepth != settings.discriminatorBatchSize && discriminatorBatchDepth > 1) {
         Error("TrainCpu","Given batch depth of %zu (specified in BatchLayout)  should be equal to given batch size %zu",discriminatorBatchDepth,settings.discriminatorBatchSize);
         return;
      }
      if (discriminatorBatchDepth == 1 && settings.discriminatorBatchSize > 1 && settings.discriminatorBatchSize != discriminatorBatchHeight ) {
         Error("TrainCpu","Given batch height of %zu (specified in BatchLayout)  should be equal to given batch size %zu",discriminatorBatchHeight,settings.discriminatorBatchSize);
         return;
      }


      //check also that input layout compatible with batch layout
      bool disBadLayout = false;
      // case batch depth == batch size
      if (discriminatorBatchDepth == settings.discriminatorBatchSize)
         disBadLayout = ( discriminatorInputDepth * discriminatorInputHeight * discriminatorInputWidth != discriminatorBatchHeight * discriminatorBatchWidth ) ;
      // case batch Height is batch size
      if (discriminatorBatchHeight == discriminatorBatchSize && discriminatorBatchDepth == 1) 
         disBadLayout |=  ( discriminatorInputDepth * discriminatorInputHeight * discriminatorInputWidth !=  discriminatorBatchWidth);
      if (disBadLayout) {
         Error("TrainCpu","Given input layout %zu x %zu x %zu is not compatible with  batch layout %zu x %zu x  %zu ",
               discriminatorInputDepth,discriminatorInputHeight,discriminatorInputWidth,discriminatorBatchDepth,discriminatorBatchHeight,discriminatorBatchWidth);
         return;
      }


      DeepNet_t discriminatorDeepNet(discriminatorBatchSize, discriminatorInputDepth, discriminatorInputHeight, discriminatorInputWidth, discriminatorBatchDepth, discriminatorBatchHeight, discriminatorBatchWidth, discriminatorJ, discriminatorI, discriminatorR, discriminatorWeightDecay);

      // create a copy of DeepNet for evaluating but with batch size = 1
      // fNet is the saved network and will be with CPU or Referrence architecture
      discriminatorFNet = std::unique_ptr<DeepNetImpl_t>(new DeepNetImpl_t(1, discriminatorInputDepth, discriminatorInputHeight, discriminatorInputWidth, discriminatorBatchDepth, discriminatorBatchHeight, discriminatorBatchWidth, discriminatorJ, discriminatorI, discriminatorR, discriminatorWeightDecay));

      // Initialize the vector of slave nets
      std::vector<DeepNet_t> discriminatorNets{};
      discriminatorNets.reserve(nThreads);
      for (size_t i = 0; i < nThreads; i++) {
         // create a copies of the master deep net
         discriminatorNets.push_back(discriminatorDeepNet);
      }

      // Add all appropriate layers to deepNet and copies to fNet
      CreateDeepNet(discriminatorDeepNet, discriminatorNets, discriminatorFNet, this->GetDiscriminatorNetworkLayoutString());

      // print the created network
      std::cout << "***** Generator Deep Learning Network *****\n";
      discriminatorDeepNet.Print();

      // Loading the training and testing datasets
      TMVAInput_t discriminatorTrainingTuple = std::tie(GetEventCollection(Types::kTraining), DataInfo());
      TensorDataLoader_t discriminatorTrainingData(discriminatorTrainingTuple, nTrainingSamples, discriminatorDeepNet.GetBatchSize(),
                                      discriminatorDeepNet.GetBatchDepth(), discriminatorDeepNet.GetBatchHeight(), discriminatorDeepNet.GetBatchWidth(),
                                      discriminatorDeepNet.GetOutputWidth(), nThreads);

      TMVAInput_t discriminatorTestTuple = std::tie(GetEventCollection(Types::kTesting), DataInfo());
      TensorDataLoader_t discriminatorTestingData(discriminatorTestTuple, nTestSamples, discriminatorDeepNet.GetBatchSize(),
                                     discriminatorDeepNet.GetBatchDepth(), discriminatorDeepNet.GetBatchHeight(), discriminatorDeepNet.GetBatchWidth(),
                                     discriminatorDeepNet.GetOutputWidth(), nThreads);

      // Initialize the minimizer
      DNN::TDLGradientDescent<Architecture_t> discriminatorMinimizer(settings.discriminatorLearningRate, settings.discriminatorConvergenceSteps,
                                                        settings.discriminatorTestInterval);

      // Initialize the vector of batches, one batch for one slave network
      std::vector<TTensorBatch<Architecture_t>> discriminatorBatches{};

      bool discriminatorConverged = false;
      // count the steps until the convergence
      size_t discriminatorStepCount = 0;
      size_t discriminatorBatchesInEpoch = nTrainingSamples / discriminatorDeepNet.GetBatchSize();

      // start measuring
      std::chrono::time_point<std::chrono::system_clock> dis_tstart, dis_tend;
      dis_tstart = std::chrono::system_clock::now();

      if (!fInteractive) {
         Log() << std::setw(10) << "Epoch"
               << " | " << std::setw(12) << "Train Err." << std::setw(12) << "Test  Err." << std::setw(12) << "GFLOP/s"
               << std::setw(16) << "time(s)/epoch" << std::setw(12) << "Conv. Steps" << Endl;
         std::string separator(62, '-');
         Log() << separator << Endl;
      }

      Double_t disMinTestError = 0;
      // use discriminator with 0 seed to get always different values 
      RandomGenerator<TRandom3> disRng(0);   
      while (!discriminatorConverged) {
         discriminatorStepCount++;
         discriminatorTrainingData.Shuffle(disRng);

         // execute all epochs
         //for (size_t i = 0; i < discriminatorBatchesInEpoch; i += nThreads) {
         //std::cout << "Loop on batches " <<  discriminatorBatchesInEpoch << std::endl;
         for (size_t i = 0; i < discriminatorBatchesInEpoch; ++i ) {
            // Clean and load new batches, one batch for one slave net
            //batches.clear();
            //batches.reserve(nThreads);
            //for (size_t j = 0; j < nThreads; j++) {
            //   batches.push_back(trainingData.GetTensorBatch());
            //}

            auto discriminatorMy_batch = discriminatorTrainingData.GetTensorBatch();

            //std::cout << "retrieve batch # " << i << " data " << my_batch.GetInput()[0](0,0) << std::endl;

            //std::cout << "input size " << my_batch.GetInput().size() << " matrix  " << my_batch.GetInput().front().GetNrows() << " x " << my_batch.GetInput().front().GetNcols()   << std::endl;

         // execute one minimization step
         // StepMomentum is currently not written for single thread, TODO write it
            if (settings.discriminatorMomentum > 0.0) {
               //minimizer.StepMomentum(deepNet, nets, batches, settings.momentum);
               discriminatorMinimizer.Step(discriminatorDeepNet, discriminatorMy_batch.GetInput(), discriminatorMy_batch.GetOutput(), discriminatorMy_batch.GetWeights());
            } else {
               //minimizer.Step(deepNet, nets, batches);
               discriminatorMinimizer.Step(discriminatorDeepNet, discriminatorMy_batch.GetInput(), discriminatorMy_batch.GetOutput(), discriminatorMy_batch.GetWeights());
            }
         }
         //}


         if ((discriminatorStepCount % discriminatorMinimizer.GetTestInterval()) == 0) {

            std::chrono::time_point<std::chrono::system_clock> dis_t1,dis_t2; 

            dis_t1 = std::chrono::system_clock::now();

            // Compute test error.
            Double_t discriminatorTestError = 0.0;
            for (auto batch : discriminatorTestingData) {
               auto inputTensor = batch.GetInput();
               auto outputMatrix = batch.GetOutput();
               auto weights = batch.GetWeights();
               discriminatorTestError += discriminatorDeepNet.Loss(inputTensor, outputMatrix, weights);
            }


            dis_t2 = std::chrono::system_clock::now();
            discriminatorTestError /= (Double_t)(nTestSamples / settings.discriminatorBatchSize);
            // copy configuration when reached a minimum error
            if (discriminatorTestError < disMinTestError ) {
               // Copy weights from deepNet to fNet
               Log() << std::setw(10) << discriminatorStepCount << " Minimun Test error found - save the configuration " << Endl;
               for (size_t i = 0; i < discriminatorDeepNet.GetDepth(); ++i) {
                  const auto & discriminatorNLayer = discriminatorFNet->GetLayerAt(i); 
                  const auto & discriminatorDLayer = discriminatorDeepNet.GetLayerAt(i); 
                  discriminatorNLayer->CopyWeights(discriminatorDLayer->GetWeights()); 
                  discriminatorNLayer->CopyBiases(discriminatorDLayer->GetBiases());
                  // std::cout << "Weights for layer " << i << std::endl;
                  // for (size_t k = 0; k < dlayer->GetWeights().size(); ++k) 
                  //    dLayer->GetWeightsAt(k).Print(); 
               }
               disMinTestError = discriminatorTestError;
            }
            else if ( disMinTestError <= 0. )
               disMinTestError = discriminatorTestError; 


            Double_t discriminatorTrainingError = 0.0;
            // Compute training error.
            for (auto batch : discriminatorTrainingData) {
               auto inputTensor = batch.GetInput();
               auto outputMatrix = batch.GetOutput();
               auto weights = batch.GetWeights();

               //std::cout << "After  size " << batch.GetInput().size() << " matrix  " << batch.GetInput().front().GetNrows() << " x " << batch.GetInput().front().GetNcols()   << std::endl;

               discriminatorTrainingError += discriminatorDeepNet.Loss(inputTensor, outputMatrix, weights);
            }
            discriminatorTrainingError /= (Double_t)(nTrainingSamples / settings.discriminatorBatchSize);

            // stop measuring
            dis_tend = std::chrono::system_clock::now();

            // Compute numerical throughput.
            std::chrono::duration<double> dis_elapsed_seconds = dis_tend - dis_tstart;
            std::chrono::duration<double> dis_elapsed1 = dis_t1 - dis_tstart;
            std::chrono::duration<double> dis_elapsed2 = dis_t2 - dis_tstart;

            double dis_seconds = dis_elapsed_seconds.count();
            double dis_nFlops = (double)(settings.discriminatorTestInterval * discriminatorBatchesInEpoch);
            // nFlops *= net.GetNFlops() * 1e-9;

            discriminatorConverged = discriminatorMinimizer.HasConverged(discriminatorTestError) || discriminatorStepCount >= settings.discriminatorMaxEpochs;

            Log() << std::setw(10) << discriminatorStepCount << " | " << std::setw(12) << discriminatorTrainingError << std::setw(12) << discriminatorTestError
                  << std::setw(12) << dis_nFlops / dis_seconds << std::setw(12)
                  << std::setw(12) << dis_seconds/settings.discriminatorTestInterval 
                  << std::setw(12) << discriminatorMinimizer.GetConvergenceCount()
                  <<  std::setw(12) << dis_elapsed1.count()
                  << std::setw(12) << dis_elapsed2.count() 
                  << std::setw(12) << dis_seconds 

                  << Endl;

            if (discriminatorConverged) {
               Log() << Endl;
            }
            dis_tstart = std::chrono::system_clock::now();
         }
      }


   }

}

////////////////////////////////////////////////////////////////////////////////
Double_t MethodGAN::GetMvaValue(Double_t * /*errLower*/, Double_t * /*errUpper*/)
{
   using Matrix_t = typename ArchitectureImpl_t::Matrix_t;

   int nVariables = GetEvent()->GetNVariables();
   int batchWidth = generatorFNet->GetBatchWidth();
   int batchDepth = generatorFNet->GetBatchDepth();
   int batchHeight = generatorFNet->GetBatchHeight();
   int nb = generatorFNet->GetBatchSize();
   int noutput = generatorFNet->GetOutputWidth();
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
   if (batchDepth == 1 && GetGeneratorInputHeight() == 1 && GetGeneratorInputDepth() == 1) n1 = 1;

   X.emplace_back(Matrix_t(n1, n2));

   if (n1 > 1) {
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
   generatorFNet->Prediction(YHat, X, fOutputFunction);

   double mvaValue = YHat(0, 0);

   // for debugging
// #ifdef DEBUG
//    TMatrixF  xInput(n1,n2, inputValues.data() ); 
//    std::cout << "Input data - class " << GetEvent()->GetClass() << std::endl;
//    xInput.Print(); 
//    std::cout << "Output of DeepNet " << mvaValue << std::endl;
//    auto & deepnet = *fNet; 
//    const auto *  rnn = deepnet.GetLayerAt(0);
//    const auto & rnn_output = rnn->GetOutput();
//    std::cout << "DNN output " << rnn_output.size() << std::endl;
//    for (size_t i = 0; i < rnn_output.size(); ++i) {
//       TMatrixD m(rnn_output[i].GetNrows(), rnn_output[i].GetNcols() , rnn_output[i].GetRawDataPointer()  );
//       m.Print();
//       //rnn_output[i].Print();
//    }
// #endif
//    std::cout << " { " << GetEvent()->GetClass() << "  , " << mvaValue << " } ";
 
   
   return (TMath::IsNaN(mvaValue)) ? -999. : mvaValue;

}

////////////////////////////////////////////////////////////////////////////////
void MethodGAN::AddWeightsXMLTo(void * parent) const
{
      // Create the parrent XML node with name "Weights"
   auto & xmlEngine = gTools().xmlengine(); 
   void* nn = xmlEngine.NewChild(parent, 0, "Weights");
   
   /*! Get all necessary information, in order to be able to reconstruct the net 
    *  if we read the same XML file. */

   // Deep Net specific info
   Int_t depth = generatorFNet->GetDepth();

   Int_t inputDepth = generatorFNet->GetInputDepth();
   Int_t inputHeight = generatorFNet->GetInputHeight();
   Int_t inputWidth = generatorFNet->GetInputWidth();

   Int_t batchSize = generatorFNet->GetBatchSize();

   Int_t batchDepth = generatorFNet->GetBatchDepth();
   Int_t batchHeight = generatorFNet->GetBatchHeight();
   Int_t batchWidth = generatorFNet->GetBatchWidth();

   char lossFunction = static_cast<char>(generatorFNet->GetLossFunction());
   char initialization = static_cast<char>(generatorFNet->GetInitialization());
   char regularization = static_cast<char>(generatorFNet->GetRegularization());

   Double_t weightDecay = generatorFNet->GetWeightDecay();

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
      generatorFNet->GetLayerAt(i) -> AddWeightsXMLTo(nn);
   }
}
//////////////////////////////////////////////////////////////////////////
void MethodGAN::ReadWeightsFromXML(void * rootXML)
{
   std::cout << "READ DL network from XML " << std::endl;
   
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

   std::cout << "lossfunction is " << lossFunctionChar << std::endl;

   // create the net

   // DeepNetCpu_t is defined in MethodDL.h

   generatorFNet = std::unique_ptr<DeepNetImpl_t>(new DeepNetImpl_t(batchSize, inputDepth, inputHeight, inputWidth, batchDepth,
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


         generatorFNet->AddDenseLayer(width, func, 0.0); // no need to pass dropout probability

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


         generatorFNet->AddConvLayer(depth, fltHeight, fltWidth, strideRows, strideCols,
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

         generatorFNet->AddMaxPoolLayer(frameHeight, frameWidth, strideRows, strideCols);
      }
      else if (layerName == "ReshapeLayer") {

         // read reshape layer info
         size_t depth, height, width = 0; 
         gTools().ReadAttr(layerXML, "Depth", depth);
         gTools().ReadAttr(layerXML, "Height", height);
         gTools().ReadAttr(layerXML, "Width", width);
         int flattening = 0;
         gTools().ReadAttr(layerXML, "Flattening",flattening );

         generatorFNet->AddReshapeLayer(depth, height, width, flattening);

      }
      else if (layerName == "RNNLayer") {

         std::cout << "add RNN layer " << std::endl;

         // read reshape layer info
         size_t  stateSize,inputSize, timeSteps = 0;
         int rememberState= 0;   
         gTools().ReadAttr(layerXML, "StateSize", stateSize);
         gTools().ReadAttr(layerXML, "InputSize", inputSize);
         gTools().ReadAttr(layerXML, "TimeSteps", timeSteps);
         gTools().ReadAttr(layerXML, "RememberState", rememberState );
         
         generatorFNet->AddBasicRNNLayer(stateSize, inputSize, timeSteps, rememberState);
         
      }


      // read eventually weights and biases
      generatorFNet->GetLayers().back()->ReadWeightsFromXML(layerXML);

      // read next layer
      layerXML = gTools().GetNextChild(layerXML);
   }
}


////////////////////////////////////////////////////////////////////////////////
void MethodGAN::ReadWeightsFromStream(std::istream & /*istr*/)
{
}

////////////////////////////////////////////////////////////////////////////////
const Ranking *TMVA::MethodGAN::CreateRanking()
{
   // TODO
   return NULL;
}

////////////////////////////////////////////////////////////////////////////////
void MethodGAN::GetHelpMessage() const
{
   // TODO
}

} // namespace TMVA
