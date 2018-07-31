// @(#)root/tmva/tmva/dnn:$Id$
// Author: Anushree Rankawat

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodGAN                                                             *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Generative Adversarial Networks                                           *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Anushree Rankawat <anushreerankawat110@gmail.com>                         *
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

#ifndef ROOT_TMVA_MethodGAN
#define ROOT_TMVA_MethodGAN

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// MethodGAN                                                            //
//                                                                      //
// Method class for all Generative Adversarial Networks                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TString.h"

#include "TMVA/MethodDL.h"
#include "TMVA/Types.h"
#include "TMVA/DNN/TensorDataLoader.h"

#include "TMVA/DNN/Architectures/Cpu.h"

#ifdef R__HAS_TMVACPU
#include "TMVA/DNN/Architectures/Cpu.h"
#endif

#ifdef R__HAS_TMVACUDA
#include "TMVA/DNN/Architectures/Cuda.h"
#endif

#ifdef R__HAS_TMVACPU
   using ArchitectureImpl_t = TMVA::DNN::TCpu<Double_t>;
#else
   using ArchitectureImpl_t = TMVA::DNN::TReference<Double_t>;
#endif
using DeepNetImpl_t = TMVA::DNN::TDeepNet<ArchitectureImpl_t>;

#include "TMVA/DNN/Architectures/Reference.h"
#include "TMVA/DNN/Functions.h"
#include "TMVA/DNN/DeepNet.h"

#include <vector>

using namespace TMVA;
using namespace TMVA::DNN::CNN;
using namespace TMVA::DNN;

using Architecture_t = TCpu<Double_t>;
using Scalar_t = Architecture_t::Scalar_t;
using DeepNet_t = TMVA::DNN::TDeepNet<Architecture_t>;
using TensorDataLoader_t = TTensorDataLoader<TMVAInput_t, Architecture_t>;

using TMVA::DNN::EActivationFunction;
using TMVA::DNN::ELossFunction;
using TMVA::DNN::EInitialization;
using TMVA::DNN::EOutputFunction;

namespace TMVA {

/*! All of the options that can be specified in the training string */
struct GANTTrainingSettings {
   size_t maxEpochs;
   size_t generatorBatchSize;
   size_t generatorTestInterval;
   size_t generatorConvergenceSteps;
   DNN::ERegularization generatorRegularization;
   Double_t generatorLearningRate;
   Double_t generatorMomentum;
   Double_t generatorWeightDecay;
   std::vector<Double_t> generatorDropoutProbabilities;
   bool generatorMultithreading;

   size_t discriminatorBatchSize;
   size_t discriminatorTestInterval;
   size_t discriminatorConvergenceSteps;
   DNN::ERegularization discriminatorRegularization;
   Double_t discriminatorLearningRate;
   Double_t discriminatorMomentum;
   Double_t discriminatorWeightDecay;
   std::vector<Double_t> discriminatorDropoutProbabilities;
   bool discriminatorMultithreading;

};

class MethodGAN : public MethodBase {

private:
   // Key-Value vector type, contining the values for the training options
   using KeyValueVector_t = std::vector<std::map<TString, TString>>;
   std::unique_ptr<DeepNetImpl_t> generatorFNet, discriminatorFNet, combinedFNet;
   using Matrix_t = typename ArchitectureImpl_t::Matrix_t;

   /*! The option handling methods */
   void DeclareOptions();
   void ProcessOptions();

   void Init();

   // Function to parse the layout of the input
   void ParseNetworkLayout();
   void ParseInputLayout();
   void ParseBatchLayout();

   /*! After calling the ProcesOptions(), all of the options are parsed,
    *  so using the parsed options, and given the architecture and the
    *  type of the layers, we build the Deep Network passed as
    *  a reference in the function. */
   template <typename Architecture_t, typename Layer_t>
   void CreateDeepNet(DNN::TDeepNet<Architecture_t, Layer_t> &deepNet,
                      std::vector<DNN::TDeepNet<Architecture_t, Layer_t>> &nets, std::unique_ptr<DeepNetImpl_t> &modelNet, TString layoutString);

   size_t fGeneratorInputDepth;  ///< The depth of the input of the generator.
   size_t fGeneratorInputHeight; ///< The height of the input of the generator.
   size_t fGeneratorInputWidth;  ///< The width of the input of the generator.

   size_t fGeneratorBatchDepth;  ///< The depth of the batch used to train the deep net for generator.
   size_t fGeneratorBatchHeight; ///< The height of the batch used to train the deep net for generator.
   size_t fGeneratorBatchWidth;  ///< The width of the batch used to train the deep net for generator.

   size_t fDiscriminatorInputDepth;  ///< The depth of the input of the discriminator.
   size_t fDiscriminatorInputHeight; ///< The height of the input of the discriminator.
   size_t fDiscriminatorInputWidth;  ///< The width of the input of the discriminator.

   size_t fDiscriminatorBatchDepth;  ///< The depth of the batch used to train the deep net for discriminator.
   size_t fDiscriminatorBatchHeight; ///< The height of the batch used to train the deep net for discriminator.
   size_t fDiscriminatorBatchWidth;  ///< The width of the batch used to train the deep net for discriminator.


   DNN::EInitialization fWeightInitialization; ///< The initialization method
   DNN::EOutputFunction fOutputFunction;       ///< The output function for making the predictions
   DNN::ELossFunction fLossFunction;           ///< The loss function

   TString fInputLayoutString;          ///< The string defining the layout of the input
   TString fBatchLayoutString;          ///< The string defining the layout of the batch
   TString fLayoutString;               ///< The string defining the layout of the deep net
   TString fErrorStrategy;              ///< The string defining the error strategy for training
   TString fTrainingStrategyString;     ///< The string defining the training strategy
   TString fWeightInitializationString; ///< The string defining the weight initialization method
   TString fArchitectureString;         ///< The string defining the architecure: CPU or GPU
   TString fGeneratorNetworkLayoutString;      ///< The string defining the network layout for generator
   TString fDiscriminatorNetworkLayoutString;  ///< The string defining the network layout for discriminator
   bool fResume;

   KeyValueVector_t fSettings;                       ///< Map for the training strategy
   std::vector<GANTTrainingSettings> fTrainingSettings; ///< The vector defining each training strategy

   ClassDef(MethodGAN, 0);

protected:
   // provide a help message
   void GetHelpMessage() const;

public:
   /*! Constructor */
   MethodGAN(const TString &jobName, const TString &methodTitle, DataSetInfo &theData, const TString &theOption);

   /*! Constructor */
   MethodGAN(DataSetInfo &theData, const TString &theWeightFile);

   /*! Virtual Destructor */
   virtual ~MethodGAN();

   /*! Function for parsing the training settings, provided as a string
    *  in a key-value form.  */
   KeyValueVector_t ParseKeyValueString(TString parseString, TString blockDelim, TString tokenDelim);

   /*! Check the type of analysis the deep learning network can do */
   Bool_t HasAnalysisType(Types::EAnalysisType type, UInt_t numberClasses, UInt_t numberTargets);

   /*! Methods for training the deep learning network */
   void Train();

   Double_t GetMvaValue(Double_t *err = 0, Double_t *errUpper = 0);
   Double_t GetMvaValueGAN(std::unique_ptr<DeepNetImpl_t> & modelNet, Double_t *err = 0, Double_t *errUpper = 0);
   void CreateNoisyMatrices(std::vector<TMatrixT<Double_t>> &inputTensor, TMatrixT<Double_t> &outputMatrix, TMatrixT<Double_t> &weights, DeepNet_t &DeepNet, size_t nSamples, size_t classLabel);
   Double_t ComputeLoss(TTensorDataLoader<TensorInput, Architecture_t> &generalDataloader, DeepNet_t &DeepNet);
   Double_t ComputeLoss(TTensorDataLoader<TMVAInput_t, Architecture_t> &generalDataloader, DeepNet_t &DeepNet);
   void CreateDiscriminatorFakeData(std::vector<TMatrixT<Double_t>> &predTensor,  TMatrixT<Double_t> &outputMatrix, TMatrixT<Double_t> &weights, TTensorDataLoader<TensorInput, Architecture_t> &trainingData, DeepNet_t &genDeepNet, DeepNet_t &disDeepNet, EOutputFunction outputFunction, size_t nSamples, size_t classLabel, size_t epochs);
   void CombineGAN(DeepNet_t &combinedDeepNet, DeepNet_t &generatorNet, DeepNet_t &discriminatorNet, std::unique_ptr<DeepNetImpl_t> & combinedNet);
   void SetDiscriminatorLayerTraining(DeepNet_t &discrimatorNet);

   /*! Methods for writing and reading weights */
   using MethodBase::ReadWeightsFromStream;
   void AddWeightsXMLTo(void *parent) const;
   void AddWeightsXMLToGenerator(void *parent) const;
   void AddWeightsXMLToDiscriminator(void *parent) const;
   void ReadWeightsFromXML(void *wghtnode);
   void ReadWeightsFromXMLGenerator(void *rootXML);
   void ReadWeightsFromXMLDiscriminator(void *rootXML);
   void ReadWeightsFromStream(std::istream &);

   /* Create ranking */
   const Ranking *CreateRanking();

   /* Getters */
   size_t GetGeneratorInputDepth() const { return fGeneratorInputDepth; }
   size_t GetGeneratorInputHeight() const { return fGeneratorInputHeight; }
   size_t GetGeneratorInputWidth() const { return fGeneratorInputWidth; }

   size_t GetGeneratorBatchDepth() const { return fGeneratorBatchDepth; }
   size_t GetGeneratorBatchHeight() const { return fGeneratorBatchHeight; }
   size_t GetGeneratorBatchWidth() const { return fGeneratorBatchWidth; }

   size_t GetDiscriminatorInputDepth() const { return fDiscriminatorInputDepth; }
   size_t GetDiscriminatorInputHeight() const { return fDiscriminatorInputHeight; }
   size_t GetDiscriminatorInputWidth() const { return fDiscriminatorInputWidth; }

   size_t GetDiscriminatorBatchDepth() const { return fDiscriminatorBatchDepth; }
   size_t GetDiscriminatorBatchHeight() const { return fDiscriminatorBatchHeight; }
   size_t GetDiscriminatorBatchWidth() const { return fDiscriminatorBatchWidth; }

   DNN::EInitialization GetWeightInitialization() const { return fWeightInitialization; }
   DNN::EOutputFunction GetOutputFunction() const { return fOutputFunction; }
   DNN::ELossFunction GetLossFunction() const { return fLossFunction; }

   TString GetInputLayoutString() const { return fInputLayoutString; }
   TString GetBatchLayoutString() const { return fBatchLayoutString; }
   TString GetLayoutString() const { return fLayoutString; }
   TString GetErrorStrategyString() const { return fErrorStrategy; }
   TString GetTrainingStrategyString() const { return fTrainingStrategyString; }
   TString GetWeightInitializationString() const { return fWeightInitializationString; }
   TString GetArchitectureString() const { return fArchitectureString; }

   TString GetGeneratorNetworkLayoutString() const {return fGeneratorNetworkLayoutString; }
   TString GetDiscriminatorNetworkLayoutString() const {return fDiscriminatorNetworkLayoutString; }

   const std::vector<GANTTrainingSettings> &GetTrainingSettings() const { return fTrainingSettings; }
   std::vector<GANTTrainingSettings> &GetTrainingSettings() { return fTrainingSettings; }
   const KeyValueVector_t &GetKeyValueSettings() const { return fSettings; }
   KeyValueVector_t &GetKeyValueSettings() { return fSettings; }

   /** Setters */
   void SetGeneratorInputDepth(size_t inputDepth) { fGeneratorInputDepth = inputDepth; }
   void SetGeneratorInputHeight(size_t inputHeight) { fGeneratorInputHeight = inputHeight; }
   void SetGeneratorInputWidth(size_t inputWidth) { fGeneratorInputWidth = inputWidth; }

   void SetGeneratorBatchDepth(size_t batchDepth) { fGeneratorBatchDepth = batchDepth; }
   void SetGeneratorBatchHeight(size_t batchHeight) { fGeneratorBatchHeight = batchHeight; }
   void SetGeneratorBatchWidth(size_t batchWidth) { fGeneratorBatchWidth = batchWidth; }

   void SetDiscriminatorInputDepth(size_t inputDepth) { fDiscriminatorInputDepth = inputDepth; }
   void SetDiscriminatorInputHeight(size_t inputHeight) { fDiscriminatorInputHeight = inputHeight; }
   void SetDiscriminatorInputWidth(size_t inputWidth) { fDiscriminatorInputWidth = inputWidth; }

   void SetDiscriminatorBatchDepth(size_t batchDepth) { fDiscriminatorBatchDepth = batchDepth; }
   void SetDiscriminatorBatchHeight(size_t batchHeight) { fDiscriminatorBatchHeight = batchHeight; }
   void SetDiscriminatorBatchWidth(size_t batchWidth) { fDiscriminatorBatchWidth = batchWidth; }

   void SetWeightInitialization(DNN::EInitialization weightInitialization)
   {
      fWeightInitialization = weightInitialization;
   }
   void SetOutputFunction(DNN::EOutputFunction outputFunction) { fOutputFunction = outputFunction; }
   void SetErrorStrategyString(TString errorStrategy) { fErrorStrategy = errorStrategy; }
   void SetTrainingStrategyString(TString trainingStrategyString) { fTrainingStrategyString = trainingStrategyString; }
   void SetWeightInitializationString(TString weightInitializationString)
   {
      fWeightInitializationString = weightInitializationString;
   }
   void SetArchitectureString(TString architectureString) { fArchitectureString = architectureString; }
   void SetLayoutString(TString layoutString) { fLayoutString = layoutString; }

   void SetGeneratorNetworkLayout(TString networkLayoutString) { fGeneratorNetworkLayoutString = networkLayoutString; }

   void SetDiscriminatorNetworkLayout(TString networkLayoutString) { fDiscriminatorNetworkLayoutString = networkLayoutString; }

};

} // namespace TMVA

#endif
