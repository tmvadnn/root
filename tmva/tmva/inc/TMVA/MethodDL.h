// @(#)root/tmva/tmva/dnn:$Id$
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

#ifndef ROOT_TMVA_MethodDL
#define ROOT_TMVA_MethodDL

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// MethodDL                                                             //
//                                                                      //
// Method class for all Deep Learning Networks                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TString.h"

#include "TMVA/MethodBase.h"
#include "TMVA/Types.h"

#include "TMVA/DNN/Architectures/Reference.h"

#ifdef R__HAS_TMVACPU
#include "TMVA/DNN/Architectures/Cpu.h"
#endif

#ifdef R__HAS_TMVAGPU
#include "TMVA/DNN/Architectures/Cuda.h"
#endif

#include "TMVA/DNN/Functions.h"
#include "TMVA/DNN/DeepNet.h"

#include <vector>

namespace TMVA {

/*! All of the options that can be specified in the training string */
struct TTrainingSettings {
   size_t batchSize;
   size_t testInterval;
   size_t convergenceSteps;
   size_t maxEpochs; 
   DNN::ERegularization regularization;
   Double_t learningRate;
   Double_t momentum;
   Double_t weightDecay;
   std::vector<Double_t> dropoutProbabilities;
   bool multithreading;
};

class MethodDL : public MethodBase {

private:
   // Key-Value vector type, contining the values for the training options
   using KeyValueVector_t = std::vector<std::map<TString, TString>>;
// #ifdef R__HAS_TMVAGPU
//    using ArchitectureImpl_t = TMVA::DNN::TCuda<Double_t>;
// #else
// do not use arch GPU for evaluation. It is too slow for batch size=1   
#ifdef R__HAS_TMVACPU
   using ArchitectureImpl_t = TMVA::DNN::TCpu<Double_t>;
#else
   using ArchitectureImpl_t = TMVA::DNN::TReference<Double_t>;
#endif  
//#endif
   using DeepNetImpl_t = TMVA::DNN::TDeepNet<ArchitectureImpl_t>;
   std::unique_ptr<DeepNetImpl_t> fNet;

   /*! The option handling methods */
   void DeclareOptions();
   void ProcessOptions();

   void Init();

   // Function to parse the layout of the input
   void ParseInputLayout();
   void ParseBatchLayout();

   /*! After calling the ProcesOptions(), all of the options are parsed,
    *  so using the parsed options, and given the architecture and the
    *  type of the layers, we build the Deep Network passed as
    *  a reference in the function. */
   template <typename Architecture_t, typename Layer_t>
   void CreateDeepNet(DNN::TDeepNet<Architecture_t, Layer_t> &deepNet,
                      std::vector<DNN::TDeepNet<Architecture_t, Layer_t>> &nets);

   template <typename Architecture_t, typename Layer_t>
   void ParseDenseLayer(DNN::TDeepNet<Architecture_t, Layer_t> &deepNet,
                        std::vector<DNN::TDeepNet<Architecture_t, Layer_t>> &nets, TString layerString, TString delim);

   template <typename Architecture_t, typename Layer_t>
   void ParseConvLayer(DNN::TDeepNet<Architecture_t, Layer_t> &deepNet,
                       std::vector<DNN::TDeepNet<Architecture_t, Layer_t>> &nets, TString layerString, TString delim);

   template <typename Architecture_t, typename Layer_t>
   void ParseMaxPoolLayer(DNN::TDeepNet<Architecture_t, Layer_t> &deepNet,
                          std::vector<DNN::TDeepNet<Architecture_t, Layer_t>> &nets, TString layerString,
                          TString delim);

   template <typename Architecture_t, typename Layer_t>
   void ParseReshapeLayer(DNN::TDeepNet<Architecture_t, Layer_t> &deepNet,
                          std::vector<DNN::TDeepNet<Architecture_t, Layer_t>> &nets, TString layerString,
                          TString delim);

   template <typename Architecture_t, typename Layer_t>
   void ParsePaddingLayer2D(DNN::TDeepNet<Architecture_t, Layer_t> &deepNet,
                          std::vector<DNN::TDeepNet<Architecture_t, Layer_t>> &nets, TString layerString,
                          TString delim);

   template <typename Architecture_t, typename Layer_t>
   void ParseRnnLayer(DNN::TDeepNet<Architecture_t, Layer_t> &deepNet,
                      std::vector<DNN::TDeepNet<Architecture_t, Layer_t>> &nets, TString layerString, TString delim);

   template <typename Architecture_t, typename Layer_t>
   void ParseLstmLayer(DNN::TDeepNet<Architecture_t, Layer_t> &deepNet,
                       std::vector<DNN::TDeepNet<Architecture_t, Layer_t>> &nets, TString layerString, TString delim);

   template <typename Architecture_t>
   void TrainDeepNet(); 
   
   size_t fInputDepth;  ///< The depth of the input.
   size_t fInputHeight; ///< The height of the input.
   size_t fInputWidth;  ///< The width of the input.

   size_t fBatchDepth;  ///< The depth of the batch used to train the deep net.
   size_t fBatchHeight; ///< The height of the batch used to train the deep net.
   size_t fBatchWidth;  ///< The width of the batch used to train the deep net.
   
   size_t fRandomSeed;  ///<The random seed used to initialize the weights and shuffling batches (default is zero)

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
   bool fResume;
   bool fBuildNet;                     ///< Flag to control whether to build fNet, the stored network used for the evaluation

   KeyValueVector_t fSettings;                       ///< Map for the training strategy
   std::vector<TTrainingSettings> fTrainingSettings; ///< The vector defining each training strategy

   ClassDef(MethodDL, 0);

protected:
   // provide a help message
   void GetHelpMessage() const;

public:
   /*! Constructor */
   MethodDL(const TString &jobName, const TString &methodTitle, DataSetInfo &theData, const TString &theOption);

   /*! Constructor */
   MethodDL(DataSetInfo &theData, const TString &theWeightFile);

   /*! Virtual Destructor */
   virtual ~MethodDL();

   /*! Function for parsing the training settings, provided as a string
    *  in a key-value form.  */
   KeyValueVector_t ParseKeyValueString(TString parseString, TString blockDelim, TString tokenDelim);

   /*! Check the type of analysis the deep learning network can do */
   Bool_t HasAnalysisType(Types::EAnalysisType type, UInt_t numberClasses, UInt_t numberTargets);

   /*! Methods for training the deep learning network */
   void Train();

   Double_t GetMvaValue(Double_t *err = 0, Double_t *errUpper = 0);

   /*! Methods for writing and reading weights */
   using MethodBase::ReadWeightsFromStream;
   void AddWeightsXMLTo(void *parent) const;
   void ReadWeightsFromXML(void *wghtnode);
   void ReadWeightsFromStream(std::istream &);

   /* Create ranking */
   const Ranking *CreateRanking();

   /* Getters */
   size_t GetInputDepth() const { return fInputDepth; }
   size_t GetInputHeight() const { return fInputHeight; }
   size_t GetInputWidth() const { return fInputWidth; }

   size_t GetBatchDepth() const { return fBatchDepth; }
   size_t GetBatchHeight() const { return fBatchHeight; }
   size_t GetBatchWidth() const { return fBatchWidth; }

   const DeepNetImpl_t & GetDeepNet() const { return *fNet; }

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

   const std::vector<TTrainingSettings> &GetTrainingSettings() const { return fTrainingSettings; }
   std::vector<TTrainingSettings> &GetTrainingSettings() { return fTrainingSettings; }
   const KeyValueVector_t &GetKeyValueSettings() const { return fSettings; }
   KeyValueVector_t &GetKeyValueSettings() { return fSettings; }

   /** Setters */
   void SetInputDepth(size_t inputDepth) { fInputDepth = inputDepth; }
   void SetInputHeight(size_t inputHeight) { fInputHeight = inputHeight; }
   void SetInputWidth(size_t inputWidth) { fInputWidth = inputWidth; }

   void SetBatchDepth(size_t batchDepth) { fBatchDepth = batchDepth; }
   void SetBatchHeight(size_t batchHeight) { fBatchHeight = batchHeight; }
   void SetBatchWidth(size_t batchWidth) { fBatchWidth = batchWidth; }

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
};

} // namespace TMVA

#endif
