// @(#)root/tmva/tmva/dnn:$Id$
// Author: Vladimir Ilievski, Saurav Shekhar, Siddhartha Rao Kamalakara

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodAE                                                              *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Deep Neural Network Method                                                *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Vladimir Ilievski  <ilievski.vladimir@live.com> - CERN, Switzerland       *
 *      Saurav Shekhar     <sauravshekhar01@gmail.com> - ETH Zurich, Switzerland  *
 *      Siddhartha Rao Kamalakara  <srk97c@gmail.com> - CERN, Switzerland         *
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

#ifndef ROOT_TMVA_MethodAE
#define ROOT_TMVA_MethodAE

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// MethodAE                                                             //
//                                                                      //
// Method class for creating Auto Encoders                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TString.h"

#include "TMVA/MethodBase.h"
#include "TMVA/Types.h"

#include "TMVA/DNN/Architectures/Reference.h"

#ifdef R__HAS_TMVACPU
#include "TMVA/DNN/Architectures/Cpu.h"
#endif

#ifdef R__HAS_TMVACUDA
#include "TMVA/DNN/Architectures/Cuda.h"
#endif

#include "TMVA/DNN/Functions.h"
#include "TMVA/DNN/DeepNet.h"

#include <vector>

namespace TMVA {

/*! All of the options that can be specified in the training string */
struct TTrainingAESettings {
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

class MethodAE : public MethodBase {

private:
   // Key-Value vector type, contining the values for the training options
   using KeyValueVector_t = std::vector<std::map<TString, TString>>;
#ifdef R__HAS_TMVACPU
   using ArchitectureImpl_t = TMVA::DNN::TCpu<Double_t>;
#else
   using ArchitectureImpl_t = TMVA::DNN::TReference<Double_t>;
#endif  
   using DeepNetImpl_t = TMVA::DNN::TDeepNet<ArchitectureImpl_t>;
   using Matrix_t       = ArchitectureImpl_t::Matrix_t;
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
   void CreateEncoder(DNN::TDeepNet<Architecture_t, Layer_t> &deepNet,
                      std::vector<DNN::TDeepNet<Architecture_t, Layer_t>> &nets, TString layoutString);

   template <typename Architecture_t, typename Layer_t>
   void CreateDecoder(DNN::TDeepNet<Architecture_t, Layer_t> &deepNet,
                      std::vector<DNN::TDeepNet<Architecture_t, Layer_t>> &nets, TString layoutString);                                   

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
   void ParseRnnLayer(DNN::TDeepNet<Architecture_t, Layer_t> &deepNet,
                      std::vector<DNN::TDeepNet<Architecture_t, Layer_t>> &nets, TString layerString, TString delim);

   template <typename Architecture_t, typename Layer_t>
   void ParseLstmLayer(DNN::TDeepNet<Architecture_t, Layer_t> &deepNet,
                       std::vector<DNN::TDeepNet<Architecture_t, Layer_t>> &nets, TString layerString, TString delim);   

   size_t fInputDepth;  ///< The depth of the input.
   size_t fInputHeight; ///< The height of the input.
   size_t fInputWidth;  ///< The width of the input.

   size_t fBatchDepth;  ///< The depth of the batch used to train the deep net.
   size_t fBatchHeight; ///< The height of the batch used to train the deep net.
   size_t fBatchWidth;  ///< The width of the batch used to train the deep net.

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

   KeyValueVector_t fSettings;                       ///< Map for the training strategy
   std::vector<TTrainingAESettings> fTrainingSettings; ///< The vector defining each training strategy

   ClassDef(MethodAE, 0);

protected:
   // provide a help message
   void GetHelpMessage() const;

public:
   /*! Constructor */
   MethodAE(const TString &jobName, const TString &methodTitle, DataSetInfo &theData, const TString &theOption);

   /*! Constructor */
   MethodAE(DataSetInfo &theData, const TString &theWeightFile);

   /*! Virtual Destructor */
   virtual ~MethodAE();

   /*! Function for parsing the training settings, provided as a string
    *  in a key-value form.  */
   KeyValueVector_t ParseKeyValueString(TString parseString, TString blockDelim, TString tokenDelim);

   /*! Check the type of analysis the deep learning network can do */
   Bool_t HasAnalysisType(Types::EAnalysisType type, UInt_t numberClasses, UInt_t numberTargets);

   /*! Methods for training the deep learning network */
   void Train();

   Double_t GetMvaValue(Double_t *err = 0, Double_t *errUpper = 0);
   virtual const std::vector<Float_t>& GetRegressionValues();
   virtual const std::vector<Float_t>& GetMulticlassValues();
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

   const std::vector<TTrainingAESettings> &GetTrainingSettings() const { return fTrainingSettings; }
   std::vector<TTrainingAESettings> &GetTrainingSettings() { return fTrainingSettings; }
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
