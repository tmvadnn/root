// @(#)root/tmva/tmva/cnn:$Id$
// Author: Anushree Rankawat

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  :                                                                       *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Testing MethodGAN for Generative Adversarial Networks                                            *
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

#ifndef TMVA_TEST_DNN_TEST_CNN_TEST_METHOD_GAN_H
#define TMVA_TEST_DNN_TEST_CNN_TEST_METHOD_GAN_H

#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TROOT.h"

#include "TMVA/MethodGAN.h"
#include "TMVA/DataLoader.h"
#include "TMVA/Factory.h"
#include "TMVA/Config.h"
#include "TMVA/ClassifierFactory.h"

#include <iostream>

using DeepNet_t = TMVA::DNN::TDeepNet<Architecture_t>;
using ArchitectureImpl_t = TMVA::DNN::TCpu<Double_t>;
using DeepNetImpl_t = TMVA::DNN::TDeepNet<ArchitectureImpl_t>;

/** Testing the entire pipeline of the Method GAN*/
//______________________________________________________________________________


void testMethodGAN_DNN(TString architectureStr)
{
   TFile *input(0);
   TString fname = "/home/anushree/GSoC/DataCreation/mnist_original1.root";

  input = TFile::Open( fname );

   TTree *signalTree     = (TTree*)input->Get("train_sig");
   TTree *background     = (TTree*)input->Get("train_bkg");

   // Create a ROOT output file where TMVA will store ntuples, histograms, etc.
   TString outfileName( "TMVA_MethodGAN.root" );
   TFile* outputFile = TFile::Open( outfileName, "RECREATE" );

   TMVA::Factory *factory = new TMVA::Factory( "TMVAGAN", outputFile,
                                               "!V:!Silent:Color:DrawProgressBar:Transformations=I;D;P;G,D:AnalysisType=Classification" );

   TMVA::DataLoader *dataloader = new TMVA::DataLoader("dataset");

   // global event weights per tree (see below for setting event-wise weights)
   Double_t signalWeight     = 1.0;
   Double_t backgroundWeight = 1.0;

   // You can add an arbitrary number of signal or background trees
   dataloader->AddSignalTree    ( signalTree,     signalWeight );
   dataloader->AddBackgroundTree( background, backgroundWeight );

   // Apply additional cuts on the signal and background samples (can be different)
   TCut mycuts = ""; // for example: TCut mycuts = "abs(var1)<0.5 && abs(var2-0.5)<1";
   TCut mycutb = ""; // for example: TCut mycutb = "abs(var1)<0.5";


   for (int i = 0; i < 28; ++i) {
      for (int j = 0; j < 28; ++j) {
         int ivar=i*28+j;
         TString varName = TString::Format("x%d",ivar);
         dataloader->AddVariable(varName,'F');
      }
   }

   dataloader->PrepareTrainingAndTestTree( mycuts, mycutb,
                                       "nTrain_Signal=1000:nTrain_Background=1000:SplitMode=Random:NormMode=NumEvents:!V" );

                                       // Input Layout
                                       TString inputLayoutString("InputLayout=1|1|784##1|1|784");

                                       // Batch Layout
                                       TString batchLayoutString("BatchLayout=32|1|784##32|1|784");

                                       //General Layout
                                       TString layoutString ("Layout=RESHAPE|1|1|784|FLAT,DENSE|256|RELU,DENSE|512|RELU,DENSE|1024|RELU,DENSE|784|TANH##RESHAPE|1|1|784|FLAT,DENSE|512|RELU,DENSE|256|RELU,DENSE|1|SIGMOID");

                                       // Training strategies.
                                       TString training0("MaxEpochs=2,GeneratorLearningRate=2e-4,GeneratorMomentum=0.9,GeneratorRepetitions=1,"
                                                         "GeneratorConvergenceSteps=20,GeneratorBatchSize=32,GeneratorTestRepetitions=10,"
                                                         "GeneratorWeightDecay=1e-4,GeneratorRegularization=L2,"
                                                         "GeneratorDropConfig=0.0+0.5+0.5+0.5, GeneratorMultithreading=True,"
                                                         "DiscriminatorLearningRate=2e-4,DiscriminatorMomentum=0.9,DiscriminatorRepetitions=1,"
                                                         "DiscriminatorConvergenceSteps=20,DiscriminatorBatchSize=32,DiscriminatorTestRepetitions=10,"
                                                         "DiscriminatorWeightDecay=1e-4,DiscriminatorRegularization=L2,"
                                                         "DiscriminatorDropConfig=0.0+0.5+0.5+0.5, DiscriminatorMultithreading=True");
                                       TString training1("MaxEpochs=2,GeneratorLearningRate=2e-5,GeneratorMomentum=0.9,GeneratorRepetitions=1,"
                                                         "GeneratorConvergenceSteps=20,GeneratorBatchSize=32,GeneratorTestRepetitions=10,"
                                                         "GeneratorWeightDecay=2e-5,GeneratorRegularization=L2,"
                                                         "GeneratorDropConfig=0.0+0.0+0.0+0.0, GeneratorMultithreading=True,"
                                                         "DiscriminatorLearningRate=1e-5,DiscriminatorMomentum=0.9,DiscriminatorRepetitions=1,"
                                                         "DiscriminatorConvergenceSteps=20,DiscriminatorBatchSize=32,DiscriminatorTestRepetitions=10,"
                                                         "DiscriminatorWeightDecay=1e-4,DiscriminatorRegularization=L2,"
                                                         "DropConfig=0.0+0.0+0.0+0.0, Multithreading=True");
                                       TString training2("MaxEpochs=2,GeneratorLearningRate=2e-6,GeneratorMomentum=0.0,GeneratorRepetitions=1,"
                                                         "GeneratorConvergenceSteps=20,GeneratorBatchSize=32,GeneratorTestRepetitions=10,"
                                                         "GeneratorWeightDecay=2e-6,GeneratorRegularization=L2,"
                                                         "GeneratorDropConfig=0.0+0.0+0.0+0.0, GeneratorMultithreading=True,"
                                                         "DiscriminatorLearningRate=1e-6,DiscriminatorMomentum=0.0,DiscriminatorRepetitions=1,"
                                                         "DiscriminatorConvergenceSteps=20, DiscriminatorBatchSize=32, DiscriminatorTestRepetitions=10,"
                                                         "DiscriminatorWeightDecay=1e-4, DiscriminatorRegularization=L2,"
                                                         "DiscriminatorDropConfig=0.0+0.0+0.0+0.0, DiscriminatorMultithreading=True");
                                       TString trainingStrategyString ("TrainingStrategy=");
   trainingStrategyString += training0 + "|" + training1 + "|" + training2;

   // General Options.DataSet
   TString ganOptions ("!H:V:ErrorStrategy=CROSSENTROPY:VarTransform=N:"
                       "WeightInitialization=XAVIERUNIFORM");

   ganOptions.Append(":");
   ganOptions.Append(inputLayoutString);

   ganOptions.Append(":");
   ganOptions.Append(batchLayoutString);
   ganOptions.Append (":");
   ganOptions.Append (layoutString);
   ganOptions.Append (":");
   ganOptions.Append (trainingStrategyString);

   TString cpuOptions = ganOptions + architectureStr;
   factory->BookMethod(dataloader, TMVA::Types::kGAN, "GAN", cpuOptions);

   // Train MVAs using the set of training events
   factory->TrainAllMethods();

   // Save the output
   outputFile->Close();

   std::cout << "==> Wrote root file: " << outputFile->GetName() << std::endl;
   std::cout << "==> TMVAGAN is done!" << std::endl;

   delete factory;
   delete dataloader;
}

TMVA::DNN::TDeepNet<Architecture_t>* createAlexNet(size_t outputWidth)
{
   /*AlexNet Architecture!!*/

   printf("Initialize AlexNet\n");

   /* TDeepNet<Architecture_t, Layer_t>(size_t BatchSize, size_t inputDepth, size_t inputHeight, size_t inputWidth, size_t batchDepth, size_t batchHeight, size_t batchWidth, ELossFunction fJ, EInitialization fI = EInitialization::kZero, ERegularization fR = ERegularization::kNone, Scalar_t fWeightDecay = 0.0, bool isTraining = false);*/
   TMVA::DNN::TDeepNet<Architecture_t> *deepNet =  new TMVA::DNN::TDeepNet<Architecture_t>(2, 3, 227, 227, 2, 3, 51529, ELossFunction::kCrossEntropy, EInitialization::kGauss, ERegularization::kL2, 0.0001, false);

   EActivationFunction activationFunction = EActivationFunction::kRelu;

   /* TConvLayer<Architecture_t> *AddConvLayer(size_t depth, size_t filterHeight, size_t filterWidth, size_t strideRows, size_t strideCols, size_t paddingHeight, size_t paddingWidth, EActivationFunction f, Scalar_t dropoutProbability = 1.0);*/
   TConvLayer<Architecture_t> *convLayer1 = deepNet->AddConvLayer(96, 11, 11, 4, 4, 0, 0, activationFunction);
   convLayer1->Initialize();

   /* TMaxPoolLayer<Architecture_t> *AddMaxPoolLayer(size_t frameHeight, size_t frameWidth, size_t strideRows, size_t strideCols, Scalar_t dropoutProbability = 1.0); */
   TMaxPoolLayer<Architecture_t> *maxPoolLayer1 = deepNet->AddMaxPoolLayer(3, 3, 2, 2, 0.0);
   maxPoolLayer1->Initialize();

   /* TConvLayer<Architecture_t> *AddConvLayer(size_t depth, size_t filterHeight, size_t filterWidth, size_t strideRows, size_t strideCols, size_t paddingHeight, size_t paddingWidth, EActivationFunction f, Scalar_t dropoutProbability = 1.0);*/
   TConvLayer<Architecture_t> *convLayer2 = deepNet->AddConvLayer(256, 5, 5, 1, 1, 2, 2, activationFunction);
   convLayer2->Initialize();

   /* TMaxPoolLayer<Architecture_t> *AddMaxPoolLayer(size_t frameHeight, size_t frameWidth, size_t strideRows, size_t strideCols, Scalar_t dropoutProbability = 1.0); */
   TMaxPoolLayer<Architecture_t> *maxPoolLayer2 = deepNet->AddMaxPoolLayer(3, 3, 2, 2, 0.0);
   maxPoolLayer2->Initialize();

   /* TConvLayer<Architecture_t> *AddConvLayer(size_t depth, size_t filterHeight, size_t filterWidth, size_t strideRows, size_t strideCols, size_t paddingHeight, size_t paddingWidth, EActivationFunction f, Scalar_t dropoutProbability = 1.0);*/
   TConvLayer<Architecture_t> *convLayer3 = deepNet->AddConvLayer(384, 3, 3, 1, 1, 1, 1, activationFunction);
   convLayer3->Initialize();

   /* TConvLayer<Architecture_t> *AddConvLayer(size_t depth, size_t filterHeight, size_t filterWidth, size_t strideRows, size_t strideCols, size_t paddingHeight, size_t paddingWidth, EActivationFunction f, Scalar_t dropoutProbability = 1.0);*/
   TConvLayer<Architecture_t> *convLayer4 = deepNet->AddConvLayer(384, 3, 3, 1, 1, 1, 1, activationFunction);
   convLayer4->Initialize();

   /* TConvLayer<Architecture_t> *AddConvLayer(size_t depth, size_t filterHeight, size_t filterWidth, size_t strideRows, size_t strideCols, size_t paddingHeight, size_t paddingWidth, EActivationFunction f, Scalar_t dropoutProbability = 1.0);*/
   TConvLayer<Architecture_t> *convLayer5 = deepNet->AddConvLayer(256, 3, 3, 1, 1, 1, 1, activationFunction);
   convLayer5->Initialize();

   /* TMaxPoolLayer<Architecture_t> *AddMaxPoolLayer(size_t frameHeight, size_t frameWidth, size_t strideRows, size_t strideCols, Scalar_t dropoutProbability = 1.0); */
   TMaxPoolLayer<Architecture_t> *maxPoolLayer3 = deepNet->AddMaxPoolLayer(3, 3, 2, 2, 0.0);
   maxPoolLayer3->Initialize();

   size_t depthReshape = 1;
   size_t heightReshape = 1;
   size_t widthReshape = deepNet->GetLayerAt(deepNet->GetDepth() - 1)->GetDepth() *
                         deepNet->GetLayerAt(deepNet->GetDepth() - 1)->GetHeight() *
                         deepNet->GetLayerAt(deepNet->GetDepth() - 1)->GetWidth();

   deepNet->AddReshapeLayer(depthReshape, heightReshape, widthReshape, true);

   /* TDenseLayer<Architecture_t> *AddDenseLayer(size_t width, EActivationFunction f, Scalar_t dropoutProbability = 1.0);*/
   TDenseLayer<Architecture_t> *denseLayer1 = deepNet->AddDenseLayer(512, activationFunction, 0.0);
   denseLayer1->Initialize();

   // Removed since computations taken after inclusion of this layer was high
   /* TDenseLayer<Architecture_t> *AddDenseLayer(size_t width, EActivationFunction f, Scalar_t dropoutProbability = 1.0);*/
   //TDenseLayer<Architecture_t> *denseLayer2 = deepNet->AddDenseLayer(4096, activationFunction, 0.0);
   //denseLayer2->Initialize();

   /* TDenseLayer<Architecture_t> *AddDenseLayer(size_t width, EActivationFunction f, Scalar_t dropoutProbability = 1.0);*/
   TDenseLayer<Architecture_t> *denseLayer3 = deepNet->AddDenseLayer(outputWidth, activationFunction, 0.0);
   denseLayer3->Initialize();

   printf("Initialization of AlexNet ends\n");

   return deepNet;
}


void testCreateNoisyMatrices()
{
  double convergenceSteps = 100;
  double batchSize = 2;
  double maxEpochs = 2000;
  double testInterval = 7;
  double learningRate = 1e-5;
  double momentum = 0.3;
  double weightDecay = 1e-4;
  bool multiThreading = false;

  DNN::EInitialization weightInitialization =  EInitialization::kGauss; ///< The initialization method
  DNN::EOutputFunction outputFunction;       ///< The output function for making the predictions
  DNN::ELossFunction lossFunction = ELossFunction::kCrossEntropy;
  DNN::ERegularization regularization = ERegularization::kL2;
  EActivationFunction activationFunction = EActivationFunction::kRelu;

  size_t nSamples = 500;
  size_t nChannels = 3;
  size_t nImgHeight = 227;
  size_t nImgWidth = 227;

  size_t inputDepth = 3;
  size_t inputHeight = 227;
  size_t inputWidth = 227;

  size_t batchDepth = batchSize;
  size_t batchHeight = nChannels;
  size_t batchWidth = nImgHeight * nImgWidth;
  size_t nOutputs = 1;

  //Get model for training
  TMVA::DNN::TDeepNet<Architecture_t> *deepNet = createAlexNet(batchWidth);

  // print the created network
  std::cout << "***** Deep Learning Network *****\n";
  deepNet->Print();

  //Class Label is 0 for fake data
  size_t fakeClassLabel = 0.0;

  std::vector<TMatrixT<Double_t>> inputTensor;
  inputTensor.reserve(nSamples);

  TMatrixT<Double_t> outputMatrix(nSamples, nOutputs);
  TMatrixT<Double_t> weights(nSamples, 1);

  TString jobName = "TestCreateNoisyMatrices";
  TString methodTitle = "MethodGAN";
  TString optionString = "";
  TString emptyString = "";
  TString typeString = "kGAN";

  DataSetInfo dataInfo(emptyString);

  MethodGAN objectGAN(jobName, methodTitle, dataInfo, optionString);
  objectGAN.SetupMethod();
  objectGAN.CreateNoisyMatrices(inputTensor, outputMatrix, weights, *deepNet, nSamples, fakeClassLabel);

  size_t weightRowVal = weights.GetNrows();
  size_t weightColVal = weights.GetNcols();
  size_t outputMatrixRowVal = outputMatrix.GetNrows();
  size_t outputMatrixColVal = outputMatrix.GetNcols();

  for(int row = 0; row < outputMatrixRowVal; row++){
     for(int col = 0; col < outputMatrixColVal; col++){
        R__ASSERT(outputMatrix(row,col) == fakeClassLabel);
     }
  }

  for(int row = 0; row < weightRowVal; row++){
     for(int col = 0; col < weightColVal; col++){
        R__ASSERT(weights(row,col) == 1);
     }
  }
}

void testCreateDiscriminatorFakeData()
{
   double convergenceSteps = 100;
   double batchSize = 2;
   double maxEpochs = 2000;
   double testInterval = 7;
   double learningRate = 1e-5;
   double momentum = 0.3;
   double weightDecay = 1e-4;
   bool multiThreading = false;

   DNN::EInitialization weightInitialization =  EInitialization::kGauss; ///< The initialization method
   DNN::EOutputFunction outputFunction = EOutputFunction::kSigmoid;       ///< The output function for making the predictions
   DNN::ELossFunction lossFunction = ELossFunction::kCrossEntropy;
   DNN::ERegularization regularization = ERegularization::kL2;
   EActivationFunction activationFunction = EActivationFunction::kRelu;

   size_t nSamples = 1;
   size_t nChannels = 3;
   size_t nImgHeight = 227;
   size_t nImgWidth = 227;

   size_t inputDepth = 3;
   size_t inputHeight = 227;
   size_t inputWidth = 227;

   size_t outputWidth = 1;

   size_t batchDepth = batchSize;
   size_t batchHeight = nChannels;
   size_t batchWidth = nImgHeight * nImgWidth;
   size_t nOutputs = 1;
   size_t nThreads = 1;
   size_t epoch = 1;

   //Class Label is 0 for fake data
   size_t fakeClassLabel = 0.0;

   //Get model for training
   TMVA::DNN::TDeepNet<Architecture_t> *genDeepNet = createAlexNet(batchWidth);

   // print the created network
   std::cout << "*****Generator Deep Learning Network *****\n";
   genDeepNet->Print();

   std::vector<TMatrixT<Double_t>> genInputTensor;
   genInputTensor.reserve(nSamples);

   TMatrixT<Double_t> genOutputMatrix(nSamples, nOutputs);
   TMatrixT<Double_t> genWeights(nSamples, 1);

   for (size_t i = 0; i < nSamples; i++)
   {
      genInputTensor.emplace_back(batchHeight, batchWidth);
   }

   size_t m, n;
   TRandom rand(clock());
   Double_t sigma = sqrt(10.0);

   for (size_t i = 0; i < nSamples; i++)
   {
      m = genInputTensor[0].GetNrows();
      n = genInputTensor[0].GetNcols();
      for (size_t j = 0; j < m; j++) {
         for (size_t k = 0; k < n; k++) {
            genInputTensor[0](j, k) = rand.Gaus(0.0, sigma);
         }
      }
   }

   // Create the output
   for (size_t i = 0; i < nSamples; i++)
   {
     // Class of fake data is 1
     genOutputMatrix(i, 0) = fakeClassLabel;
   }

   // Create the weights
   for (size_t i = 0; i < nSamples; i++)
   {
      genWeights(i, 0) = 1;
   }

   TensorInput generatorTuple(genInputTensor, genOutputMatrix, genWeights);

   // Loading the training and testing datasets
   TTensorDataLoader<TensorInput, Architecture_t> generatorData(generatorTuple, nSamples, batchSize,
                                   batchDepth, batchHeight, batchWidth, outputWidth, nThreads);

   TMatrixT<Double_t> disOutputMatrix(nSamples, nOutputs);
   TMatrixT<Double_t> disWeights(nSamples, 1);

   std::vector<TMatrixT<Double_t>> disPredTensor;
   disPredTensor.reserve(nSamples);

   //Get model for training
   TMVA::DNN::TDeepNet<Architecture_t> *disDeepNet = createAlexNet(nOutputs);

   // print the created network
   std::cout << "*****Discriminator Deep Learning Network *****\n";
   disDeepNet->Print();

   TString jobName = "TestCreateDiscriminatorFakeData";
   TString methodTitle = "MethodGAN";
   TString optionString = "";
   TString emptyString = "";
   TString typeString = "kGAN";

   DataSetInfo dataInfo(emptyString);

   MethodGAN objectGAN(jobName, methodTitle, dataInfo, optionString);
   objectGAN.SetupMethod();
   objectGAN.CreateDiscriminatorFakeData(disPredTensor, disOutputMatrix, disWeights,
                               generatorData, *genDeepNet, *disDeepNet, outputFunction, nSamples, fakeClassLabel, epoch);

  size_t disWeightRowVal = disWeights.GetNrows();
  size_t disWeightColVal = disWeights.GetNcols();
  size_t disOutputMatrixRowVal = disOutputMatrix.GetNrows();
  size_t disOutputMatrixColVal = disOutputMatrix.GetNcols();
  size_t sizeDisPredTensor = disPredTensor.size();
  size_t disPredTensorRowVal = disPredTensor[0].GetNrows();
  size_t disPredTensorColVal = disPredTensor[0].GetNcols();

  R__ASSERT(sizeDisPredTensor == nSamples);
  R__ASSERT(disPredTensorRowVal == genDeepNet->GetBatchSize());
  R__ASSERT(disPredTensorColVal == genDeepNet->GetOutputWidth());

  for(int row = 0; row < disOutputMatrixRowVal; row++){
     for(int col = 0; col < disOutputMatrixColVal; col++){
        R__ASSERT(disOutputMatrix(row,col) == fakeClassLabel);
     }
  }

  for(int row = 0; row < disWeightRowVal; row++){
     for(int col = 0; col < disWeightColVal; col++){
        R__ASSERT(disWeights(row,col) == 1);
     }
  }
}

void testCombineGAN()
{
  double convergenceSteps = 100;
  double batchSize = 2;
  double maxEpochs = 2000;
  double testInterval = 7;
  double learningRate = 1e-5;
  double momentum = 0.3;
  double weightDecay = 1e-4;
  bool multiThreading = false;

  DNN::EInitialization weightInitialization =  EInitialization::kGauss; ///< The initialization method
  DNN::EOutputFunction outputFunction = EOutputFunction::kSigmoid;       ///< The output function for making the predictions
  DNN::ELossFunction lossFunction = ELossFunction::kCrossEntropy;
  DNN::ERegularization regularization = ERegularization::kL2;
  EActivationFunction activationFunction = EActivationFunction::kRelu;

  size_t nSamples = 1;
  size_t nChannels = 3;
  size_t nImgHeight = 227;
  size_t nImgWidth = 227;

  size_t inputDepth = 3;
  size_t inputHeight = 227;
  size_t inputWidth = 227;

  size_t outputWidth = 1;

  size_t batchDepth = batchSize;
  size_t batchHeight = nChannels;
  size_t batchWidth = nImgHeight * nImgWidth;
  size_t nOutputs = 1;
  size_t nThreads = 1;
  size_t combineLayerNum = 0;

  //Class Label is 0 for fake data
  size_t fakeClassLabel = 0.0;

  //Get model for training
  TMVA::DNN::TDeepNet<Architecture_t> *genDeepNet = createAlexNet(batchWidth);

  // print the created network
  std::cout << "*****Discriminator Deep Learning Network *****\n";
  genDeepNet->Print();

  //Get model for training
  TMVA::DNN::TDeepNet<Architecture_t> *disDeepNet = createAlexNet(nOutputs);

  // print the created network
  std::cout << "*****Discriminator Deep Learning Network *****\n";
  disDeepNet->Print();

  DeepNet_t combinedDeepNet(genDeepNet->GetBatchSize(), genDeepNet->GetInputDepth(), genDeepNet->GetInputHeight(), genDeepNet->GetInputWidth(),
                    genDeepNet->GetBatchDepth(), genDeepNet->GetBatchHeight(), genDeepNet->GetBatchWidth(), lossFunction, weightInitialization,
                    regularization, weightDecay);

  std::unique_ptr<TMVA::DNN::TDeepNet<TMVA::DNN::TCpu<Double_t>>> combinedNet = std::unique_ptr<TMVA::DNN::TDeepNet<TMVA::DNN::TCpu<Double_t>>>
                                                                          (new TMVA::DNN::TDeepNet<TMVA::DNN::TCpu<Double_t>>(1, genDeepNet->GetInputDepth(),
                                                                          genDeepNet->GetInputHeight(), genDeepNet->GetInputWidth(),
                                                                          genDeepNet->GetBatchDepth(), genDeepNet->GetBatchHeight(), genDeepNet->GetBatchWidth(),
                                                                          lossFunction, weightInitialization, regularization, weightDecay));

  TString jobName = "TestCombineGAN";
  TString methodTitle = "MethodGAN";
  TString optionString = "";
  TString emptyString = "";
  TString typeString = "kGAN";

  DataSetInfo dataInfo(emptyString);

  MethodGAN objectGAN(jobName, methodTitle, dataInfo, optionString);
  objectGAN.SetupMethod();
  objectGAN.CombineGAN(combinedDeepNet, *genDeepNet, *disDeepNet, combinedNet);

  // print the created network
  std::cout << "*****Combined Deep Learning Network *****\n";
  combinedDeepNet.Print();

  combineLayerNum = genDeepNet->GetDepth()+disDeepNet->GetDepth();

  R__ASSERT(combinedDeepNet.GetDepth() == (combineLayerNum-1));

}

#endif
