// @(#)root/tmva/tmva/cnn:$Id$
// Author: Vladimir Ilievski

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  :                                                                       *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Testing Method DL for Conv Net                                            *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Vladimir Ilievski      <ilievski.vladimir@live.com>  - CERN, Switzerland  *
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

#include <iostream>

/** Testing the entire pipeline of the Method GAN*/
//______________________________________________________________________________
void testMethodGAN_DNN(TString architectureStr)
{
   // Load the photon input
   TFile *photonInput(0);
   TString photonFileName =
        "/Users/vladimirilievski/Desktop/Vladimir/GSoC/ROOT-CI/common-version/root/tmva/tmva/test/DNN/CNN/"
        "dataset/SinglePhotonPt50_FEVTDEBUG_n250k_IMG_CROPS32.root";
   photonInput = TFile::Open(photonFileName);

   // Load the electron input
   TFile *electronInput(0);
   TString electronFileName = "/Users/vladimirilievski/Desktop/Vladimir/GSoC/ROOT-CI/common-version/root/tmva/tmva/"
                              "test/DNN/CNN/dataset/SingleElectronPt50_FEVTDEBUG_n250k_IMG_CROPS32.root";
   electronInput = TFile::Open(electronFileName);

   // Get the trees
   TTree *signalTree = (TTree *)photonInput->Get("RHTree");
   TTree *background = (TTree *)electronInput->Get("RHTree");

   // Create a ROOT output file where TMVA will store ntuples, histograms, etc.
   TString outfileName("TMVA_MethodGAN.root");
   TFile *outputFile = TFile::Open(outfileName, "RECREATE");

   TMVA::DataLoader *dataloader = new TMVA::DataLoader("dataset");

   // global event weights per tree
   Double_t signalWeight = 1.0;
   Double_t backgroundWeight = 1.0;

   // Add signal and background trees
   dataloader->AddSignalTree(signalTree, signalWeight);
   dataloader->AddBackgroundTree(background, backgroundWeight);

   // dataloader->SetBackgroundWeightExpression("weight");

   TCut mycuts = "";
   TCut mycutb = "";

   dataloader->PrepareTrainingAndTestTree(
      mycuts, mycutb, "nTrain_Signal=10000:nTrain_Background=10000:SplitMode=Random:NormMode=NumEvents:!V");

   // Input Layout
   TString inputLayoutString("InputLayout=1|1|4##1|1|4");

   // Batch Layout
   TString batchLayoutString("BatchLayout=256|1|4##256|1|4");

   //General Layout
   TString layoutString ("Layout=RESHAPE|1|1|4|FLAT,DENSE|128|TANH,DENSE|128|TANH,DENSE|128|"
   "TANH,DENSE|1|LINEAR##RESHAPE|1|1|4|FLAT,DENSE|128|TANH,DENSE|128|TANH,DENSE|128|TANH,DENSE|1|LINEAR");

   // Training strategies.
   TString training0("GeneratorLearningRate=1e-1,GeneratorMomentum=0.9,GeneratorRepetitions=1,"
                     "GeneratorConvergenceSteps=20,GeneratorBatchSize=256,GeneratorTestRepetitions=10,"
                     "GeneratorWeightDecay=1e-4,GeneratorRegularization=L2,"
                     "GeneratorDropConfig=0.0+0.5+0.5+0.5, GeneratorMultithreading=True,"
   "DiscriminatorLearningRate=1e-1,DiscriminatorMomentum=0.9,DiscriminatorRepetitions=1,"
                     "DiscriminatorConvergenceSteps=20,DiscriminatorBatchSize=256,DiscriminatorTestRepetitions=10,"
                     "DiscriminatorWeightDecay=1e-4,DiscriminatorRegularization=L2,"
                     "DiscriminatorDropConfig=0.0+0.5+0.5+0.5, DiscriminatorMultithreading=True");
   TString training1("GeneratorLearningRate=1e-2,GeneratorMomentum=0.9,GeneratorRepetitions=1,"
                     "GeneratorConvergenceSteps=20,GeneratorBatchSize=256,GeneratorTestRepetitions=10,"
                     "GeneratorWeightDecay=1e-4,GeneratorRegularization=L2,"
                     "GeneratorDropConfig=0.0+0.0+0.0+0.0, GeneratorMultithreading=True,"
   "DiscriminatorLearningRate=1e-2,DiscriminatorMomentum=0.9,DiscriminatorRepetitions=1,"
                     "DiscriminatorConvergenceSteps=20,DiscriminatorBatchSize=256,DiscriminatorTestRepetitions=10,"
                     "DiscriminatorWeightDecay=1e-4,DiscriminatorRegularization=L2,"
                     "DropConfig=0.0+0.0+0.0+0.0, Multithreading=True");
   TString training2("GeneratorLearningRate=1e-3,GeneratorMomentum=0.0,GeneratorRepetitions=1,"
                     "GeneratorConvergenceSteps=20,GeneratorBatchSize=256,GeneratorTestRepetitions=10,"
                     "GeneratorWeightDecay=1e-4,GeneratorRegularization=L2,"
                     "GeneratorDropConfig=0.0+0.0+0.0+0.0, GeneratorMultithreading=True,"
   "DiscriminatorLearningRate=1e-3,DiscriminatorMomentum=0.0,DiscriminatorRepetitions=1,"
                      "DiscriminatorConvergenceSteps=20, DiscriminatorBatchSize=256, DiscriminatorTestRepetitions=10,"
                      "DiscriminatorWeightDecay=1e-4, DiscriminatorRegularization=L2,"
                      "DiscriminatorDropConfig=0.0+0.0+0.0+0.0, DiscriminatorMultithreading=True");
   TString trainingStrategyString ("TrainingStrategy=");
   trainingStrategyString += training0 + "|" + training1 + "|" + training2;

   // General Options.
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
   std::cout << "==> TMVAClassification is done!" << std::endl;

   delete factory;
   delete dataloader;
}

#endif
