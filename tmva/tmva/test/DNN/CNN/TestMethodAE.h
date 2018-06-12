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

#ifndef TMVA_TEST_DNN_TEST_CNN_TEST_METHOD_AE_H
#define TMVA_TEST_DNN_TEST_CNN_TEST_METHOD_AE_H

#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TROOT.h"

#include "TMVA/MethodAE.h"
#include "TMVA/DataLoader.h"
#include "TMVA/Factory.h"
#include "TMVA/Config.h"

#include "MakeImageData.h"

#include <iostream>


/** Testing the entire pipeline of the Method DL, when only a Multilayer Percepton
 *  is constructed. */
//______________________________________________________________________________
void testMethodAE_DNN(TString architectureStr)
{

   ROOT::EnableImplicitMT(1);
   TMVA::Config::Instance();
   
   TFile *input(0);
   // TString fname = "/Users/vladimirilievski/Desktop/Vladimir/GSoC/ROOT-CI/common-version/root/tmva/tmva/test/DNN/CNN/"
   //                 "dataset/tmva_class_example.root";
   /*
   TString fname = "http://root.cern.ch/files/tmva_class_example.root";
   TString fopt = "CACHEREAD";
   input = TFile::Open(fname,fopt);
   */
   input = TFile::Open("http://root.cern.ch/files/tmva_reg_example.root", "CACHEREAD");


   TString outfileName("TMVA_DNN.root");
   TFile *outputFile = TFile::Open(outfileName, "RECREATE");

   TMVA::DataLoader *dataloader = new TMVA::DataLoader("dataset");

   dataloader->AddVariable( "var1", "Variable 1", "units", 'F' );
   dataloader->AddVariable( "var2", "Variable 2", "units", 'F' );
   dataloader->AddSpectator( "spec1:=var1*2",  "Spectator 1", "units", 'F' );
   dataloader->AddSpectator( "spec2:=var1*3",  "Spectator 2", "units", 'F' );

   dataloader->AddTarget("var1");
   dataloader->AddTarget("var2");

   TTree *regTree = (TTree*)input->Get("TreeR");

   Double_t regWeight  = 1.0;

   dataloader->AddRegressionTree( regTree, regWeight );

   TCut mycut = "";

   dataloader->PrepareTrainingAndTestTree( mycut,
                                         "nTrain_Regression=1000:nTest_Regression=0:SplitMode=Random:NormMode=NumEvents:!V" );

   // Input Layout
   TString inputLayoutString("InputLayout=1|1|2");

   // Batch Layout
   TString batchLayoutString("BatchLayout=256|1|2");

   // General layout.
   TString layoutString("Layout=Encoder={RESHAPE|1|1|2|FLAT,DENSE|128|TANH,DENSE|64|TANH}Decoder={DENSE|128|TANH,DENSE|2|LINEAR,LINEAR}");

   // Training strategies.
   TString training0("LearningRate=1e-1,Momentum=0.9,Repetitions=1,"
                     "ConvergenceSteps=20,BatchSize=256,TestRepetitions=10,"
                     "WeightDecay=1e-4,Regularization=L2,"
                     "DropConfig=0.0+0.5+0.5+0.5, Multithreading=True");
   TString training1("LearningRate=1e-2,Momentum=0.9,Repetitions=1,"
                     "ConvergenceSteps=20,BatchSize=256,TestRepetitions=10,"
                     "WeightDecay=1e-4,Regularization=L2,"
                     "DropConfig=0.0+0.0+0.0+0.0, Multithreading=True");
   TString training2("LearningRate=1e-3,Momentum=0.9,Repetitions=1,"
                     "ConvergenceSteps=20,BatchSize=256,TestRepetitions=10,"
                     "WeightDecay=1e-4,Regularization=L2,"
                     "DropConfig=0.0+0.0+0.0+0.0, Multithreading=True");
   TString trainingStrategyString("TrainingStrategy=");
   trainingStrategyString += training0 + "|" + training1 + "|" + training2;

   // General Options.
   TString dnnOptions("!H:V:ErrorStrategy=SUMOFSQUARES:"
                      "WeightInitialization=XAVIERUNIFORM");

   
   // Concatenate all option strings
   dnnOptions.Append(":");
   dnnOptions.Append(inputLayoutString);

   dnnOptions.Append(":");
   dnnOptions.Append(batchLayoutString);
   
   dnnOptions.Append(":");
   dnnOptions.Append(layoutString);

   dnnOptions.Append(":");
   dnnOptions.Append(trainingStrategyString);

   dnnOptions.Append(":Architecture=");
   dnnOptions.Append(architectureStr);

   // create factory
   TMVA::Factory *factory = new TMVA::Factory( "TMVARegression", outputFile,
                                               "!V:!Silent:Color:DrawProgressBar:AnalysisType=Regression" );

   TString methodTitle = "AE_" + architectureStr;
   factory->BookMethod(dataloader, TMVA::Types::kAE, methodTitle, dnnOptions);

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
