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
 *      Siddhartha Rao Kamalakara       <srk97c@gmail.com>   - CERN, Switzerland  *
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

#include "TMVA/MethodDNN.h"
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

   input = TFile::Open("~/Documents/gsoc/root/tree.root", "CACHEREAD");


   TString outfileName("TMVA_DNN.root");
   TFile *outputFile = TFile::Open(outfileName, "RECREATE");

   TMVA::DataLoader *dataloader = new TMVA::DataLoader("dataset");

   dataloader->AddVariable( "uniform1", "Variable 1", "units", 'F' );
   dataloader->AddVariable( "uniform2", "Variable 2", "units", 'F' );
   dataloader->AddVariable( "uniform_add", "Variable 3", "units", 'F' );
   dataloader->AddVariable( "uniform_sub", "Variable 4", "units", 'F' );

   dataloader->AddTarget("uniform1");
   dataloader->AddTarget("uniform2");
   dataloader->AddTarget("uniform_add");
   dataloader->AddTarget("uniform_sub");


   TTree *regTree = (TTree*)input->Get("name_of_tree");

   Double_t regWeight  = 1.0;

   dataloader->AddRegressionTree( regTree, regWeight );

   TCut mycut = "";

   dataloader->PrepareTrainingAndTestTree( mycut,"nTrain_Regression=9000:nTest_Regression=1000:SplitMode=Random:NormMode=NumEvents:!V" );

   // Input Layout
   TString inputLayoutString("InputLayout=1|1|4");

   // Batch Layout
   TString batchLayoutString("BatchLayout=16|1|4");

   // General layout.
   TString layoutString("Layout=Encoder={RESHAPE|1|1|4|FLAT,DENSE|2|SIGMOID}Decoder={DENSE|4|LINEAR}");

   // Training strategies.
   TString training0("LearningRate=1e-1,Momentum=0.9,Repetitions=100,"
                     "ConvergenceSteps=10,BatchSize=16,TestRepetitions=10,"
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
   TString dnnOptions("!H:V:ErrorStrategy=SUMOFSQUARES:""WeightInitialization=XAVIERUNIFORM");
   
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
   TMVA::Factory *factory = new TMVA::Factory( "TMVARegression", outputFile,"!V:!Silent:Color:DrawProgressBar:AnalysisType=Regression" );

   TString methodTitle = "AE_" + architectureStr;
   factory->BookMethod(dataloader, TMVA::Types::kAE, methodTitle, dnnOptions);

   // Train MVAs using the set of training events
   factory->TrainAllMethods();

   // Save the output
   outputFile->Close();

   std::cout << "==> Wrote root file: " << outputFile->GetName() << std::endl;
   std::cout << "==> TMVARegression is done!" << std::endl;

   delete factory;
   delete dataloader;
}

#endif
