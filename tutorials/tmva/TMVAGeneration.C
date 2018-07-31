/// \file
/// \ingroup tutorial_tmva
/// \notebook -nodraw
/// This macro provides examples for the training and testing of the
/// TMVA generative models. This is specifically for unsupervised learning
/// based models.
///
/// If the user wants to train a Generative Adversarial Network for unsupervised learning,
/// the following command can be used:
///
///     root -l ./TMVAGeneration.C\(\"GAN\"\)
///
/// (note that the backslashes are mandatory)
/// If no method given, a default set of classifiers is used.
/// The output file "TMVA.root" can be analysed with the use of dedicated
/// macros (simply say: root -l <macro.C>), which can be conveniently
/// invoked through a GUI that will appear at the end of the run of this macro.
/// Launch the GUI via the command:
///
///     root -l ./TMVAGui.C
///
/// You can also compile and run the example with the following commands
///
///     make
///     ./TMVAGeneration <Methods>
///
/// where: `<Methods> = "method1 method2"` are the TMVA generative model names
/// example:
///
///     ./TMVAGeneration GAN
///
/// If no method given, a default set is of classifiers is used
///
/// - Project   : TMVA - a ROOT-integrated toolkit for multivariate data analysis
/// - Package   : TMVA
/// - Root Macro: TMVAGeneration
///
/// \macro_output
/// \macro_code
/// \author Anushree Rankawat


#include <cstdlib>
#include <iostream>
#include <map>
#include <string>

#include "TChain.h"
#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TObjString.h"
#include "TSystem.h"
#include "TROOT.h"

#include "TMVA/Factory.h"
#include "TMVA/DataLoader.h"
#include "TMVA/Tools.h"
#include "TMVA/TMVAGui.h"

int TMVAGeneration( TString myMethodList = "" )
{
   // The explicit loading of the shared libTMVA is done in TMVAlogon.C, defined in .rootrc
   // if you use your private .rootrc, or run from a different directory, please copy the
   // corresponding lines from .rootrc

   // Methods to be processed can be given as an argument; use format:
   //
   //     mylinux~> root -l TMVAGeneration.C\(\"myMethod1,myMethod2,myMethod3\"\)
   // Currently there is just a single unsupervised method to be tested i.e., GAN

   //---------------------------------------------------------------
   // This loads the library
   TMVA::Tools::Instance();

   // Default MVA methods to be trained + tested
   std::map<std::string,int> Use;


   //
   // Neural Networks (all are feed-forward Multilayer Perceptrons)
   Use["GAN"]		  = 0; // Generative Adversarial Networks
   // ---------------------------------------------------------------

   std::cout << std::endl;
   std::cout << "==> Start TMVAGeneration" << std::endl;

   // Select methods (don't look at this code - not of interest)
   if (myMethodList != "") {
      for (std::map<std::string,int>::iterator it = Use.begin(); it != Use.end(); it++) it->second = 0;

      std::vector<TString> mlist = TMVA::gTools().SplitString( myMethodList, ',' );
      for (UInt_t i=0; i<mlist.size(); i++) {
         std::string regMethod(mlist[i]);

         if (Use.find(regMethod) == Use.end()) {
            std::cout << "Method \"" << regMethod << "\" not known in TMVA under this name. Choose among the following:" << std::endl;
            for (std::map<std::string,int>::iterator it = Use.begin(); it != Use.end(); it++) std::cout << it->first << " ";
            std::cout << std::endl;
            return 1;
         }
         Use[regMethod] = 1;
      }
   }

   // --------------------------------------------------------------------------------------------------

   // Here the preparation phase begins

   // Read training and test data
   // (it is also possible to use ASCII format as input -> see TMVA Users Guide)
   TFile *input(0);
   TString fname = "~/root/tutorials/mnist.root";
   //if (!gSystem->AccessPathName( fname )) {
   input = TFile::Open( fname ); // check if file in local directory exists
   /*}
   else {
      TFile::SetCacheFileDir(".");
      //input = TFile::Open("http://root.cern.ch/files/tmva_class_example.root", "CACHEREAD");
   }*/
   if (!input) {
      std::cout << "ERROR: could not open data file" << std::endl;
      exit(1);
   }
   std::cout << "--- TMVAGeneration       : Using input file: " << input->GetName() << std::endl;

   // Register the training and test trees

   TTree *signalTree     = (TTree*)input->Get("train_sig");
   TTree *background     = (TTree*)input->Get("train_bkg");

   // Create a ROOT output file where TMVA will store ntuples, histograms, etc.
   TString outfileName( "TMVA.root" );
   TFile* outputFile = TFile::Open( outfileName, "RECREATE" );

   // Create the factory object. Later you can choose the methods
   // whose performance you'd like to investigate. The factory is
   // the only TMVA object you have to interact with
   //
   // The first argument is the base of the name of all the
   // weightfiles in the directory weight/
   //
   // The second argument is the output file for the training results
   // All TMVA output can be suppressed by removing the "!" (not) in
   // front of the "Silent" argument in the option string
   TMVA::Factory *factory = new TMVA::Factory( "TMVAGeneration", outputFile,
                                               "!V:!Silent:Color:DrawProgressBar:Transformations=I;D;P;G,D:AnalysisType=Classification" );

   TMVA::DataLoader *dataloader=new TMVA::DataLoader("dataset");
   // If you wish to modify default settings
   // (please check "src/Config.h" to see all available global options)
   //
   //    (TMVA::gConfig().GetVariablePlotting()).fTimesRMS = 8.0;
   //    (TMVA::gConfig().GetIONames()).fWeightFileDir = "myWeightDirectory";

   // Define the input variables that shall be used for the MVA training
   // note that you may also use variable expressions, such as: "3*var1/var2*abs(var3)"
   // [all types of expressions that can also be parsed by TTree::Draw( "expression" )]
   //dataloader->AddVariable( "myvar1 := var1+var2", 'F' );
   //dataloader->AddVariable( "myvar2 := var1-var2", "Expression 2", "", 'F' );
   //dataloader->AddVariable( "var3",                "Variable 3", "units", 'F' );
   //dataloader->AddVariable( "var4",                "Variable 4", "units", 'F' );

   // You can add so-called "Spectator variables", which are not used in the MVA training,
   // but will appear in the final "TestTree" produced by TMVA. This TestTree will contain the
   // input variables, the response values of all trained MVAs, and the spectator variables



   // global event weights per tree (see below for setting event-wise weights)
   Double_t signalWeight     = 1.0;
   Double_t backgroundWeight = 1.0;

   // You can add an arbitrary number of signal or background trees
   dataloader->AddSignalTree    ( signalTree,     signalWeight );
   dataloader->AddBackgroundTree( background, backgroundWeight );

   // To give different trees for training and testing, do as follows:
   //
   //     dataloader->AddSignalTree( signalTrainingTree, signalTrainWeight, "Training" );
   //     dataloader->AddSignalTree( signalTestTree,     signalTestWeight,  "Test" );

   // Use the following code instead of the above two or four lines to add signal and background
   // training and test events "by hand"
   // NOTE that in this case one should not give expressions (such as "var1+var2") in the input
   //      variable definition, but simply compute the expression before adding the event
   // ```cpp
   // // --- begin ----------------------------------------------------------
   // std::vector<Double_t> vars( 4 ); // vector has size of number of input variables
   // Float_t  treevars[4], weight;
   //
   // // Signal
   // for (UInt_t ivar=0; ivar<4; ivar++) signalTree->SetBranchAddress( Form( "var%i", ivar+1 ), &(treevars[ivar]) );
   // for (UInt_t i=0; i<signalTree->GetEntries(); i++) {
   //    signalTree->GetEntry(i);
   //    for (UInt_t ivar=0; ivar<4; ivar++) vars[ivar] = treevars[ivar];
   //    // add training and test events; here: first half is training, second is testing
   //    // note that the weight can also be event-wise
   //    if (i < signalTree->GetEntries()/2.0) dataloader->AddSignalTrainingEvent( vars, signalWeight );
   //    else                              dataloader->AddSignalTestEvent    ( vars, signalWeight );
   // }
   //
   // // Background (has event weights)
   // background->SetBranchAddress( "weight", &weight );
   // for (UInt_t ivar=0; ivar<4; ivar++) background->SetBranchAddress( Form( "var%i", ivar+1 ), &(treevars[ivar]) );
   // for (UInt_t i=0; i<background->GetEntries(); i++) {
   //    background->GetEntry(i);
   //    for (UInt_t ivar=0; ivar<4; ivar++) vars[ivar] = treevars[ivar];
   //    // add training and test events; here: first half is training, second is testing
   //    // note that the weight can also be event-wise
   //    if (i < background->GetEntries()/2) dataloader->AddBackgroundTrainingEvent( vars, backgroundWeight*weight );
   //    else                                dataloader->AddBackgroundTestEvent    ( vars, backgroundWeight*weight );
   // }
   // // --- end ------------------------------------------------------------
   // ```
   // End of tree registration

   // Set individual event weights (the variables must exist in the original TTree)
   // -  for signal    : `dataloader->SetSignalWeightExpression    ("weight1*weight2");`
   // -  for background: `dataloader->SetBackgroundWeightExpression("weight1*weight2");`
   //dataloader->SetBackgroundWeightExpression( "weight" );

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

   //dataloader->AddTarget("y",'F');
   // Tell the dataloader how to use the training and testing events
   //
   // If no numbers of events are given, half of the events in the tree are used
   // for training, and the other half for testing:
   //
   //    dataloader->PrepareTrainingAndTestTree( mycut, "SplitMode=random:!V" );
   //
   // To also specify the number of testing events, use:
   //
   //    dataloader->PrepareTrainingAndTestTree( mycut,
   //         "NSigTrain=3000:NBkgTrain=3000:NSigTest=3000:NBkgTest=3000:SplitMode=Random:!V" );
   dataloader->PrepareTrainingAndTestTree( mycuts, mycutb,
                                        "nTrain_Signal=1000:nTrain_Background=1000:SplitMode=Random:NormMode=NumEvents:!V" );

   // ### Book MVA methods
   //
   // Please lookup the various method configuration options in the corresponding cxx files, eg:
   // src/MethoCuts.cxx, etc, or here: http://tmva.sourceforge.net/optionRef.html
   // it is possible to preset ranges in the option string in which the cut optimisation should be done:
   // "...:CutRangeMin[2]=-1:CutRangeMax[2]=1"...", where [2] is the third input variable


   if(Use["GAN"]) {

      // Input Layout
      TString inputLayoutString("InputLayout=1|1|784##1|1|784");

      // Batch Layout
      TString batchLayoutString("BatchLayout=32|1|784##32|1|784");

      //General Layout
      TString layoutString ("Layout=RESHAPE|1|1|784|FLAT,DENSE|256|RELU,DENSE|512|RELU,DENSE|1024|RELU,DENSE|784|TANH##RESHAPE|1|1|784|FLAT,DENSE|512|RELU,DENSE|256|RELU,DENSE|1|SIGMOID");

      // Training strategies.
      TString training0("MaxEpochs=10000,GeneratorLearningRate=2e-4,GeneratorMomentum=0.9,GeneratorRepetitions=1,"
                        "GeneratorConvergenceSteps=20,GeneratorBatchSize=32,GeneratorTestRepetitions=10,"
                        "GeneratorWeightDecay=1e-4,GeneratorRegularization=L2,"
                        "GeneratorDropConfig=0.0+0.5+0.5+0.5, GeneratorMultithreading=True,"
 			                  "DiscriminatorLearningRate=2e-4,DiscriminatorMomentum=0.9,DiscriminatorRepetitions=1,"
                        "DiscriminatorConvergenceSteps=20,DiscriminatorBatchSize=32,DiscriminatorTestRepetitions=10,"
                        "DiscriminatorWeightDecay=1e-4,DiscriminatorRegularization=L2,"
                        "DiscriminatorDropConfig=0.0+0.5+0.5+0.5, DiscriminatorMultithreading=True");
      TString training1("MaxEpochs=10000,GeneratorLearningRate=2e-5,GeneratorMomentum=0.9,GeneratorRepetitions=1,"
                        "GeneratorConvergenceSteps=20,GeneratorBatchSize=32,GeneratorTestRepetitions=10,"
                        "GeneratorWeightDecay=2e-5,GeneratorRegularization=L2,"
                        "GeneratorDropConfig=0.0+0.0+0.0+0.0, GeneratorMultithreading=True,"
			                  "DiscriminatorLearningRate=1e-5,DiscriminatorMomentum=0.9,DiscriminatorRepetitions=1,"
                        "DiscriminatorConvergenceSteps=20,DiscriminatorBatchSize=32,DiscriminatorTestRepetitions=10,"
                        "DiscriminatorWeightDecay=1e-4,DiscriminatorRegularization=L2,"
                        "DropConfig=0.0+0.0+0.0+0.0, Multithreading=True");
      TString training2("MaxEpochs=10000,GeneratorLearningRate=2e-6,GeneratorMomentum=0.0,GeneratorRepetitions=1,"
                        "GeneratorConvergenceSteps=20,GeneratorBatchSize=32,GeneratorTestRepetitions=10,"
                        "GeneratorWeightDecay=2e-6,GeneratorRegularization=L2,"
                        "GeneratorDropConfig=0.0+0.0+0.0+0.0, GeneratorMultithreading=True,"
			                  "DiscriminatorLearningRate=1e-6,DiscriminatorMomentum=0.0,DiscriminatorRepetitions=1,"
                        "DiscriminatorConvergenceSteps=20, DiscriminatorBatchSize=32, DiscriminatorTestRepetitions=10,"
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

      TString cpuOptions = ganOptions + ":Architecture=CPU";
      factory->BookMethod(dataloader, TMVA::Types::kGAN, "GAN", cpuOptions);
   }


   // Now you can tell the factory to train, test, and evaluate the MVAs
   //
   // Train MVAs using the set of training events
   factory->TrainAllMethods();

   // Evaluate all MVAs using the set of test events
   factory->TestAllMethods();

   // Evaluate and compare performance of all configured MVAs
   factory->EvaluateAllMethods();

   // --------------------------------------------------------------

   // Save the output
   outputFile->Close();

   std::cout << "==> Wrote root file: " << outputFile->GetName() << std::endl;
   std::cout << "==> TMVAGeneration is done!" << std::endl;

   delete factory;
   delete dataloader;
   // Launch the GUI for the root macros
   if (!gROOT->IsBatch()) TMVA::TMVAGui( outfileName );

   return 0;
}

int main( int argc, char** argv )
{
   // Select methods (don't look at this code - not of interest)
   TString methodList;
   for (int i=1; i<argc; i++) {
      TString regMethod(argv[i]);
      if(regMethod=="-b" || regMethod=="--batch") continue;
      if (!methodList.IsNull()) methodList += TString(",");
      methodList += regMethod;
   }
   return TMVAGeneration(methodList);
}
