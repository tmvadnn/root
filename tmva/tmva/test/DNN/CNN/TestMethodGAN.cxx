// @(#)root/tmva/tmva/cnn:$Id$
// Author: Anushree Rankawat

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  :                                                                       *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Testing Method GAN                                                        *
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

#include "TestMethodGAN.h"
#include "TString.h"

int main()
{
   std::cout << "Testing Method GAN for CPU backend: " << std::endl;

   TString archCPU = "CPU";

   testMethodGAN_DNN(archCPU);
   testCreateNoisyMatrices();
   testCreateDiscriminatorFakeData();
   testCombineGAN();

}
