// @(#)root/tmva $Id$
// Author: Vladimir Ilievski

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  :                                                                       *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Testing Flatten function for Reference backend                            *
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

////////////////////////////////////////////////////////////////////
// Testing Flatten/Deflatten on the Reference architecture        //
////////////////////////////////////////////////////////////////////

#include <iostream>

#include "TMVA/DNN/Architectures/Reference.h"
#include "TestReshape.h"

using namespace TMVA::DNN;
using namespace TMVA::DNN::CNN;
using Matrix_t = typename TReference<double>::Matrix_t;


int main()
{
   using Scalar_t = Double_t;
   std::cout << "Testing Flatten/Deflatten on the Reference architecture:" << std::endl;

   bool status = true;

   std::cout << "Test Reshape: " << std::endl;
   status &= testReshape<TReference<Scalar_t>>();
   if (!status) {
      std::cerr << "ERROR - testReshape failed " << std::endl;
      return 1;
   }

   std::cout << "Test Flatten: " << std::endl;
   status &= testFlatten<TReference<Scalar_t>>();
   if (!status) {
      std::cerr << "ERROR - testFlatten failed " << std::endl;
      return 1;
   }

   std::cout << "Test Deflatten: " << std::endl;
   status &= testDeflatten<TReference<Scalar_t>>();
   if (!status) {
      std::cerr << "ERROR - testDeflatten failed " << std::endl;
      return 1;
   }
}
