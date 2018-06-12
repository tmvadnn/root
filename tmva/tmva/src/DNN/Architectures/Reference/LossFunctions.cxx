// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 10/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

 ////////////////////////////////////////////////////////////
 // Implementation of the loss functions for the reference //
 // implementation.                                        //
 ////////////////////////////////////////////////////////////

#include "TMVA/DNN/Architectures/Reference.h"

namespace TMVA
{
namespace DNN
{
//______________________________________________________________________________
template <typename AReal>
AReal TReference<AReal>::MeanSquaredError(const TMatrixT<AReal> &Y, const TMatrixT<AReal> &output,
                                          const TMatrixT<AReal> &weights)
{
   size_t m,n;
   m = Y.GetNrows();
   n = Y.GetNcols();
   AReal result = 0.0;

   for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
         AReal dY = (Y(i,j) - output(i,j));
         result += weights(i, 0) * dY * dY;
      }
   }
   result /= static_cast<AReal>(m * n);
   return result;
}

//______________________________________________________________________________
template <typename AReal>
void TReference<AReal>::MeanSquaredErrorGradients(TMatrixT<AReal> &dY, const TMatrixT<AReal> &Y,
                                                  const TMatrixT<AReal> &output, const TMatrixT<AReal> &weights)
{
   size_t m,n;
   m = Y.GetNrows();
   n = Y.GetNcols();

   dY.Minus(Y, output);
   dY *= -2.0 / static_cast<AReal>(m * n);

   for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
         dY(i, j) *= weights(i, 0);
      }
   }
}

//______________________________________________________________________________
template <typename AReal>
AReal TReference<AReal>::CrossEntropy(const TMatrixT<AReal> &Y, const TMatrixT<AReal> &output,
                                      const TMatrixT<AReal> &weights)
{
   size_t m,n;
   m = Y.GetNrows();
   n = Y.GetNcols();
   AReal result = 0.0;

   for (size_t i = 0; i < m; i++) {
      AReal w = weights(i, 0);
      for (size_t j = 0; j < n; j++) {
         AReal sig = 1.0 / (1.0 + std::exp(-output(i,j)));
         result += w * (Y(i, j) * std::log(sig) + (1.0 - Y(i, j)) * std::log(1.0 - sig));
      }
   }
   result /= -static_cast<AReal>(m * n);
   return result;
}

//______________________________________________________________________________
template <typename AReal>
void TReference<AReal>::CrossEntropyGradients(TMatrixT<AReal> &dY, const TMatrixT<AReal> &Y,
                                              const TMatrixT<AReal> &output, const TMatrixT<AReal> &weights)
{
   size_t m,n;
   m = Y.GetNrows();
   n = Y.GetNcols();

   AReal norm = 1.0 / static_cast<AReal>(m * n);
   for (size_t i = 0; i < m; i++)
   {
      AReal w = weights(i, 0);
      for (size_t j = 0; j < n; j++)
      {
         AReal y   = Y(i,j);
         AReal sig = 1.0 / (1.0 + std::exp(-output(i,j)));
         dY(i, j) = norm * w * (sig - y);
      }
   }
}

//______________________________________________________________________________
template <typename AReal>
AReal TReference<AReal>::SoftmaxCrossEntropy(const TMatrixT<AReal> &Y, const TMatrixT<AReal> &output,
                                             const TMatrixT<AReal> &weights)
{
   size_t m,n;
   m = Y.GetNrows();
   n = Y.GetNcols();
   AReal result = 0.0;

   for (size_t i = 0; i < m; i++) {
      AReal sum = 0.0;
      AReal w = weights(i, 0);
      for (size_t j = 0; j < n; j++) {
         sum += exp(output(i,j));
      }
      for (size_t j = 0; j < n; j++) {
         result += w * Y(i, j) * log(exp(output(i, j)) / sum);
      }
   }
   result /= -static_cast<AReal>(m);
   return result;
}

//______________________________________________________________________________
template <typename AReal>
void TReference<AReal>::SoftmaxCrossEntropyGradients(TMatrixT<AReal> &dY, const TMatrixT<AReal> &Y,
                                                     const TMatrixT<AReal> &output, const TMatrixT<AReal> &weights)
{
   size_t m,n;
   m = Y.GetNrows();
   n = Y.GetNcols();
   AReal norm = 1.0 / m ;

   for (size_t i = 0; i < m; i++)
   {
      AReal sum  = 0.0;
      AReal sumY = 0.0;
      AReal w = weights(i, 0);
      for (size_t j = 0; j < n; j++) {
         sum  += exp(output(i,j));
         sumY += Y(i,j);
      }
      for (size_t j = 0; j < n; j++) {
         dY(i, j) = w * norm * (exp(output(i, j)) / sum * sumY - Y(i, j));
      }
   }
}

//______________________________________________________________________________
template <typename AReal>
AReal TReference<AReal>::KLDivergence(const TMatrixT<AReal> &Mean, const TMatrixT<AReal> &StandardDeviation,
                                      const TMatrixT<AReal> &weights)
{
   size_t m,n;
   m = Mean.GetNrows();
   n = Mean.GetNcols();

   AReal result = 0.0;

   for(size_t i=0; i<m; i++){
      for(size_t j=0; j<n; j++){
         AReal sum = 1 + StandardDeviation(i, j) - (Mean(i, j) * Mean(i, j)) - std::pow(std::exp(StandardDeviation(i, j)) , 2);
         result += weights(i,0) * sum;
      }
   }

   result /= static_cast<AReal>(m);

   return result;
}

template <typename AReal>
void TReference<AReal>::KLDivergenceGradients(TMatrixT<AReal> &dMean, TMatrixT<AReal> &dStandardDeviation,
                                               const TMatrixT<AReal> &Mean, const TMatrixT<AReal> &StandardDeviation, 
                                               const TMatrixT<AReal> &weights)
{
   size_t m,n;
   m = Mean.GetNrows();
   n = Mean.GetNcols();
   AReal norm = 1.0 / m ;

   for(size_t i = 0; i < m; i++){
      for(size_t j = 0; j < n; j++){
         dMean(i,j) = -2.0 * norm * Mean(i,j) * weights(i, 0);
         dStandardDeviation(i,j) = 1.0 - (2.0 * std::exp(2.0 * StandardDeviation(i,j)));
         dStandardDeviation(i,j) *= norm * weights(i, 0);
      }
   }
}

} // namespace DNN
} // namespace TMVA
