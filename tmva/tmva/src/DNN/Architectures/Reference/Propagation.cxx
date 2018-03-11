// @(#)root/tmva/tmva/dnn:$Id$ // Author: Simon Pfreundschuh 10/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/////////////////////////////////////////////////////////////////////
// Implementation of the functions required for the forward and    //
// backward propagation of activations through a neural network in //
// the reference implementation.                                   //
/////////////////////////////////////////////////////////////////////

#include "TMVA/DNN/Architectures/Reference.h"

namespace TMVA {
namespace DNN {

template <typename AReal>
void TReference<AReal>::MultiplyTranspose(TMatrixT<AReal> &output, const TMatrixT<AReal> &input,
                                          const TMatrixT<AReal> &weights)
{
   output.MultT(input, weights);
}

template <typename AReal>
void TReference<AReal>::AddRowWise(TMatrixT<AReal> &output, const TMatrixT<AReal> &biases)
{
   for (size_t i = 0; i < (size_t)output.GetNrows(); i++) {
      for (size_t j = 0; j < (size_t)output.GetNcols(); j++) {
         output(i, j) += biases(j, 0);
      }
   }
}

template <typename AReal>
void TReference<AReal>::Backward(TMatrixT<AReal> &activation_gradients_backward, TMatrixT<AReal> &weight_gradients,
                                 TMatrixT<AReal> &bias_gradients, TMatrixT<AReal> &df,
                                 const TMatrixT<AReal> &activation_gradients, const TMatrixT<AReal> &weights,
                                 const TMatrixT<AReal> &activations_backward)
{

   // Compute element-wise product.
   for (size_t i = 0; i < (size_t)df.GetNrows(); i++) {
      for (size_t j = 0; j < (size_t)df.GetNcols(); j++) {
         df(i, j) *= activation_gradients(i, j);
      }
   }

   // Activation gradients.
   if (activation_gradients_backward.GetNoElements() > 0) {
      activation_gradients_backward.Mult(df, weights);
   }

   // Weights gradients.
   if (weight_gradients.GetNoElements() > 0) {
      weight_gradients.TMult(df, activations_backward);
   }

   // Bias gradients.
   if (bias_gradients.GetNoElements() > 0) {
      for (size_t j = 0; j < (size_t)df.GetNcols(); j++) {
         AReal sum = 0.0;
         for (size_t i = 0; i < (size_t)df.GetNrows(); i++) {
            sum += df(i, j);
         }
         bias_gradients(j, 0) = sum;
      }
   }
}

template <typename AReal>
void TReference<AReal>::ScaleAdd(TMatrixT<AReal> &A, const TMatrixT<AReal> &B, AReal beta)
{
   for (size_t i = 0; i < (size_t)A.GetNrows(); i++) {
      for (size_t j = 0; j < (size_t)A.GetNcols(); j++) {
         A(i, j) += beta * B(i, j);
      }
   }
}

template <typename AReal>
void TReference<AReal>::Copy(TMatrixT<AReal> &A, const TMatrixT<AReal> &B)
{
   A = B;
}

template <typename AReal>
void TReference<AReal>::ScaleAdd(std::vector<TMatrixT<AReal>> &A, const std::vector<TMatrixT<AReal>> &B, AReal beta)
{
   for (size_t i = 0; i < A.size(); ++i) {
      ScaleAdd(A[i], B[i], beta);
   }
}

template <typename AReal>
void TReference<AReal>::Copy(std::vector<TMatrixT<AReal>> &A, const std::vector<TMatrixT<AReal>> &B)
{
   for (size_t i = 0; i < A.size(); ++i) {
      Copy(A[i], B[i]);
   }
}

template <typename AReal>
void TReference<AReal>::SumColumns(TMatrixT<AReal> &B, const TMatrixT<AReal> &A)
{
   B = 0.0;
   for (Int_t i = 0; i < A.GetNrows(); i++) {
      for (Int_t j = 0; j < A.GetNcols(); j++) {
         B(0, j) += A(i, j);
      }
   }
}

//______________________________________________________________________________
template <typename AReal>
void TReference<AReal>::Im2col(TMatrixT<AReal> &A, TMatrixT<AReal> &B, size_t imgHeight, size_t imgWidth,
                               size_t fltHeight, size_t fltWidth, size_t strideRows, size_t strideCols,
                               size_t zeroPaddingHeight, size_t zeroPaddingWidth)
{
   // image boudaries
   int imgHeightBound = imgHeight + zeroPaddingHeight - (fltHeight - 1) / 2 - 1;
   int imgWidthBound = imgWidth + zeroPaddingWidth - (fltWidth - 1) / 2 - 1;
   size_t currLocalView = 0;

   // convolution centers
   for (int i = -zeroPaddingHeight + fltHeight / 2; i <= imgHeightBound; i += strideRows) {
      for (int j = -zeroPaddingWidth + fltWidth / 2; j <= imgWidthBound; j += strideCols) {
         size_t currLocalViewPixel = 0;

         // within the local view
         for (int m = 0; m < B.GetNrows(); m++) {
            for (Int_t k = i - Int_t(fltHeight) / 2; k <= i + (Int_t(fltHeight) - 1) / 2; k++) {
               for (Int_t l = j - Int_t(fltWidth) / 2; l <= j + (Int_t(fltWidth) - 1) / 2; l++) {

                  // Check the boundaries
                  if (k < 0 || k >= Int_t(imgHeight) || l < 0 || l >= Int_t(imgWidth))
                     A(currLocalView, currLocalViewPixel++) = 0;
                  else
                     A(currLocalView, currLocalViewPixel++) = B(m, k * imgWidth + l);
               }
            }
         }

         currLocalView++;
      }
   }
}

//______________________________________________________________________________
template <typename AReal>
void TReference<AReal>::RotateWeights(TMatrixT<AReal> &A, const TMatrixT<AReal> &B, size_t filterDepth,
                                      size_t filterHeight, size_t filterWidth, size_t numFilters)
{
   size_t jump = filterHeight * filterWidth;
   for (size_t j = 0; j < filterDepth; j++) {
      for (size_t k = 0; k < numFilters; k++) {
         for (size_t i = 0; i < jump; i++) {
            A(j, k * jump + i) = B(k, ((j + 1) * jump - 1) - i);
         }
      }
   }
}

//______________________________________________________________________________
template <typename AReal>
void TReference<AReal>::AddConvBiases(TMatrixT<AReal> &output, const TMatrixT<AReal> &biases)
{
   for (size_t i = 0; i < (size_t)output.GetNrows(); i++) {
      for (size_t j = 0; j < (size_t)output.GetNcols(); j++) {
         output(i, j) += biases(i, 0);
      }
   }
}

//______________________________________________________________________________
template <typename AReal>
void TReference<AReal>::ConvLayerBackward(std::vector<TMatrixT<AReal>> &activation_gradients_backward,
                                          TMatrixT<AReal> &weight_gradients, TMatrixT<AReal> &bias_gradients,
                                          std::vector<TMatrixT<AReal>> &df,
                                          const std::vector<TMatrixT<AReal>> &activation_gradients,
                                          const TMatrixT<AReal> &weights,
                                          const std::vector<TMatrixT<AReal>> &activations_backward, size_t batchSize,
                                          size_t inputHeight, size_t inputWidth, size_t depth, size_t height,
                                          size_t width, size_t filterDepth, size_t filterHeight, size_t filterWidth,
                                          size_t nLocalViews)
{

   // Update derivatives
   size_t m, n;
   m = activation_gradients[0].GetNrows();
   n = activation_gradients[0].GetNcols();

   for (size_t i = 0; i < batchSize; i++) {
      for (size_t j = 0; j < (size_t)m; j++) {
         for (size_t k = 0; k < (size_t)n; k++) {
            df[i](j, k) *= activation_gradients[i](j, k);
         }
      }
   }

   // Calculate the activation gradients of the previous layer
   CalculateConvActivationGradients(activation_gradients_backward, df, weights, batchSize, inputHeight, inputWidth,
                                    depth, height, width, filterDepth, filterHeight, filterWidth);

   // Calculate the weight gradients
   CalculateConvWeightGradients(weight_gradients, df, activations_backward, batchSize, inputHeight, inputWidth, depth,
                                height, width, filterDepth, filterHeight, filterWidth, nLocalViews);

   // Calculate the bias gradients
   CalculateConvBiasGradients(bias_gradients, df, batchSize, depth, nLocalViews);
}

//______________________________________________________________________________
template <typename AReal>
void TReference<AReal>::CalculateConvActivationGradients(std::vector<TMatrixT<AReal>> &activation_gradients_backward,
                                                         const std::vector<TMatrixT<AReal>> &df,
                                                         const TMatrixT<AReal> &weights, size_t batchSize,
                                                         size_t inputHeight, size_t inputWidth, size_t depth,
                                                         size_t height, size_t width, size_t filterDepth,
                                                         size_t filterHeight, size_t filterWidth)
{
   if (activation_gradients_backward.size() == 0) return;
   // need to implement
#if 0
   // Transform the weights
   TMatrixT<AReal> rotWeights(filterDepth, depth * filterHeight * filterWidth);
   RotateWeights(rotWeights, weights, filterDepth, filterHeight, filterWidth, weights.GetNrows());

   // Calculate the zero paddings
   size_t tempZeroPaddingHeight = (size_t)(floor((inputHeight - height + filterHeight - 1) / 2));
   size_t tempZeroPaddingWidth = (size_t)(floor((inputWidth - width + filterWidth - 1) / 2));

   // Calculate the number of local views and the number of pixles in each view
   size_t tempNLocalViews = inputHeight * inputWidth;
   size_t tempNLocalViewPixels = depth * filterHeight * filterWidth;

   size_t tempStrideRows = 1;
   size_t tempStrideCols = 1;

   // An entire convolution follows
   for (size_t i = 0; i < batchSize; i++) {
      TMatrixT<AReal> dfTr(tempNLocalViews, tempNLocalViewPixels);
      Im2col(dfTr, df[i], inputHeight, inputWidth, filterHeight, filterWidth, tempStrideRows, tempStrideCols,
             tempZeroPaddingHeight, tempZeroPaddingWidth);

      activation_gradients_backward[i].MultT(rotWeights, dfTr);
   }
#endif
   return ;
}

//______________________________________________________________________________
template <typename AReal>
void TReference<AReal>::CalculateConvWeightGradients(TMatrixT<AReal> &weight_gradients,
                                                     const std::vector<TMatrixT<AReal>> &df,
                                                     const std::vector<TMatrixT<AReal>> &activations_backward,
                                                     size_t batchSize, size_t inputHeight, size_t inputWidth,
                                                     size_t depth, size_t height, size_t width, size_t filterDepth,
                                                     size_t filterHeight, size_t filterWidth, size_t nLocalViews)
{
   // reinitialize the weight gradients to 0
   for (Int_t i = 0; i < weight_gradients.GetNrows(); i++) {
      for (Int_t j = 0; j < weight_gradients.GetNcols(); j++) {
         weight_gradients(i, j) = 0;
      }
   }
#if 0
   for (size_t i = 0; i < batchSize; i++) {
      // Calculate the zero paddings
      size_t tempZeroPaddingHeight = (filterHeight - height + inputHeight - 1) / 2;
      size_t tempZeroPaddingWidth = (filterWidth - width + inputWidth - 1) / 2;

      size_t tempNLocalViews = filterHeight * filterWidth;
      size_t tempNLocalViewPixels = inputHeight * inputWidth;

      size_t tempStrideRows = 1;
      size_t tempStrideCols = 1;

      for (size_t j = 0; j < depth; j++) {

         // row matrix
         TMatrixT<AReal> rowDelta(1, nLocalViews);
         for (size_t k = 0; k < nLocalViews; k++) {
            rowDelta(0, k) = df[i](j, k);
         }

         // convolution
         TMatrixT<AReal> res(filterDepth, filterHeight * filterWidth);

         TMatrixT<AReal> rowDeltaTr(tempNLocalViews, tempNLocalViewPixels);
         Im2col(rowDeltaTr, rowDelta, height, width, inputHeight, inputWidth, tempStrideRows, tempStrideCols,
                tempZeroPaddingHeight, tempZeroPaddingWidth);

         res.MultT(activations_backward[i], rowDeltaTr);

         for (size_t k = 0; k < filterDepth; k++) {
            for (size_t l = 0; l < filterHeight * filterWidth; l++) {
               weight_gradients(j, k * (filterHeight * filterWidth) + l) += res(k, (tempNLocalViews - 1) - l);
            }
         }
      }
   }
#endif  
}

//______________________________________________________________________________
template <typename AReal>
void TReference<AReal>::CalculateConvBiasGradients(TMatrixT<AReal> &bias_gradients, const std::vector<TMatrixT<AReal>> &df,
                                                   size_t batchSize, size_t depth, size_t nLocalViews)
{
   for (size_t i = 0; i < depth; i++) {
      AReal sum = 0;
      for (size_t j = 0; j < nLocalViews; j++) {
         for (size_t k = 0; k < batchSize; k++) {
            sum += df[k](i, j);
         }
      }
      bias_gradients(i, 0) = sum;
   }
}

//______________________________________________________________________________
template<typename AReal>
void
TReference<AReal>::Downsample(CNN::TPoolLayer <TReference> *layer, const TMatrixT <AReal> &input, size_t batchIndex)
{
   // unpack frequently used layer internals
   size_t inHeight = layer->GetInputHeight(), inWidth = layer->GetInputWidth();
   size_t fltHeight = layer->GetFrameHeight(), fltWidth = layer->GetFrameWidth();
   size_t strideRows = layer->GetStrideRows(), strideCols = layer->GetStrideCols();
   size_t outWidth = layer->GetWidth();
   std::string method = layer->GetMethod();

   // image boundaries
   int imgHeightBound = inHeight - (fltHeight - 1) / 2 - 1;
   int imgWidthBound = inWidth - (fltWidth - 1) / 2 - 1;
   size_t outRow = 0, outCol = 0;

   // centers
   for (int i = fltHeight / 2; i <= imgHeightBound; i += strideRows) {
      outCol = 0;
      for (int j = fltWidth / 2; j <= imgWidthBound; j += strideCols) {
         // within local views
         for (int m = 0; m < (Int_t) input.GetNrows(); m++) {
            AReal value = 0.0;
            if (method == "max") {
               value = -std::numeric_limits<AReal>::max();

               for (int k = i - fltHeight / 2; k <= Int_t(i + (fltHeight - 1) / 2); k++) {
                  for (int l = j - fltWidth / 2; l <= Int_t(j + (fltWidth - 1) / 2); l++) {
                     if (input(m, k * inWidth + l) > value) {
                        value = input(m, k * inWidth + l);
                        layer->GetIndexMatrix()[batchIndex](m, outCol + outRow * outWidth) = k * inWidth + l;
                     }
                  }
               }
            }
            else if (method == "avg") {
               unsigned int counter = 0;
               for (int k = i - fltHeight / 2; k <= Int_t(i + (fltHeight - 1) / 2); k++) {
                  for (int l = j - fltWidth / 2; l <= Int_t(j + (fltWidth - 1) / 2); l++) {
                     value += input(m, k * inWidth + l);
                     counter += 1;
                  }
               }
               value /= counter;
            }
            else {
               throw std::invalid_argument("The method argument can be either 'max' or 'avg', not " + method);
            }
            layer->GetOutputAt(batchIndex)(m, outCol + outRow * outWidth) = value;
         }
         outCol++;
      }
      outRow++;
   }
}

//______________________________________________________________________________
template <typename AReal>
void TReference<AReal>::PoolLayerBackward(std::vector<TMatrixT<AReal>> &activationGradientsBackward,
                                           CNN::TPoolLayer<TReference> const * const layer)
{
   size_t fHeight = layer->GetFrameHeight(), fWidth = layer->GetFrameWidth();
   size_t inHeight = layer->GetInputHeight(), inWidth = layer->GetInputWidth();
   size_t strideRows = layer->GetStrideRows(), strideCols = layer->GetStrideCols();
   int inHeightBound = inHeight - (fHeight - 1) / 2 - 1;
   int inWidthBound = inWidth - (fWidth - 1) / 2 - 1;

   size_t currLocalView = 0;

   for (size_t b = 0; b < layer->GetBatchSize(); b++) {
      for (size_t d = 0; d < layer->GetDepth(); d++) {

         // initialize to zeros
         for (size_t t = 0; t < (size_t)activationGradientsBackward[b].GetNcols(); t++) {
            activationGradientsBackward[b](d, t) = 0;
         }

         // centers
         for (int i = fHeight / 2; i <= inHeightBound; i += strideRows) {
            for (int j = fWidth / 2; j <= inWidthBound; j += strideCols) {
               AReal grad =  layer->GetActivationGradients()[b](d, currLocalView);
               if (layer->GetMethod() == "avg") {
                  for (int k = i - fHeight / 2; k <= Int_t(i + (fHeight - 1) / 2); k++) {
                     for (int l = j - fWidth / 2; l <= Int_t(j + (fWidth - 1) / 2); l++) {
                        activationGradientsBackward[b](d, k * inWidth + l) += grad / (fHeight * fWidth);
                     }
                  }
               }
               else {
                  size_t winningIdx = layer->GetIndexMatrix()[b](d, currLocalView);
                  activationGradientsBackward[b](d, winningIdx) += grad;
               }
               currLocalView++;
            }
         }
      }
   }
}

//______________________________________________________________________________
template <typename AReal>
void TReference<AReal>::Reshape(TMatrixT<AReal> &A, const TMatrixT<AReal> &B)
{
   auto nColsA = A.GetNcols();
   auto nColsB = B.GetNcols();

   for (Int_t i = 0; i < A.GetNrows(); i++) {
      for (Int_t j = 0; j < A.GetNcols(); j++) {
         auto nElem = i * nColsA + j;
         A(i, j) = B(nElem / nColsB, (nElem - 1) % nColsB);
      }
   }
}

//______________________________________________________________________________
template <typename AReal>
void TReference<AReal>::Flatten(TMatrixT<AReal> &A, const std::vector<TMatrixT<AReal>> &B, size_t size, size_t nRows,
                                size_t nCols)
{
   for (size_t i = 0; i < (size_t)size; i++) {
      for (size_t j = 0; j < (size_t)nRows; j++) {
         for (size_t k = 0; k < (size_t)nCols; k++) {
            A(i, j * nCols + k) = B[i](j, k);
         }
      }
   }
}

//______________________________________________________________________________
template <typename AReal>
void TReference<AReal>::Deflatten(std::vector<TMatrixT<AReal>> &A, const TMatrixT<AReal> &B, size_t size, size_t nRows,
                                  size_t nCols)
{
   for (size_t i = 0; i < (size_t)size; i++) {
      for (size_t j = 0; j < (size_t)nRows; j++) {
         for (size_t k = 0; k < (size_t)nCols; k++) {
            A[i](j, k) = B(i, j * nCols + k);
         }
      }
   }
}

//______________________________________________________________________________
template <typename AReal>
void TReference<AReal>::Rearrange(std::vector<TMatrixT<AReal>> &out, const std::vector<TMatrixT<AReal>> &in)
{
   // B x T x D out --- T x B x D in*/
   auto B = out.size();
   auto T = out[0].GetNrows();
   auto D = out[0].GetNcols();
   if ((T != (Int_t)in.size()) || (Int_t(B) != in[0].GetNrows()) || (D != in[0].GetNcols())) {
      std::cout << "Incompatible Dimensions\n"
                << in.size() << "x" << in[0].GetNrows() << "x" << in[0].GetNcols() << " --> " << B << "x" << T << "x"
                << D << "\n";
      return;
   }
   for (size_t i = 0; i < B; ++i) {
      for (Int_t j = 0; j < T; ++j) {
         for (Int_t k = 0; k < D; ++k) {
            out[i](j, k) = in[j](i, k);
         }
      }
   }
   return;
}

} // namespace DNN
} // namespace TMVA
