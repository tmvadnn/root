// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 05/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

///////////////////////////////////////////////////////////////////
// Definition of the TCuda architecture class, which provides an //
// implementation of the low-level functionality for neural      //
// networks for the CUDA computing architectures.                //
///////////////////////////////////////////////////////////////////

#ifndef TMVA_DNN_ARCHITECTURES_CUDA
#define TMVA_DNN_ARCHITECTURES_CUDA

#include "TMVA/DNN/Functions.h"


#include "cuda.h"
#include "Cuda/CudaBuffers.h"
#include "Cuda/CudaMatrix.h"
#include "TMVA/DNN/DataLoader.h"
#include <utility>
#include <vector>

class TRandom;

namespace TMVA
{
namespace DNN
{

/** The TCuda architecture class.
 *
 * Low-level interface class for CUDA computing architectures. Contains as
 * public types the declaration of the scalar, matrix and buffer types
 * for this architecture as well as the remaining functions in the low-level
 * interface in the form of static members.
 */
template<typename AFloat = Real_t>
class TCuda
{
private:
   static TRandom * fgRandomGen;
public:

    using Scalar_t       = AFloat;
    using Matrix_t       = TCudaMatrix<AFloat>;
    using DeviceBuffer_t = TCudaDeviceBuffer<AFloat>;
    using HostBuffer_t   = TCudaHostBuffer<AFloat>;

   //____________________________________________________________________________
   //
   // Propagation
   //____________________________________________________________________________

   /** @name Forward Propagation
    * Low-level functions required for the forward propagation of activations
    * through the network.
    */
   ///@{
   /** Matrix-multiply \p input with the transpose of \pweights and
    *  write the results into \p output. */
   static void MultiplyTranspose(TCudaMatrix<AFloat> &output,
                                 const TCudaMatrix<AFloat> &input,
                                 const TCudaMatrix<AFloat> &weights);
   /** Add the vectors biases row-wise to the matrix output */
   static void AddRowWise(TCudaMatrix<AFloat> &output,
                          const TCudaMatrix<AFloat> &biases);
   ///@}

   /** @name Backward Propagation
    * Low-level functions required for the forward propagation of activations
    * through the network.
    */
   ///@{
   /** Perform the complete backward propagation step. If the provided
    *  \p activationGradientsBackward matrix is not empty, compute the
    *  gradients of the objective function with respect to the activations
    *  of the previous layer (backward direction).
    *  Also compute the weight and the bias gradients. Modifies the values
    *  in \p df and thus produces only a valid result, if it is applied the
    *  first time after the corresponding forward propagation has been per-
    *  formed. */
   static void Backward(TCudaMatrix<AFloat> & activationGradientsBackward,
                        TCudaMatrix<AFloat> & weightGradients,
                        TCudaMatrix<AFloat> & biasGradients,
                        TCudaMatrix<AFloat> & df,
                        const TCudaMatrix<AFloat> & activationGradients,
                        const TCudaMatrix<AFloat> & weights,
                        const TCudaMatrix<AFloat> & activationBackward);
   /** Backward pass for Recurrent Networks */
  static Matrix_t & RecurrentLayerBackward(TCudaMatrix<AFloat> & state_gradients_backward, // BxH
                                           TCudaMatrix<AFloat> & input_weight_gradients,
                                           TCudaMatrix<AFloat> & state_weight_gradients,
                                           TCudaMatrix<AFloat> & bias_gradients,
                                           TCudaMatrix<AFloat> & df, //DxH
                                           const TCudaMatrix<AFloat> & state, // BxH
                                           const TCudaMatrix<AFloat> & weights_input, // HxD 
                                           const TCudaMatrix<AFloat> & weights_state, // HxH
                                           const TCudaMatrix<AFloat> & input,  // BxD
                                           TCudaMatrix<AFloat> & input_gradient);
   /** Adds a the elements in matrix B scaled by c to the elements in
    *  the matrix A. This is required for the weight update in the gradient
    *  descent step.*/
   static void ScaleAdd(TCudaMatrix<AFloat> & A,
                        const TCudaMatrix<AFloat> & B,
                        Scalar_t beta = 1.0);
   /** Copy the elements of matrix A into matrix B. */
   static void Copy(TCudaMatrix<AFloat> & B,
                    const TCudaMatrix<AFloat> & A);

   // copy from another type of matrix
   template<typename AMatrix_t>
   static void CopyDiffArch(TCudaMatrix<Scalar_t> & B, const AMatrix_t & A); 


   /** Above functions extended to vectors */
   static void ScaleAdd(std::vector<TCudaMatrix<Scalar_t>> & A,
                        const std::vector<TCudaMatrix<Scalar_t>> & B,
                        Scalar_t beta = 1.0);

   static void Copy(std::vector<TCudaMatrix<Scalar_t>> & A,
                    const std::vector<TCudaMatrix<Scalar_t>> & B);

   // copy from another architecture
   template<typename AMatrix_t>
   static void CopyDiffArch(std::vector<TCudaMatrix<Scalar_t>> & A,
                    const std::vector<AMatrix_t> & B);


   ///@}

   //____________________________________________________________________________
   //
   // Activation Functions
   //____________________________________________________________________________

   /** @name Activation Functions
    * For each activation function, the low-level interface contains two routines.
    * One that applies the acitvation function to a matrix and one that evaluate
    * the derivatives of the activation function at the elements of a given matrix
    * and writes the results into the result matrix.
    */
   ///@{
   static void Identity(TCudaMatrix<AFloat> & B);
   static void IdentityDerivative(TCudaMatrix<AFloat> & B,
                                  const TCudaMatrix<AFloat> & A);

   static void Relu(TCudaMatrix<AFloat> & B);
   static void ReluDerivative(TCudaMatrix<AFloat> & B,
                              const TCudaMatrix<AFloat> & A);

   static void Sigmoid(TCudaMatrix<AFloat> & B);
   static void SigmoidDerivative(TCudaMatrix<AFloat> & B,
                                 const TCudaMatrix<AFloat> & A);

   static void Tanh(TCudaMatrix<AFloat> & B);
   static void TanhDerivative(TCudaMatrix<AFloat> & B,
                              const TCudaMatrix<AFloat> & A);

   static void SymmetricRelu(TCudaMatrix<AFloat> & B);
   static void SymmetricReluDerivative(TCudaMatrix<AFloat> & B,
                                       const TCudaMatrix<AFloat> & A);

   static void SoftSign(TCudaMatrix<AFloat> & B);
   static void SoftSignDerivative(TCudaMatrix<AFloat> & B,
                                  const TCudaMatrix<AFloat> & A);

   static void Gauss(TCudaMatrix<AFloat> & B);
   static void GaussDerivative(TCudaMatrix<AFloat> & B,
                               const TCudaMatrix<AFloat> & A);
   ///@}

   //____________________________________________________________________________
   //
   // Loss Functions
   //____________________________________________________________________________

   /** @name Loss Functions
    * Loss functions compute a scalar value given the \p output of the network
    * for a given training input and the expected network prediction \p Y that
    * quantifies the quality of the prediction. For each function also a routing
    * that computes the gradients (suffixed by Gradients) must be provided for
    * the starting of the backpropagation algorithm.
    */
   ///@{

   static AFloat MeanSquaredError(const TCudaMatrix<AFloat> &Y, const TCudaMatrix<AFloat> &output,
                                  const TCudaMatrix<AFloat> &weights);
   static void MeanSquaredErrorGradients(TCudaMatrix<AFloat> &dY, const TCudaMatrix<AFloat> &Y,
                                         const TCudaMatrix<AFloat> &output, const TCudaMatrix<AFloat> &weights);

   /** Sigmoid transformation is implicitly applied, thus \p output should
    *  hold the linear activations of the last layer in the net. */
   static AFloat CrossEntropy(const TCudaMatrix<AFloat> &Y, const TCudaMatrix<AFloat> &output,
                              const TCudaMatrix<AFloat> &weights);

   static void CrossEntropyGradients(TCudaMatrix<AFloat> &dY, const TCudaMatrix<AFloat> &Y,
                                     const TCudaMatrix<AFloat> &output, const TCudaMatrix<AFloat> &weights);

   /** Softmax transformation is implicitly applied, thus \p output should
    *  hold the linear activations of the last layer in the net. */
   static AFloat SoftmaxCrossEntropy(const TCudaMatrix<AFloat> &Y, const TCudaMatrix<AFloat> &output,
                                     const TCudaMatrix<AFloat> &weights);
   static void SoftmaxCrossEntropyGradients(TCudaMatrix<AFloat> &dY, const TCudaMatrix<AFloat> &Y,
                                            const TCudaMatrix<AFloat> &output, const TCudaMatrix<AFloat> &weights);
   ///@}

   //____________________________________________________________________________
   //
   // Output Functions
   //____________________________________________________________________________

   /** @name Output Functions
    * Output functions transform the activations \p output of the
    * output layer in the network to a valid prediction \p YHat for
    * the desired usage of the network, e.g.  the identity function
    * for regression or the sigmoid transformation for two-class
    * classification.
    */
   ///@{
   static void Sigmoid(TCudaMatrix<AFloat> &YHat,
                       const TCudaMatrix<AFloat> & );
   static void Softmax(TCudaMatrix<AFloat> &YHat,
                       const TCudaMatrix<AFloat> & );
   ///@}

   //____________________________________________________________________________
   //
   // Regularization
   //____________________________________________________________________________

   /** @name Regularization
    * For each regularization type two functions are required, one named
    * <tt><Type>Regularization</tt> that evaluates the corresponding
    * regularization functional for a given weight matrix and the
    * <tt>Add<Type>RegularizationGradients</tt>, that adds the regularization
    * component in the gradients to the provided matrix.
    */
   ///@{

   static AFloat L1Regularization(const TCudaMatrix<AFloat> & W);
   static void AddL1RegularizationGradients(TCudaMatrix<AFloat> & A,
                                            const TCudaMatrix<AFloat> & W,
                                            AFloat weightDecay);

   static AFloat L2Regularization(const TCudaMatrix<AFloat> & W);
   static void AddL2RegularizationGradients(TCudaMatrix<AFloat> & A,
                                            const TCudaMatrix<AFloat> & W,
                                            AFloat weightDecay);
   ///@}

   //____________________________________________________________________________
   //
   // Initialization
   //____________________________________________________________________________

   /** @name Initialization
    * For each initialization method, one function in the low-level interface
    * is provided. The naming scheme is <p>Initialize<Type></p> for a given
    * initialization method Type.
    */
   ///@{

   static void InitializeGauss(TCudaMatrix<AFloat> & A);
   static void InitializeUniform(TCudaMatrix<AFloat> & A);
   static void InitializeIdentity(TCudaMatrix<AFloat> & A);
   static void InitializeZero(TCudaMatrix<AFloat> & A);
   static void InitializeGlorotUniform(TCudaMatrix<AFloat> & A);
   static void InitializeGlorotNormal(TCudaMatrix<AFloat> & A);
   // return static instance of random generator used for initialization
   // if generator does not exist it is created the first time with a random seed (e.g. seed = 0)
   static TRandom & GetRandomGenerator(); 
   // set random seed for the static geenrator
   // if the static geneerator does not exists it is created
   static void SetRandomSeed(size_t seed); 


   ///@}

   //____________________________________________________________________________
   //
   // Dropout
   //____________________________________________________________________________

   /** @name Dropout
    */
   ///@{

   /** Apply dropout with activation probability \p p to the given
    *  matrix \p A and scale the result by reciprocal of \p p. */
   static void Dropout(TCudaMatrix<AFloat> & A, AFloat p);

   ///@}

   //____________________________________________________________________________
   //
   //  Convolutional Layer Propagation
   //____________________________________________________________________________

   /** @name Forward Propagation in Convolutional Layer
    */
   ///@{

   /** Transform the matrix \p B in local view format, suitable for
    *  convolution, and store it in matrix \p A. */
   static void Im2col(TCudaMatrix<AFloat> &A, const TCudaMatrix<AFloat> &B, size_t imgHeight, size_t imgWidth,
                      size_t fltHeight, size_t fltWidth, size_t strideRows, size_t strideCols, size_t zeroPaddingHeight,
                      size_t zeroPaddingWidth);

   static void Im2colIndices(std::vector<int> &V, const TCudaMatrix<AFloat> &B, size_t nLocalViews, size_t imgHeight, size_t imgWidth, size_t fltHeight,
                      size_t fltWidth, size_t strideRows, size_t strideCols, size_t zeroPaddingHeight,
                             size_t zeroPaddingWidth) {}
   static void Im2colFast(TCudaMatrix<AFloat> &A, const TCudaMatrix<AFloat> &B, const std::vector<int> & V) {}


   /** Rotates the matrix \p B, which is representing a weights,
    *  and stores them in the matrix \p A. */
   static void RotateWeights(TCudaMatrix<AFloat> &A, const TCudaMatrix<AFloat> &B, size_t filterDepth,
                             size_t filterHeight, size_t filterWidth, size_t numFilters);

   /** Add the biases in the Convolutional Layer.  */
   static void AddConvBiases(TCudaMatrix<AFloat> &output, const TCudaMatrix<AFloat> &biases);

   ///@}
   /** Forward propagation in the Convolutional layer */
   static void ConvLayerForward(std::vector<TCudaMatrix<AFloat>> & output, std::vector<TCudaMatrix<AFloat>> & derivatives,
                                const std::vector<TCudaMatrix<AFloat>> &input,
                                const TCudaMatrix<Scalar_t> & weights, const TCudaMatrix<Scalar_t> & biases,
                                EActivationFunction func, const std::vector<int> & vIndices,
                                size_t nlocalViews, size_t nlocalViewPixels,
                                Scalar_t dropoutProbability, bool applyDropout) {}

   /** @name Backward Propagation in Convolutional Layer
    */
   ///@{

   /** Perform the complete backward propagation step in a Convolutional Layer.
    *  If the provided \p activationGradientsBackward matrix is not empty, compute the
    *  gradients of the objective function with respect to the activations
    *  of the previous layer (backward direction).
    *  Also compute the weight and the bias gradients. Modifies the values
    *  in \p df and thus produces only a valid result, if it is applied the
    *  first time after the corresponding forward propagation has been per-
    *  formed. */
   static void ConvLayerBackward(std::vector<TCudaMatrix<AFloat>> &activationGradientsBackward,
                                 TCudaMatrix<AFloat> &weightGradients, TCudaMatrix<AFloat> &biasGradients,
                                 std::vector<TCudaMatrix<AFloat>> &df,
                                 const std::vector<TCudaMatrix<AFloat>> &activationGradients,
                                 const TCudaMatrix<AFloat> &weights,
                                 const std::vector<TCudaMatrix<AFloat>> &activationBackward, size_t batchSize,
                                 size_t inputHeight, size_t inputWidth, size_t depth, size_t height, size_t width,
                                 size_t filterDepth, size_t filterHeight, size_t filterWidth, size_t nLocalViews);

   /** Utility function for calculating the activation gradients of the layer
    *  before the convolutional layer. */
   static void CalculateConvActivationGradients(std::vector<TCudaMatrix<AFloat>> &activationGradientsBackward,
                                                std::vector<TCudaMatrix<AFloat>> &df,
                                                const TCudaMatrix<AFloat> &weights, size_t batchSize,
                                                size_t inputHeight, size_t inputWidth, size_t depth, size_t height,
                                                size_t width, size_t filterDepth, size_t filterHeight,
                                                size_t filterWidth);

   /** Utility function for calculating the weight gradients of the convolutional
    * layer. */
   static void CalculateConvWeightGradients(TCudaMatrix<AFloat> &weightGradients, std::vector<TCudaMatrix<AFloat>> &df,
                                            const std::vector<TCudaMatrix<AFloat>> &activations_backward,
                                            size_t batchSize, size_t inputHeight, size_t inputWidth, size_t depth,
                                            size_t height, size_t width, size_t filterDepth, size_t filterHeight,
                                            size_t filterWidth, size_t nLocalViews);

   /** Utility function for calculating the bias gradients of the convolutional
    *  layer */
   static void CalculateConvBiasGradients(TCudaMatrix<AFloat> &biasGradients, std::vector<TCudaMatrix<AFloat>> &df,
                                          size_t batchSize, size_t depth, size_t nLocalViews);

   ///@}

   //____________________________________________________________________________
   //
   //  Max Pooling Layer Propagation
   //____________________________________________________________________________
   /** @name Forward Propagation in Max Pooling Layer
    */
   ///@{

   /** Downsample the matrix \p C to the matrix \p A, using max
    *  operation, such that the winning indices are stored in matrix
    *  \p B. */
   static void Downsample(TCudaMatrix<AFloat> &A, TCudaMatrix<AFloat> &B, const TCudaMatrix<AFloat> &C,
                          size_t imgHeight, size_t imgWidth, size_t fltHeight, size_t fltWidth,
                          size_t strideRows, size_t strideCols);
   ///@}

   /** @name Backward Propagation in Max Pooling Layer
    */
   ///@{
       
   /** Perform the complete backward propagation step in a Pooling Layer. Based on the
    *  winning idices stored in the index matrix, it just forwards the actiovation
    *  gradients to the previous layer. */
   static void MaxPoolLayerBackward(std::vector<TCudaMatrix<AFloat>> &activationGradientsBackward,
                                    const std::vector<TCudaMatrix<AFloat>> &activationGradients,
                                    const std::vector<TCudaMatrix<AFloat>> &indexMatrix, size_t batchSize, size_t depth,
                                    size_t nLocalViews);

   ///@}

   //____________________________________________________________________________
   //
   //  Zero Padding Layer Propagation
   //____________________________________________________________________________
   /** @name Forward Propagation in Zero Padding Layer
    */
   ///@{

  /** Zero Pad the matrix \p B to the matrix \p A, using the
    *  padding dimensions specified.
    *   */
   static void ZeroPad2DForward(TCudaMatrix<AFloat> &A, const TCudaMatrix<AFloat> &B, 
                                size_t topPad, size_t bottomPad, size_t leftPad,
                                size_t rightPad, size_t outputHeight, size_t outputWidth);

   ///@}

   /** @name Backward Propagation in Zero Padding Layer
    */
   ///@{

   /** Perform the complete backward propagation step in a Zero Padding Layer. The gradients
    *  at the padded positions get discarded. */
   static void ZeroPad2DBackward(std::vector<TCudaMatrix<AFloat>> &activationGradientsBackward,
                                 const std::vector<TCudaMatrix<AFloat>> &activationGradients,
                                 size_t topPad, size_t bottomPad, size_t leftPad,
                                 size_t rightPad, size_t outputHeight, size_t outputWidth,
                                 size_t batchSize, size_t depth);
   ///@}

   //____________________________________________________________________________
   //
   //  Reshape Layer Propagation
   //____________________________________________________________________________
   /** @name Forward and Backward Propagation in Reshape Layer
    */
   ///@{

   /** Transform the matrix \p B to a matrix with different dimensions \p A */
   static void Reshape(TCudaMatrix<AFloat> &A, const TCudaMatrix<AFloat> &B);

   /** Flattens the tensor \p B, such that each matrix, is stretched in
    *  one row, resulting with a matrix \p A. */
   static void Flatten(TCudaMatrix<AFloat> &A, const std::vector<TCudaMatrix<AFloat>> &B, size_t size, size_t nRows,
                       size_t nCols);

   /** Transforms each row of \p B to a matrix and stores it in the tensor \p B. */
   static void Deflatten(std::vector<TCudaMatrix<AFloat>> &A, const TCudaMatrix<AFloat> &B, size_t index, size_t nRows,
                         size_t nCols);
   /** Rearrage data accoring to time fill B x T x D out with T x B x D matrix in*/
   static void Rearrange(std::vector<TCudaMatrix<AFloat>> &out, const std::vector<TCudaMatrix<AFloat>> &in); 

   ///@}

   //____________________________________________________________________________
   //
   // Additional Arithmetic Functions
   //____________________________________________________________________________

   /** @name Additional Arithmetic Functions
    *
    * Additional arithmetic on CUDA matrices  used to implement the low-level
    * interface.
    */
   ///@{

   /** Standard multiplication of two matrices \p A and \p B with the result being
    *  written into C.
    */
   static void Multiply(TCudaMatrix<AFloat> & C,
                        const TCudaMatrix<AFloat> & A,
                        const TCudaMatrix<AFloat> & B);
   /** Matrix multiplication of two matrices \p A and \p B^T (transposed) with the
    *  result being written into C.
    */
   static void TransposeMultiply(TCudaMatrix<AFloat> & output,
                                 const TCudaMatrix<AFloat> & input,
                                 const TCudaMatrix<AFloat> & Weights);
   /** In-place Hadamard (element-wise) product of matrices \p A and \p B
    *  with the result being written into \p A.
    */
   static void Hadamard(TCudaMatrix<AFloat> & A, const TCudaMatrix<AFloat> & B);

   /** Sum columns of (m x n) matrixx \p A and write the results into the first
    * m elements in \p A.
    */
   static void SumColumns(TCudaMatrix<AFloat> & B, const TCudaMatrix<AFloat> & A);

   /** Compute the sum of all elements in \p A */
   static AFloat Sum(const TCudaMatrix<AFloat> &A);
};

//____________________________________________________________________________
template <typename AFloat>
template <typename AMatrix_t>
void TCuda<AFloat>::CopyDiffArch(TCudaMatrix<AFloat> &B,
                        const AMatrix_t &A)
{
   // copy from another architecture using the reference one
   // this is not very efficient since creates temporary objects
   TMatrixT<AFloat> tmp = A;
   Copy(B, TCudaMatrix<AFloat>(tmp) ); 
}

//____________________________________________________________________________
template <typename AFloat>
template <typename AMatrix_t>
void TCuda<AFloat>::CopyDiffArch(std::vector<TCudaMatrix<AFloat>> &B,
                            const std::vector<AMatrix_t> &A)
{
   for (size_t i = 0; i < B.size(); ++i) {
      CopyDiffArch(B[i], A[i]);
   }
}

} // namespace DNN
} // namespace TMVA

#endif
